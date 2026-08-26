from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Condition
from .recurrence import ChunkedSelectiveScan, ScanResult, selective_scan_reference


def _linear_without_bias(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    method = getattr(module, "forward_without_bias", None)
    if method is not None:
        return method(inputs)
    return F.linear(inputs, module.weight, None)


def _causal_convolution(mixer: nn.Module, inputs: torch.Tensor, length: int) -> torch.Tensor:
    pass
    if inputs.is_cuda:
        from einops import rearrange
        from causal_conv1d import causal_conv1d_fn

        return causal_conv1d_fn(
            x=inputs,
            weight=rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
            bias=mixer.conv1d.bias,
            activation=mixer.activation,
        )
    return mixer.act(mixer.conv1d(inputs)[..., :length])


class FusedScanMambaMixer(nn.Module):
    pass

    def __init__(self, base_mixer: nn.Module) -> None:
        super().__init__()
        self.base = base_mixer

    @property
    def layer_idx(self) -> int | None:
        return getattr(self.base, "layer_idx", None)

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("training mixer expects full sequences")
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        batch, length, _ = hidden_states.shape
        mixer = self.base
        xz = mixer.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = _causal_convolution(mixer, x, length)
        x_dbl = mixer.x_proj(x.transpose(1, 2).reshape(batch * length, -1))
        dt, B, C = torch.split(x_dbl, [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1)
        dt = _linear_without_bias(mixer.dt_proj, dt)
        dt = dt.reshape(batch, length, mixer.d_inner).transpose(1, 2).contiguous()
        B = B.reshape(batch, length, mixer.d_state).transpose(1, 2).contiguous()
        C = C.reshape(batch, length, mixer.d_state).transpose(1, 2).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            -torch.exp(mixer.A_log.float()),
            B,
            C,
            mixer.D.float(),
            z=z,
            delta_bias=mixer.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return mixer.out_proj(y.transpose(1, 2))


class ProjectedMambaMixer(nn.Module):
    pass

    def __init__(
        self,
        base_mixer: nn.Module,
        *,
        condition: Condition | str,
        radius: float,
        epsilon: float = 1e-6,
        alpha: float = 1.0,
        scan_chunk_size: int = 4,
        compile_scan: bool = False,
        checkpoint_scan_chunks: bool = False,
    ) -> None:
        super().__init__()
        self.base = base_mixer
        self.condition = Condition(condition)
        self.epsilon = epsilon
        self.alpha = alpha
        radius_device = getattr(base_mixer, "A_log").device
        self.register_buffer(
            "radius", torch.tensor(float(radius), dtype=torch.float32, device=radius_device)
        )
        self.capture_states = False
        self.capture_steps: set[int] = set()
        self.captured_states: dict[int, torch.Tensor] = {}
        self.state_transform: Callable[[int, torch.Tensor], torch.Tensor] | None = None
        self.last_scan: ScanResult | None = None
        self.scan_chunk_size = scan_chunk_size
        self.compile_scan = compile_scan
        self.checkpoint_scan_chunks = checkpoint_scan_chunks
        self.scan_engine = self._make_scan_engine()

    def _make_scan_engine(self) -> ChunkedSelectiveScan:
        return ChunkedSelectiveScan(
            self.condition,
            epsilon=self.epsilon,
            chunk_size=self.scan_chunk_size,
            compile_chunks=self.compile_scan,
            checkpoint_chunks=self.checkpoint_scan_chunks,
        )

    def set_condition(self, condition: Condition | str) -> None:
        self.condition = Condition(condition)
        self.scan_engine = self._make_scan_engine()

    @property
    def layer_idx(self) -> int | None:
        return getattr(self.base, "layer_idx", None)

    def set_projection_strength(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("projection strength must be in [0, 1]")
        self.alpha = float(alpha)

    def set_radius(self, radius: float) -> None:
        if radius <= 0:
            raise ValueError("calibrated radius must be positive")
        self.radius.fill_(float(radius))

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("instrumented layers require full-sequence reference recurrence")
        batch, length, _ = hidden_states.shape
        mixer = self.base
        xz = mixer.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = _causal_convolution(mixer, x, length)
        x_dbl = mixer.x_proj(x.transpose(1, 2).reshape(batch * length, -1))
        dt, B, C = torch.split(x_dbl, [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1)
        dt = _linear_without_bias(mixer.dt_proj, dt)
        dt = dt.reshape(batch, length, mixer.d_inner).transpose(1, 2).contiguous()
        B = B.reshape(batch, length, mixer.d_state).transpose(1, 2).contiguous()
        C = C.reshape(batch, length, mixer.d_state).transpose(1, 2).contiguous()
        A = -torch.exp(mixer.A_log.float())
        use_reference = self.capture_states or bool(self.capture_steps) or self.state_transform is not None
        if use_reference:
            self.captured_states.clear()

            def transform(step: int, state: torch.Tensor) -> torch.Tensor:
                if self.state_transform is not None:
                    state = self.state_transform(step, state)
                if step in self.capture_steps:
                    self.captured_states[step] = state.detach().clone()
                return state

            scan = selective_scan_reference(
                x,
                dt,
                A,
                B,
                C,
                mixer.D.float(),
                z,
                delta_bias=mixer.dt_proj.bias.float(),
                delta_softplus=True,
                condition=self.condition,
                radius=self.radius,
                alpha=self.alpha,
                epsilon=self.epsilon,
                state_transform=transform,
                return_states=self.capture_states,
            )
        else:
            scan = self.scan_engine(
                x,
                dt,
                A,
                B,
                C,
                mixer.D.float(),
                z,
                delta_bias=mixer.dt_proj.bias.float(),
                radius=self.radius,
                alpha=self.alpha,
            )
        self.last_scan = scan
        y = scan.output.transpose(1, 2)
        return mixer.out_proj(y)


def instrument_model(
    model: nn.Module,
    layer_indices: Iterable[int],
    radii: dict[int, float],
    condition: Condition | str,
    *,
    epsilon: float = 1e-6,
    scan_chunk_size: int = 4,
    compile_scan: bool = False,
    checkpoint_scan_chunks: bool = False,
) -> list[ProjectedMambaMixer]:
    pass
    try:
        layers = model.backbone.layers
    except AttributeError as exc:
        raise TypeError("expected an official MambaLMHeadModel with backbone.layers") from exc
    adapters: list[ProjectedMambaMixer] = []
    for index in layer_indices:
        if index not in radii:
            raise ValueError(f"missing calibrated radius for layer {index}")
        mixer = layers[index].mixer
        adapter = ProjectedMambaMixer(
            mixer,
            condition=condition,
            radius=radii[index],
            epsilon=epsilon,
            alpha=0.0 if Condition(condition) is Condition.SPHERE else 1.0,
            scan_chunk_size=scan_chunk_size,
            compile_scan=compile_scan,
            checkpoint_scan_chunks=checkpoint_scan_chunks,
        )
        layers[index].mixer = adapter
        adapters.append(adapter)
    return adapters


def make_remaining_mixers_lora_compatible(model: nn.Module) -> list[FusedScanMambaMixer]:
    pass
    try:
        layers = model.backbone.layers
    except AttributeError as exc:
        raise TypeError("expected an official MambaLMHeadModel with backbone.layers") from exc
    wrappers: list[FusedScanMambaMixer] = []
    for layer in layers:
        if isinstance(layer.mixer, ProjectedMambaMixer):
            continue
        if isinstance(layer.mixer, FusedScanMambaMixer):
            wrappers.append(layer.mixer)
            continue
        wrapper = FusedScanMambaMixer(layer.mixer)
        layer.mixer = wrapper
        wrappers.append(wrapper)
    return wrappers


def set_projection_strength(model: nn.Module, alpha: float) -> None:
    for module in model.modules():
        if isinstance(module, ProjectedMambaMixer):
            module.set_projection_strength(alpha)


class FixedRowGain(nn.Module):
    pass

    def __init__(self, weight: torch.Tensor, epsilon: float = 1e-12) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("gain", torch.linalg.vector_norm(weight.detach().float(), dim=1, keepdim=True))

    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(direction.float(), dim=1, keepdim=True).clamp_min(self.epsilon)
        return (direction.float() * (self.gain / norm)).to(direction.dtype)


def apply_fixed_row_gain(linears: Iterable[nn.Linear]) -> None:
    from torch.nn.utils import parametrize

    for linear in linears:
        parametrize.register_parametrization(linear, "weight", FixedRowGain(linear.weight))
