from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


LORA_TARGETS = ("in_proj", "x_proj", "dt_proj", "out_proj")


class MambaLoRALinear(nn.Module):
    pass

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        self.lora_A.to(dtype=base.weight.dtype)
        self.lora_B.to(dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def lora_delta(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.lora_dropout(inputs))) * self.scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_delta(inputs)

    def forward_without_bias(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.base.weight, None) + self.lora_delta(inputs)


class CheckpointedBlock(nn.Module):
    pass

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.layer_idx = getattr(base, "layer_idx", None)

    def forward(self, hidden_states, residual=None, inference_params=None, **mixer_kwargs):
        if not self.training or not torch.is_grad_enabled() or inference_params is not None:
            return self.base(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if residual is None:
            def run_without_residual(hidden):
                return self.base(hidden, None, inference_params=None, **mixer_kwargs)

            return checkpoint(
                run_without_residual,
                hidden_states,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        def run(hidden, saved_residual):
            return self.base(hidden, saved_residual, inference_params=None, **mixer_kwargs)

        return checkpoint(
            run,
            hidden_states,
            residual,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def allocate_inference_cache(self, *args, **kwargs):
        return self.base.allocate_inference_cache(*args, **kwargs)


def enable_activation_checkpointing(model: nn.Module) -> None:
    from .adapter import ProjectedMambaMixer

    try:
        layers = model.backbone.layers
    except AttributeError as exc:
        raise TypeError("expected an official MambaLMHeadModel with backbone.layers") from exc
    for index, layer in enumerate(layers):
        if not isinstance(layer, CheckpointedBlock):
            mixer = getattr(layer, "mixer", None)
            
            
            
            if isinstance(mixer, ProjectedMambaMixer) and mixer.compile_scan:
                continue
            layers[index] = CheckpointedBlock(layer)


@dataclass(frozen=True)
class ParameterManifest:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    count: int

    def write(self, path: str | Path) -> None:
        payload = {"names": self.names, "shapes": self.shapes, "count": self.count}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def inject(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, MambaLoRALinear):
                continue
            if isinstance(child, nn.Linear) and name in LORA_TARGETS:
                setattr(
                    module,
                    name,
                    MambaLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
            else:
                inject(child)

    inject(model)
    return model


def enable_recurrence_parameters(model: nn.Module) -> None:
    pass
    for name, parameter in model.named_parameters():
        canonical = name.replace(".base.", ".")
        if canonical.endswith("A_log") or canonical.endswith(".D"):
            parameter.requires_grad_(True)
        elif canonical.endswith("dt_proj.bias"):
            parameter.requires_grad_(True)
        elif parameter.ndim == 1 and ("norm" in canonical.lower()):
            parameter.requires_grad_(True)


def parameter_manifest(model: nn.Module) -> ParameterManifest:
    rows = [(name.replace(".base.", "."), tuple(parameter.shape), parameter.numel())
            for name, parameter in model.named_parameters() if parameter.requires_grad]
    rows.sort(key=lambda row: row[0])
    return ParameterManifest(
        names=tuple(row[0] for row in rows),
        shapes=tuple(row[1] for row in rows),
        count=sum(row[2] for row in rows),
    )


def assert_parameter_parity(*models: nn.Module) -> None:
    manifests = [parameter_manifest(model) for model in models]
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise AssertionError("trainable parameter names, shapes, or counts differ across conditions")


def initial_parameter_hashes(model: nn.Module) -> dict[str, str]:
    pass
    result: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            canonical = name.replace(".base.", ".")
            value = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            result[canonical] = hashlib.sha256(value).hexdigest()
    return result


def assert_initial_tensor_parity(*models: nn.Module) -> None:
    hashes = [initial_parameter_hashes(model) for model in models]
    if any(item != hashes[0] for item in hashes[1:]):
        raise AssertionError("initial trainable tensors differ across paired conditions")


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def load_trainable_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
) -> None:
    pass
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError("checkpoint must contain a tensor state dictionary")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "trainable checkpoint does not match reconstructed model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint keys: {incompatible.unexpected_keys[:5]}")


def load_official_model(
    repository: str,
    *,
    revision: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> tuple[nn.Module, str]:
    pass
    try:
        from huggingface_hub import snapshot_download
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before loading the checkpoint") from exc
    snapshot = snapshot_download(repository, revision=revision)
    model = MambaLMHeadModel.from_pretrained(snapshot, device=device, dtype=dtype)
    return model, snapshot


def package_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
