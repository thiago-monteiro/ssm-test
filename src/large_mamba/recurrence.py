from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import Condition


StateTransform = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass
class ScanResult:
    output: torch.Tensor
    last_state: torch.Tensor
    states: torch.Tensor | None = None


class _ScanChunk(nn.Module):
    def __init__(self, condition: Condition, epsilon: float) -> None:
        super().__init__()
        self.condition = condition
        self.epsilon = epsilon

    def forward(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        radius: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for step in range(u.shape[2]):
            dA = torch.exp(delta[:, :, step, None] * A[None])
            update = (
                dA * state
                + delta[:, :, step, None]
                * B[:, None, :, step]
                * u[:, :, step, None]
            )
            if self.condition is Condition.SPHERE:
                norm = torch.linalg.vector_norm(update.flatten(start_dim=1), dim=1)
                scale = (radius / (norm + self.epsilon)).view(-1, 1, 1)
                state = update + alpha * (update * scale - update)
            else:
                state = update
            if self.condition is Condition.READ:
                norm = torch.linalg.vector_norm(state.flatten(start_dim=1), dim=1)
                read_state = state * (radius / (norm + self.epsilon)).view(-1, 1, 1)
            else:
                read_state = state
            outputs.append((read_state * C[:, None, :, step]).sum(dim=-1))
        return state, torch.stack(outputs, dim=2)


class ChunkedSelectiveScan(nn.Module):
    pass

    def __init__(
        self,
        condition: Condition | str,
        *,
        epsilon: float = 1e-6,
        chunk_size: int = 4,
        compile_chunks: bool = True,
        checkpoint_chunks: bool = False,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("scan chunk size must be positive")
        self.condition = Condition(condition)
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        chunk: nn.Module = _ScanChunk(self.condition, epsilon)
        if compile_chunks:
            
            
            
            try:
                from torch._dynamo import config as dynamo_config

                dynamo_config.cache_size_limit = max(dynamo_config.cache_size_limit, 128)
            except (ImportError, AttributeError):
                pass
            chunk = torch.compile(chunk, fullgraph=True, dynamic=False, mode="default")
        self.chunk = chunk

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None,
        z: torch.Tensor | None,
        *,
        delta_bias: torch.Tensor,
        radius: torch.Tensor,
        alpha: float,
    ) -> ScanResult:
        if B.ndim != 3 or C.ndim != 3:
            raise ValueError("optimized scan expects token-dependent (batch, state, length) B/C")
        input_dtype = u.dtype
        u32 = u.float()
        delta32 = F.softplus(delta.float() + delta_bias.float().view(1, -1, 1))
        state = torch.zeros(
            u.shape[0], u.shape[1], A.shape[1], device=u.device, dtype=torch.float32
        )
        outputs: list[torch.Tensor] = []
        radius_batch = radius.float().expand(u.shape[0])
        alpha_tensor = torch.as_tensor(alpha, device=u.device, dtype=torch.float32)
        for start in range(0, u.shape[2], self.chunk_size):
            stop = min(start + self.chunk_size, u.shape[2])
            args = (
                state,
                u32[:, :, start:stop],
                delta32[:, :, start:stop],
                A.float(),
                B.float()[:, :, start:stop],
                C.float()[:, :, start:stop],
                radius_batch,
                alpha_tensor,
            )
            if self.checkpoint_chunks and torch.is_grad_enabled():
                state, output = checkpoint(
                    self.chunk,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                state, output = self.chunk(*args)
            outputs.append(output)
        output = torch.cat(outputs, dim=2)
        if D is not None:
            output = output + u32 * D.float().view(1, -1, 1)
        if z is not None:
            output = output * F.silu(z.float())
        return ScanResult(output.to(input_dtype), state)


def project_state(state: torch.Tensor, radius: float | torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    pass
    if state.ndim < 2:
        raise ValueError("state must have a leading batch dimension")
    state_fp32 = state.float()
    norm = torch.linalg.vector_norm(state_fp32.flatten(start_dim=1), dim=1)
    shape = (state.shape[0],) + (1,) * (state.ndim - 1)
    radius_tensor = torch.as_tensor(radius, device=state.device, dtype=torch.float32)
    projected = state_fp32 * (radius_tensor / (norm + epsilon)).reshape(shape)
    return projected.to(state.dtype)


def _variable_term(term: torch.Tensor, step: int, dim: int) -> torch.Tensor:
    if term.ndim == 2:  
        return term.unsqueeze(0)
    if term.ndim == 3:  
        return term[:, :, step].unsqueeze(1)
    if term.ndim == 4:  
        value = term[:, :, :, step]
        if dim % value.shape[1]:
            raise ValueError("state channels must be divisible by B/C groups")
        return value.repeat_interleave(dim // value.shape[1], dim=1)
    raise ValueError(f"unsupported B/C rank: {term.ndim}")


def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    *,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    condition: Condition | str = Condition.ORD,
    radius: float | torch.Tensor = 1.0,
    alpha: float = 1.0,
    epsilon: float = 1e-6,
    state_transform: StateTransform | None = None,
    return_states: bool = False,
) -> ScanResult:
    pass
    condition = Condition(condition)
    if condition is Condition.WEIGHT:
        condition = Condition.ORD
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must both have shape (batch, channels, length)")
    if A.shape[0] != u.shape[1]:
        raise ValueError("A channel dimension does not match u")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    input_dtype = u.dtype
    u32 = u.float()
    delta32 = delta.float()
    if delta_bias is not None:
        delta32 = delta32 + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta32 = F.softplus(delta32)
    A32, B32, C32 = A.float(), B.float(), C.float()
    batch, channels, length = u.shape
    state = torch.zeros(batch, channels, A.shape[1], dtype=torch.float32, device=u.device)
    outputs: list[torch.Tensor] = []
    state_history: list[torch.Tensor] = []

    for step in range(length):
        bt = _variable_term(B32, step, channels)
        ct = _variable_term(C32, step, channels)
        dA = torch.exp(delta32[:, :, step].unsqueeze(-1) * A32.unsqueeze(0))
        update = dA * state + (
            delta32[:, :, step].unsqueeze(-1)
            * bt
            * u32[:, :, step].unsqueeze(-1)
        )
        if condition is Condition.SPHERE:
            spherical = project_state(update, radius, epsilon).float()
            state = (1.0 - alpha) * update + alpha * spherical
        else:
            state = update
        if state_transform is not None:
            state = state_transform(step, state)
        read_state = project_state(state, radius, epsilon).float() if condition is Condition.READ else state
        outputs.append((read_state * ct).sum(dim=-1))
        if return_states:
            state_history.append(state)

    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + u32 * D.float().view(1, -1, 1)
    if z is not None:
        output = output * F.silu(z.float())
    states = torch.stack(state_history, dim=2) if return_states else None
    return ScanResult(output.to(input_dtype), state, states)
