from __future__ import annotations

import torch

from .config import Condition
from .recurrence import selective_scan_reference


def random_scan_inputs(
    *, batch: int = 2, channels: int = 5, length: int = 7, state_size: int = 3,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    u = torch.randn(batch, channels, length, generator=generator, dtype=dtype, requires_grad=True)
    delta = torch.randn(batch, channels, length, generator=generator, dtype=dtype, requires_grad=True)
    A = -torch.exp(torch.randn(channels, state_size, generator=generator, dtype=torch.float32)).requires_grad_()
    B = torch.randn(batch, state_size, length, generator=generator, dtype=dtype, requires_grad=True)
    C = torch.randn(batch, state_size, length, generator=generator, dtype=dtype, requires_grad=True)
    D = torch.randn(channels, generator=generator, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, channels, length, generator=generator, dtype=dtype, requires_grad=True)
    bias = torch.randn(channels, generator=generator, dtype=torch.float32, requires_grad=True)
    return u, delta, A, B, C, D, z, bias


def validate_norm_invariant(radius: float = 3.25) -> float:
    inputs = random_scan_inputs()
    result = selective_scan_reference(
        *inputs[:7], delta_bias=inputs[7], condition=Condition.SPHERE,
        radius=radius, alpha=1.0, return_states=True,
    )
    assert result.states is not None
    norms = torch.linalg.vector_norm(
        result.states.permute(0, 2, 1, 3).flatten(start_dim=2), dim=2
    )
    relative_error = ((norms - radius).abs() / radius).max().item()
    if relative_error > 1e-4:
        raise AssertionError(f"norm invariant error {relative_error:.3g} exceeds 1e-4")
    return relative_error
