
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def calibrate_ranges(
    z: torch.Tensor,
    mode: str = "per_coord",
    global_lo: float = -1.05,
    global_hi: float = 1.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "global":
        lo = torch.full((z.shape[1],), global_lo, device=z.device, dtype=z.dtype)
        hi = torch.full((z.shape[1],), global_hi, device=z.device, dtype=z.dtype)
        return lo, hi
    lo = z.min(dim=0).values
    hi = z.max(dim=0).values
    same = (hi - lo).abs() < 1e-8
    hi = torch.where(same, lo + 1.0, hi)
    return lo, hi


@torch.no_grad()
def quantize_uniform(
    z: torch.Tensor,
    levels: int,
    lo: torch.Tensor,
    hi: torch.Tensor,
    round_mode: str = "round",
) -> torch.Tensor:
    if levels < 2:
        raise ValueError("levels must be >= 2")
    lo = lo.view(1, -1)
    hi = hi.view(1, -1)
    z_clamped = torch.clamp(z, lo, hi)
    scale = (hi - lo) / (levels - 1)
    scale = scale.clamp_min(1e-12)
    idx = (z_clamped - lo) / scale
    if round_mode == "ste":
        idx_round = torch.round(idx).detach() + idx - idx.detach()
        idx_round = torch.clamp(idx_round, 0, levels - 1)
    else:
        idx_round = torch.round(idx)
        idx_round = torch.clamp(idx_round, 0, levels - 1)
    return lo + idx_round * scale


@torch.no_grad()
def quantize_fixed_bitwidth(
    z: torch.Tensor,
    bits: int,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    levels = 2 ** bits
    return quantize_uniform(z, levels, lo, hi, round_mode="ste")
@torch.no_grad()
def matched_noise_inject(
    z: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    levels: int,
    noise_dist: str = "uniform",
) -> torch.Tensor:
    lo = lo.view(1, -1)
    hi = hi.view(1, -1)
    if levels < 2:
        return z
    scale = (hi - lo) / (levels - 1)
    scale = scale.clamp_min(1e-12)
    quant_mse = (scale ** 2) / 12.0
    if noise_dist == "uniform":
        half_width = (quant_mse * 3.0).sqrt()
        noise = torch.rand_like(z) * 2 * half_width - half_width
    else:
        noise = torch.randn_like(z) * quant_mse.sqrt()
    return z + noise
@torch.no_grad()
def entropy_estimate(z: torch.Tensor, bins: int = 30) -> float:
    z = z.detach().cpu().float()
    N, k = z.shape
    entropies = []
    for j in range(k):
        col = z[:, j]
        lo, hi = col.min().item(), col.max().item()
        if hi - lo < 1e-8:
            continue
        h = torch.histc(col, bins=bins, min=lo, max=hi)
        p = h / h.sum()
        p = p[p > 0]
        ent = -(p * p.log()).sum().item()
        entropies.append(ent)
    return float(np.mean(entropies)) if entropies else 0.0
