from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

EPS = 1e-08
HEAD_KEYS = ("head", "lm_head")
ALWAYS_EXCLUDED_KEYS = ("embed", "pos_embed", "a_raw", "norm")


@dataclass(frozen=True)
class NormalizationPolicy:
    weight_rows: bool = False
    state_sphere: bool = False
    include_head: bool = False
    state_ramp_fraction: float = 0.1

    @property
    def is_plain(self) -> bool:
        return not self.weight_rows and (not self.state_sphere)

    @property
    def label(self) -> str:
        if self.is_plain:
            return "plain"
        if self.weight_rows and self.state_sphere:
            return "normWS"
        if self.weight_rows:
            return "normW"
        return "normS"

    def state_strength(self, step: int, total: int) -> float:
        if not self.state_sphere:
            return 0.0
        ramp = max(1, round(total * self.state_ramp_fraction))
        return min(1.0, max(0.0, (step + 1) / ramp))


def iter_weight_row_targets(
    model: nn.Module, include_head: bool = False
) -> list[tuple[str, nn.Parameter]]:
    targets: list[tuple[str, nn.Parameter]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight
        if weight.ndim != 2 or not weight.requires_grad:
            continue
        if any((key in name for key in ALWAYS_EXCLUDED_KEYS)):
            continue
        if not include_head and any((key in name for key in HEAD_KEYS)):
            continue
        targets.append((name + ".weight", weight))
    return targets


def project_rows_(
    weight: torch.Tensor, strength: float = 1.0, eps: float = EPS
) -> torch.Tensor:
    if weight.ndim != 2:
        return weight
    if strength <= 0.0:
        return weight
    unit = weight.data / (weight.data.norm(dim=1, keepdim=True) + eps)
    if strength >= 1.0:
        weight.data.copy_(unit)
    else:
        weight.data.mul_(1.0 - strength).add_(unit, alpha=strength)
    return weight


def project_weight_rows_(
    model: nn.Module, include_head: bool = False, strength: float = 1.0
) -> int:
    n = 0
    for _, weight in iter_weight_row_targets(model, include_head=include_head):
        project_rows_(weight, strength=strength)
        n += 1
    return n


def normalize_state(h: torch.Tensor, strength: float, eps: float = EPS) -> torch.Tensor:
    if strength <= 0.0:
        return h
    unit = h / (h.norm(dim=-1, keepdim=True) + eps)
    if strength >= 1.0:
        return unit
    return (1.0 - strength) * h + strength * unit


def tangentialize(
    update: torch.Tensor, weight: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    w_hat = weight / (weight.norm(dim=1, keepdim=True) + eps)
    radial = (update * w_hat).sum(dim=1, keepdim=True)
    return update - radial * w_hat


def row_radial_fraction(
    grads: dict[str, torch.Tensor], weights: dict[str, torch.Tensor], eps: float = EPS
) -> float:
    radial_sq = 0.0
    total_sq = 0.0
    for name, g in grads.items():
        w = weights.get(name)
        if w is None:
            continue
        w_hat = w / (w.norm(dim=1, keepdim=True) + eps)
        radial = (g * w_hat).sum(dim=1, keepdim=True)
        radial_sq += float((radial**2).sum().item())
        total_sq += float((g**2).sum().item())
    if total_sq <= 0.0:
        return 0.0
    return radial_sq / total_sq
