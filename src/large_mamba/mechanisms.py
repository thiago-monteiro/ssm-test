from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from scipy.stats import spearmanr


@dataclass(frozen=True)
class FaithfulnessMetrics:
    e_all: float
    e_eff: float
    rank_correlation: float
    sign_accuracy: float
    false_positive_rate: float
    false_negative_rate: float


def tangent_directions(state: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    pass
    state_flat = state.float().flatten()
    state_unit = state_flat / state_flat.norm().clamp_min(1e-12)
    direction_flat = directions.float().flatten(start_dim=1)
    tangent = direction_flat - (direction_flat @ state_unit)[:, None] * state_unit[None]
    tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return tangent.reshape_as(directions).to(directions.dtype)


def perturb_state(
    state: torch.Tensor,
    directions: torch.Tensor,
    theta: float,
    *,
    spherical: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass
    expanded = state.unsqueeze(0).expand_as(directions)
    if spherical:
        tangent = tangent_directions(state, directions)
        perturbed = math.cos(theta) * expanded + math.sin(theta) * state.norm() * tangent
    else:
        flat_norm = directions.flatten(start_dim=1).norm(dim=1).clamp_min(1e-12)
        norm_shape = (-1,) + (1,) * (directions.ndim - 1)
        unit = directions / flat_norm.view(norm_shape)
        perturbed = expanded + theta * state.norm() * unit
    return perturbed, perturbed - expanded


def faithfulness_metrics(predicted: torch.Tensor, actual: torch.Tensor) -> FaithfulnessMetrics:
    pass
    predicted = predicted.detach().float()
    actual = actual.detach().float()
    if predicted.shape != actual.shape or predicted.ndim not in (1, 2):
        raise ValueError("effects need equal (pairs,) or (examples, directions) shape")
    if predicted.ndim == 1:
        predicted, actual = predicted.unsqueeze(0), actual.unsqueeze(0)
    effective = actual.abs() >= actual.abs().median(dim=1, keepdim=True).values
    error = (actual - predicted).abs() / (actual.abs() + 1e-6)
    pred_eff, actual_eff = predicted[effective], actual[effective]
    if actual_eff.numel() == 0:
        raise ValueError("no effective perturbation directions")
    pred_abs, actual_abs = pred_eff.abs(), actual_eff.abs()
    pred_quartiles = torch.quantile(pred_abs, torch.tensor([0.25, 0.75]))
    actual_quartiles = torch.quantile(actual_abs, torch.tensor([0.25, 0.75]))
    rank = float(spearmanr(pred_abs.cpu().numpy(), actual_abs.cpu().numpy()).statistic)
    return FaithfulnessMetrics(
        e_all=float(error.mean()),
        e_eff=float(error[effective].mean()),
        rank_correlation=rank,
        sign_accuracy=float(pred_eff.sign().eq(actual_eff.sign()).float().mean()),
        false_positive_rate=float(
            ((pred_abs >= pred_quartiles[1]) & (actual_abs <= actual_quartiles[0])).float().mean()
        ),
        false_negative_rate=float(
            ((pred_abs <= pred_quartiles[0]) & (actual_abs >= actual_quartiles[1])).float().mean()
        ),
    )


def matched_additive_noise(
    clean: torch.Tensor,
    quantized: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    pass
    variance = (quantized.float() - clean.float()).square().mean()
    noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=torch.float32)
    return (clean.float() + noise * variance.sqrt()).to(clean.dtype)


def causal_pruning_recovery(
    state: torch.Tensor,
    gradient: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0),
) -> dict[float, float]:
    pass
    flat_state, flat_gradient = state.flatten(), gradient.flatten()
    order = flat_gradient.abs().argsort(descending=True)
    zero_score = score_fn(torch.zeros_like(state)).reshape(()).float()
    full_effect = score_fn(state).reshape(()).float() - zero_score
    if full_effect.abs() < 1e-12:
        raise ValueError("clean state has no measurable causal effect")
    recovery: dict[float, float] = {}
    for fraction in fractions:
        keep = max(1, math.ceil(fraction * flat_state.numel()))
        pruned = torch.zeros_like(flat_state)
        pruned[order[:keep]] = flat_state[order[:keep]]
        effect = score_fn(pruned.reshape_as(state)).reshape(()).float() - zero_score
        recovery[fraction] = float(effect / full_effect)
    return recovery


def source_capture(
    attribution: torch.Tensor,
    causal_effect: torch.Tensor,
    top_k: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[int, float]:
    if attribution.shape != causal_effect.shape or attribution.ndim != 1:
        raise ValueError("source vectors must have equal one-dimensional shape")
    total = causal_effect.abs().sum().clamp_min(1e-12)
    order = attribution.abs().argsort(descending=True)
    return {k: float(causal_effect[order[:k]].abs().sum() / total) for k in top_k}
