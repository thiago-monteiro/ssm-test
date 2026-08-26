from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AnswerMetrics:
    exact_match: float
    cross_entropy: float
    mean_margin: float
    predictions: torch.Tensor


def answer_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pass
    if logits.shape[:2] != labels.shape:
        raise ValueError("logit batch/sequence dimensions must match labels")
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def answer_metrics(logits: torch.Tensor, labels: torch.Tensor) -> AnswerMetrics:
    shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
    mask = shifted_labels.ne(-100)
    selected_logits = shifted_logits[mask]
    targets = shifted_labels[mask]
    if targets.numel() == 0:
        raise ValueError("batch has no answer labels")
    predictions = selected_logits.argmax(dim=-1)
    target_logits = selected_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    without_target = selected_logits.clone()
    without_target.scatter_(1, targets.unsqueeze(1), -torch.inf)
    margins = target_logits - without_target.max(dim=1).values
    return AnswerMetrics(
        exact_match=float(predictions.eq(targets).float().mean()),
        cross_entropy=float(F.cross_entropy(selected_logits, targets)),
        mean_margin=float(margins.mean()),
        predictions=predictions,
    )


def stress_auc(cell_accuracies: list[float]) -> float:
    if not cell_accuracies:
        raise ValueError("stress AUC requires at least one valid cell")
    return sum(cell_accuracies) / len(cell_accuracies)


def symmetric_quantize_state(
    state: torch.Tensor, *, bits: int, maximum: torch.Tensor | float
) -> torch.Tensor:
    if bits not in (2, 4, 8):
        raise ValueError("state quantization sweep is fixed to 2, 4, and 8 bits")
    max_value = torch.as_tensor(maximum, dtype=state.dtype, device=state.device).clamp_min(1e-12)
    qmax = 2 ** (bits - 1) - 1
    scale = max_value / qmax
    return (torch.round(state / scale).clamp(-qmax, qmax) * scale).to(state.dtype)
