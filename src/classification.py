from __future__ import annotations

import torch
import torch.nn.functional as F


def curriculum_length(step: int, steps: int, length: int, *, delay: int = 0) -> int:
    if step < steps // 5:
        return max(8, length // 4, delay + 3) if delay else max(8, length // 4)
    if step < steps // 2:
        return max(16, length // 2, delay + 3) if delay else max(16, length // 2)
    return length


def classification_step(model, batch, optimizer, scheduler, grad_clip):
    out = model(batch["input_ids"], batch["query_pos"])
    loss = F.cross_entropy(out["logits"], batch["target"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    return out, loss
