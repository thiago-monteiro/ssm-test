from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.expC.data import make_copy_batch
from src.expC.model import CopySSM
from src.seed import seed_everything


def train_copy_ssm(
    seed: int,
    variant: str,
    L: int = 32,
    k: int = 128,
    V: int = 16,
    d_model: int = 64,
    steps: int = 4000,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    device: str | torch.device | None = None,
    log_every: int = 1000,
    delay: int = 0,
) -> tuple[CopySSM, dict[str, Any]]:
    assert variant in ("ordinary", "sphere")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)
    device = torch.device(device)

    model = CopySSM(V=V, L=L, d_model=d_model, k=k, sphere=(variant == "sphere")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    final_acc: float | None = None
    for step in range(1, steps + 1):

        if step < steps // 5:
            L_step = max(max(8, L // 4), delay + 3)
        elif step < steps // 2:
            L_step = max(max(16, L // 2), delay + 3)
        else:
            L_step = L

        batch = make_copy_batch(batch_size, L_step, V=V, delay=delay, device=device)
        out = model(batch["input_ids"], batch["query_pos"])
        loss = F.cross_entropy(out["logits"], batch["target"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        if log_every and step % log_every == 0:
            with torch.no_grad():
                acc = (out["logits"].argmax(-1) == batch["target"]).float().mean().item()
            print(
                f"  [C seed={seed} variant={variant}] step {step}/{steps} "
                f"loss={loss.item():.4f} acc={acc:.3f}",
                flush=True,
            )

    final_acc = quick_accuracy(model, L=L, V=V, device=device, n=1024, delay=delay)
    meta = {"seed": seed, "variant": variant, "L": L, "k": k, "steps": steps,
            "final_acc": final_acc, "delay": delay}
    return model, meta


@torch.no_grad()
def quick_accuracy(
    model: CopySSM,
    L: int,
    V: int,
    device: torch.device | str,
    n: int = 1024,
    delay: int = 0,
) -> float:
    model.eval()
    device = torch.device(device)
    batch = make_copy_batch(n, L, V=V, delay=delay, device=device)
    out = model(batch["input_ids"], batch["query_pos"])
    acc = (out["logits"].argmax(-1) == batch["target"]).float().mean().item()
    return float(acc)
