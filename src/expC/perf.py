from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.expC.data import make_copy_batch
from src.expC.model import CopySSM


def _gen_eval_batches(n: int, L: int, V: int, seed: int, device: torch.device, delay: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for i in range(0, n, 256):
        bs = min(256, n - i)
        yield make_copy_batch(bs, L, V=V, delay=delay, device=device, generator=gen)


@torch.no_grad()
def eval_perf(
    model: CopySSM,
    n: int = 2048,
    L: int = 32,
    V: int = 16,
    seed: int = 0,
    device: str | torch.device = "cpu",
    delay: int = 0,
) -> dict:
    model = model.to(device).eval()
    device = torch.device(device)

    correct_all = []
    p_top1_all = []
    margin_correct = []
    brier_sum = 0.0
    n_seen = 0

    for batch in _gen_eval_batches(n, L, V, seed=seed + 90000, device=device, delay=delay):
        out = model(batch["input_ids"], batch["query_pos"])
        logits = out["logits"]
        target = batch["target"].to(device)
        probs = F.softmax(logits, dim=-1)
        p_top1 = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        correct = (logits.argmax(-1) == target)

        correct_all.append(correct.cpu().numpy())
        p_top1_all.append(p_top1.cpu().numpy())
        brier_sum += float(((probs ** 2).sum(-1) - 2 * p_top1).sum().item())
        n_seen += len(target)

        z = logits.clone()
        z[torch.arange(len(target), device=device), target] = -1e9
        alt = z.argmax(-1)
        margins = (logits.gather(-1, target.unsqueeze(-1)) - logits.gather(-1, alt.unsqueeze(-1))).squeeze(-1)
        margin_correct.append(margins[correct].cpu().numpy())

    correct_all = np.concatenate(correct_all)
    p_top1_all = np.concatenate(p_top1_all)
    acc = float(correct_all.mean())


    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        hi = p_top1_all <= bins[i + 1] if i == 14 else p_top1_all < bins[i + 1]
        m = (p_top1_all >= bins[i]) & hi
        if m.sum() > 0:
            conf = p_top1_all[m].mean()
            acc_b = correct_all[m].mean()
            ece += (m.sum() / len(p_top1_all)) * abs(conf - acc_b)

    mc = np.concatenate(margin_correct) if margin_correct else np.array([np.nan])
    return {
        "n": n_seen,
        "accuracy": acc,
        "mean_margin_correct": float(mc.mean()) if len(mc) and np.isfinite(mc).all() else float("nan"),
        "median_margin_correct": float(np.median(mc)) if len(mc) and np.isfinite(mc).all() else float("nan"),
        "mean_p_target": float(p_top1_all.mean()),
        "ece": float(ece),
        "brier": brier_sum / max(n_seen, 1),
    }


@torch.no_grad()
def eval_robustness(
    model: CopySSM,
    n: int = 1024,
    L: int = 32,
    V: int = 16,
    seed: int = 0,
    device: str | torch.device = "cpu",
    delay: int = 0,
) -> dict:
    model = model.to(device).eval()
    device = torch.device(device)

    acc_clean_list, acc_corrupt_list = [], []
    for batch in _gen_eval_batches(n, L, V, seed=seed + 91000, device=device, delay=delay):
        input_ids = batch["input_ids"].to(device)
        query_pos = batch["query_pos"].to(device)
        target = batch["target"].to(device)

        out_c = model(input_ids, query_pos)
        acc_clean_list.append((out_c["logits"].argmax(-1) == target).float().mean().item())

        tokens = input_ids[:, :-1].clone()
        Bsz, Ltok = tokens.shape
        gen = torch.Generator(device="cpu").manual_seed(seed + 92000)
        for i in range(Bsz):
            q = int(query_pos[i])
            candidates = [t for t in range(Ltok) if t != q]
            pos = int(torch.randint(0, len(candidates), (1,), generator=gen).item())
            new_tok = int(torch.randint(0, V, (1,), generator=gen).item())
            tokens[i, candidates[pos]] = new_tok
        input_ids_corrupt = torch.cat([tokens, input_ids[:, -1:]], dim=1)
        out_k = model(input_ids_corrupt, query_pos)
        acc_corrupt_list.append((out_k["logits"].argmax(-1) == target).float().mean().item())

    acc_clean = float(np.mean(acc_clean_list))
    acc_corrupt = float(np.mean(acc_corrupt_list))
    return {
        "acc_clean": acc_clean,
        "acc_one_token_corrupted": acc_corrupt,
        "drop": acc_clean - acc_corrupt,
    }


@torch.no_grad()
def eval_geometry(
    model: CopySSM,
    n: int = 512,
    L: int = 32,
    V: int = 16,
    seed: int = 0,
    device: str | torch.device = "cpu",
    delay: int = 0,
) -> dict:
    model = model.to(device).eval()
    device = torch.device(device)

    norms = []
    top10_energy = []
    for batch in _gen_eval_batches(n, L, V, seed=seed + 93000, device=device, delay=delay):
        out = model(batch["input_ids"], batch["query_pos"].to(device))
        h_q = out["h_q"]
        norms.append(h_q.norm(dim=-1).cpu().numpy())
        e = (h_q ** 2).sum(-1, keepdim=True).clamp_min(1e-12)
        top10_energy.append((h_q.abs().sort(dim=-1, descending=True).values[:, :10] ** 2).sum(-1) / e.squeeze(-1))

    norms = np.concatenate(norms)
    top10_energy = np.concatenate(top10_energy)
    return {
        "h_norm_mean": float(norms.mean()),
        "h_norm_median": float(np.median(norms)),
        "h_norm_p90": float(np.percentile(norms, 90)),
        "h_norm_cv": float(norms.std() / max(norms.mean(), 1e-9)),
        "top10_dim_energy_frac": float(top10_energy.mean()),
    }
