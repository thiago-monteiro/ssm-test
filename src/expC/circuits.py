from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.expC.data import make_copy_batch
from src.expC.faith import _runner_up
from src.expC.model import CopySSM


def eval_pruning_recovery(
    model: CopySSM,
    n_seq: int,
    L: int,
    V: int,
    seed: int,
    fracs: list[float],
    device: str | torch.device = "cpu",
    delay: int = 0,
) -> dict:
    model = model.to(device).eval()
    device = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(seed + 12345)

    recs: dict[float, list[float]] = {f: [] for f in fracs}
    n_skipped = 0

    for s in range(n_seq):
        with torch.no_grad():
            batch = make_copy_batch(1, L, V=V, delay=delay, device=device, generator=gen)
            input_ids = batch["input_ids"].to(device)
            query_pos = batch["query_pos"].to(device)
            target = int(batch["target"][0].item())

            out = model(input_ids, query_pos)
            logits = out["logits"][0]
            h_q = out["h_q"][0]
            x_q = out["x_q"][0]
            alt = _runner_up(logits, target)
            S_clean = float(logits[target].item() - logits[alt].item())

            zero = torch.zeros(1, model.k, device=device)
            lz = model.logits_from_final_state(zero, x_q.unsqueeze(0))[0]
            S_zero = float(lz[target].item() - lz[alt].item())
        dS_full = S_clean - S_zero
        if abs(dS_full) < 1e-6:
            n_skipped += 1
            continue


        h_leaf = h_q.detach().clone().requires_grad_(True)
        lp = model.logits_from_final_state(h_leaf.unsqueeze(0), x_q.unsqueeze(0))[0]
        S_p = lp[target] - lp[alt]
        g = torch.autograd.grad(S_p, h_leaf)[0].detach()

        order = torch.argsort(g.abs(), descending=True)
        with torch.no_grad():
            for f in fracs:
                n_keep = max(1, int(np.ceil(f * model.k)))
                keep = order[:n_keep]
                h_f = torch.zeros_like(h_q)
                h_f[keep] = h_q[keep]
                lp2 = model.logits_from_final_state(h_f.unsqueeze(0), x_q.unsqueeze(0))[0]
                S_f = float(lp2[target].item() - lp2[alt].item())
                recs[f].append((S_f - S_zero) / dS_full)

    n_used = n_seq - n_skipped
    out = {"n_used": n_used, "n_skipped": n_skipped}
    for f in fracs:
        arr = np.array(recs[f])
        out[f"f={f:.2f}_recovery_mean"] = float(arr.mean()) if len(arr) else float("nan")
        out[f"f={f:.2f}_abs_recovery_mean"] = float(np.abs(arr).mean()) if len(arr) else float("nan")

    f95 = None
    for f in sorted(fracs):
        key = f"f={f:.2f}_abs_recovery_mean"
        if np.isfinite(out[key]) and out[key] >= 0.95:
            f95 = f
            break
    out["f95"] = f95
    return out


def eval_path_recovery(
    model: CopySSM,
    n_seq: int,
    L: int,
    V: int,
    seed: int,
    top_p_max: int = 8,
    device: str | torch.device = "cpu",
    delay: int = 0,
) -> dict:
    model = model.to(device).eval()
    device = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(seed + 54321)

    attr_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    n_pos_total = 0

    with torch.no_grad():
        mean_emb = model.mean_token_embed()

    for s in range(n_seq):
        with torch.no_grad():
            batch = make_copy_batch(1, L, V=V, delay=delay, device=device, generator=gen)
            input_ids = batch["input_ids"].to(device)
            query_pos = batch["query_pos"].to(device)
            q = int(query_pos[0].item())
            target = int(batch["target"][0].item())
            tokens = batch["tokens"][0]

            pos_idx = torch.arange(L, device=device).unsqueeze(0)
            x0 = model.embed(tokens.unsqueeze(0)) + model.pos_embed(pos_idx)

            out = model.forward_x0(x0, query_pos)
            logits = out["logits"][0]
            alt = _runner_up(logits, target)
            S_clean = float(logits[target].item() - logits[alt].item())


        x0_leaf = x0.detach().clone().requires_grad_(True)
        out_g = model.forward_x0(x0_leaf, query_pos)
        lp = out_g["logits"][0]
        S_p = lp[target] - lp[alt]
        G_e = torch.autograd.grad(S_p, x0_leaf)[0][0].detach()

        a_t = G_e[: q + 1].norm(dim=-1).cpu().numpy()

        with torch.no_grad():

            emb_part = model.embed(tokens.unsqueeze(0))[:, : q + 1]
            delta_e = mean_emb.unsqueeze(0).unsqueeze(0) - emb_part
        pred_t = (G_e[: q + 1].cpu().numpy() * delta_e[0].cpu().numpy()).sum(axis=-1)

        with torch.no_grad():

            x0_patched = x0.expand(q + 1, -1, -1).clone()
            for t in range(q + 1):
                x0_patched[t, t] = mean_emb + model.pos_embed(torch.tensor([t], device=device))[0]
            out_p = model.forward_x0(x0_patched, query_pos.expand(q + 1))
            lp2 = out_p["logits"]
        act_t = (lp2[:, target].cpu().numpy() - lp2[:, alt].cpu().numpy()) - S_clean

        attr_list.append(a_t)
        pred_list.append(pred_t)
        act_list.append(act_t)
        n_pos_total += q + 1

    A = np.concatenate(attr_list)
    P = np.concatenate(pred_list)
    C = np.concatenate(act_list)

    out: dict = {"n_pairs": int(n_pos_total)}
    if len(A) >= 8 and np.std(A) > 0 and np.std(C) > 0:
        out["rho_rank_attr"] = float(spearmanr(A, np.abs(C)).statistic)
        med = np.median(np.abs(C))
        eff = np.abs(C) >= max(med, 1e-9)
        if eff.sum() >= 8:
            out["sign_acc_pos"] = float(np.mean(np.sign(P[eff]) == np.sign(C[eff])))
            out["calib_r_pos"] = float(
                np.corrcoef(P[eff], C[eff])[0, 1]
            ) if np.std(P[eff]) > 1e-12 else float("nan")
        else:
            out["sign_acc_pos"], out["calib_r_pos"] = float("nan"), float("nan")
    else:
        out["rho_rank_attr"] = float("nan")


    cap_grad: dict[int, list[float]] = {p: [] for p in range(1, top_p_max + 1)}
    cap_act: dict[int, list[float]] = {p: [] for p in range(1, top_p_max + 1)}
    for a_t, c_t in zip(attr_list, act_list):
        total = np.abs(c_t).sum()
        if total < 1e-9:
            continue
        order_g = np.argsort(a_t)[::-1]
        order_c = np.argsort(np.abs(c_t))[::-1]
        for p in range(1, top_p_max + 1):
            cap_grad[p].append(float(np.abs(c_t[order_g[:p]]).sum() / total))
            cap_act[p].append(float(np.abs(c_t[order_c[:p]]).sum() / total))
    out["top_p_capture_gradient"] = {p: float(np.mean(v)) for p, v in cap_grad.items()}
    out["top_p_capture_actual"] = {p: float(np.mean(v)) for p, v in cap_act.items()}
    return out
