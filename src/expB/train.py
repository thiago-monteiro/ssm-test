
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.expB.data import V_DEFAULT, make_batch
from src.expB.ssm import DiagonalSSM
from src.seed import seed_everything
from src.snr import (
    effective_snr,
    signal_noise_rho,
    train_decode_probe,
    variance_fraction,
)


def train_ssm(
    seed: int,
    mode: str = "B0",
    L: int = 32,
    k: int = 128,
    V: int = V_DEFAULT,
    d_model: int = 64,
    steps: int = 4000,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    device: str | torch.device | None = None,
    log_every: int = 500,
    eval_every: int = 500,
    no_pos_embed: bool = False,
    with_replacement: bool = True,
) -> tuple[DiagonalSSM, dict[str, Any]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)
    device = torch.device(device)

    if not with_replacement and L > V:
        V = L * 2

    model = DiagonalSSM(
        V=V, L_max=max(L, 256), d_model=d_model, k=k, mode=mode, n_layers=2,
        no_pos_embed=no_pos_embed,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best_acc = -1.0
    best_state = None
    history: list[dict[str, float]] = []

    model.train()
    for step in range(1, steps + 1):
        if step < steps // 5:
            L_step = max(8, L // 4)
        elif step < steps // 2:
            L_step = max(16, L // 2)
        else:
            L_step = L

        batch = make_batch(batch_size, L_step, V=V, device=device, with_replacement=with_replacement)
        out = model(batch["input_ids"], batch["query_pos"])
        loss = F.cross_entropy(out["logits"], batch["target"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        model.row_normalize_weights_()

        if log_every and step % log_every == 0:
            with torch.no_grad():
                pred = out["logits"].argmax(-1)
                acc = (pred == batch["target"]).float().mean().item()
            print(
                f"  [B seed={seed} mode={mode} L={L} k={k}] "
                f"step {step}/{steps} loss={loss.item():.4f} acc={acc:.3f} L_step={L_step}",
                flush=True,
            )

        if eval_every and step % eval_every == 0:
            metrics = _quick_acc(model, L=L, V=V, device=device, n=512)
            history.append({"step": float(step), **metrics})
            if metrics["overall_acc"] > best_acc:
                best_acc = metrics["overall_acc"]
                best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final = _quick_acc(model, L=L, V=V, device=device, n=1024)
    meta = {
        "seed": seed,
        "mode": mode,
        "L": L,
        "k": k,
        "best_val_acc": best_acc,
        "final_acc": final["overall_acc"],
        "history": history,
    }
    return model, meta


@torch.no_grad()
def _quick_acc(
    model: DiagonalSSM,
    L: int,
    V: int,
    device: torch.device,
    n: int = 512,
) -> dict[str, float]:
    model.eval()
    batch = make_batch(n, L, V=V, device=device)
    out = model(batch["input_ids"], batch["query_pos"])
    pred = out["logits"].argmax(-1)
    acc = float((pred == batch["target"]).float().mean().item())
    model.train()
    return {"overall_acc": acc}


@torch.no_grad()
def _over_smoothing(states: torch.Tensor, use_raw: bool = True) -> float:
    N, L, k = states.shape
    if L > 32:
        idx = torch.linspace(0, L - 1, 32).long()
        states = states[:, idx]
        L = states.shape[1]
    s = F.normalize(states, dim=-1)
    sim = torch.einsum("nld,nmd->nlm", s, s)
    iu = torch.triu_indices(L, L, offset=1)
    vals = sim[:, iu[0], iu[1]]
    return float(vals.mean().item())
@torch.no_grad()
def _task_conditioned_os(
    states: torch.Tensor,
    target_positions: torch.Tensor,
    n_pairs: int = 500,
) -> float:
    N, L, k = states.shape
    s = F.normalize(states, dim=-1)
    cos_sim = torch.einsum("nld,nmd->nlm", s, s)
    distinct_mask = target_positions.unsqueeze(1) != target_positions.unsqueeze(2)
    vals = cos_sim[distinct_mask]
    if vals.numel() == 0:
        return 0.0
    return float(vals.mean().item())


@torch.no_grad()
def _intervention_drop(
    model: DiagonalSSM,
    L: int,
    V: int,
    device: torch.device,
    n_sequences: int = 256,
) -> float:
    model.eval()
    batch = make_batch(n_sequences, L, V=V, device=device)
    out_clean = model(batch["input_ids"], batch["query_pos"], return_states=True)
    pred_clean = out_clean["logits"].argmax(-1)
    acc_clean = (pred_clean == batch["target"]).float().mean().item()
    states = out_clean.get("states")
    if states is None:
        return 0.0
    mean_state = states.mean(dim=1, keepdim=True)
    mid_start, mid_end = L // 4, 3 * L // 4
    states_intervened = states.clone()
    states_intervened[:, mid_start:mid_end] = mean_state.expand(-1, mid_end - mid_start, -1)
    Bsz, L_seq = batch["input_ids"].shape
    pos = torch.arange(L, device=device).unsqueeze(0).expand(Bsz, L)
    x = model.embed(batch["tokens"])
    if model.pos_embed is not None:
        x = x + model.pos_embed(pos.clamp(0, model.L_max - 1))
    h = states_intervened
    for i in range(model.n_layers):
        y = model._readout_h(h, i)
        y = model.out_proj[i](y)
        noshort = model.mode in ("B0_noshort", "BR_noshort")
        if not noshort:
            x = x + y
        else:
            x = y
    idx = batch["query_pos"].view(Bsz, 1, 1).expand(Bsz, 1, model.k)
    h_q = h.gather(1, idx).squeeze(1)
    y_q = model._readout_h(h_q, model.n_layers - 1)
    noshort = model.mode in ("B0_noshort", "BR_noshort")
    if not noshort:
        idx_d = batch["query_pos"].view(Bsz, 1, 1).expand(Bsz, 1, model.d_model)
        x_q = x.gather(1, idx_d).squeeze(1)
        feat = y_q + x_q
    else:
        feat = y_q
    logits = model.head(feat)
    pred_int = logits.argmax(-1)
    acc_int = (pred_int == batch["target"]).float().mean().item()
    return acc_clean - acc_int
def eval_position(
    model: DiagonalSSM,
    seed: int,
    L: int,
    V: int = V_DEFAULT,
    queries_per_pos: int = 200,
    device: str | torch.device | None = None,
    noise_sigma_frac: float = 0.1,
    do_intervention: bool = True,
    do_decode_probe: bool = True,
    do_task_os: bool = True,
) -> dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed + 20_000)
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    correct = torch.zeros(L, device=device)
    total = torch.zeros(L, device=device)
    states_list: list[torch.Tensor] = []
    h_final_list: list[torch.Tensor] = []
    max_state_batches = 8
    all_h: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for ell in range(L):
            remaining = queries_per_pos
            while remaining > 0:
                bs = min(64, remaining)
                tokens = torch.randint(0, V, (bs, L), device=device)
                query_pos = torch.full((bs,), ell, device=device, dtype=torch.long)
                target = tokens[torch.arange(bs, device=device), query_pos]
                query_tok = torch.full((bs, 1), V, device=device, dtype=tokens.dtype)
                input_ids = torch.cat([tokens, query_tok], dim=1)
                need_states = len(states_list) < max_state_batches and ell == 0
                out = model(input_ids, query_pos, return_states=need_states)
                pred = out["logits"].argmax(-1)
                correct[ell] += (pred == target).sum()
                total[ell] += bs
                if need_states and "states" in out:
                    states_list.append(out["states"].cpu())
                if len(h_final_list) < max_state_batches * 2 and "h_final" in out:
                    hf = out["h_final"]
                    if hf.shape[-1] == model.k:
                        h_final_list.append(hf.cpu())
                if do_decode_probe:
                    all_h.append(out["h_final"].cpu())
                    all_labels.append(target.cpu())
                remaining -= bs
    acc = (correct / total.clamp_min(1)).cpu().numpy()
    a0, amid, alast = float(acc[0]), float(acc[L // 2]), float(acc[L - 1])
    udepth = 0.5 * (a0 + alast) - amid
    udepth_abs = 0.5 * (a0 + alast) - float(acc[L // 4]) if L >= 4 else udepth
    endpoint = 0.5 * (a0 + alast)
    overall = float(acc.mean())
    if states_list:
        states = torch.cat(states_list, dim=0).to(device)
        os_score = _over_smoothing(states, use_raw=True)
        readout_h = states[:, L // 2, :]
        readout_y = model._readout_h(readout_h, model.n_layers - 1)
        rf = F.normalize(readout_y, dim=-1)
        cos_mat = rf @ rf.T
        n = cos_mat.shape[0]
        iu = torch.triu_indices(n, n, offset=1)
        os_readout = float(cos_mat[iu[0], iu[1]].mean().item()) if n > 1 else float("nan")
    else:
        os_score = float("nan")
        os_readout = float("nan")
    tau = model.effective_tau().detach().cpu().numpy()
    tau_mean = float(tau.mean())
    tau_median = float(np.median(tau))
    if h_final_list:
        h = torch.cat(h_final_list, dim=0).to(device)
        p_clean = model.probe_products(h)
        std = h.std().clamp_min(1e-6)
        noise = torch.randn_like(h) * (noise_sigma_frac * std)
        p_obs = model.probe_products(h + noise)
        snr = signal_noise_rho(p_clean, p_obs, max_coords=min(64, h.shape[1]))
        esnr = effective_snr(p_clean, p_obs)
        p_noise = p_obs - p_clean
        vf = variance_fraction(p_clean, p_noise, n_components=min(4, h.shape[1]))
    else:
        snr = {"rho_signal": 0.0, "rho_noise": 0.0, "delta_rho": 0.0}
        esnr = {"snr_effective": 0.0, "alignment": 0.0}
        vf = {"signal_topvar_frac": 0.0, "noise_topvar_frac": 0.0, "var_ratio": 1.0}
    result: dict[str, Any] = {
        "seed": seed,
        "mode": model.mode,
        "L": L,
        "k": model.k,
        "acc_curve": acc.tolist(),
        "udepth": udepth,
        "udepth_abs": udepth_abs,
        "endpoint_acc": endpoint,
        "mid_acc": amid,
        "overall_acc": overall,
        "over_smoothing": os_score,
        "over_smoothing_readout": os_readout,
        "tau_mean": tau_mean,
        "tau_median": tau_median,
        "tau": tau.tolist(),
        **snr,
        **esnr,
        **vf,
    }
    if do_decode_probe and all_h:
        h_all = torch.cat(all_h, dim=0)
        labels_all = torch.cat(all_labels, dim=0)
        probe_acc, _ = train_decode_probe(
            h_all, labels_all, n_classes=V, device=device, steps=100
        )
        result["decode_probe_acc"] = probe_acc
    if do_intervention:
        int_drop = _intervention_drop(model, L, V, device, n_sequences=128)
        result["intervention_drop"] = int_drop
    if do_task_os and states_list:
        states_all = torch.cat(states_list, dim=0).to(device)
        pos_labels = torch.arange(L, device=device).unsqueeze(0).expand(states_all.shape[0], L)
        task_os = _task_conditioned_os(states_all, pos_labels)
        result["task_conditioned_os"] = task_os
    return result
