from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.expC.data import make_copy_batch
from src.expC.faith import (
    FaithResult,
    _runner_up,
    _tangent_project,
    _unit_random,
)
from src.expC.model import CopySSM
from src.expC.train import quick_accuracy, train_copy_ssm
from src.normopt.normalize import tangentialize
from src.normopt.optim import GraftedTangentMuon, MuonMomentum
from src.stats import paired_stats

CONFIG = {
    "vocab": 16,
    "state_dim": 64,
    "d_model": 64,
    "seq_len": 16,
    "train_steps": 1500,
    "eval_seqs": 48,
    "eval_seqs_why": 24,
    "n_dirs": 16,
    "n_dirs_wide": 32,
    "thetas": [0.02, 0.05, 0.1, 0.2, 0.4],
    "thetas_wide": {
        "ordinary": [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        "sphere": [0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
    },
    "thetas_method": [0.05, 0.1, 0.2, 0.4],
    "theta_why": 0.1,
    "theta_full_effect": 0.2,
    "opt_steps": 30,
    "capture_ks": [1, 2, 4, 8],
    "prune_fracs": [0.05, 0.1, 0.25, 0.5],
    "effect_bins": np.arange(-7.0, 1.5, 0.5),
    "floor": 1e-3,
}

QUICK = {
    "eval_seqs": 4,
    "eval_seqs_why": 4,
    "n_dirs": 4,
    "n_dirs_wide": 4,
    "thetas": [0.1],
    "opt_steps": 2,
}


def settings(args, **extra):
    cfg = dict(CONFIG)
    if getattr(args, "quick", False):
        cfg.update(QUICK)
        cfg["thetas_wide"] = {"ordinary": [0.05, 0.1], "sphere": [0.05, 0.1]}
        cfg["thetas_method"] = [0.1]
    cfg.update(extra)
    return cfg


def tangent_dirs(h, m, gen):
    hat = (h / h.norm().clamp_min(1e-9)).float()
    dirs = _tangent_project(_unit_random(m, h.numel(), gen, hat.device), hat)
    return dirs.to(h.dtype)


def split_tangent_normal(update, weight):
    tangent = tangentialize(update, weight)
    axis = weight / (weight.norm(dim=1, keepdim=True) + 1e-8)
    radial = ((update - tangent) * axis).sum(dim=1)
    return radial, tangent


def peek_proposal(opt, param):
    saved = {
        k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
        for k, v in opt.state[param].items()
    }
    try:
        with torch.no_grad():
            proposal = opt.peek_update(param).detach().clone()
    finally:
        current = opt.state[param]
        for k in list(current.keys()):
            if k not in saved:
                del current[k]
        for k, v in saved.items():
            current[k] = v
    for k, v in saved.items():
        if torch.is_tensor(v):
            assert torch.equal(v, opt.state[param][k])
    return proposal


def fit_quad(thetas, maes):
    th = np.array(thetas, dtype=float)
    m = np.array(maes, dtype=float)
    slope = (
        np.polyfit(np.log(th[m > 0]), np.log(m[m > 0]), 1)[0]
        if (m > 0).sum() >= 2
        else float("nan")
    )
    return float((m * th**2).sum() / (th**4).sum()), float(slope)


def pair_metrics(pred, actual):
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(actual) == 0:
        return {"n": 0}
    acc = FaithResult(variant="tangent", seed=0, n_seq=0, thetas=[0.0])
    acc.add_effects(0.0, np.asarray(pred, dtype=float), np.asarray(actual, dtype=float))
    base = acc.metrics(0.0)
    err = np.abs(np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float))
    mag = np.abs(np.asarray(actual, dtype=float))
    q75 = float(np.quantile(mag, 0.75))
    strong = mag >= max(q75, 1e-30)
    return {
        "n": base["n_pairs"],
        "MAE": float(np.mean(err)),
        "E_all_floor": base["E_all"],
        "E_eff": base["E_eff"],
        "median_abs_actual": float(np.median(mag)),
        "pearson": base["calib_r"],
        "spearman_abs": base["rho_rank"],
        "sign_acc": float(np.mean(np.sign(actual) == np.sign(pred))),
        "sign_acc_eff": base["sign_acc"],
        "E_strong": float(np.mean(err[strong] / mag[strong]))
        if strong.sum() >= 8
        else float("nan"),
    }


def check_math():
    rng = torch.Generator().manual_seed(0)
    out = {}
    worst = 0.0
    for _ in range(20):
        x = torch.randn(32, generator=rng, dtype=torch.float64)
        v = torch.randn(32, generator=rng, dtype=torch.float64)
        hat = x / x.norm()
        proj = torch.eye(32, dtype=torch.float64) - torch.outer(hat, hat)
        h = 1e-7
        fd = ((x + h * v) / (x + h * v).norm() - x / x.norm()) / h
        worst = max(worst, float(((proj @ v / x.norm()) - fd).norm()))
    out["DN_finite_difference_max_err"] = worst
    x = torch.randn(16, generator=rng, dtype=torch.float64)
    hat = x / x.norm()
    proj = torch.eye(16, dtype=torch.float64) - torch.outer(hat, hat)
    out["projector_idempotent_err"] = float((proj @ proj - proj).norm())
    out["projector_symmetric_err"] = float((proj - proj.T).norm())
    out["projector_kills_radial"] = float((proj @ hat).norm())
    worst_angle, worst_first = 0.0, 0.0
    for _ in range(200):
        w = torch.randn(32, generator=rng, dtype=torch.float64)
        w = w / w.norm()
        u = torch.randn(32, generator=rng, dtype=torch.float64)
        a = float(u @ w)
        t = u - a * w
        for eta in (0.003, 0.02, 0.1):
            formula = math.atan2(eta * float(t.norm()), 1.0 - eta * a)
            moved = w - eta * u
            moved = moved / moved.norm()
            measured = math.acos(float(torch.clamp(w @ moved, -1.0, 1.0)))
            worst_angle = max(worst_angle, abs(formula - measured))
            if eta == 0.003:
                worst_first = max(worst_first, float(((moved - w) + eta * t).norm()))
    out["theta_formula_max_abs_err_rad"] = worst_angle
    out["first_order_direction_err_eta0.003"] = worst_first
    out["theta_formula_holds"] = bool(worst_angle < 1e-9)
    return out


def sample_sequence(model, site, seq_len, gen, device):
    with torch.no_grad():
        batch = make_copy_batch(
            1, seq_len, V=CONFIG["vocab"], device=device, generator=gen
        )
        ids = batch["input_ids"].to(device)
        pos = batch["query_pos"].to(device)
        q = int(pos[0].item())
        if q < 2:
            return None
        out = model(ids, pos, return_all=True)
        logits = out["logits"][0]
        target = int(batch["target"][0].item())
    alt = _runner_up(logits, target)
    clean = float(logits[target].item() - logits[alt].item())
    with torch.no_grad():
        ctx = {
            "q": q,
            "target": target,
            "alt": alt,
            "clean": clean,
            "x_q": out["x_q"][0],
            "logits": logits,
        }
        if site == "mid":
            t0 = max(1, q // 2)
            if t0 >= q:
                return None
            ctx["t0"] = t0
            ctx["h"] = out["h_last"][0][t0].detach().clone()
            ctx["h_last"] = out["h_last"][0].detach().clone()
            ctx["x1"] = out["layer_inputs"][1][0]
        else:
            ctx["h"] = out["h_q"][0].detach().clone()
    return ctx


def state_gradient(model, ctx, site, create_graph=False):
    leaf = ctx["h"].detach().clone().requires_grad_(True)
    if site == "mid":
        hq = model.tail_scan_to_q(ctx["x1"], ctx["t0"], leaf.unsqueeze(0), ctx["q"])
    else:
        hq = leaf.unsqueeze(0)
    lp = model.logits_from_final_state(hq, ctx["x_q"].unsqueeze(0))[0]
    g = torch.autograd.grad(
        lp[ctx["target"]] - lp[ctx["alt"]], leaf, create_graph=create_graph
    )[0]
    return leaf, g


def rescore(model, ctx, site, states):
    with torch.no_grad():
        if site == "mid":
            final = model.tail_scan_to_q(ctx["x1"], ctx["t0"], states, ctx["q"])
        else:
            final = states
        lg = model.logits_from_final_state(
            final, ctx["x_q"].unsqueeze(0).expand(len(states), -1)
        )
        return (lg[:, ctx["target"]] - lg[:, ctx["alt"]]).cpu().numpy() - ctx["clean"]


def geodesic(h, dirs, theta):
    r = float(h.norm().item())
    hat = h / r
    return r, r * (math.cos(theta) * hat.unsqueeze(0) + math.sin(theta) * dirs)


def public_config(cfg):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()}


def train_variant(seed, variant, cfg, device, out_dir, log_every):
    model, meta = train_copy_ssm(
        seed=seed,
        variant=variant,
        L=cfg["seq_len"],
        k=cfg["state_dim"],
        d_model=cfg["d_model"],
        steps=cfg["train_steps"],
        batch_size=64,
        device=str(device),
        log_every=log_every,
    )
    torch.save(model.state_dict(), out_dir / f"ckpt_seed{seed}_{variant}.pt")
    acc = quick_accuracy(
        model, L=cfg["seq_len"], V=CONFIG["vocab"], device=device, n=512
    )
    return model, meta["final_acc"], acc


def fresh_model(seed, variant, cfg, ckpt_dir, dtype=None):
    model = CopySSM(
        V=CONFIG["vocab"],
        L=cfg["seq_len"],
        d_model=cfg["d_model"],
        k=cfg["state_dim"],
        sphere=(variant == "sphere"),
    )
    model.load_state_dict(
        torch.load(
            ckpt_dir / f"ckpt_seed{seed}_{variant}.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def run_partA(model, seed, cfg, device):
    model = model.to(device).eval()
    gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    collected = {"ANG": {}, "ABS": {}}
    h_norms, margins, n_correct, n_used = [], [], 0, 0
    for _ in range(cfg["eval_seqs"]):
        ctx = sample_sequence(model, "mid", cfg["seq_len"], gen, device)
        if ctx is None:
            continue
        n_correct += int(ctx["logits"].argmax().item() == ctx["target"])
        r = float(ctx["h"].norm().item())
        if not math.isfinite(r) or r < 1e-9:
            continue
        h_norms.append(r)
        margins.append(ctx["clean"])
        n_used += 1
        _, g = state_gradient(model, ctx, "mid")
        dirs = tangent_dirs(ctx["h"].cpu(), cfg["n_dirs"], gen)
        dots = g.cpu() @ dirs.T
        for theta in cfg["thetas"]:
            _, pert = geodesic(ctx["h"].cpu(), dirs, theta)
            actual = rescore(model, ctx, "mid", pert.to(device))
            collected["ANG"].setdefault(theta, []).append(
                ((theta * r * dots).numpy(), actual)
            )
            flat = ctx["h"].cpu().unsqueeze(0) + theta * dirs
            actual_abs = rescore(model, ctx, "mid", flat.to(device))
            collected["ABS"].setdefault(theta, []).append(
                ((theta * dots).numpy(), actual_abs)
            )
    summary = {"n_used": n_used, "acc_eval": n_correct / max(1, cfg["eval_seqs"])}
    for proto, groups in collected.items():
        summary[proto] = {}
        for theta, pairs in groups.items():
            p = np.concatenate([x[0] for x in pairs])
            a = np.concatenate([x[1] for x in pairs])
            err = np.abs(a - p)
            mag = np.abs(a)
            summary[proto][str(theta)] = {
                "n": int(len(a)),
                "MAE": float(np.mean(err)),
                "E_sig": float(np.mean(err / (mag + CONFIG["floor"]))),
                "median_abs_actual": float(np.median(mag)),
                "mean_h_norm": float(np.mean(h_norms)),
                "mean_margin": float(np.mean(margins)),
            }
    summary["h_norm_median"] = float(np.median(h_norms)) if h_norms else float("nan")
    return summary


def aggregate_partA(per_seed, seeds, thetas):
    paired = {}
    for proto in ("ANG", "ABS"):
        paired[proto] = {}
        for th in thetas:

            def get(v, m, th=th):
                return [per_seed[f"seed{s}_{v}"][proto][str(th)][m] for s in seeds]

            paired[proto][str(th)] = {
                "MAE": paired_stats(get("ordinary", "MAE"), get("sphere", "MAE")),
                "E_sig": paired_stats(get("ordinary", "E_sig"), get("sphere", "E_sig")),
            }
    quad = {}
    for variant in ("ordinary", "sphere"):
        for proto in ("ANG", "ABS"):
            coefs, slopes = [], []
            for s in seeds:
                maes = [
                    per_seed[f"seed{s}_{variant}"][proto][str(t)]["MAE"]
                    for t in thetas[:3]
                ]
                c, sl = fit_quad(thetas[:3], maes)
                coefs.append(c)
                slopes.append(sl)
            quad[f"{variant}_{proto}"] = {
                "c_mean": float(np.mean(coefs)),
                "c_per_seed": coefs,
                "loglog_slope_mean": float(np.mean(slopes)),
            }
    return paired, quad


def make_optimizer(name, param):
    if name == "muon":
        return MuonMomentum([param], lr=0.02, ns_steps=5)
    return GraftedTangentMuon([param], lr=0.003, ns_steps=3)


def seeded_generator(salt):
    return torch.Generator().manual_seed(salt)


def trace_optimizer(name, seed, cfg, device):
    w0 = torch.randn(16, 32, generator=seeded_generator(1000 + seed))
    w0 = F.normalize(w0, dim=1)
    w_true = torch.randn(16, 32, generator=seeded_generator(2000 + seed))
    x = torch.randn(256, 32, generator=seeded_generator(3000 + seed))
    y = x @ w_true.T
    x, y = x.to(device), y.to(device)
    w = w0.clone().to(device).requires_grad_(True)
    qs = w0.clone().to(device).requires_grad_(True)
    opt = make_optimizer(name, w)
    shadow = make_optimizer("gtmuon" if name == "muon" else "muon", qs)
    eta = opt.param_groups[0]["lr"]
    eta_other = shadow.param_groups[0]["lr"]
    steps, probes = [], []
    for step in range(cfg["opt_steps"]):
        opt.zero_grad(set_to_none=True)
        shadow.zero_grad(set_to_none=True)
        loss = F.mse_loss(x @ w.T, y)
        loss.backward()
        g = w.grad.detach().clone()
        before = w.detach().clone()
        with torch.no_grad():
            qs.copy_(before)
        qs.grad = g.clone()
        own = peek_proposal(opt, w)
        _, t_own = split_tangent_normal(own, before)
        alt = peek_proposal(shadow, qs)
        _, t_alt = split_tangent_normal(alt, before)
        budget = float((eta * t_own).norm())
        alt_budget = float((eta_other * t_alt).norm())
        rescale = (budget / max(alt_budget, 1e-24)) if alt_budget > 0 else 1.0
        with torch.no_grad():
            hypo_own = F.normalize(before - eta * t_own, dim=1)
            hypo_alt = F.normalize(before + (-eta_other * t_alt) * rescale, dim=1)
            loss_own = float(F.mse_loss(x @ hypo_own.T, y))
            loss_alt = float(F.mse_loss(x @ hypo_alt.T, y))
        current = float(loss.detach())
        opt.step()
        shadow.step()
        with torch.no_grad():
            raw = w.detach().clone()
            committed = (before - raw) / eta
            radial, tangent = split_tangent_normal(committed, before)
            w.copy_(F.normalize(raw, dim=1))
        t_norm = float(tangent.norm())
        u_norm = float(committed.norm())
        with torch.no_grad():
            loss_new = float(F.mse_loss(x @ w.detach().T, y))
            c0 = F.normalize(before, dim=1)
            c1 = F.normalize(w.detach(), dim=1)
            measured = torch.acos((c0 * c1).sum(dim=1).clamp(-1, 1))
            formula = torch.atan2(eta * tangent.norm(dim=1), 1.0 - eta * radial)
            angle_err = (measured - formula).abs()
        steps.append(
            {
                "step": step,
                "loss": current,
                "loss_new": loss_new,
                "dL_actual": loss_new - current,
                "dL_pred_tan": float((g * (-eta * tangent)).sum()),
                "dL_pred_full": float((g * (-eta * committed)).sum()),
                "tan_frac": float(t_norm**2 / max(u_norm**2, 1e-24)),
                "theta_meas_mean": float(measured.mean()),
                "theta_formula_maxerr": float(angle_err.max()),
            }
        )
        probes.append(
            {
                "seed": seed,
                "dL_own": loss_own - current,
                "dL_alt_matched": loss_alt - current,
            }
        )
    with torch.no_grad():
        final_state = {"opt_state": opt.state_dict(), "W": w.detach().cpu()}
    return steps, probes, final_state


def summarize_traces(traces):
    flat = [s for tr in traces for s in tr["steps"]]
    return {
        "traces": traces,
        "tan_frac_mean": float(np.mean([s["tan_frac"] for s in flat])),
        "tan_frac_min": float(np.min([s["tan_frac"] for s in flat])),
        "theta_formula_maxerr_overall": float(
            np.max([s["theta_formula_maxerr"] for s in flat])
        ),
        "loss_pred_err_tan_mean": float(
            np.mean([abs(s["dL_actual"] - s["dL_pred_tan"]) for s in flat])
        ),
        "loss_pred_err_full_mean": float(
            np.mean([abs(s["dL_actual"] - s["dL_pred_full"]) for s in flat])
        ),
        "loss_drop_mean": float(np.mean([tr["loss0"] - tr["lossT"] for tr in traces])),
    }


def run_partB(seeds, cfg, device, out_dir):
    per_opt, probe_rows = {}, []
    for name in ("muon", "gtmuon"):
        traces = []
        for seed in seeds:
            steps, probes, final_state = trace_optimizer(name, seed, cfg, device)
            torch.save(final_state, out_dir / f"opt_{name}_seed{seed}.pt")
            traces.append(
                {
                    "seed": seed,
                    "steps": steps,
                    "loss0": steps[0]["loss"],
                    "lossT": steps[-1]["loss_new"],
                }
            )
            if name == "gtmuon":
                probe_rows.extend(probes)
        per_opt[name] = summarize_traces(traces)
    own = np.array([r["dL_own"] for r in probe_rows])
    alt = np.array([r["dL_alt_matched"] for r in probe_rows])
    per_opt["matched_budget_probe"] = {
        "n": int(len(own)),
        "mean_dL_gt": float(own.mean()) if len(own) else float("nan"),
        "mean_dL_muon_matched": float(alt.mean()) if len(alt) else float("nan"),
        "mean_diff_gt_minus_mu": float((own - alt).mean())
        if len(own)
        else float("nan"),
        "frac_steps_gt_better": float((own < alt).mean()) if len(own) else float("nan"),
    }
    return per_opt


def collect_site(model, site, seed, cfg, device):
    model = model.eval().to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed + 50_000)
    thetas = cfg["thetas_wide"]["sphere" if model.sphere else "ordinary"]
    n_dirs = cfg["n_dirs_wide"]
    pairs = {th: [] for th in thetas}
    captures, gnorms, n_used = [], [], 0
    for _ in range(cfg["eval_seqs"]):
        ctx = sample_sequence(model, site, cfg["seq_len"], gen, device)
        if ctx is None or (site == "mid" and "t0" not in ctx):
            continue
        n_used += 1
        _, g = state_gradient(model, ctx, site)
        gnorms.append(float(g.norm().item()))
        dirs = tangent_dirs(ctx["h"].cpu(), n_dirs, gen).to(dtype=torch.float64)
        dots = g.cpu().to(dtype=torch.float64) @ dirs.T
        r = float(ctx["h"].norm().item())
        hat = (ctx["h"] / r).cpu()
        for th in thetas:
            pert = r * (
                math.cos(th) * hat.unsqueeze(0).to(torch.float64) + math.sin(th) * dirs
            )
            actual = rescore(model, ctx, site, pert.to(device))
            pred = (th * r * dots).numpy()
            pairs[th].append((pred, actual.astype(np.float64)))
            mag = np.abs(actual)
            total = mag.sum()
            row = {"theta": th}
            if total > 0:
                by_pred = np.argsort(-np.abs(pred))
                by_true = np.argsort(-mag)
                for k in cfg["capture_ks"]:
                    kk = min(k, len(mag))
                    row[f"cap{k}"] = float(mag[by_pred[:kk]].sum() / total)
                    row[f"oracle{k}"] = float(mag[by_true[:kk]].sum() / total)
            captures.append(row)
    P = {
        th: np.concatenate([p for p, _ in v]) if v else np.array([])
        for th, v in pairs.items()
    }
    A = {
        th: np.concatenate([a for _, a in v]) if v else np.array([])
        for th, v in pairs.items()
    }
    return P, A, captures, np.array(gnorms), n_used


def pooled_binning(store, site, seeds, wide):
    bank = {}
    for variant in ("ordinary", "sphere"):
        bank[variant] = (
            np.concatenate(
                [
                    store[f"{site}/seed{s}_{variant}/{th}"][0]
                    for s in seeds
                    for th in wide[variant]
                ]
            ),
            np.concatenate(
                [
                    store[f"{site}/seed{s}_{variant}/{th}"][1]
                    for s in seeds
                    for th in wide[variant]
                ]
            ),
        )
    rows = []
    for i in range(len(CONFIG["effect_bins"]) - 1):
        lo, hi = 10 ** CONFIG["effect_bins"][i], 10 ** CONFIG["effect_bins"][i + 1]
        row = {"bin": [float(lo), float(hi)]}
        for variant in ("ordinary", "sphere"):
            P, A = bank[variant]
            hit = (np.abs(A) >= lo) & (np.abs(A) < hi)
            row[f"{variant}_n"] = int(hit.sum())
            row[f"{variant}_relerr"] = (
                float(np.mean(np.abs(A[hit] - P[hit]) / np.abs(A[hit])))
                if hit.sum() >= 20
                else float("nan")
            )
        a, b = row["ordinary_relerr"], row["sphere_relerr"]
        row["ratio_sph_over_ord"] = (
            float(b / a)
            if np.isfinite(a) and np.isfinite(b) and a > 0
            else float("nan")
        )
        rows.append(row)
    overlap = [r for r in rows if r["ordinary_n"] >= 100 and r["sphere_n"] >= 100]
    ratios = [
        r["ratio_sph_over_ord"] for r in overlap if np.isfinite(r["ratio_sph_over_ord"])
    ]
    return {
        "bins": rows,
        "n_overlap_bins": len(overlap),
        "mean_ratio_sph_over_ord": float(np.mean(ratios)) if ratios else float("nan"),
        "frac_bins_sphere_better": float(np.mean([v < 1 for v in ratios]))
        if ratios
        else float("nan"),
    }


def run_mech(ckpt_dir, out_dir, seeds, cfg, device, fname="mech.json", save_pairs=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    store = {}
    summary = {"sites": {}, "signal_matched": {}, "capture": {}, "paired_focus": {}}
    for site in ("mid", "final"):
        summary["sites"][site] = {}
        for s in seeds:
            for variant in ("ordinary", "sphere"):
                model = fresh_model(s, variant, cfg, ckpt_dir, dtype=torch.float64)
                P, A, captures, gnorms, n_used = collect_site(
                    model, site, s, cfg, device
                )
                key = f"seed{s}_{variant}"
                for th in cfg["thetas_wide"][variant]:
                    store[f"{site}/{key}/{th}"] = np.stack([P[th], A[th]])
                per_theta = {
                    str(th): pair_metrics(P[th], A[th])
                    for th in cfg["thetas_wide"][variant]
                }
                for th in cfg["thetas_wide"][variant]:
                    rows = [c for c in captures if c.get("theta") == th]
                    for k in cfg["capture_ks"]:
                        vals = [c[f"cap{k}"] for c in rows if f"cap{k}" in c]
                        refs = [c[f"oracle{k}"] for c in rows if f"oracle{k}" in c]
                        per_theta[str(th)][f"cap{k}_mean"] = (
                            float(np.mean(vals)) if vals else float("nan")
                        )
                        per_theta[str(th)][f"oracle{k}_mean"] = (
                            float(np.mean(refs)) if refs else float("nan")
                        )
                per_theta["grad_norm_median"] = (
                    float(np.median(gnorms)) if len(gnorms) else float("nan")
                )
                per_theta["n_used"] = n_used
                summary["sites"][site][key] = per_theta
                print(
                    f"  {site} seed={s} {variant}: |g|={per_theta['grad_norm_median']:.2e} n={n_used}",
                    flush=True,
                )
        summary["signal_matched"][site] = pooled_binning(
            store, site, seeds, cfg["thetas_wide"]
        )
        matched = summary["signal_matched"][site]
        print(
            f"  [{site}] overlap={matched['n_overlap_bins']} "
            f"ratio={matched['mean_ratio_sph_over_ord']:.3f} "
            f"sph_better={matched['frac_bins_sphere_better']:.2f}",
            flush=True,
        )
        caught = {}
        for variant in ("ordinary", "sphere"):
            for th in cfg["thetas_wide"][variant]:
                for k in cfg["capture_ks"]:
                    vals = [
                        summary["sites"][site][f"seed{s}_{variant}"][str(th)][
                            f"cap{k}_mean"
                        ]
                        for s in seeds
                    ]
                    refs = [
                        summary["sites"][site][f"seed{s}_{variant}"][str(th)][
                            f"oracle{k}_mean"
                        ]
                        for s in seeds
                    ]
                    caught.setdefault(str(th), {})[f"{variant}_cap{k}"] = float(
                        np.nanmean(vals)
                    )
                    caught[str(th)][f"{variant}_oracle{k}"] = float(np.nanmean(refs))
        summary["capture"][site] = caught
        for th in sorted(
            set(cfg["thetas_wide"]["ordinary"]) & set(cfg["thetas_wide"]["sphere"])
        ):
            row = {}
            for metric in (
                "MAE",
                "E_strong",
                "E_eff",
                "pearson",
                "spearman_abs",
                "cap4_mean",
            ):

                def get(v, metric=metric, th=th):
                    return [
                        summary["sites"][site][f"seed{s}_{v}"][str(th)][metric]
                        for s in seeds
                    ]

                row[metric] = paired_stats(get("ordinary"), get("sphere"))
            summary["paired_focus"].setdefault(site, {})[str(th)] = row
    summary["config"] = {
        "thetas_wide": cfg["thetas_wide"],
        "eval_seqs": cfg["eval_seqs"],
        "n_dirs_wide": cfg["n_dirs_wide"],
        "seeds": list(seeds),
        "seq_len": cfg["seq_len"],
        "state_dim": cfg["state_dim"],
        "d_model": cfg["d_model"],
    }
    with open(out_dir / fname, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    if save_pairs:
        np.savez_compressed(
            out_dir / "mech_pairs.npz",
            **{k.replace("/", "_"): v for k, v in store.items()},
        )
    return summary


def trajectory_prenorms(model, x1):
    with torch.no_grad():
        A = model.A_bar(model.n_layers - 1)
        Bu = F.linear(x1, model.B[model.n_layers - 1])
        h = torch.zeros(model.k, device=x1.device, dtype=x1.dtype)
        pres = []
        for t in range(x1.shape[0]):
            pre = A * h + Bu[t]
            pres.append(float(pre.norm().item()))
            h = pre / (pre.norm() + 1e-8) if model.sphere else pre
    return pres


def run_why(ckpt_dir, seeds, cfg, device):
    out = {}
    for variant in ("ordinary", "sphere"):
        rows, c_norms = [], []
        for s in seeds:
            model = fresh_model(s, variant, cfg, ckpt_dir).eval().to(device)
            with torch.no_grad():
                c_norms.append(
                    float(model.C[model.n_layers - 1].detach().norm().item())
                )
            gen = torch.Generator(device="cpu").manual_seed(s + 99_000)
            for _ in range(cfg["eval_seqs_why"]):
                ctx = sample_sequence(model, "mid", cfg["seq_len"], gen, device)
                if ctx is None or ctx["q"] < 4:
                    continue
                pres = trajectory_prenorms(model, ctx["x1"])
                for t0 in range(1, ctx["q"]):
                    ctx_t0 = dict(ctx, t0=t0, h=ctx["h_last"][t0].detach().clone())
                    _, g = state_gradient(model, ctx_t0, "mid")
                    dirs = tangent_dirs(ctx_t0["h"].cpu(), cfg["n_dirs"], gen)
                    _, pert = geodesic(ctx_t0["h"].cpu(), dirs, cfg["theta_why"])
                    actual = rescore(model, ctx_t0, "mid", pert.to(device))
                    rows.append(
                        {
                            "d": ctx["q"] - t0,
                            "gnorm": float(g.norm().item()),
                            "medA": float(np.median(np.abs(actual))),
                            "prenorm": float(np.mean(pres[t0 : ctx["q"]])),
                        }
                    )
        dist = np.array([r["d"] for r in rows])
        gmag = np.array([r["gnorm"] for r in rows])
        if len(rows) == 0:
            out[variant] = {"C_last_fro": float(np.mean(c_norms)), "n_rows": 0}
            print(f"{variant}: no usable sequences", flush=True)
            continue
        slope = float(np.polyfit(dist[gmag > 0], np.log(gmag[gmag > 0]), 1)[0])
        ds = sorted(set(r["d"] for r in rows))
        out[variant] = {
            "C_last_fro": float(np.mean(c_norms)),
            "damping_per_step": float(np.exp(slope)),
            "log_slope": slope,
            "gnorm_vs_d": [
                {
                    "d": int(d),
                    "gnorm_med": float(
                        np.median([r["gnorm"] for r in rows if r["d"] == d])
                    ),
                    "medA_med": float(
                        np.median([r["medA"] for r in rows if r["d"] == d])
                    ),
                }
                for d in ds
            ],
            "prenorm_mean": float(
                np.mean([r["prenorm"] for r in rows if r["prenorm"] == r["prenorm"]])
            ),
        }
        print(
            f"{variant}: C_fro={out[variant]['C_last_fro']:.2f} "
            f"damping={out[variant]['damping_per_step']:.4f} "
            f"prenorm={out[variant]['prenorm_mean']:.2f}",
            flush=True,
        )
    return out


def run_method(ckpt_dir, seeds, cfg, device):
    res = {"T2 prediction": {}, "T1 ranking": {}, "radial": {}}
    for variant in ("ordinary", "sphere"):
        errs = {th: [] for th in cfg["thetas_method"]}
        radfracs = []
        rec = {
            "euclid": {f: [] for f in cfg["prune_fracs"]},
            "tangent": {f: [] for f in cfg["prune_fracs"]},
        }
        for s in seeds:
            model = (
                fresh_model(s, variant, cfg, ckpt_dir, dtype=torch.float64)
                .eval()
                .to(device)
            )
            gen = torch.Generator(device="cpu").manual_seed(s + 777_000)
            for _ in range(cfg["eval_seqs"]):
                ctx = sample_sequence(model, "mid", cfg["seq_len"], gen, device)
                if ctx is None:
                    continue
                r = float(ctx["h"].norm().item())
                hat = ctx["h"] / r
                leaf, g = state_gradient(model, ctx, "mid", create_graph=True)
                g_det = g.detach()
                grad_rad = float((g_det @ hat).item())
                radfracs.append(abs(grad_rad) / max(float(g_det.norm().item()), 1e-30))
                dirs = tangent_dirs(ctx["h"].cpu(), cfg["n_dirs_wide"], gen).to(
                    device=device, dtype=torch.float64
                )
                dots = g_det @ dirs.T
                hess = torch.stack(
                    [
                        torch.autograd.grad(
                            (g * dirs[j]).sum(), leaf, retain_graph=True
                        )[0].detach()
                        for j in range(cfg["n_dirs_wide"])
                    ]
                )
                quad = (dirs * hess).sum(dim=1)
                for th in cfg["thetas_method"]:
                    _, pert = geodesic(ctx["h"].cpu(), dirs.cpu(), th)
                    base = (th * r * dots).detach().cpu().numpy()
                    first = base + (-(th**2) * r / 2.0 * grad_rad)
                    second = first + (
                        (th**2) * r**2 / 2.0 * quad.detach().cpu().numpy()
                    )
                    actual = rescore(model, ctx, "mid", pert.to(device))
                    errs[th].append(
                        (
                            np.abs(actual - base),
                            np.abs(actual - first),
                            np.abs(actual - second),
                            np.abs(actual),
                        )
                    )
                with torch.no_grad():
                    probe = tangent_dirs(ctx["h"].cpu(), 1, gen).to(
                        device=device, dtype=torch.float64
                    )[0]
                    _, probe_pert = geodesic(
                        ctx["h"].cpu(), probe.cpu(), cfg["theta_full_effect"]
                    )
                    dfull = float(rescore(model, ctx, "mid", probe_pert.to(device))[0])
                gtan = g_det - float(g_det @ hat) * hat
                orders = {
                    "euclid": torch.argsort(g_det.abs(), descending=True),
                    "tangent": torch.argsort(gtan.abs(), descending=True),
                }
                with torch.no_grad():
                    for f in cfg["prune_fracs"]:
                        kept = max(1, int(round(f * model.k)))
                        for name, order in orders.items():
                            mask = torch.zeros(model.k, dtype=torch.bool, device=device)
                            mask[order[:kept]] = True
                            patched = torch.where(
                                mask.unsqueeze(0),
                                probe_pert.to(device),
                                ctx["h"].unsqueeze(0),
                            )
                            dp = float(rescore(model, ctx, "mid", patched)[0])
                            rec[name][f].append(abs(dp) / max(abs(dfull), 1e-30))
        res["T2 prediction"][variant] = {}
        for th in cfg["thetas_method"]:
            if not errs[th]:
                res["T2 prediction"][variant][str(th)] = {"n": 0}
                continue
            e0 = np.concatenate([x[0] for x in errs[th]])
            e1 = np.concatenate([x[1] for x in errs[th]])
            e2 = np.concatenate([x[2] for x in errs[th]])
            aa = np.concatenate([x[3] for x in errs[th]])
            strong = aa >= max(float(np.quantile(aa, 0.75)), 1e-30)
            res["T2 prediction"][variant][str(th)] = {
                "MAE_C0": float(e0.mean()),
                "MAE_C1": float(e1.mean()),
                "MAE_C2": float(e2.mean()),
                "Estrong_C0": float(np.mean(e0[strong] / aa[strong])),
                "Estrong_C1": float(np.mean(e1[strong] / aa[strong])),
                "Estrong_C2": float(np.mean(e2[strong] / aa[strong])),
            }
        res["T1 ranking"][variant] = {
            str(f): {
                "euclid": float(np.mean(rec["euclid"][f])),
                "tangent": float(np.mean(rec["tangent"][f])),
            }
            for f in cfg["prune_fracs"]
        }
        res["radial"][variant] = {
            "median_grad_radial_fraction": float(np.median(radfracs))
        }
        print(
            f"{variant}: radfrac={res['radial'][variant]['median_grad_radial_fraction']:.3f}",
            flush=True,
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--device", default="cpu")
    base.add_argument("--quick", action="store_true")
    p = sub.add_parser("full", parents=[base])
    p.add_argument("--out", default="results/tangent_geometry")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, default=3)
    p = sub.add_parser("mech", parents=[base])
    p.add_argument("--inp", default="results/tangent_geometry")
    p.add_argument("--out", default="results/tangent_geometry")
    p.add_argument("--seeds", type=int, default=3)
    p = sub.add_parser("scale", parents=[base])
    p.add_argument("--out", default="results/tangent_scale")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=2)
    p = sub.add_parser("why", parents=[base])
    p.add_argument("--inp", default="results/tangent_geometry")
    p.add_argument("--seeds", type=int, default=3)
    p = sub.add_parser("method", parents=[base])
    p.add_argument("--inp", default="results/tangent_geometry")
    p.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    device = torch.device(a.device)
    seeds = list(range(a.seeds))

    if a.cmd == "full":
        t_all = time.time()
        cfg = settings(a, train_steps=a.steps)
        print(json.dumps(check_math(), indent=2), flush=True)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        per_seed = {}
        for seed in seeds:
            for variant in ("ordinary", "sphere"):
                t0 = time.time()
                model, train_acc, acc = train_variant(
                    seed, variant, cfg, device, out, log_every=0
                )
                ev = run_partA(model, seed, cfg, device)
                ev["train_final_acc"] = train_acc
                ev["quick_acc_512"] = acc
                per_seed[f"seed{seed}_{variant}"] = ev
                print(
                    f"  seed={seed} {variant}: train={train_acc:.3f} eval={acc:.3f} "
                    f"h={ev['h_norm_median']:.3f} ({time.time() - t0:.1f}s)",
                    flush=True,
                )
        paired, quad = aggregate_partA(per_seed, seeds, cfg["thetas"])
        partB = run_partB(seeds, cfg, device, out)
        summary = {
            "experiment": "tangent_space_fidelity_under_normalization",
            "config": public_config(cfg),
            "analytic_checks": check_math(),
            "partA": {
                "config": public_config(cfg),
                "per_seed": per_seed,
                "paired": paired,
                "quadratic_fit_small_thetas": quad,
            },
            "partB": partB,
            "runtime_seconds": round(time.time() - t_all, 1),
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        for proto in ("ANG", "ABS"):
            for th in cfg["thetas"]:
                m = paired[proto][str(th)]["MAE"]
                print(
                    f"[{proto} {th}] ord={m['mean_ordinary']:.4f} "
                    f"sph={m['mean_sphere']:.4f} p={m.get('p_value')}",
                    flush=True,
                )
        print(f"Wrote {out / 'summary.json'}", flush=True)

    if a.cmd == "mech":
        t0 = time.time()
        cfg = settings(a)
        summary = run_mech(Path(a.inp), Path(a.out), seeds, cfg, device)
        summary["runtime_seconds"] = round(time.time() - t0, 1)
        with open(Path(a.out) / "mech.json", "w") as f:
            json.dump(summary, f, indent=1, default=float)
        print(f"Wrote {Path(a.out) / 'mech.json'}", flush=True)

    if a.cmd == "scale":
        t_all = time.time()
        cfg = settings(a, train_steps=a.steps, state_dim=128, seq_len=32)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        parity = {}
        for s in seeds:
            for variant in ("ordinary", "sphere"):
                t0 = time.time()
                _, train_acc, acc = train_variant(
                    s, variant, cfg, device, out, log_every=1000
                )
                parity[f"seed{s}_{variant}"] = {"train_acc": train_acc, "quick512": acc}
                print(
                    f"  seed={s} {variant}: train={train_acc:.3f} eval={acc:.3f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
        summary = run_mech(
            out, out, seeds, cfg, device, fname="mech_scale.json", save_pairs=False
        )
        summary["parity"] = parity
        summary["runtime_seconds"] = round(time.time() - t_all, 1)
        with open(out / "mech_scale.json", "w") as f:
            json.dump(summary, f, indent=1, default=float)
        print(f"Wrote {out / 'mech_scale.json'}", flush=True)

    if a.cmd == "why":
        t0 = time.time()
        out = run_why(Path(a.inp), seeds, settings(a), device)
        out["runtime_seconds"] = round(time.time() - t0, 1)
        with open(Path(a.inp) / "why.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"Wrote {Path(a.inp) / 'why.json'}", flush=True)

    if a.cmd == "method":
        t0 = time.time()
        out = run_method(Path(a.inp), seeds, settings(a), device)
        out["runtime_seconds"] = round(time.time() - t0, 1)
        with open(Path(a.inp) / "method.json", "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"Wrote {Path(a.inp) / 'method.json'}", flush=True)


if __name__ == "__main__":
    main()
