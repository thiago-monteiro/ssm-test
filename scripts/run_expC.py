from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.expC.circuits import eval_path_recovery, eval_pruning_recovery
from src.expC.faith import FaithfulnessEvaluator
from src.expC.model import CopySSM
from src.expC.perf import eval_geometry, eval_perf, eval_robustness
from src.expC.train import train_copy_ssm
from src.stats import compare_groups, improvement, paired_stats

ROOT = Path(__file__).resolve().parents[1]

THETA_SWEEP = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
MID_THETAS = [0.02, 0.05, 0.1, 0.2, 0.4]
MID_T0_FRACS = [0.25, 0.5, 0.75]
PRUNE_FRACS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
VARIANTS = ("ordinary", "sphere")


def load_model(
    ckpt_path: Path, variant: str, L: int, k: int, V: int, d_model: int
) -> CopySSM:
    model = CopySSM(V=V, L=L, d_model=d_model, k=k, sphere=variant == "sphere")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model.eval()


def phase_train(args) -> None:
    ckpt_dir = Path(args.out) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for seed in range(args.seeds):
        for variant in VARIANTS:
            t0 = time.time()
            model, meta = train_copy_ssm(
                seed=seed,
                variant=variant,
                L=args.L,
                k=args.k,
                steps=args.steps,
                batch_size=64,
                device=args.device,
                log_every=1000,
                delay=args.delay,
            )
            torch.save(model.state_dict(), ckpt_dir / f"seed{seed}_{variant}.pt")
            meta["train_seconds"] = round(time.time() - t0, 1)
            metas.append(meta)
            print(
                f"  saved {ckpt_dir / f'seed{seed}_{variant}.pt'} ({meta['final_acc']:.4f} acc)",
                flush=True,
            )
    with open(Path(args.out) / "train_meta.json", "w") as f:
        json.dump(metas, f, indent=2)


def phase_eval_one(seed: int, variant: str, args) -> dict:
    ckpt = Path(args.out) / "ckpt" / f"seed{seed}_{variant}.pt"
    model = load_model(ckpt, variant, args.L, args.k, 16, 64)
    res: dict = {"seed": seed, "variant": variant}
    t0 = time.time()
    res["perf"] = eval_perf(
        model,
        n=args.n_perf,
        L=args.L,
        V=16,
        seed=seed,
        device=args.device,
        delay=args.delay,
    )
    res["robustness"] = eval_robustness(
        model,
        n=args.n_robust,
        L=args.L,
        V=16,
        seed=seed,
        device=args.device,
        delay=args.delay,
    )
    res["geometry"] = eval_geometry(
        model,
        n=args.n_geom,
        L=args.L,
        V=16,
        seed=seed,
        device=args.device,
        delay=args.delay,
    )
    ev = FaithfulnessEvaluator(model, device=args.device)
    faith_final = ev.eval_final_state(
        n_seq=args.n_faith,
        L=args.L,
        V=16,
        seed=seed,
        thetas=THETA_SWEEP,
        n_random_dirs=args.n_random_dirs,
        use_basis=True,
        delay=args.delay,
    )
    res["faith_final"] = {th: faith_final.metrics(th) for th in THETA_SWEEP}
    res["faith_final_diag"] = {
        "h_norm_median": float(np.median(faith_final.h_norms)),
        "margin_median": float(np.median(faith_final.margins_clean)),
        "frac_correct": float(np.mean(faith_final.correct)),
    }
    if args.delay > 0:
        faith_mid = ev.eval_mid_state(
            n_seq=args.n_faith,
            L=args.L,
            V=16,
            seed=seed,
            thetas=MID_THETAS,
            n_random_dirs=max(16, args.n_random_dirs // 2),
            delay=args.delay,
            mid_mode="downstream",
        )
    else:
        faith_mid = ev.eval_mid_state(
            n_seq=args.n_faith,
            L=args.L,
            V=16,
            seed=seed,
            thetas=MID_THETAS,
            n_random_dirs=max(16, args.n_random_dirs // 2),
            delay=args.delay,
            mid_mode="upstream",
            t0_fracs=MID_T0_FRACS,
        )
    res["faith_mid"] = {
        lab: {th: fr.metrics(th) for th in MID_THETAS} for lab, fr in faith_mid.items()
    }
    res["prune"] = eval_pruning_recovery(
        model,
        n_seq=args.n_faith,
        L=args.L,
        V=16,
        seed=seed,
        fracs=PRUNE_FRACS,
        device=args.device,
        delay=args.delay,
    )
    res["path"] = eval_path_recovery(
        model,
        n_seq=args.n_faith,
        L=args.L,
        V=16,
        seed=seed,
        top_p_max=8,
        device=args.device,
        delay=args.delay,
    )
    res["eval_seconds"] = round(time.time() - t0, 1)
    return res


def _theta_star(metrics_by_theta: dict, threshold: float = 0.5) -> float | None:
    best = None
    for th in sorted(metrics_by_theta):
        m = metrics_by_theta[th]
        e = m.get("E_eff", m.get("E_all"))
        if np.isfinite(e) and e <= threshold:
            best = th
    return best


def phase_aggregate(per_seed: dict, args) -> dict:
    out_dir = Path(args.out)
    rows_perf, rows_ff, rows_fm, rows_prune, rows_path = ([], [], [], [], [])
    for seed in range(args.seeds):
        for variant in VARIANTS:
            r = per_seed[seed, variant]
            p = r["perf"]
            rows_perf.append(
                {
                    "seed": seed,
                    "variant": variant,
                    **p,
                    "acc_clean_robust": r["robustness"]["acc_clean"],
                    "acc_corrupted": r["robustness"]["acc_one_token_corrupted"],
                    "robust_drop": r["robustness"]["drop"],
                    **{f"geom_{k2}": v for k2, v in r["geometry"].items()},
                }
            )
            for th, m in r["faith_final"].items():
                rows_ff.append(
                    {"seed": seed, "variant": variant, "level": "final", **m}
                )
            for lab, by_th in r["faith_mid"].items():
                for th, m in by_th.items():
                    rows_fm.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "level": "mid",
                            "t0_label": lab,
                            **m,
                        }
                    )
            pr = r["prune"]
            for f in PRUNE_FRACS:
                rows_prune.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "f": f,
                        "recovery_mean": pr[f"f={f:.2f}_recovery_mean"],
                        "abs_recovery_mean": pr[f"f={f:.2f}_abs_recovery_mean"],
                    }
                )
            rows_prune.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "f": None,
                    "recovery_mean": None,
                    "abs_recovery_mean": pr["f95"],
                }
            )
            pa = r["path"]
            row_path = {
                "seed": seed,
                "variant": variant,
                "rho_rank_attr": pa.get("rho_rank_attr"),
                "sign_acc_pos": pa.get("sign_acc_pos"),
                "calib_r_pos": pa.get("calib_r_pos"),
            }
            for pidx in range(1, 9):
                row_path[f"cap_grad_p{pidx}"] = pa["top_p_capture_gradient"].get(
                    str(pidx)
                ) or pa["top_p_capture_gradient"].get(pidx)
                row_path[f"cap_act_p{pidx}"] = pa["top_p_capture_actual"].get(
                    str(pidx)
                ) or pa["top_p_capture_actual"].get(pidx)
            rows_path.append(row_path)
    df_perf = pd.DataFrame(rows_perf)
    df_ff = pd.DataFrame(rows_ff)
    df_fm = pd.DataFrame(rows_fm)
    df_prune = pd.DataFrame(rows_prune)
    df_path = pd.DataFrame(rows_path)
    df_perf.to_csv(out_dir / "perf.csv", index=False)
    df_ff.to_csv(out_dir / "faith_final_theta.csv", index=False)
    df_fm.to_csv(out_dir / "faith_mid_theta.csv", index=False)
    df_prune.to_csv(out_dir / "prune_recovery.csv", index=False)
    df_path.to_csv(out_dir / "path_recovery.csv", index=False)
    stats_out: dict = {
        "config": vars(args),
        "parity": {},
        "faithfulness": {},
        "circuits": {},
    }
    for metric in ["accuracy", "mean_margin_correct", "mean_p_target", "ece", "brier"]:
        stats_out["parity"][metric] = paired_stats(
            [per_seed[s, "ordinary"]["perf"][metric] for s in range(args.seeds)],
            [per_seed[s, "sphere"]["perf"][metric] for s in range(args.seeds)],
        )
    stats_out["parity"]["robust_drop"] = paired_stats(
        [per_seed[s, "ordinary"]["robustness"]["drop"] for s in range(args.seeds)],
        [per_seed[s, "sphere"]["robustness"]["drop"] for s in range(args.seeds)],
    )
    faith_metrics = (
        "E_all",
        "E_eff",
        "rho_rank",
        "calib_r",
        "slope",
        "sign_acc",
        "fn_rate",
        "fp_rate",
    )
    faith_stats = {
        f"theta={th}": compare_groups(df_ff[df_ff["theta"] == th], faith_metrics)
        for th in THETA_SWEEP
    }
    stats_out["faithfulness"]["final_state"] = faith_stats
    t0_labels = (
        sorted(df_fm["t0_label"].dropna().unique())
        if "t0_label" in df_fm.columns and df_fm["t0_label"].notna().any()
        else [""]
    )
    mid_stats = {}
    for lab in t0_labels:
        frame = df_fm[df_fm["t0_label"] == lab] if lab else df_fm
        mid_stats[str(lab)] = {
            f"theta={th}": compare_groups(
                frame[frame["theta"] == th], ("E_all", "E_eff", "rho_rank", "sign_acc")
            )
            for th in MID_THETAS
        }
    stats_out["faithfulness"]["mid_state"] = mid_stats
    theta_star = {}
    for variant in VARIANTS:
        vals = []
        for s in range(args.seeds):
            ts = _theta_star(per_seed[s, variant]["faith_final"])
            vals.append(ts)
        finite = [v for v in vals if v is not None]
        theta_star[variant] = {
            "per_seed": vals,
            "mean_theta_star": float(np.mean(finite)) if finite else None,
            "n_reached_max": int(sum(v == max(THETA_SWEEP) for v in vals)),
        }
    stats_out["faithfulness"]["theta_star_final"] = theta_star
    prune_stats = {
        f"f={f:.2f}": compare_groups(
            df_prune[df_prune["f"] == f], ("recovery_mean", "abs_recovery_mean")
        )
        for f in PRUNE_FRACS
    }
    f95_o = [per_seed[s, "ordinary"]["prune"]["f95"] for s in range(args.seeds)]
    f95_s = [per_seed[s, "sphere"]["prune"]["f95"] for s in range(args.seeds)]
    prune_stats["f95"] = {
        "ordinary_per_seed": f95_o,
        "sphere_per_seed": f95_s,
        "mean_ordinary": float(np.mean([v for v in f95_o if v is not None]))
        if any(v is not None for v in f95_o)
        else None,
        "mean_sphere": float(np.mean([v for v in f95_s if v is not None]))
        if any(v is not None for v in f95_s)
        else None,
    }
    stats_out["circuits"]["pruning"] = prune_stats
    path_stats = compare_groups(
        df_path,
        (
            "rho_rank_attr",
            "sign_acc_pos",
            "calib_r_pos",
            *(f"cap_grad_p{i}" for i in range(1, 9)),
        ),
    )
    stats_out["circuits"]["path_recovery"] = path_stats
    acc_p = stats_out["parity"]["accuracy"]
    parity_ok = (
        acc_p.get("mean_diff_sphere_minus_ordinary") is not None
        and abs(acc_p["mean_diff_sphere_minus_ordinary"]) < 0.01
    )
    ref_th = 0.1
    mid_verdicts = {}
    for lab, by_th in mid_stats.items():
        verdict = improvement(
            by_th.get(f"theta={ref_th}", {}).get("E_all"), higher=False
        )
        if verdict is not None:
            verdict["E_all_diff"] = verdict.pop("diff")
            mid_verdicts[lab] = verdict
    verdicts = {
        "performance_parity": {
            "holds": bool(parity_ok),
            "accuracy_diff": acc_p.get("mean_diff_sphere_minus_ordinary"),
            "note": "|acc_sphere - acc_ordinary| < 0.01",
        },
        "final_state_faithfulness_at_theta0.1": {
            "E_all_lower_for_sphere": improvement(
                faith_stats["theta=0.1"].get("E_all"), higher=False
            ),
            "rho_rank_higher_for_sphere": improvement(
                faith_stats["theta=0.1"].get("rho_rank"), higher=True
            ),
            "sign_acc_higher_for_sphere": improvement(
                faith_stats["theta=0.1"].get("sign_acc"), higher=True
            ),
            "fn_rate_lower_for_sphere": improvement(
                faith_stats["theta=0.1"].get("fn_rate"), higher=False
            ),
        },
        "mid_state_dynamics_faithfulness_at_theta0.1": mid_verdicts,
        "path_capture_top2_higher_for_sphere": improvement(
            path_stats.get("cap_grad_p2"), higher=True
        ),
        "radius_of_validity": {
            "theta_star_ordinary_mean": theta_star["ordinary"]["mean_theta_star"],
            "theta_star_sphere_mean": theta_star["sphere"]["mean_theta_star"],
            "sphere_larger_radius": bool(
                (theta_star["sphere"]["mean_theta_star"] or 0)
                > (theta_star["ordinary"]["mean_theta_star"] or 0)
            ),
        },
        "circuit_tracing": {
            "f95_ordinary_mean": prune_stats["f95"]["mean_ordinary"],
            "f95_sphere_mean": prune_stats["f95"]["mean_sphere"],
            "sphere_needs_fewer_dims": bool(
                (prune_stats["f95"]["mean_sphere"] or 1)
                < (prune_stats["f95"]["mean_ordinary"] or 0)
            ),
        },
    }
    stats_out["verdicts"] = verdicts
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats_out, f, indent=2, default=float)
    return stats_out


def phase_plots(per_seed: dict, stats_out: dict, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out)
    df_ff = pd.read_csv(out_dir / "faith_final_theta.csv")
    df_fm = pd.read_csv(out_dir / "faith_mid_theta.csv")
    df_prune = pd.read_csv(out_dir / "prune_recovery.csv")
    df_path = pd.read_csv(out_dir / "path_recovery.csv")
    df_perf = pd.read_csv(out_dir / "perf.csv")
    colors = {"ordinary": "#4c72b0", "sphere": "#dd8452"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for level, df in (("final", df_ff), ("mid", df_fm)):
        ax = axes[0 if level == "final" else 1]
        if level == "final":
            groups = [(variant, None) for variant in VARIANTS]
        elif "t0_label" in df.columns and df["t0_label"].notna().any():
            groups = [
                (v, lab)
                for v in VARIANTS
                for lab in sorted(df["t0_label"].dropna().unique())
            ]
        else:
            groups = [(variant, None) for variant in VARIANTS]
        labs_sorted = (
            sorted(df["t0_label"].dropna().unique()) if "t0_label" in df.columns else []
        )
        for variant, lab in groups:
            sub_df = df[df["variant"] == variant]
            if lab is not None:
                sub_df = sub_df[sub_df["t0_label"] == lab]
            sub = sub_df.groupby("theta")["E_all"].agg(["mean", "std"])
            th = sorted(sub.index)
            mean = [sub.loc[t, "mean"] for t in th]
            sd = [
                sub.loc[t, "std"] / np.sqrt(max(len(sub_df[sub_df["theta"] == t]), 1))
                for t in th
            ]
            solid = lab is None or (
                labs_sorted and lab == labs_sorted[len(labs_sorted) // 2]
            )
            ls = "-" if solid else "--"
            lbl = (
                f"{variant}"
                if level == "final"
                else f"{variant} t0={lab}"
                if lab is not None
                else variant
            )
            ax.plot(th, mean, marker="o", ls=ls, color=colors[variant], label=lbl)
            ax.fill_between(
                th,
                [m - s for m, s in zip(mean, sd)],
                [m + s for m, s in zip(mean, sd)],
                alpha=0.12,
                color=colors[variant],
            )
        ts = stats_out["faithfulness"]["theta_star_final"]
        if level == "final":
            for variant in VARIANTS:
                v = ts[variant]["mean_theta_star"]
                if v is not None:
                    ax.axvline(v, color=colors[variant], ls="--", lw=1)
                    ax.text(
                        v,
                        ax.get_ylim()[1] * 0.9,
                        f"θ*={v}",
                        rotation=90,
                        fontsize=8,
                        color=colors[variant],
                    )
        ax.set_xscale("log")
        ax.set_xlabel("perturbation scale θ (rad / relative)")
        ax.set_ylabel(
            "$E(\\theta)=\\mathbb{E}\\,|\\Delta S-\\widehat{\\Delta S}|/(|\\Delta S|+\\epsilon)$"
        )
        ax.set_title(f"Radius of validity ({level} state)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "radius_of_validity.png", dpi=140)
    plt.close(fig)
    ref_th = 0.1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(
        axes,
        ["rho_rank", "sign_acc", "calib_r"],
        [
            "Rank recovery (Spearman |ΔŜ| vs |ΔS|)",
            "Sign accuracy (effective subset)",
            "Magnitude calibration (Pearson r)",
        ],
    ):
        vals = []
        labels = []
        for level in ("final", "mid"):
            dfx = df_ff if level == "final" else df_fm
            for variant in VARIANTS:
                v = dfx[(dfx["variant"] == variant) & (dfx["theta"] == ref_th)][
                    metric
                ].mean()
                vals.append(v)
                labels.append(f"{level}/{variant}")
        bars = ax.bar(
            range(len(vals)),
            vals,
            color=[colors[label.split("/")[1]] for label in labels],
            alpha=0.85,
        )
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{title}\n(θ={ref_th})")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "faithfulness_metrics.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in VARIANTS:
        sub = df_prune[(df_prune["variant"] == variant) & df_prune["f"].notna()]
        f = sorted(sub["f"])
        rec = [sub[sub["f"] == x]["abs_recovery_mean"].mean() for x in f]
        ax.plot(f, rec, marker="o", color=colors[variant], label=variant)
    ax.axhline(0.95, color="k", ls="--", lw=1)
    ax.text(0.62, 0.955, "95% recovery", fontsize=8)
    f95 = stats_out["circuits"]["pruning"]["f95"]
    for variant in VARIANTS:
        v = f95[f"mean_{variant}"]
        if v is not None:
            ax.annotate(
                f"{variant}: f₉₅={v:.0%}",
                xy=(v, 0.95),
                xytext=(v * 1.6, 0.8),
                fontsize=9,
                color=colors[variant],
                arrowprops=dict(arrowstyle="->", color=colors[variant]),
            )
    ax.set_xscale("log")
    ax.set_xlabel("fraction of state dimensions retained (by gradient ranking)")
    ax.set_ylabel("mean |recovered causal effect| / |full effect|")
    ax.set_title("Circuit tracing: top-k% retention vs recovered effect")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "circuit_recovery.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, variant in zip(axes, VARIANTS):
        sub = df_path[df_path["variant"] == variant]
        ps = list(range(1, 9))
        cap_g = [sub[f"cap_grad_p{p}"].mean() for p in ps]
        cap_a = [sub[f"cap_act_p{p}"].mean() for p in ps]
        ax.plot(ps, cap_a, marker="s", color="#55a868", label="actual ranking (oracle)")
        ax.plot(
            ps,
            cap_g,
            marker="o",
            color=colors[variant],
            label=f"gradient ranking ({variant})",
        )
        ax.set_xlabel("number of top positions p retained")
        ax.set_ylabel("fraction of total |effect| mass captured")
        ax.set_title(f"Position-level path recovery — {variant}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "path_recovery.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    for i in range(args.seeds):
        a_o = df_perf[(df_perf["seed"] == i) & (df_perf["variant"] == "ordinary")][
            "accuracy"
        ].iloc[0]
        a_s = df_perf[(df_perf["seed"] == i) & (df_perf["variant"] == "sphere")][
            "accuracy"
        ].iloc[0]
        ax.plot([a_o, a_s], [i, i], color="gray", lw=1.5, zorder=1)
        ax.scatter([a_o], [i], color=colors["ordinary"], s=60, zorder=2)
        ax.scatter([a_s], [i], color=colors["sphere"], s=60, zorder=2)
    ax.set_yticks(range(args.seeds))
    ax.set_ylabel("seed")
    ax.set_xlabel("eval accuracy (n=2048)")
    ax.set_title("Performance parity per seed")
    ax.grid(alpha=0.3)
    metrics = [
        ("mean_margin_correct", "mean margin (correct)", 1),
        ("mean_p_target", "mean P(target)", 1),
        ("ece", "ECE (lower better)", -1),
        ("brier", "Brier (lower better)", -1),
    ]
    ax = axes[1]
    width = 0.38
    x = np.arange(len(metrics))
    for j, variant in enumerate(VARIANTS):
        sub = df_perf[df_perf["variant"] == variant]
        vals = [sub[m].mean() for m, _, _ in metrics]
        ax.bar(x + (j - 0.5) * width, vals, width, color=colors[variant], label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t, _ in metrics], rotation=20, fontsize=8)
    ax.set_title("Confidence / calibration (means over seeds)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax = axes[2]
    sub_o = df_perf[df_perf["variant"] == "ordinary"]["robust_drop"].mean()
    sub_s = df_perf[df_perf["variant"] == "sphere"]["robust_drop"].mean()
    bars = ax.bar(
        ["ordinary", "sphere"],
        [sub_o, sub_s],
        color=[colors["ordinary"], colors["sphere"]],
    )
    for b, v in zip(bars, [sub_o, sub_s]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max(sub_o, sub_s) * 0.01,
            f"{v:.4f}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("accuracy drop under one-token corruption")
    ax.set_title("Robustness control")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "parity.png", dpi=140)
    plt.close(fig)


def phase_report(per_seed: dict, stats_out: dict, args) -> None:
    out_dir = Path(args.out)
    v = stats_out["verdicts"]
    faith = stats_out["faithfulness"]["final_state"]

    def fmt(x, spec=".4f"):
        return "n/a" if x is None else format(x, spec)

    lines = []
    lines.extend(
        """# expC: Hyperspherical state geometry and gradient faithfulness

## Hypothesis (H1)

> Constraining neural computation to hyperspherical state geometry increases the causal
> faithfulness and usable radius of first-order gradient attribution, enabling more accurate
> execution-level circuit tracing — at equal task performance.

## Design
""".split("\n")
    )
    lines.append(
        f"- Models: `M_ordinary` (h_t = A h + B x) vs `M_sphere` (h_t = normalize(A h + B x)), 2-layer diagonal SSM, k={args.k}, d={64}, L={args.L}, V={16}."
    )
    lines.append(
        f"- Identical parameter count, initialization and data stream per seed ({args.seeds} seeds); only the per-step projection differs."
    )
    lines.extend(
        """- Behavioral target: margin S = z_y − z_alt (runner-up alt fixed from clean pass).
- First-order prediction ΔŜ = ⟨g, δ⟩ with g tangent-projected for M_sphere; actual effect ΔS measured by exact re-evaluation of the forward pass.
- Perturbations: geodesic steps h' = cosθ·h + sinθ·u (sphere) vs matched relative-scale Euclidean steps δ = θ‖h‖v (ordinary).""".split(
            "\n"
        )
    )
    lines.append(
        f"- Eval sets: n_faith={args.n_faith} sequences for faithfulness/circuits, n_perf={args.n_perf} for performance controls."
    )
    lines.extend("\n## Performance parity (control)\n".split("\n"))
    p = stats_out["parity"]
    lines.extend(
        """| metric | ordinary mean | sphere mean | diff (sphere−ord) | 95% CI | paired t p |
|---|---|---|---|---|---|""".split("\n")
    )
    for m in [
        "accuracy",
        "mean_margin_correct",
        "mean_p_target",
        "ece",
        "brier",
        "robust_drop",
    ]:
        e = p[m]
        ci = e.get("ci95", [None, None])
        lines.append(
            f"| {m} | {fmt(e.get('mean_ordinary'))} | {fmt(e.get('mean_sphere'))} | {fmt(e.get('mean_diff_sphere_minus_ordinary'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(e.get('p_value'), '.3f')} |"
        )
    lines.append("")
    lines.append(
        f"**Parity holds (|Δacc| < 0.01): {v['performance_parity']['holds']}**"
    )
    lines.extend(
        """
## Local faithfulness — final state, E(θ) radius of validity

E_all = all random directions; E_eff = restricted to directions with |ΔS| ≥ median (where first-order
theory is expected to make a real prediction). Near-orthogonal directions inflate E_all for both variants.

| θ | E_all ord | E_all sph | E_eff ord | E_eff sph | ρ_rank ord | ρ_rank sph | sign acc ord | sign acc sph |
|---|---|---|---|---|---|---|---|---|""".split("\n")
    )
    for th in THETA_SWEEP:
        e = faith[f"theta={th}"]
        eo, es = (e["E_all"]["mean_ordinary"], e["E_all"]["mean_sphere"])
        eo2, es2 = (
            e.get("E_eff", {}).get("mean_ordinary"),
            e.get("E_eff", {}).get("mean_sphere"),
        )
        lines.append(
            f"| {th} | {fmt(eo)} | {fmt(es)} | {fmt(eo2)} | {fmt(es2)} | {fmt(e['rho_rank']['mean_ordinary'])} | {fmt(e['rho_rank']['mean_sphere'])} | {fmt(e['sign_acc']['mean_ordinary'], '.3f')} | {fmt(e['sign_acc']['mean_sphere'], '.3f')} |"
        )
    ts = stats_out["faithfulness"]["theta_star_final"]
    lines.append("")
    lines.append(
        f"- θ* (largest θ with E_eff ≤ 0.5): ordinary mean = {fmt(ts['ordinary']['mean_theta_star'])}, sphere mean = {fmt(ts['sphere']['mean_theta_star'])}"
    )
    lines.append(
        f"- Sphere has larger radius of validity: **{v['radius_of_validity']['sphere_larger_radius']}**"
    )
    lines.extend(
        """
## Local faithfulness — mid state (gradients through scan dynamics)

Intervention at intermediate last-layer states h_{t0} with exact re-scan to q; the gradient must be
back-propagated through (q − t0) recurrence steps. For delay=0 the target token is only input AT q, so
this isolates faithfulness of multi-step dynamical attribution.
""".split("\n")
    )
    mid = stats_out["faithfulness"]["mid_state"]
    for lab, by_th in mid.items():
        lines.append(f"**Intervention at t0 = {lab.replace('t0=', '')}**")
        lines.append("")
        lines.append(
            "| θ | E ord | E sph | ρ_rank ord | ρ_rank sph | sign acc ord | sign acc sph |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for th in MID_THETAS:
            e = by_th.get(f"theta={th}", {})
            if not e:
                continue
            eo, es = (
                e.get("E_all", {}).get("mean_ordinary"),
                e.get("E_all", {}).get("mean_sphere"),
            )
            ro, rs = (
                e.get("rho_rank", {}).get("mean_ordinary"),
                e.get("rho_rank", {}).get("mean_sphere"),
            )
            so, ss = (
                e.get("sign_acc", {}).get("mean_ordinary"),
                e.get("sign_acc", {}).get("mean_sphere"),
            )
            lines.append(
                f"| {th} | {fmt(eo)} | {fmt(es)} | {fmt(ro)} | {fmt(rs)} | {fmt(so, '.3f')} | {fmt(ss, '.3f')} |"
            )
        lines.append("")
    lines.extend("## Circuit tracing (execution-level)\n".split("\n"))
    pr = stats_out["circuits"]["pruning"]
    lines.extend(
        "| retained fraction f | recovery ord | recovery sph |\n|---|---|---|".split(
            "\n"
        )
    )
    for f in PRUNE_FRACS:
        e = pr[f"f={f:.2f}"]["abs_recovery_mean"]
        lines.append(
            f"| {f:.0%} | {fmt(e['mean_ordinary'])} | {fmt(e['mean_sphere'])} |"
        )
    lines.append("")
    f95 = pr["f95"]
    lines.append(
        f"- f₉₅ (smallest retention reaching 95% mean recovery): ordinary = {fmt(f95['mean_ordinary'], '.3f')}, sphere = {fmt(f95['mean_sphere'], '.3f')}"
    )
    lines.append("")
    pa = stats_out["circuits"]["path_recovery"]
    lines.append(
        "- Position-level path recovery (gradient attribution vs neutral-patch causal effects):"
    )
    for m in ["rho_rank_attr", "sign_acc_pos"]:
        e = pa[m]
        lines.append(
            f"  - {m}: ordinary={fmt(e.get('mean_ordinary'))}, sphere={fmt(e.get('mean_sphere'))}"
        )
    cap1_o = pa["cap_grad_p1"]["mean_ordinary"]
    cap1_s = pa["cap_grad_p1"]["mean_sphere"]
    lines.append(
        f"  - top-1 position capture (gradient ranking): ordinary={fmt(cap1_o)}, sphere={fmt(cap1_s)}"
    )
    lines.extend("\n## Verdicts\n".split("\n"))
    lines.append(
        f"- Performance parity (|Δacc| < 0.01): **{v['performance_parity']['holds']}** (diff = {fmt(v['performance_parity']['accuracy_diff'])})"
    )
    fv = v["final_state_faithfulness_at_theta0.1"]
    lines.append("- Final-state faithfulness at θ=0.1:")
    for k2, e in fv.items():
        if e is None:
            continue
        lines.append(
            f"  - {k2}: diff={fmt(e['diff'])}, p={fmt(e.get('p'), '.3f')}, sphere better={e['sphere_better']}, consistent across seeds={e['consistent_across_seeds']}"
        )
    mv = v["mid_state_dynamics_faithfulness_at_theta0.1"]
    lines.append(
        "- Mid-state (dynamics) faithfulness at θ=0.1, E_all diff (sphere−ordinary):"
    )
    for lab, e in mv.items():
        lines.append(
            f"  - {lab}: diff={fmt(e['E_all_diff'])}, p={fmt(e.get('p'), '.3f')}, sphere better={e['sphere_better']}, consistent across seeds={e['consistent_across_seeds']}"
        )
    pc = v["path_capture_top2_higher_for_sphere"]
    if pc is not None:
        lines.append(
            f"- Path capture top-2 (gradient ranking): diff={fmt(pc['diff'])}, p={fmt(pc.get('p'), '.3f')}, sphere better={pc['sphere_better']}, consistent across seeds={pc['consistent_across_seeds']}"
        )
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument(
        "--delay",
        type=int,
        default=0,
        help="target = t_{q-delay}; 0 = copy-current (matched-performance regime), >=1 = short-term memory (supplementary regime analysis)",
    )
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-faith", type=int, default=256)
    ap.add_argument("--n-perf", type=int, default=2048)
    ap.add_argument("--n-robust", type=int, default=1024)
    ap.add_argument("--n-geom", type=int, default=512)
    ap.add_argument("--n-random-dirs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "expC"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="load per-seed results from <out>/per_seed/*.json instead of re-evaluating",
    )
    args = ap.parse_args()
    if args.quick:
        args.seeds = 2
        args.steps = 800
        args.n_faith = 64
        args.n_perf = 512
        args.n_robust = 256
        args.n_geom = 128
        args.n_random_dirs = 32
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        print("=== Phase 1: training ===", flush=True)
        phase_train(args)
    print("=== Phase 2: evaluation ===", flush=True)
    per_seed = {}
    ps_dir = out_dir / "per_seed"
    ps_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        for variant in VARIANTS:
            if args.skip_eval:
                with open(ps_dir / f"seed{seed}_{variant}.json") as fh:
                    r = json.load(fh)
                r["faith_final"] = {float(k): v for k, v in r["faith_final"].items()}
                fm = r["faith_mid"]
                if (
                    fm
                    and isinstance(next(iter(fm.values())), dict)
                    and ("E_all" in next(iter(fm.values())))
                ):
                    r["faith_mid"] = {"legacy": {float(k): v for k, v in fm.items()}}
                else:
                    r["faith_mid"] = {
                        lab: {float(th): m for th, m in by_th.items()}
                        for lab, by_th in fm.items()
                    }
                print(
                    f"  [C eval seed={seed} {variant}] loaded from disk acc={r['perf']['accuracy']:.4f}",
                    flush=True,
                )
                per_seed[seed, variant] = r
                continue
            t0 = time.time()
            r = phase_eval_one(seed, variant, args)
            per_seed[seed, variant] = r
            with open(ps_dir / f"seed{seed}_{variant}.json", "w") as fh:
                json.dump(r, fh, indent=1, default=float)
            print(
                f"  [C eval seed={seed} {variant}] acc={r['perf']['accuracy']:.4f} f95={r['prune']['f95']} E(0.1)={r['faith_final'][0.1]['E_all']:.3f} ({time.time() - t0:.1f}s)",
                flush=True,
            )
    print("=== Phase 3: aggregation ===", flush=True)
    stats_out = phase_aggregate(per_seed, args)
    print("=== Phase 4: plots + report ===", flush=True)
    phase_plots(per_seed, stats_out, args)
    phase_report(per_seed, stats_out, args)
    print("\n=== expC verdicts ===")
    print(json.dumps(stats_out["verdicts"], indent=2, default=float))
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
