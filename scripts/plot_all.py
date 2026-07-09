
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plot_expA(expA: Path) -> None:
    mse_path = expA / "mse_by_quant.csv"
    if not mse_path.exists():
        print("Skip ExpA plots: no mse_by_quant.csv")
        return
    df = pd.read_csv(mse_path)
    order = ["fp32", "16-level", "8-level", "4-level", "2-level"]
    df["quant_label"] = pd.Categorical(df["quant_label"], categories=order, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {False: "#c0392b", True: "#2980b9"}
    labels = {False: "Standard", True: "Normalized"}

    ax = axes[0, 0]
    for norm in (False, True):
        sub = df[df["normalized"] == norm]
        g = sub.groupby("quant_label", observed=True)["mse"]
        mean = g.mean()
        sem = g.sem()
        x = np.arange(len(mean))
        ax.errorbar(x, mean.values, yerr=sem.values, marker="o", label=labels[norm],
                     color=colors[norm], capsize=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45)
    ax.set_xlabel("Quantization level")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("MSE vs latent quantization")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "k" in df.columns:
        for norm in (False, True):
            sub = df[df["normalized"] == norm]
            sub = sub[sub["quant_label"] == "2-level"]
            g = sub.groupby("k")["mse"]
            ax.plot(g.mean().index, g.mean().values, marker="o", label=labels[norm],
                    color=colors[norm])
        ax.set_xlabel("Latent dimension k")
        ax.set_ylabel("MSE at 2-level quant")
        ax.set_title("Cliff by state width")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if "snr_effective" in df.columns:
        for norm in (False, True):
            sub = df[(df["normalized"] == norm) & (df["quant_label"] != "fp32")]
            g = sub.groupby("quant_label", observed=True)["snr_effective"]
            mean = g.mean()
            sem = g.sem()
            x = np.arange(len(mean))
            ax.errorbar(x, mean.values, yerr=sem.values, marker="o", label=labels[norm],
                         color=colors[norm], capsize=3)
        ax.set_xticks(range(len(order) - 1))
        ax.set_xticklabels(order[1:], rotation=45)
        ax.set_xlabel("Quantization level")
        ax.set_ylabel("Effective SNR")
        ax.set_title("SNR vs quantization")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "mse_matched_noise" in df.columns:
        for norm in (False, True):
            sub = df[(df["normalized"] == norm) & (df["quant_label"] != "fp32")]
            g = sub.groupby("quant_label", observed=True)["mse_matched_noise"]
            mean = g.mean()
            sem = g.sem()
            x = np.arange(len(mean))
            ax.errorbar(x, mean.values, yerr=sem.values, marker="s", label=f"{labels[norm]} (matched)",
                         color=colors[norm], capsize=3, linestyle="--")
        ax.set_xticks(range(len(order) - 1))
        ax.set_xticklabels(order[1:], rotation=45)
        ax.set_xlabel("Quantization level")
        ax.set_ylabel("MSE (matched noise)")
        ax.set_title("Matched-noise comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment A — Enhanced SNR + matched-noise suite", fontsize=14)
    fig.tight_layout()
    fig.savefig(expA / "mse_curves_enhanced.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {expA / 'mse_curves_enhanced.png'}")


def plot_expB(expB: Path) -> None:
    metrics_path = expB / "metrics.csv"
    if not metrics_path.exists():
        print("Skip ExpB plots: no metrics.csv")
        return
    df = pd.read_csv(metrics_path)

    if "grid_label" not in df.columns:
        df["grid_label"] = df["L"].astype(str) + "_" + df["k"].astype(str)

    for gl in df["grid_label"].unique():
        sub = df[df["grid_label"] == gl]
        L = int(sub["L"].mode().iloc[0])
        k = int(sub["k"].mode().iloc[0])

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        colors = {"B0": "#7f8c8d", "BW": "#27ae60", "BR": "#8e44ad",
                  "BX": "#e67e22", "BW_BR": "#1abc9c", "B0_noshort": "#95a5a6",
                  "BR_noshort": "#9b59b6", "sphere_on_z": "#f39c12"}

        modes_list = [m for m in sub["mode"].unique() if m in colors]

        ax = axes[0, 0]
        means = [sub[sub["mode"] == m]["udepth"].mean() for m in modes_list]
        sems = [sub[sub["mode"] == m]["udepth"].sem() for m in modes_list]
        ax.bar(modes_list, means, yerr=sems, color=[colors.get(m, "#333") for m in modes_list], capsize=4)
        ax.set_title("U-shape depth")
        ax.set_ylabel("UDepth")
        ax.tick_params(axis="x", rotation=45)

        ax = axes[0, 1]
        means = [sub[sub["mode"] == m]["over_smoothing"].mean() for m in modes_list]
        sems = [sub[sub["mode"] == m]["over_smoothing"].sem() for m in modes_list]
        ax.bar(modes_list, means, yerr=sems, color=[colors.get(m, "#333") for m in modes_list], capsize=4)
        ax.set_title("Over-smoothing (raw h)")
        ax.set_ylabel("Mean pairwise cosine")
        ax.tick_params(axis="x", rotation=45)

        ax = axes[0, 2]
        if "decode_probe_acc" in sub.columns:
            means = [sub[sub["mode"] == m]["decode_probe_acc"].mean() for m in modes_list]
            sems = [sub[sub["mode"] == m]["decode_probe_acc"].sem() for m in modes_list]
            ax.bar(modes_list, means, yerr=sems, color=[colors.get(m, "#333") for m in modes_list], capsize=4)
        ax.set_title("Decode probe accuracy")
        ax.set_ylabel("Probe acc")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)

        ax = axes[1, 0]
        if "intervention_drop" in sub.columns:
            means = [sub[sub["mode"] == m]["intervention_drop"].mean() for m in modes_list]
            sems = [sub[sub["mode"] == m]["intervention_drop"].sem() for m in modes_list]
            ax.bar(modes_list, means, yerr=sems, color=[colors.get(m, "#333") for m in modes_list], capsize=4)
        ax.set_title("Intervention drop")
        ax.set_ylabel("Acc drop")
        ax.tick_params(axis="x", rotation=45)

        ax = axes[1, 1]
        if "task_conditioned_os" in sub.columns:
            means = [sub[sub["mode"] == m]["task_conditioned_os"].mean() for m in modes_list]
            sems = [sub[sub["mode"] == m]["task_conditioned_os"].sem() for m in modes_list]
            ax.bar(modes_list, means, yerr=sems, color=[colors.get(m, "#333") for m in modes_list], capsize=4)
        ax.set_title("Task-conditioned OS")
        ax.set_ylabel("Mean cosine (distinct)")
        ax.tick_params(axis="x", rotation=45)

        ax = axes[1, 2]
        means = [sub[sub["mode"] == m]["endpoint_acc"].mean() for m in modes_list]
        sems = [sub[sub["mode"] == m]["endpoint_acc"].sem() for m in modes_list]
        ax.bar(modes_list, means, yerr=sems, color=[colors.get(m, "#333") for m in modes_list], capsize=4)
        ax.set_title("Endpoint accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)

        b0_endpoint = sub[sub["mode"] == "B0"]["endpoint_acc"].mean() if "B0" in sub["mode"].values else None
        if b0_endpoint is not None:
            ax.axhline(y=b0_endpoint - 0.05, color="gray", linestyle="--", alpha=0.5, label="Guardrail (B0 - 5pp)")
            ax.legend(fontsize=8)

        fig.suptitle(f"Exp B metrics — {gl} (L={L}, k={k})", fontsize=14)
        fig.tight_layout()
        fig.savefig(expB / f"metrics_{gl}_L{L}_k{k}.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {expB / f'metrics_{gl}_L{L}_k{k}.png'}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in df["mode"].unique():
        vals = df[df["mode"] == mode]["tau_mean"].values
        if len(vals):
            ax.hist(vals, bins=min(10, max(3, len(vals))), alpha=0.5, label=mode)
    ax.set_xlabel("Mean effective τ (steps)")
    ax.set_ylabel("Count (seeds)")
    ax.set_title("Learned decay (mean τ per seed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(expB / "decay_histograms.png", dpi=150)
    plt.close(fig)


def write_conclusion(root: Path) -> None:
    expA = root / "results" / "expA"
    expB = root / "results" / "expB"
    lines = [
        "# Experiment Conclusion (Enhanced)",
        "",
        "Auto-generated from `results/` after full runs. Applies decision rules",
        "from the gap-closing analysis (v2 decision table).",
        "",
    ]

    lines.append("## Experiment A — Bottleneck quantization cliff")
    lines.append("")
    stats_path = expA / "cliff_stats.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text(encoding="utf-8"))
        for k_label, v in s.get("by_k", {}).items():
            cs = v.get("cliff_standard_mean")
            cn = v.get("cliff_normalized_mean")
            p = v.get("p_value")
            md = v.get("mean_diff_norm_minus_std")
            lines.append(f"### k={k_label}")
            lines.append(f"- Standard mean cliff: **{cs:.4f}**" if cs is not None else "")
            lines.append(f"- Normalized mean cliff: **{cn:.4f}**" if cn is not None else "")
            lines.append(f"- Mean diff: **{md:.4f}** (CI {v.get('ci95')})")
            lines.append(f"- Paired t-test p: **{p}**")
            lines.append("")

        snr_path = expA / "snr_probe.csv"
        if snr_path.exists():
            snr = pd.read_csv(snr_path)
            for k_val in snr["k"].unique():
                subk = snr[snr["k"] == k_val]
                d0 = subk[(~subk["normalized"]) & (subk["quant_label"] == "2-level")]["delta_rho"].mean()
                d1 = subk[(subk["normalized"]) & (subk["quant_label"] == "2-level")]["delta_rho"].mean()
                s0 = subk[(~subk["normalized"]) & (subk["quant_label"] == "2-level")]["snr_effective"].mean()
                s1 = subk[(subk["normalized"]) & (subk["quant_label"] == "2-level")]["snr_effective"].mean()
                lines.append(f"### k={k_val} SNR at 2-level")
                lines.append(f"- Δρ Standard={d0:.4f}, Normalized={d1:.4f} (higher={ 'Norm' if d1 > d0 else 'Std' })")
                lines.append(f"- SNR effective Standard={s0:.4f}, Normalized={s1:.4f} (higher={ 'Norm' if s1 > s0 else 'Std' })")
                lines.append("")
    else:
        lines.append("No Exp A results found.")
        lines.append("")

    lines.append("## Experiment B — SSM serial-position (enhanced)")
    lines.append("")
    bstats = expB / "stats.json"
    if bstats.exists():
        s = json.loads(bstats.read_text(encoding="utf-8"))
        for gl, gs in s.get("grids", {}).items():
            lines.append(f"### Grid: {gl} (L={gs.get('L')}, k={gs.get('k')})")
            lines.append(f"Baseline: endpoint={gs.get('baseline_endpoint_mean', 'n/a')}, "
                         f"overall={gs.get('baseline_overall_mean', 'n/a')}")
            lines.append("")
            for mode, g in gs.get("guardrail", {}).items():
                lines.append(f"**{mode}**: guardrail={g['pass_guardrail']}, "
                             f"udepth={g['udepth_reduced']}, H1={g['h1_support']}")
                cmp_ = gs.get("comparisons", {}).get(mode, {})
                if cmp_:
                    lines.append(f"  - UDepth B0→{mode}: {cmp_.get('udepth_b0_mean'):.4f}→{cmp_.get('udepth_mode_mean'):.4f}, "
                                 f"p={cmp_.get('p_value')}")
                    lines.append(f"  - OS: {cmp_.get('os_b0_mean'):.4f}→{cmp_.get('os_mode_mean'):.4f}")
                    lines.append(f"  - Probe acc: {cmp_.get('probe_acc_b0_mean'):.4f}→{cmp_.get('probe_acc_mode_mean'):.4f}")
                    lines.append(f"  - Int drop: {cmp_.get('int_drop_b0_mean'):.4f}→{cmp_.get('int_drop_mode_mean'):.4f}")
                lines.append("")
    else:
        lines.append("No Exp B results found.")
        lines.append("")

    out = root / "results" / "CONCLUSION.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    root = ROOT
    plot_expA(root / "results" / "expA")
    plot_expB(root / "results" / "expB")
    write_conclusion(root)


if __name__ == "__main__":
    main()
