from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting import bar_metric, errorbar_metric, finish_figure

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plot_expA(expA: Path) -> None:
    mse_path = expA / "mse_by_quant.csv"
    if not mse_path.exists():
        print("Skip ExpA plots: no mse_by_quant.csv")
        return
    df = pd.read_csv(mse_path)
    order = ["fp32", "16-level", "8-level", "4-level", "2-level"]
    df["quant_label"] = pd.Categorical(
        df["quant_label"], categories=order, ordered=True
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {False: "#c0392b", True: "#2980b9"}
    labels = {False: "Standard", True: "Normalized"}
    ax = axes[0, 0]
    errorbar_metric(ax, df, "mse", (False, True), colors, labels)
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
            ax.plot(
                g.mean().index,
                g.mean().values,
                marker="o",
                label=labels[norm],
                color=colors[norm],
            )
        ax.set_xlabel("Latent dimension k")
        ax.set_ylabel("MSE at 2-level quant")
        ax.set_title("Cliff by state width")
        ax.legend()
        ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    if "snr_effective" in df.columns:
        errorbar_metric(
            ax,
            df[df["quant_label"] != "fp32"],
            "snr_effective",
            (False, True),
            colors,
            labels,
        )
        ax.set_xticks(range(len(order) - 1))
        ax.set_xticklabels(order[1:], rotation=45)
        ax.set_xlabel("Quantization level")
        ax.set_ylabel("Effective SNR")
        ax.set_title("SNR vs quantization")
        ax.legend()
        ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    if "mse_matched_noise" in df.columns:
        errorbar_metric(
            ax,
            df[df["quant_label"] != "fp32"],
            "mse_matched_noise",
            (False, True),
            colors,
            labels,
            marker="s",
            linestyle="--",
            label_suffix=" (matched)",
        )
        ax.set_xticks(range(len(order) - 1))
        ax.set_xticklabels(order[1:], rotation=45)
        ax.set_xlabel("Quantization level")
        ax.set_ylabel("MSE (matched noise)")
        ax.set_title("Matched-noise comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Experiment A — Enhanced SNR + matched-noise suite", fontsize=14)
    finish_figure(fig, expA / "mse_curves_enhanced.png")
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
        colors = {
            "B0": "#7f8c8d",
            "BW": "#27ae60",
            "BR": "#8e44ad",
            "BX": "#e67e22",
            "BW_BR": "#1abc9c",
            "B0_noshort": "#95a5a6",
            "BR_noshort": "#9b59b6",
            "sphere_on_z": "#f39c12",
        }
        modes_list = [m for m in sub["mode"].unique() if m in colors]
        panels = [
            ("udepth", "U-shape depth", "UDepth", None),
            ("over_smoothing", "Over-smoothing (raw h)", "Mean pairwise cosine", None),
            ("decode_probe_acc", "Decode probe accuracy", "Probe acc", (0, 1.05)),
            ("intervention_drop", "Intervention drop", "Acc drop", None),
            (
                "task_conditioned_os",
                "Task-conditioned OS",
                "Mean cosine (distinct)",
                None,
            ),
            ("endpoint_acc", "Endpoint accuracy", "Accuracy", (0, 1.05)),
        ]
        for ax, (metric, title, ylabel, ylim) in zip(axes.flat, panels, strict=True):
            bar_metric(
                ax,
                sub,
                metric,
                modes_list,
                colors,
                title=title,
                ylabel=ylabel,
                ylim=ylim,
            )
        b0_endpoint = (
            sub[sub["mode"] == "B0"]["endpoint_acc"].mean()
            if "B0" in sub["mode"].values
            else None
        )
        if b0_endpoint is not None:
            ax.axhline(
                y=b0_endpoint - 0.05,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Guardrail (B0 - 5pp)",
            )
            ax.legend(fontsize=8)
        fig.suptitle(f"Exp B metrics — {gl} (L={L}, k={k})", fontsize=14)
        finish_figure(fig, expB / f"metrics_{gl}_L{L}_k{k}.png")
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
    finish_figure(fig, expB / "decay_histograms.png")


def main() -> None:
    root = ROOT
    plot_expA(root / "results" / "expA")
    plot_expB(root / "results" / "expB")


if __name__ == "__main__":
    main()
