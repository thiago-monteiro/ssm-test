from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]

def ci95_mean(arr):
    if len(arr) < 2:
        return (np.nan, np.nan)
    m = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(len(arr))
    h = se * sp_stats.t.ppf(0.975, df=len(arr) - 1)
    return (m - h, m + h)

def fmt_ci(mean, ci, decimals=2):
    lo, hi = ci
    return f"${mean:.{decimals}f}\,({lo:.{decimals}f},\,{hi:.{decimals}f})$"

expA = ROOT / "results" / "expA"
cliff = pd.read_csv(expA / "cliff_scores.csv")

print("=== Experiment A: Cliff scores (Table 1) ===")
for k in sorted(cliff["k"].unique()):
    sub = cliff[cliff["k"] == k]
    for norm, label in [(False, "Std"), (True, "Norm")]:
        vals = sub[sub["normalized"] == norm]["cliff"].values
        m = np.mean(vals)
        ci = ci95_mean(vals)
        print(f"  k={k:>2d} {label}: mean={m:.2f}, 95%CI=[{ci[0]:.2f}, {ci[1]:.2f}]")
    std = sub[~sub["normalized"]].sort_values("seed")["cliff"].values
    nrm = sub[sub["normalized"]].sort_values("seed")["cliff"].values
    if len(std) > 1 and len(nrm) > 1:
        t, p = sp_stats.ttest_rel(std, nrm)
        ratio = np.mean(std) / np.mean(nrm)
        print(f"  k={k:>2d} Std/Norm ratio={ratio:.1f}x, t={t:.2f}, p={p:.2e}")
    print()

snr = pd.read_csv(expA / "snr_probe.csv")
print("=== SNR Probe (2-level quantization) ===")
for k in sorted(snr["k"].unique()):
    sub = snr[(snr["k"] == k) & (snr["quant_label"] == "2-level")]
    for norm, label in [(False, "Std"), (True, "Norm")]:
        vals = sub[sub["normalized"] == norm]
        if len(vals) == 0: continue
        rho_s = vals["rho_signal"].mean()
        rho_n = vals["rho_noise"].mean()
        snr_eff = vals["snr_effective"].mean()
        delta = vals["delta_rho"].mean()
        print(f"  k={k:>2d} {label}: SNR_eff={snr_eff:.2f}, delta_rho={delta:.4f}, rho_signal={rho_s:.4f}, rho_noise={rho_n:.4f}")
    print()

expB = ROOT / "results" / "expB"
metrics = pd.read_csv(expB / "metrics.csv")

sub = metrics[(metrics["L"] == 128) & (metrics["k"] == 128)]
print("=== Experiment B: SSM metrics (Table 2) L=128, k=128 ===")
modes_display = {"B0": "B0", "BW": "BW", "BR": "BR"}
fields = [
    ("endpoint_acc", "End.", 2),
    ("udepth", "UDepth", 2),
    ("over_smoothing_readout", "OS", 2),
    ("decode_probe_acc", "Probe", 2),
    ("intervention_drop", "Drop", 2),
]
for mode in ["B0", "BW", "BR"]:
    vals = sub[sub["mode"] == mode]
    if len(vals) == 0: continue
    parts = [f"  {mode}"]
    for col, label, dec in fields:
        if col not in vals:
            parts.append(f"{label}=--")
            continue
        arr = vals[col].dropna().values
        if len(arr) == 0:
            parts.append(f"{label}=--")
            continue
        m = np.mean(arr)
        ci = ci95_mean(arr)
        parts.append(f"{label}={m:.{dec}f} [{ci[0]:.{dec}f}, {ci[1]:.{dec}f}]")
    print(" & ".join(parts))
    print()

expB_hard = ROOT / "results" / "expB_hard"
if (expB_hard / "metrics.csv").exists():
    hmetrics = pd.read_csv(expB_hard / "metrics.csv")
    hsub = hmetrics[(hmetrics["L"] == 128) & (hmetrics["k"] == 128)]
    print("=== ExpB_hard L=128, k=128 ===")
    for mode in sorted(hsub["mode"].unique()):
        vals = hsub[hsub["mode"] == mode]
        parts = [f"  {mode}"]
        for col, label, dec in fields:
            if col not in vals:
                parts.append(f"{label}=--")
                continue
            arr = vals[col].dropna().values
            if len(arr) == 0:
                parts.append(f"{label}=--")
                continue
            m = np.mean(arr)
            ci = ci95_mean(arr)
            parts.append(f"{label}={m:.{dec}f} [{ci[0]:.{dec}f}, {ci[1]:.{dec}f}]")
        print(" & ".join(parts))
        print()

print("=== Summary stats for text ===")
for mode in ["B0", "BW", "BR"]:
    vals = sub[sub["mode"] == mode]
    if len(vals) == 0: continue
    e = vals["endpoint_acc"].mean()
    ei = ci95_mean(vals["endpoint_acc"].dropna().values)
    u = vals["udepth"].mean()
    ui = ci95_mean(vals["udepth"].dropna().values)
    os = vals["over_smoothing_readout"].mean()
    osi = ci95_mean(vals["over_smoothing_readout"].dropna().values)
    p = vals["decode_probe_acc"].mean()
    pi = ci95_mean(vals["decode_probe_acc"].dropna().values)
    d = vals["intervention_drop"].mean()
    di_ = ci95_mean(vals["intervention_drop"].dropna().values)
    print(f"  {mode}: End={e:.2f}[{ei[0]:.2f},{ei[1]:.2f}], UDepth={u:.2f}[{ui[0]:.2f},{ui[1]:.2f}], OS={os:.2f}[{osi[0]:.2f},{osi[1]:.2f}], Probe={p:.2f}[{pi[0]:.2f},{pi[1]:.2f}], Drop={d:.2f}[{di_[0]:.2f},{di_[1]:.2f}]")

print("\n=== B0 vs. BW/BR comparisons (L=128, k=128) ===")
for mode in ["BW", "BR"]:
    b0 = sub[sub["mode"] == "B0"].sort_values("seed")
    m = sub[sub["mode"] == mode].sort_values("seed")
    if len(b0) < 2 or len(m) < 2: continue
    n = min(len(b0), len(m))
    for col in ["endpoint_acc", "udepth", "intervention_drop", "decode_probe_acc"]:
        x = b0[col].values[:n]
        y = m[col].values[:n]
        t, p = sp_stats.ttest_rel(x, y)
        diff = y - x
        dci = ci95_mean(diff)
        print(f"  {mode} vs B0, {col}: diff={np.mean(diff):.4f} [{dci[0]:.4f},{dci[1]:.4f}], p={p:.2e}")
    print()
