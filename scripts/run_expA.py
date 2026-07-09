
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expA.train import eval_quant_sweep, train_ae
from src.parallel import run_parallel


def _expA_worker(task: tuple) -> list[dict]:
    k, seed, normalized, steps, device = task
    model, meta = train_ae(
        seed=seed, normalized=normalized, k=k, steps=steps, device=device,
    )
    rows = eval_quant_sweep(model, seed=seed, device=device)
    by_label = {r["quant_label"]: r["mse"] for r in rows}
    cliff = by_label["2-level"] / max(by_label["fp32"], 1e-12)
    for r in rows:
        r["final_train_mse"] = meta["final_mse"]
        r["k"] = k
        r["cliff"] = cliff
    cliff_row = {
        "seed": seed, "k": k, "normalized": normalized,
        "mse_fp32": by_label["fp32"], "mse_2level": by_label["2-level"],
        "cliff": cliff, "final_train_mse": meta["final_mse"],
    }
    return [rows, cliff_row]
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "expA"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--k-sweep", type=str, default="8", help="comma-separated k values")
    p.add_argument("--parallel", type=int, default=0, help="parallel workers (0=sequential)")
    args = p.parse_args()
    k_sweep = [int(k) for k in args.k_sweep.split(",") if k.strip()]
    if args.quick:
        args.seeds = 2
        args.steps = 300
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for k in k_sweep:
        for seed in range(args.seeds):
            for normalized in (False, True):
                tasks.append((k, seed, normalized, args.steps, args.device))
    n_workers = args.parallel if args.parallel > 0 else None
    results = run_parallel(_expA_worker, tasks, n_workers=n_workers)
    all_rows: list[dict] = []
    cliffs: list[dict] = []
    for rows, cliff_row in results:
        all_rows.extend(rows)
        cliffs.append(cliff_row)
    df = pd.DataFrame(all_rows)
    df_cliff = pd.DataFrame(cliffs)
    df.to_csv(out_dir / "mse_by_quant.csv", index=False)
    df_cliff.to_csv(out_dir / "cliff_scores.csv", index=False)
    summary = {"n_seeds": int(args.seeds), "k_sweep": k_sweep, "by_k": {}}
    for k in k_sweep:
        sub = df_cliff[df_cliff["k"] == k]
        std = sub[~sub["normalized"]].sort_values("seed")["cliff"].values
        nrm = sub[sub["normalized"]].sort_values("seed")["cliff"].values
        if len(std) >= 2 and len(std) == len(nrm):
            tstat, pval = stats.ttest_rel(std, nrm)
            diff = nrm - std
            mean_diff = float(diff.mean())
            sem = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
            ci_lo = mean_diff - 1.96 * sem
            ci_hi = mean_diff + 1.96 * sem
        else:
            tstat = pval = mean_diff = ci_lo = ci_hi = float("nan")
        summary["by_k"][str(k)] = {
            "cliff_standard_mean": float(std.mean()) if len(std) else None,
            "cliff_normalized_mean": float(nrm.mean()) if len(nrm) else None,
            "mean_diff_norm_minus_std": mean_diff,
            "ci95": [ci_lo, ci_hi],
            "t_stat": float(tstat) if tstat == tstat else None,
            "p_value": float(pval) if pval == pval else None,
        }
    all_std = df_cliff[~df_cliff["normalized"]].groupby("seed")["cliff"].mean().values
    all_nrm = df_cliff[df_cliff["normalized"]].groupby("seed")["cliff"].mean().values
    if len(all_std) >= 2 and len(all_std) == len(all_nrm):
        tstat, pval = stats.ttest_rel(all_std, all_nrm)
        diff = all_nrm - all_std
        mean_diff = float(diff.mean())
        sem = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        summary["mean_diff_norm_minus_std"] = mean_diff
        summary["ci95"] = [mean_diff - 1.96 * sem, mean_diff + 1.96 * sem]
        summary["t_stat"] = float(tstat)
        summary["p_value"] = float(pval)
    summary["note"] = "negative mean_diff means Normalized has lower cliff (supports A-H1)"
    with open(out_dir / "cliff_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    snr_cols = ["seed", "k", "normalized", "quant_label", "rho_signal", "rho_noise", "delta_rho",
                "snr_effective", "alignment", "signal_topvar_frac", "noise_topvar_frac", "var_ratio",
                "mse", "mse_matched_noise", "matched_noise_rho_signal", "matched_noise_delta_rho",
                "matched_noise_snr", "matched_noise_alignment", "mse_fixedbit"]
    snr_avail = [c for c in snr_cols if c in df.columns]
    snr = df[snr_avail]
    snr.to_csv(out_dir / "snr_probe.csv", index=False)
    print("\n=== ExpA summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {out_dir}")
if __name__ == "__main__":
    main()
