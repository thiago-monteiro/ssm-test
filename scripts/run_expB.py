
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

from src.expB.train import eval_position, train_ssm
from src.parallel import run_parallel


def _expB_worker(task: tuple) -> tuple:
    seed, mode, L, k, steps, d_model, device, queries_per_pos, no_pos_embed, with_replacement = task
    model, meta = train_ssm(
        seed=seed, mode=mode, L=L, k=k, steps=steps,
        d_model=d_model, device=device,
        no_pos_embed=no_pos_embed, with_replacement=with_replacement,
    )
    metrics = eval_position(
        model, seed=seed, L=L, V=model.V, queries_per_pos=queries_per_pos,
        device=device, do_intervention=True, do_decode_probe=True, do_task_os=True,
    )
    metrics["final_train_acc"] = meta["final_acc"]
    metrics["best_val_acc"] = meta["best_val_acc"]
    curve = metrics.pop("acc_curve")
    tau = metrics.pop("tau", None)
    return (metrics, curve, tau)
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--modes", type=str, default="B0,BW,BR")
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "expB"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--queries-per-pos", type=int, default=200)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--scale", action="store_true")
    p.add_argument("--all-ablations", action="store_true")
    p.add_argument("--parallel", type=int, default=0, help="parallel workers (0=sequential)")
    p.add_argument("--long-L", type=int, nargs="+", default=None, help="list of L values for sweep (overrides --L)")
    p.add_argument("--no-pos-embed", action="store_true", help="disable position embeddings")
    p.add_argument("--no-replacement", action="store_true", help="sample tokens without replacement")
    args = p.parse_args()
    if args.quick:
        args.seeds = 2
        args.steps = 400
        args.queries_per_pos = 40
    if args.all_ablations:
        args.modes = "B0,BW,BR,BX,BW_BR,B0_noshort,BR_noshort,sphere_on_z"
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    l_values = args.long_L if args.long_L is not None else [args.L]
    grids = []
    if args.scale and not args.quick:
        scale_grid = [
            (Lv, kv, args.steps)
            for Lv in l_values
            for kv in ([64, 128, 256] if Lv >= 128 else [64, 128])
        ]
        grids.extend(scale_grid)
    else:
        for Lv in l_values:
            grids.append((Lv, args.k, args.steps))
    tasks = []
    for L, k, steps in grids:
        for seed in range(args.seeds):
            for mode in modes:
                tasks.append((seed, mode, L, k, steps, args.d_model, args.device,
                              args.queries_per_pos, args.no_pos_embed, not args.no_replacement))
    n_workers = args.parallel if args.parallel > 0 else None
    raw_results = run_parallel(_expB_worker, tasks, n_workers=n_workers)
    all_metrics: list[dict] = []
    curves: list[dict] = []
    for metrics, curve, tau in raw_results:
        all_metrics.append(metrics)
        L_val = metrics["L"]
        k_val = metrics["k"]
        mode_val = metrics["mode"]
        seed_val = metrics["seed"]
        for ell, a in enumerate(curve):
            curves.append({
                "seed": seed_val, "mode": mode_val, "L": L_val, "k": k_val,
                "position": ell, "accuracy": a,
            })
    df = pd.DataFrame(all_metrics)
    df_curves = pd.DataFrame(curves)
    df.to_csv(out_dir / "metrics.csv", index=False)
    df_curves.to_csv(out_dir / "position_curves.csv", index=False)
    stats_out = {"grids": {}}
    stats_grids = [("primary", grids[0])]
    for idx, g in enumerate(grids[1:]):
        stats_grids.append((f"scale_{idx}_L{g[0]}_k{g[1]}", g))
    for glabel, (L0, k0, _) in stats_grids:
        sub = df[(df["L"] == L0) & (df["k"] == k0)]
        if len(sub) == 0:
            continue
        gs = {"L": L0, "k": k0, "comparisons": {}, "guardrail": {}}
        if "B0" in sub["mode"].values:
            b0 = sub[sub["mode"] == "B0"].sort_values("seed")
            e0 = float(b0["endpoint_acc"].mean()) if len(b0) else 0.0
            o0 = float(b0["overall_acc"].mean()) if len(b0) else 0.0
            gs["baseline_endpoint_mean"] = e0
            gs["baseline_overall_mean"] = o0
            delta = 0.05
            for mode in [m for m in modes if m != "B0"]:
                mdf = sub[sub["mode"] == mode].sort_values("seed")
                if len(b0) < 2 or len(mdf) == 0:
                    continue
                u0 = b0["udepth"].values[:len(mdf)]
                um = mdf["udepth"].values[:len(mdf)]
                if len(u0) < 2:
                    continue
                tstat, pval = stats.ttest_rel(u0, um)
                diff = um - u0
                mean_diff = float(diff.mean())
                sem = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
                gs["comparisons"][mode] = {
                    "udepth_b0_mean": float(u0.mean()),
                    "udepth_mode_mean": float(um.mean()),
                    "mean_diff_mode_minus_b0": mean_diff,
                    "ci95": [mean_diff - 1.96 * sem, mean_diff + 1.96 * sem],
                    "t_stat": float(tstat),
                    "p_value": float(pval),
                    "os_b0_mean": float(b0["over_smoothing"].mean()),
                    "os_mode_mean": float(mdf["over_smoothing"].mean()),
                    "os_readout_b0_mean": float(b0["over_smoothing_readout"].mean()),
                    "os_readout_mode_mean": float(mdf["over_smoothing_readout"].mean()),
                    "probe_acc_b0_mean": float(b0["decode_probe_acc"].mean()),
                    "probe_acc_mode_mean": float(mdf["decode_probe_acc"].mean()),
                    "int_drop_b0_mean": float(b0["intervention_drop"].mean()),
                    "int_drop_mode_mean": float(mdf["intervention_drop"].mean()),
                    "task_os_b0_mean": float(b0["task_conditioned_os"].mean()),
                    "task_os_mode_mean": float(mdf["task_conditioned_os"].mean()),
                    "snr_effective_b0_mean": float(b0["snr_effective"].mean()),
                    "snr_effective_mode_mean": float(mdf["snr_effective"].mean()),
                }
                e = float(mdf["endpoint_acc"].mean())
                o = float(mdf["overall_acc"].mean())
                pass_guard = (e >= e0 - delta) and (o >= o0 - delta)
                udepth_better = mean_diff < 0
                gs["guardrail"][mode] = {
                    "endpoint_mean": e,
                    "overall_mean": o,
                    "pass_guardrail": bool(pass_guard),
                    "udepth_reduced": bool(udepth_better),
                    "h1_support": bool(pass_guard and udepth_better and pval < 0.05),
                    "h1_support_descriptive": bool(pass_guard and udepth_better),
                }
        stats_out["grids"][glabel] = gs
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2)
    lines = ["# Guardrail report (parallel)", ""]
    for gl, gs in stats_out.get("grids", {}).items():
        lines.append(f"## {gl} (L={gs.get('L')}, k={gs.get('k')})")
        lines.append(f"Baseline endpoint: {gs.get('baseline_endpoint_mean', 'n/a')}")
        lines.append(f"Baseline overall: {gs.get('baseline_overall_mean', 'n/a')}")
        lines.append("")
        for mode, g in gs.get("guardrail", {}).items():
            lines.append(f"### {mode}")
            lines.append(f"- pass_guardrail: {g['pass_guardrail']}")
            lines.append(f"- udepth_reduced: {g['udepth_reduced']}")
            lines.append(f"- H1: {g['h1_support']}")
            lines.append("")
    (out_dir / "guardrail_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== ExpB summary ===")
    print(json.dumps(stats_out, indent=2))
    print(f"Wrote results to {out_dir}")
if __name__ == "__main__":
    main()
