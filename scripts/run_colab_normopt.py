from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification import curriculum_length
from src.expB.data import make_batch
from src.expB.ssm import DiagonalSSM
from src.normopt.normalize import project_weight_rows_, row_radial_fraction
from src.normopt.optim import HybridOptimizer, MuonMomentum, TangentRowMomentum

ROOT = Path(__file__).resolve().parents[1]


def compute_spectral_metrics(matrix: torch.Tensor) -> tuple[float, float]:
    w = matrix.detach().float()
    s = torch.linalg.svdvals(w)
    s = s.clamp_min(1e-12)
    cond = float((s[0] / s[-1]).item())
    probs = s / s.sum()
    ent = float((-torch.sum(probs * torch.log(probs))).item())
    max_ent = math.log(len(s))
    norm_ent = ent / max_ent if max_ent > 0 else 1.0
    return (cond, norm_ent)


def build_optimizer(model: DiagonalSSM, opt_name: str) -> torch.optim.Optimizer:
    matrix_params = []
    matrix_names = []
    rest_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim == 2
            and any(k in name for k in ("out_proj", "head"))
            and ("weight" in name)
        ):
            matrix_params.append(p)
            matrix_names.append(name)
        else:
            rest_params.append(p)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    matrix_optimizer = {"muon": MuonMomentum, "rmo": TangentRowMomentum}[opt_name]
    return HybridOptimizer(
        matrix_optimizer(matrix_params, lr=0.02),
        torch.optim.AdamW(rest_params, lr=0.002, weight_decay=0.01),
        names=matrix_names,
    )


def evaluate_model(
    model: DiagonalSSM,
    L: int,
    V: int,
    device: torch.device,
    n_batches: int = 4,
    batch_size: int = 128,
) -> tuple[float, float]:
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = make_batch(batch_size, L=L, V=V, device=device)
            out = model(batch["input_ids"], batch["query_pos"])
            loss = F.cross_entropy(out["logits"], batch["target"])
            pred = out["logits"].argmax(-1)
            total_correct += (pred == batch["target"]).sum().item()
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return (total_loss / total_samples, total_correct / total_samples)


def run_single_experiment(
    geometry: str,
    optimizer_name: str,
    seed: int,
    steps: int,
    batch_size: int,
    L: int,
    V: int,
    d_model: int,
    k: int,
    device: torch.device,
    is_xla: bool = False,
) -> dict:
    if is_xla:
        import torch_xla
        import torch_xla.core.xla_model as xm

    torch.manual_seed(seed)
    np.random.seed(seed)
    ssm_mode = (
        "BW" if geometry == "normW" else "sphere_on_z" if geometry == "normWS" else "B0"
    )
    model = DiagonalSSM(
        V=V, L_max=max(L * 2, 256), d_model=d_model, k=k, mode=ssm_mode, n_layers=2
    ).to(device)
    opt = build_optimizer(model, optimizer_name)
    model.train()
    step_times = []
    radial_fractions = []
    best_val_acc = 0.0
    steps_to_50 = -1
    t0_all = time.perf_counter()
    for step in range(1, steps + 1):
        L_step = curriculum_length(step, steps, L)
        batch = make_batch(batch_size, L=L_step, V=V, device=device)
        t_step = time.perf_counter()
        opt.zero_grad()
        out = model(batch["input_ids"], batch["query_pos"])
        loss = F.cross_entropy(out["logits"], batch["target"])
        loss.backward()
        if step % 50 == 0 or step == steps:
            grads = {
                n: p.grad.detach()
                for n, p in model.named_parameters()
                if p.grad is not None and p.ndim == 2
            }
            weights = {
                n: p.detach() for n, p in model.named_parameters() if p.ndim == 2
            }
            rad_frac = row_radial_fraction(grads, weights)
            radial_fractions.append(rad_frac)
        if is_xla:
            xm.optimizer_step(opt)
            if hasattr(torch_xla, "sync"):
                torch_xla.sync()
            else:
                xm.mark_step()
        else:
            opt.step()
        if geometry in ("normW", "normWS"):
            project_weight_rows_(model, include_head=False)
        step_times.append(time.perf_counter() - t_step)
        if step % 100 == 0:
            _, q_acc = evaluate_model(
                model, L=L, V=V, device=device, n_batches=1, batch_size=128
            )
            model.train()
            if q_acc > best_val_acc:
                best_val_acc = q_acc
            if q_acc >= 0.5 and steps_to_50 == -1:
                steps_to_50 = step
    total_time = time.perf_counter() - t0_all
    val_loss, val_acc = evaluate_model(
        model, L=L, V=V, device=device, n_batches=4, batch_size=128
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    w_target = next(
        (
            p
            for n, p in model.named_parameters()
            if "out_proj.0.0.weight" in n or ("out_proj" in n and "weight" in n)
        )
    )
    cond_num, spectral_ent = compute_spectral_metrics(w_target)
    warm = min(20, len(step_times) // 5)
    lat_slice = step_times[warm:] if len(step_times) > warm else step_times
    return {
        "geometry": geometry,
        "optimizer": optimizer_name,
        "seed": seed,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "best_acc": best_val_acc,
        "steps_to_50": steps_to_50 if steps_to_50 != -1 else steps,
        "cond_num": cond_num,
        "spectral_entropy": spectral_ent,
        "radial_fraction": float(np.mean(radial_fractions))
        if radial_fractions
        else 0.0,
        "ms_per_step": float(np.mean(lat_slice)) * 1000.0,
        "total_time_sec": total_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "tpu", "xla"), default="auto"
    )
    parser.add_argument("--out", default=str(ROOT / "results" / "colab_normopt"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_xla = False
    if args.device in ("tpu", "xla") or (
        args.device == "auto"
        and (
            "COLAB_TPU_ADDR" in os.environ
            or "TPU_NAME" in os.environ
            or "TPU_ACCELERATOR_TYPE" in os.environ
        )
    ):
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
        is_xla = True
        dev_name = f"TPU ({xm.xla_device_hw(device)})"
    elif args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        dev_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        dev_name = "CPU"
    geometries = ["plain", "normW", "normWS"]
    optimizers = ["adamw", "muon", "rmo"]
    seeds = list(range(args.seeds))
    print("=" * 80, flush=True)
    print("NORM-OPT BENCHMARK: RMO vs. MUON vs. ADAMW", flush=True)
    print(
        f"Device: {dev_name} | Vectorized DiagonalSSM | L={args.L}, k={args.k}, d={64}",
        flush=True,
    )
    print(
        f"Factorial design: {len(geometries)} Geometries x {len(optimizers)} Optimizers x {len(seeds)} Seeds",
        flush=True,
    )
    print("=" * 80, flush=True)
    records = []
    for geom in geometries:
        for opt in optimizers:
            for seed in seeds:
                res = run_single_experiment(
                    geometry=geom,
                    optimizer_name=opt,
                    seed=seed,
                    steps=args.steps,
                    batch_size=64,
                    L=args.L,
                    V=16,
                    d_model=64,
                    k=args.k,
                    device=device,
                    is_xla=is_xla,
                )
                records.append(res)
                print(
                    f"[{geom:6s} | {opt:5s} | seed {seed}] -> Acc: {res['val_acc'] * 100:5.1f}% (Best: {res['best_acc'] * 100:5.1f}%) | Loss: {res['val_loss']:6.4f} | Steps->50%: {res['steps_to_50']:4d} | Radial: {res['radial_fraction'] * 100:4.1f}% | Cond: {res['cond_num']:5.1f} | Latency: {res['ms_per_step']:5.2f} ms",
                    flush=True,
                )
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metrics.csv", index=False)
    print(f"\nSaved raw metrics to {out_dir / 'metrics.csv'}", flush=True)
    summary = (
        df.groupby(["geometry", "optimizer"])
        .agg(
            mean_acc=("val_acc", "mean"),
            std_acc=("val_acc", "std"),
            mean_best=("best_acc", "mean"),
            mean_steps50=("steps_to_50", "mean"),
            mean_radial=("radial_fraction", "mean"),
            mean_cond=("cond_num", "mean"),
            mean_entropy=("spectral_entropy", "mean"),
            mean_latency=("ms_per_step", "mean"),
        )
        .reset_index()
    )
    print("\n" + "=" * 90, flush=True)
    print(f"STATISTICAL BENCHMARK SUMMARY (N={args.seeds} seeds)", flush=True)
    print("=" * 90, flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n" + "=" * 90, flush=True)
    print("HYPOTHESIS TESTS: RMO vs. MUON", flush=True)
    print("=" * 90, flush=True)
    for geom in geometries:
        sub = df[df["geometry"] == geom]
        m_muon = sub[sub["optimizer"] == "muon"]["val_acc"].values
        m_rmo = sub[sub["optimizer"] == "rmo"]["val_acc"].values
        diff = m_rmo - m_muon
        lat_muon = sub[sub["optimizer"] == "muon"]["ms_per_step"].mean()
        lat_rmo = sub[sub["optimizer"] == "rmo"]["ms_per_step"].mean()
        speedup = lat_muon / (lat_rmo + 1e-08)
        print(f"\nCondition [{geom}]:", flush=True)
        print(
            f"  Accuracy: Muon = {np.mean(m_muon) * 100:.2f}% | RMO = {np.mean(m_rmo) * 100:.2f}%",
            flush=True,
        )
        print(f"  Delta (RMO - Muon): {np.mean(diff) * 100:+.2f}%", flush=True)
        print(
            f"  Speedup: RMO is {speedup:.2f}x faster than Muon ({lat_rmo:.2f}ms vs {lat_muon:.2f}ms/step)",
            flush=True,
        )
        if len(diff) > 1 and np.std(diff) > 0:
            t_stat, p_val_t = stats.ttest_rel(m_rmo, m_muon)
            ci95 = stats.t.interval(
                0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
            )
            print(
                f"  Paired t-test: t = {t_stat:.3f}, p-value = {p_val_t:.4f}",
                flush=True,
            )
            print(
                f"  95% CI: [{ci95[0] * 100:+.2f}%, {ci95[1] * 100:+.2f}%]", flush=True
            )
            w_stat, p_val_w = stats.wilcoxon(m_rmo, m_muon)
            print(
                f"  Wilcoxon signed-rank: W = {w_stat:.1f}, p-value = {p_val_w:.4f}",
                flush=True,
            )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panel_specs = [
        ("val_acc", "Validation Accuracy", axes[0, 0]),
        ("radial_fraction", "Radial Gradient Waste (Collinear Fraction)", axes[0, 1]),
        ("cond_num", "Condition Number (kappa)", axes[1, 0]),
        ("ms_per_step", "Step Latency (ms)", axes[1, 1]),
    ]
    for col, title, ax in panel_specs:
        piv = df.pivot_table(
            index="geometry", columns="optimizer", values=col, aggfunc="mean"
        )
        piv.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=160)
    print(
        f"\nSaved comprehensive 4-panel diagnostic plot to {out_dir / 'comparison.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
