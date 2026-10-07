from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("ssm-l40s-full")
vol = modal.Volume.from_name("ssm-l40s", create_if_missing=True)
_CAUSAL = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0%2Bcu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
_MAMBA = "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers==4.45.2",
        "huggingface-hub==0.25.2",
        "einops==0.8.0",
        "packaging>=24.0",
        "numpy>=1.24.0",
        "tqdm",
    )
    .run_commands(f"pip install {_CAUSAL} {_MAMBA}")
    .add_local_dir(
        ROOT,
        "/repo",
        ignore=["results/**", ".git/**", ".venv*/**", "**/__pycache__/**", "*.zip"],
    )
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/repo")


_USD_PER_HR = 2.2


def _sanitize(value):
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _emit(tag: str, payload: dict) -> dict:
    payload = _sanitize(payload)
    print(f"[{tag}_JSON_BEGIN]", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"[{tag}_JSON_END]", flush=True)
    return payload


def _peak_gib(metrics_path: Path) -> float:
    rows = metrics_path.read_text().splitlines()
    return max(
        (json.loads(line).get("peak_allocated_gib", 0.0) or 0.0 for line in rows)
    )


@app.function(
    gpu="L40S",
    cpu=2,
    memory=32768,
    timeout=6 * 3600,
    volumes={"/vol": vol},
    image=image,
    env={"HF_HOME": "/vol/hf-cache"},
)
def train_full(
    steps: int = 1000,
    benchmark_steps: int = 20,
    seed: int = 0,
    geometry: str = "normW",
    benchmark_only: bool = False,
    matrix_lrs: str = "2e-4",
    optimizers: str = "muon,rmo",
) -> dict:
    import shutil

    import torch

    lr_list = [float(x) for x in matrix_lrs.split(",") if x.strip()]
    opt_list = [o.strip() for o in optimizers.split(",") if o.strip()]
    assert lr_list and opt_list, "need at least one lr and one optimizer"
    assert torch.cuda.is_available(), "L40S function has no CUDA"
    print(
        f"GPU: {torch.cuda.get_device_name(0)} sweep: {opt_list} x {lr_list}",
        flush=True,
    )
    out = f"/vol/l40s-full/seed{seed}"
    shutil.rmtree(f"{out}-bench", ignore_errors=True)
    if not benchmark_only:
        for lr in lr_list:
            shutil.rmtree(
                f"{out}-lr{lr}" if len(lr_list) > 1 else out, ignore_errors=True
            )
    bench: dict[str, dict] = {}
    for opt in opt_list:
        _run(
            [
                sys.executable,
                "scripts/run_t4_mamba_normopt.py",
                "--full-model",
                "--geometry",
                geometry,
                "--steps",
                str(benchmark_steps),
                "--optimizers",
                opt,
                "--seeds",
                str(seed),
                "--eval-every",
                str(benchmark_steps),
                "--eval-batches",
                "1",
                "--no-save-checkpoint",
                "--output",
                f"{out}-bench",
            ]
        )
        cond = Path(f"{out}-bench/seed{seed}/{opt}")
        summary = json.loads((cond / "summary.json").read_text())
        cfg = json.loads((cond / "resolved_config.json").read_text())
        sec_per_step = summary["wall_seconds"] / summary["completed_steps"]
        peak = _peak_gib(cond / "metrics.jsonl")
        bench[opt] = {
            "steps": benchmark_steps,
            "sec_per_step": sec_per_step,
            "peak_gib": peak,
            "matrix_lr": cfg["matrix_lr"],
            "rest_lr": cfg["adamw_lr"],
            "geometry": cfg["geometry"],
            "extrap_hours_per_1000": sec_per_step * 1000 / 3600,
            "extrap_cost_usd_per_1000": sec_per_step * 1000 / 3600 * _USD_PER_HR,
        }
        print(
            f"BENCH {opt}: {sec_per_step:.2f}s/step peak={peak:.1f}GiB -> 1000 steps ~{sec_per_step * 1000 / 3600:.2f}h ~${sec_per_step * 1000 / 3600 * _USD_PER_HR:.2f}",
            flush=True,
        )
    _emit(
        "BENCHMARK",
        {"seed": seed, "benchmark": bench, "usd_per_hr_assumed": _USD_PER_HR},
    )
    if benchmark_only:
        return {"benchmark": bench, "status": "BENCH-ONLY"}
    runs = []
    for opt in opt_list:
        for lr in lr_list:
            run_out = f"{out}-lr{lr}" if len(lr_list) > 1 else out
            _run(
                [
                    sys.executable,
                    "scripts/run_t4_mamba_normopt.py",
                    "--full-model",
                    "--geometry",
                    geometry,
                    "--steps",
                    str(steps),
                    "--matrix-lr",
                    str(lr),
                    "--optimizers",
                    opt,
                    "--seeds",
                    str(seed),
                    "--output",
                    run_out,
                ]
            )
            cond = Path(f"{run_out}/seed{seed}/{opt}")
            summary = json.loads((cond / "summary.json").read_text())
            wall = summary["wall_seconds"]
            runs.append(
                {
                    "optimizer": opt,
                    "matrix_lr": lr,
                    "status": summary["status"],
                    "completed_steps": summary["completed_steps"],
                    "processed_tokens": summary["processed_tokens"],
                    "tokens_per_second": summary["processed_tokens"] / wall,
                    "wall_hours": wall / 3600,
                    "cost_usd": wall / 3600 * _USD_PER_HR,
                    "clipped_steps": summary["clipped_steps"],
                    "peak_gib": _peak_gib(cond / "metrics.jsonl"),
                    "initial_validation": summary["initial_validation"],
                    "final_validation": summary.get("final_validation"),
                }
            )
    vol.commit()
    return _emit(
        "RESULT",
        {
            "seed": seed,
            "steps": steps,
            "geometry": geometry,
            "matrix_lrs": lr_list,
            "optimizers": opt_list,
            "benchmark": bench,
            "runs": runs,
            "total_cost_usd": sum((r["cost_usd"] for r in runs)),
            "usd_per_hr_assumed": _USD_PER_HR,
            "status": "DONE",
        },
    )


@app.local_entrypoint()
def main(
    steps: int = 1000,
    benchmark_steps: int = 20,
    seed: int = 0,
    geometry: str = "normW",
    benchmark_only: bool = False,
    matrix_lrs: str = "2e-4",
    optimizers: str = "muon,rmo",
) -> None:
    print(
        json.dumps(
            _sanitize(
                train_full.remote(
                    steps,
                    benchmark_steps,
                    seed,
                    geometry,
                    benchmark_only,
                    matrix_lrs,
                    optimizers,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )
