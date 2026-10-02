"""Short, paired Mamba 2.8B optimizer stability validation on a Colab T4."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.large_mamba.config import ModelConfig
from src.large_mamba.data import TokenPools, derived_example_seed, generate_recall_example
from src.normopt.normalize import project_rows_
from src.normopt.optim import HybridOptimizer, MuonMomentum, TangentRowMomentum


class MasterLinear(torch.nn.Linear):
    """Keep optimized matrices/momentum in FP32 while the backbone uses FP16."""

    def forward(self, inputs):
        return F.linear(inputs.float(), self.weight, self.bias).to(inputs.dtype)


def prepare_model(model, layers: int, geometry: str):
    blocks = model.backbone.layers
    if not 1 <= layers <= len(blocks):
        raise ValueError(f"--train-layers must be between 1 and {len(blocks)}")
    model.requires_grad_(False)
    names, params = [], []
    for index in range(len(blocks) - layers, len(blocks)):
        mixer = blocks[index].mixer
        base = mixer.out_proj
        linear = MasterLinear(base.in_features, base.out_features, bias=base.bias is not None)
        linear.weight = torch.nn.Parameter(base.weight.detach().float())
        if base.bias is not None:
            linear.bias = torch.nn.Parameter(base.bias.detach().float(), requires_grad=False)
        mixer.out_proj = linear
        if geometry == "normW":
            project_rows_(linear.weight)
        names.append(f"backbone.layers.{index}.mixer.out_proj.weight")
        params.append(linear.weight)
    return names, params


def build_optimizer(params, name, args):
    if name == "rmo":
        return TangentRowMomentum(params, lr=args.matrix_lr, momentum=0.95)
    if name == "muon":
        return MuonMomentum(params, lr=args.matrix_lr, ns_steps=5, ns_dtype=torch.float16)
    return torch.optim.AdamW(params, lr=args.adamw_lr, weight_decay=0.01)


def prepare_full_model(model, geometry: str):
    model.requires_grad_(True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [dict(model.named_parameters())[n] for n in names]
    model.float()
    if geometry == "normW":
        with torch.no_grad():
            for n, p in zip(names, params):
                if p.ndim == 2 and "embed" not in n:
                    project_rows_(p)
    matrix = [(n, p) for n, p in zip(names, params) if p.ndim == 2 and "embed" not in n]
    rest = [p for n, p in zip(names, params) if not (p.ndim == 2 and "embed" not in n)]
    return names, params, [p for _, p in matrix], rest


def build_full_optimizer(matrix_params, rest_params, name, args):
    main = (TangentRowMomentum(matrix_params, lr=args.matrix_lr, momentum=0.95)
            if name == "rmo"
            else MuonMomentum(matrix_params, lr=args.matrix_lr, ns_steps=5, ns_dtype=torch.float16))
    rest_opt = torch.optim.AdamW(rest_params, lr=args.adamw_lr, weight_decay=0.01) if rest_params else None
    matrix_names = [f"matrix[{i}]" for i in range(len(matrix_params))]
    return HybridOptimizer(main, rest_opt, names=matrix_names, spec=None)


def load_model(smoke: bool):
    from transformers import MambaConfig, MambaForCausalLM

    if smoke:
        return MambaForCausalLM(MambaConfig(
            hidden_size=16, state_size=4, num_hidden_layers=2, vocab_size=128,
            expand=2, time_step_rank=2, use_cache=False, use_mambapy=False,
        ))
    # The official checkpoint shares tensor names with Transformers. CUDA kernels
    # come from mamba_ssm and causal_conv1d, verified before checkpoint loading.
    from huggingface_hub import hf_hub_download

    spec = ModelConfig()
    raw = json.loads(Path(hf_hub_download(
        spec.repository, "config.json", revision=spec.revision,
    )).read_text())
    if raw["d_model"] != 2560 or raw["n_layer"] != 64 or raw.get("ssm_cfg", {}):
        raise ValueError("expected the pinned standard Mamba 2.8B architecture")
    multiple = raw.get("pad_vocab_size_multiple", 8)
    config = MambaConfig(
        hidden_size=raw["d_model"], num_hidden_layers=raw["n_layer"],
        vocab_size=math.ceil(raw["vocab_size"] / multiple) * multiple,
        residual_in_fp32=raw.get("residual_in_fp32", True),
        use_cache=False, use_mambapy=False,
    )
    weights = hf_hub_download(spec.repository, "pytorch_model.bin", revision=spec.revision)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    with torch.device("meta"):
        model = MambaForCausalLM(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    return model.to(dtype=torch.float16)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def require_cuda_kernels():
    from transformers.models.mamba import modeling_mamba

    if not modeling_mamba.is_fast_path_available:
        raise RuntimeError(
            "Mamba CUDA kernels are unavailable. Run scripts/setup_colab_t4.py, "
            "restart the Colab session, and retry. Refusing the slow reference scan."
        )
    # Import the extensions explicitly so ABI/driver failures occur before loading 2.8B.
    import selective_scan_cuda
    import causal_conv1d_cuda


def batch(pools, args, split_seed, index, device):
    rows = [generate_recall_example(
        pools, seed=derived_example_seed(split_seed, index * args.micro_batch_size + row),
        sequence_length=args.sequence_length, associations=args.associations,
        queries=4, lag_bucket=("near", "middle", "far")[(index + row) % 3],
    ) for row in range(args.micro_batch_size)]
    return (torch.stack([row.input_ids for row in rows]).to(device),
            torch.stack([row.labels for row in rows]).to(device))


def loss_and_accuracy(model, inputs, labels):
    # Only project the four answer-prediction positions into the 50K vocabulary.
    hidden = model.backbone(inputs, use_cache=False).last_hidden_state
    positions = torch.arange(inputs.shape[1] - 12 + 2, inputs.shape[1], 3, device=inputs.device)
    targets = labels.index_select(1, positions).reshape(-1)
    answer_hidden = hidden.index_select(1, positions - 1).to(model.lm_head.weight.dtype)
    logits = model.lm_head(answer_hidden).float().reshape(-1, model.config.vocab_size)
    return F.cross_entropy(logits, targets), (logits.argmax(-1) == targets).float().mean()


def evaluate(model, pools, args, device, sync, amp: bool = True):
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for index in range(args.eval_batches):
            with autocast(device) if amp else nullcontext():
                loss, accuracy = loss_and_accuracy(model, *batch(pools, args, 30007, index, device))
            sync()
            losses.append(float(loss))
            accuracies.append(float(accuracy))
    model.train()
    return {"loss": sum(losses) / len(losses), "accuracy": sum(accuracies) / len(accuracies)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_condition(args, seed, name, device, sync, pools):
    output = args.output / f"seed{seed}" / name
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    print(f"Loading {'tiny smoke model' if args.smoke else 'Mamba 2.8B'} for {name} seed={seed}...", flush=True)
    model = load_model(args.smoke)
    full = bool(args.full_model)
    if full:
        names, params, matrix_params, rest_params = prepare_full_model(model, args.geometry)
        model.to(device)
        by_name = dict(model.named_parameters())
        params = [by_name[n] for n in names]
        matrix_params = [p for n, p in zip(names, params) if p.ndim == 2 and "embed" not in n]
        rest_params = [p for n, p in zip(names, params) if not (p.ndim == 2 and "embed" not in n)]
    else:
        names, params = prepare_model(model, args.train_layers, args.geometry)
        model.to(device)
        params = [dict(model.named_parameters())[name] for name in names]
        matrix_params = params
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if full:
        if name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=args.adamw_lr, weight_decay=0.01)
        else:
            optimizer = build_full_optimizer(matrix_params, rest_params, name, args)
        scaler = None
    else:
        optimizer = build_optimizer(params, name, args)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.0)
    initial = evaluate(model, pools, args, device, sync, amp=not full)
    manifest = {
        **vars(args), "output": str(output),
        "archive_dir": str(args.archive_dir) if args.archive_dir else None,
        "seed": seed, "optimizer": name,
        "device": str(device), "model": "tiny-random" if args.smoke else ModelConfig().repository,
        "revision": None if args.smoke else ModelConfig().revision,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in params), "trainable_names": names,
        "initial_validation": initial, "torch_version": torch.__version__,
        "train_split_seed": derived_example_seed(20003, seed), "validation_split_seed": 30007,
        "claim_scope": ("full-parameter training" if full
                        else "output-projection fine-tuning stability; not full-parameter pretraining"),
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        commit = None
    manifest["git_commit"] = commit.stdout.strip() if commit is not None and commit.returncode == 0 else None
    manifest["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ("scripts/run_t4_mamba_normopt.py", "src/normopt/optim.py")}
    if device.type == "cuda":
        from importlib.metadata import version
        manifest.update(gpu=torch.cuda.get_device_name(device), cuda_version=torch.version.cuda,
                        precision=("FP32 full-model, FP32 optimizer state" if full
                                   else "FP16 autocast, FP32 optimized weights and optimizer state"),
                        mamba_version=version("mamba-ssm"), causal_conv1d_version=version("causal-conv1d"),
                        cuda_kernels=True)
    write_json(output / "resolved_config.json", manifest)
    started = time.perf_counter()
    clipped_steps = 0
    step = 0
    summary = {"status": "RUNNING", "seed": seed, "optimizer": name, "smoke": args.smoke}
    try:
        with (output / "metrics.jsonl").open("w") as metrics:
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                scale = min(1.0, step / max(1, round(args.steps * 0.05)))
                if isinstance(optimizer, HybridOptimizer):
                    optimizer.set_scale(scale)
                else:
                    for group in optimizer.param_groups:
                        group["lr"] = (args.adamw_lr if name == "adamw" else args.matrix_lr) * scale
                losses = []
                for micro in range(args.accumulation_steps):
                    inputs, labels = batch(pools, args, derived_example_seed(20003, seed),
                                          (step - 1) * args.accumulation_steps + micro, device)
                    with autocast(device) if not full else nullcontext():
                        loss, _ = loss_and_accuracy(model, inputs, labels)
                    if full:
                        (loss / args.accumulation_steps).backward()
                    else:
                        scaler.scale(loss / args.accumulation_steps).backward()
                    losses.append(loss.detach())
                    sync()
                loss = torch.stack(losses).mean()
                if any(parameter.grad is None for parameter in params):
                    raise RuntimeError(f"missing gradient for an optimized matrix at step {step}")
                if not full:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, foreach=False)
                sync()
                loss_value, norm_value = float(loss), float(grad_norm)
                if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                    raise FloatingPointError(f"nonfinite loss/gradient at step {step}")
                clipped_steps += int(norm_value > args.max_grad_norm)
                if full:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                if args.geometry == "normW":
                    for parameter in matrix_params:
                        project_rows_(parameter)
                finite = torch.stack([torch.isfinite(p.detach()).all() for p in params]).all()
                row_error = torch.stack([(p.detach().float().norm(dim=1) - 1).abs().max() for p in matrix_params]).max()
                sync()
                if not bool(finite):
                    raise FloatingPointError(f"nonfinite parameters at step {step}")
                error_value = float(row_error)
                if args.geometry == "normW" and error_value > 1e-4:
                    raise FloatingPointError(f"row constraint violated at step {step}: {error_value}")
                row = {"step": step, "loss": loss_value, "grad_norm_before_clip": norm_value,
                       "row_norm_max_error": error_value,
                       "loss_scale": None if full else scaler.get_scale(),
                       "elapsed_seconds": time.perf_counter() - started}
                if device.type == "cuda":
                    row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                if step == 1 or step % args.log_every == 0:
                    print(f"{name} seed={seed} step={step}/{args.steps} loss={loss_value:.4f} grad={norm_value:.4f}", flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    row["validation"] = evaluate(model, pools, args, device, sync, amp=not full)
                    print(f"{name} seed={seed} step={step} loss={loss_value:.4f} validation={row['validation']}", flush=True)
                    if not all(math.isfinite(v) for v in row["validation"].values()):
                        raise FloatingPointError(f"nonfinite validation at step {step}: {row['validation']}")
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 100 == 0 or step == args.steps:
                    metrics.flush()
                periodic_checkpoint = args.checkpoint_every > 0 and step % args.checkpoint_every == 0
                if args.save_checkpoint and (periodic_checkpoint or step == args.steps):
                    torch.save({"step": step, "trainable_state": {n: p.detach().cpu() for n, p in zip(names, params)},
                                "optimizer": _cpu_state(optimizer.state_dict()),
                                "scaler": None if full else scaler.state_dict()}, output / f"checkpoint-{step}.pt")
        summary.update(status="PASS", completed_steps=step, clipped_steps=clipped_steps,
                       final_validation=row["validation"], initial_validation=initial,
                       processed_tokens=args.steps * args.accumulation_steps * args.micro_batch_size * args.sequence_length,
                       wall_seconds=time.perf_counter() - started,
                       interpretation="finite training and preserved constraints on this seed and trainable scope")
    except Exception as error:
        summary.update(status="FAIL", failed_step=step, error=str(error))
        raise
    finally:
        write_json(output / "summary.json", summary)
    del model, optimizer, params, scaler, inputs, labels, loss, losses, grad_norm, finite, row_error
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_state(item) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/t4_mamba_normopt")
    parser.add_argument("--archive-dir", type=Path,
                        help="copy each completed condition here once; keep live metrics on local disk")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--optimizers", nargs="+", choices=("rmo", "adamw", "muon"), default=["rmo", "muon"])
    parser.add_argument("--geometry", choices=("plain", "normW"), default="normW")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-layers", type=int, default=8)
    parser.add_argument("--full-model", action="store_true",
                        help="train all parameters (Muon/RMO on dense 2D, AdamW fallback); ignores --train-layers")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--associations", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--matrix-lr", type=float, default=None,
                        help="matrix LR; defaults to 0.02 (8-layer) or 2e-4 (--full-model)")
    parser.add_argument("--adamw-lr", type=float, default=None,
                        help="AdamW LR, incl. full-model fallback; defaults to 0.002 or 2e-5 (--full-model)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="periodic checkpoint interval; 0 saves only at the end")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                        help="save trainable weights/optimizer at the end; disable for preflight")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU implementation check; never research evidence")
    args = parser.parse_args()
    if args.smoke:
        args.train_layers = 2
    if args.matrix_lr is None:
        args.matrix_lr = 2e-4 if args.full_model else 0.02
    if args.adamw_lr is None:
        args.adamw_lr = 2e-5 if args.full_model else 0.002
    if any(getattr(args, key) <= 0 for key in (
        "steps", "train_layers", "sequence_length", "associations", "micro_batch_size",
        "accumulation_steps", "eval_every", "eval_batches", "log_every",
        "matrix_lr", "adamw_lr", "max_grad_norm",
    )):
        parser.error("budgets, learning rates, and clipping norm must be positive")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    if len(args.seeds) > 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("provide one to three distinct seeds for this bounded validation")
    if len(set(args.optimizers)) != len(args.optimizers):
        parser.error("optimizers must be distinct")
    if args.associations < 4 or 3 * (args.associations + 4) > args.sequence_length:
        parser.error("recall examples require >=4 associations and 3*(associations+4) tokens")
    if args.smoke:
        torch.set_num_threads(1)
        device, sync = torch.device("cpu"), lambda: None
        pools = TokenPools(tuple(range(4, 36)), tuple(range(36, 68)), tuple(range(68, 100)), 2, 3)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime in Colab; CUDA is required for the 2.8B run")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        require_cuda_kernels()
        sync = lambda: torch.cuda.synchronize(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}; eager CUDA kernels, FP16 AMP", flush=True)
        from transformers import AutoTokenizer
        spec = ModelConfig()
        pools = TokenPools.from_tokenizer(AutoTokenizer.from_pretrained(
            spec.tokenizer_repository, revision=spec.tokenizer_revision,
        ))
    # Reject syntactically valid but unrealizable lag cells before downloading weights.
    for index in range(3):
        batch(pools, args, 30007, index, torch.device("cpu"))
    if args.archive_dir and args.archive_dir.resolve().is_relative_to(args.output.resolve()):
        parser.error("--archive-dir must be outside --output")
    results = []
    for seed in args.seeds:
        for name in args.optimizers:
            archive = args.archive_dir / f"seed{seed}" / name if args.archive_dir else None
            if archive and archive.exists():
                raise FileExistsError(f"archive already exists: {archive}")
            results.append(run_condition(args, seed, name, device, sync, pools))
            if archive:
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(args.output / f"seed{seed}" / name, archive)
                print(f"Archived completed {name} seed={seed} to {archive}", flush=True)
    write_json(args.output / "comparison.json", {
        "design": "paired seeds, identical initial weights, train stream and validation examples",
        "scope": "bounded stability validation; no significance or seed-robustness claim",
        "smoke": args.smoke, "runs": results,
    })
    if args.archive_dir:
        shutil.copy2(args.output / "comparison.json", args.archive_dir / "comparison.json")


if __name__ == "__main__":
    main()
