from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .adapter import set_projection_strength
from .config import Condition, ExperimentConfig, projection_strength
from .data import batch_hash
from .metrics import answer_token_loss
from .modeling import trainable_state_dict


BatchFactory = Callable[[int, int], dict[str, torch.Tensor]]


def build_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    lora, recurrence = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (lora if "lora_" in name else recurrence).append(parameter)
    if not lora or not recurrence:
        raise ValueError("both LoRA and recurrence parameter groups must be non-empty")
    all_trainable = lora + recurrence
    return torch.optim.AdamW(
        [
            {"params": lora, "lr": config.training.lora_learning_rate},
            {"params": recurrence, "lr": config.training.recurrence_learning_rate},
        ],
        weight_decay=config.training.weight_decay,
        fused=bool(all_trainable) and all(parameter.device.type == "cuda" for parameter in all_trainable),
    )


def learning_rate_scale(step: int, total: int, warmup_fraction: float) -> float:
    warmup = max(1, round(total * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def run_training(
    model: torch.nn.Module,
    *,
    condition: Condition | str,
    seed: int,
    config: ExperimentConfig,
    batch_factory: BatchFactory,
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    pass
    config.validate()
    condition = Condition(condition)
    torch.manual_seed(seed)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_scale(
            step, config.training.optimizer_steps, config.training.warmup_fraction
        ),
    )
    micro_batch_sizes = config.training.micro_batch_sizes
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_path = output_dir / "batch_hashes.jsonl"
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if 0 in config.training.checkpoint_steps:
        torch.save(trainable_state_dict(model), output_dir / "checkpoint-0.pt")

    hash_file = hash_path.open("w")
    progress = tqdm(
        range(config.training.optimizer_steps),
        desc=f"train {condition} seed={seed}",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        for step in progress:
            if condition is Condition.SPHERE:
                set_projection_strength(
                    model,
                    projection_strength(
                        step, config.training.optimizer_steps,
                        config.training.projection_ramp_fraction,
                    ),
                )
            loss_sum = 0.0
            step_hashes: list[str] = []
            for micro, planned_batch_size in enumerate(micro_batch_sizes):
                batch = batch_factory(step, micro)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                if input_ids.shape[0] != planned_batch_size:
                    raise ValueError(
                        f"microbatch {micro} has {input_ids.shape[0]} examples; "
                        f"expected {planned_batch_size}"
                    )
                step_hashes.append(batch_hash(input_ids))
                amp_enabled = device.type == "cuda"
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    logits = model(input_ids).logits
                    loss = answer_token_loss(logits, labels) * (
                        planned_batch_size / config.training.effective_batch_size
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at optimizer step {step}")
                loss.backward()
                loss_sum += float(loss.detach())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.training.max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient at optimizer step {step}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            hash_file.write(json.dumps({"step": step, "microbatch_hashes": step_hashes}) + "\n")
            history.append({"step": step + 1, "loss": loss_sum, "gradient_norm": float(gradient_norm)})
            if step + 1 in config.training.checkpoint_steps:
                torch.save(trainable_state_dict(model), output_dir / f"checkpoint-{step + 1}.pt")
            step_elapsed = max(time.perf_counter() - start, 1e-9)
            tokens_done = (
                (step + 1)
                * config.task.train_sequence_length
                * config.training.effective_batch_size
            )
            peak_gib = (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else 0.0
            )
            progress.set_postfix(
                loss=f"{loss_sum:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                tok_s=f"{tokens_done / step_elapsed:.0f}",
                vram=f"{peak_gib:.1f}G",
            )
    finally:
        progress.close()
        hash_file.close()

    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    processed_tokens = (
        config.task.train_sequence_length
        * config.training.effective_batch_size
        * config.training.optimizer_steps
    )
    summary = {
        "condition": str(condition),
        "seed": seed,
        "optimizer_steps": config.training.optimizer_steps,
        "processed_tokens": processed_tokens,
        "wall_seconds": elapsed,
        "tokens_per_second": processed_tokens / elapsed,
        "peak_allocated_gib": peak_allocated / 2**30,
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def assert_paired_hash_files(left: str | Path, right: str | Path) -> None:
    if Path(left).read_bytes() != Path(right).read_bytes():
        raise AssertionError("paired conditions did not receive identical token batches")
