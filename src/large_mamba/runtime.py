from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .adapter import (
    instrument_model,
    make_remaining_mixers_lora_compatible,
    set_projection_strength,
)
from .config import Condition, ExperimentConfig
from .data import TokenPools
from .modeling import (
    apply_lora,
    enable_activation_checkpointing,
    enable_recurrence_parameters,
    load_official_model,
    load_trainable_checkpoint,
)


def load_token_pools(config: ExperimentConfig) -> TokenPools:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_repository, revision=config.model.tokenizer_revision
    )
    return TokenPools.from_tokenizer(tokenizer)


def load_radii(path: Path, layers, condition: Condition) -> dict[int, float]:
    if condition not in (Condition.SPHERE, Condition.READ):
        return dict.fromkeys(layers, 1.0)
    return {
        int(key): float(value) for key, value in json.loads(path.read_text()).items()
    }


def prepare_model(
    model, config: ExperimentConfig, seed: int, *, activation_checkpointing: bool
):
    make_remaining_mixers_lora_compatible(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    apply_lora(
        model,
        rank=config.training.lora_rank,
        alpha=config.training.lora_alpha,
        dropout=config.training.lora_dropout,
    )
    enable_recurrence_parameters(model)
    if activation_checkpointing:
        enable_activation_checkpointing(model)


@dataclass
class LoadedRun:
    model: torch.nn.Module
    adapters: list
    condition: Condition
    config: ExperimentConfig
    device: torch.device
    runtime: dict
    snapshot: str
    checkpoint: Path


def load_run(
    run_dir: Path,
    checkpoint: Path | None = None,
    *,
    radii_path: Path | None = None,
    freeze: bool = False,
) -> LoadedRun:
    runtime = json.loads((run_dir / "runtime_config.json").read_text())
    config = ExperimentConfig()
    condition = Condition(runtime["condition"])
    layers = tuple(int(index) for index in runtime["layers"])
    checkpoint = checkpoint or run_dir / "checkpoint-1000.pt"
    device = torch.device("cuda", 0)
    model, snapshot = load_official_model(
        config.model.repository,
        revision=config.model.revision,
        dtype=torch.bfloat16,
        device=device,
    )
    radii_path = (
        radii_path
        or Path(config.output_dir) / "calibration" / f"radii-{runtime['scope']}.json"
    )
    adapters = instrument_model(
        model,
        layers,
        load_radii(radii_path, layers, condition),
        condition,
        epsilon=config.model.epsilon,
        scan_chunk_size=int(runtime["scan_chunk_size"]),
        compile_scan=bool(runtime["compile_projected_scan"]),
    )
    prepare_model(
        model,
        config,
        int(runtime["seed"]),
        activation_checkpointing=bool(runtime.get("activation_checkpointing", False)),
    )
    load_trainable_checkpoint(model, checkpoint)
    set_projection_strength(model, 1.0)
    if freeze:
        model.requires_grad_(False).eval()
    return LoadedRun(
        model, adapters, condition, config, device, runtime, snapshot, checkpoint
    )
