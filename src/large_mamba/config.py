from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Condition(StrEnum):
    ORD = "ord"
    SPHERE = "sphere"
    READ = "read"
    WEIGHT = "weight"


class LagBucket(StrEnum):
    NEAR = "near"
    MIDDLE = "middle"
    FAR = "far"


@dataclass(frozen=True)
class ModelConfig:
    repository: str = "state-spaces/mamba-2.8b"
    revision: str = "e886be8192cbb383b01559a3877dfd5e6bfb3e55"
    tokenizer_repository: str = "EleutherAI/gpt-neox-20b"
    tokenizer_revision: str = "c292233c833e336628618a88a648727eb3dff0a7"
    mamba_revision: str = "95d8aba8a8c75aedcaa6143713b11e745e7cd0d9"
    preferred_scope: str = "all"
    fallback_layers: tuple[int, ...] = (7, 15, 23, 31, 39, 47, 55, 63)
    epsilon: float = 1e-6


@dataclass(frozen=True)
class TaskConfig:
    train_sequence_length: int = 256
    train_associations: int = 16
    queries: int = 4
    calibration_examples: int = 1024
    validation_examples: int = 2048
    test_examples_per_cell: int = 256
    extended_validation_examples_per_cell: int = 128
    extended_test_examples_per_cell: int = 256
    sequence_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    association_counts: tuple[int, ...] = (4, 16, 64)
    lag_buckets: tuple[LagBucket, ...] = tuple(LagBucket)
    split_seeds: dict[str, int] = field(
        default_factory=lambda: {
            "calibration": 10_001,
            "train": 20_003,
            "validation": 30_007,
            "test": 40_009,
            "preflight": 50_021,
            "extended_validation": 60_013,
            "extended_test": 70_001,
        }
    )


@dataclass(frozen=True)
class TrainingConfig:
    optimizer_steps: int = 1000
    effective_batch_size: int = 8
    micro_batch_size: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_learning_rate: float = 2e-4
    recurrence_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.05
    projection_ramp_fraction: float = 0.10
    checkpoint_steps: tuple[int, ...] = (0, 100, 300, 1000)

    @property
    def gradient_accumulation_steps(self) -> int:
        return len(self.micro_batch_sizes)

    @property
    def micro_batch_sizes(self) -> tuple[int, ...]:
        full_batches, remainder = divmod(self.effective_batch_size, self.micro_batch_size)
        sizes = (self.micro_batch_size,) * full_batches
        return sizes + ((remainder,) if remainder else ())


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "results/large_mamba"
    seeds: tuple[int, ...] = (0, 1, 2)

    def validate(self) -> None:
        if not 0 < self.training.micro_batch_size <= self.training.effective_batch_size:
            raise ValueError("micro_batch_size must be in [1, effective_batch_size]")
        expected_tokens = 256 * 8 * 1000
        actual_tokens = (
            self.task.train_sequence_length
            * self.training.effective_batch_size
            * self.training.optimizer_steps
        )
        if actual_tokens != expected_tokens:
            raise ValueError(f"processed-token budget changed: {actual_tokens} != {expected_tokens}")
        split_seeds = list(self.task.split_seeds.values())
        if len(split_seeds) != len(set(split_seeds)):
            raise ValueError("data split seeds must be disjoint")
        if self.model.epsilon != 1e-6:
            raise ValueError("the preregistered recurrence epsilon is 1e-6")
        if set(self.seeds) != {0, 1, 2}:
            raise ValueError("the confirmatory continuation seeds must be 0, 1, and 2")

    @property
    def processed_tokens(self) -> int:
        return (
            self.task.train_sequence_length
            * self.training.effective_batch_size
            * self.training.optimizer_steps
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def write_resolved(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, StrEnum):
        return str(value)
    return value


def choose_layer_scope(
    n_layers: int,
    seconds_per_step: float,
    peak_allocated_gib: float,
    fallback_layers: tuple[int, ...] = ModelConfig().fallback_layers,
) -> tuple[int, ...]:
    pass
    projected_hours = seconds_per_step * 1000 / 3600
    if projected_hours <= 12.0 and peak_allocated_gib <= 22.5:
        return tuple(range(n_layers))
    if n_layers == 64:
        return fallback_layers
    
    count = min(8, n_layers)
    return tuple(round((i + 1) * n_layers / count) - 1 for i in range(count))


def projection_strength(step: int, total_steps: int, ramp_fraction: float = 0.10) -> float:
    ramp_steps = max(1, round(total_steps * ramp_fraction))
    return min(1.0, max(0.0, step / ramp_steps))
