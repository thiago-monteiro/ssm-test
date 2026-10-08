from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import ExperimentConfig, LagBucket
from .data import (
    TokenPools,
    cell_is_valid,
    collate_examples,
    derived_example_seed,
    generate_recall_example,
)
from .metrics import answer_tokens

StressCell = tuple[int, int, LagBucket]


def stress_grid_cells(config: ExperimentConfig) -> tuple[StressCell, ...]:
    return tuple(
        (length, associations, lag)
        for length in config.task.sequence_lengths
        for associations in config.task.association_counts
        for lag in config.task.lag_buckets
        if cell_is_valid(length, associations, config.task.queries)
    )


def extended_stress_grid_cells(config: ExperimentConfig) -> tuple[StressCell, ...]:
    return tuple(
        (length, associations, lag)
        for length in (512, 1024, 2048)
        for associations in (64, 128, 256)
        for lag in config.task.lag_buckets
        if cell_is_valid(length, associations, config.task.queries)
    )


def evaluation_grid_cells(
    config: ExperimentConfig, grid: str
) -> tuple[StressCell, ...]:
    if grid == "standard":
        return stress_grid_cells(config)
    if grid == "extended":
        return extended_stress_grid_cells(config)
    raise ValueError("grid must be standard or extended")


def validation_examples_by_cell(config: ExperimentConfig) -> dict[StressCell, int]:
    cells = stress_grid_cells(config)
    base, remainder = divmod(config.task.validation_examples, len(cells))
    return {cell: base + (index < remainder) for index, cell in enumerate(cells)}


def _selected_cells(
    config: ExperimentConfig,
    *,
    grid: str,
    sequence_lengths: Iterable[int] | None,
    association_counts: Iterable[int] | None,
    lag_buckets: Iterable[LagBucket | str] | None,
) -> tuple[StressCell, ...]:
    available = evaluation_grid_cells(config, grid)
    lengths = set(sequence_lengths or (cell[0] for cell in available))
    loads = set(association_counts or (cell[1] for cell in available))
    lags = {LagBucket(item) for item in lag_buckets or config.task.lag_buckets}
    return tuple(
        cell
        for cell in available
        if cell[0] in lengths and cell[1] in loads and (cell[2] in lags)
    )


def _cell_counts(
    config: ExperimentConfig,
    cells: tuple[StressCell, ...],
    split: str,
    grid: str,
    examples_per_cell: int | None,
) -> dict[StressCell, int]:
    if examples_per_cell is not None:
        if examples_per_cell <= 0:
            raise ValueError("examples_per_cell must be positive")
        return {cell: examples_per_cell for cell in cells}
    if split == "test":
        count = (
            config.task.extended_test_examples_per_cell
            if grid == "extended"
            else config.task.test_examples_per_cell
        )
        return {cell: count for cell in cells}
    if grid == "extended":
        return {
            cell: config.task.extended_validation_examples_per_cell for cell in cells
        }
    validation_counts = validation_examples_by_cell(config)
    return {cell: validation_counts[cell] for cell in cells}


def _batch_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch_index in range(labels.shape[0]):
        selected, targets, predictions, margins = answer_tokens(
            logits[batch_index : batch_index + 1].float(),
            labels[batch_index : batch_index + 1],
        )
        correct = predictions.eq(targets)
        rows.append(
            {
                "exact_match": bool(correct.all()),
                "answer_token_accuracy": float(correct.float().mean()),
                "cross_entropy": float(F.cross_entropy(selected, targets)),
                "mean_margin": float(margins.mean()),
                "predictions": predictions.detach().cpu().tolist(),
                "targets": targets.detach().cpu().tolist(),
            }
        )
    return rows


def _mean_metric(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


@torch.inference_mode()
def evaluate_stress_grid(
    model: torch.nn.Module,
    pools: TokenPools,
    config: ExperimentConfig,
    *,
    split: str = "validation",
    grid: str = "standard",
    output_dir: str | Path,
    device: torch.device,
    max_batch_size: int = 8,
    max_batch_tokens: int = 2048,
    examples_per_cell: int | None = None,
    sequence_lengths: Iterable[int] | None = None,
    association_counts: Iterable[int] | None = None,
    lag_buckets: Iterable[LagBucket | str] | None = None,
) -> dict[str, object]:
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if max_batch_size <= 0 or max_batch_tokens <= 0:
        raise ValueError("evaluation batch limits must be positive")
    cells = _selected_cells(
        config,
        grid=grid,
        sequence_lengths=sequence_lengths,
        association_counts=association_counts,
        lag_buckets=lag_buckets,
    )
    if not cells:
        raise ValueError("the requested filters select no valid stress-grid cells")
    counts = _cell_counts(config, cells, split, grid, examples_per_cell)
    all_cells = evaluation_grid_cells(config, grid)
    canonical_index = {cell: index for index, cell in enumerate(all_cells)}
    split_seed_key = f"extended_{split}" if grid == "extended" else split
    split_seed = config.task.split_seeds[split_seed_key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    cell_rows: list[dict[str, object]] = []
    total_examples = sum(counts.values())
    with (
        tqdm(
            total=total_examples,
            desc=f"evaluate {grid} {split}",
            unit="example",
            dynamic_ncols=True,
        ) as progress,
        predictions_path.open("w") as prediction_file,
    ):
        model.eval()
        for length, associations, lag in cells:
            per_example: list[dict[str, object]] = []
            count = counts[length, associations, lag]
            batch_size = min(max_batch_size, max(1, max_batch_tokens // length))
            cell_offset = canonical_index[length, associations, lag] * 1000000
            for start in range(0, count, batch_size):
                examples = [
                    generate_recall_example(
                        pools,
                        seed=derived_example_seed(split_seed, cell_offset + index),
                        sequence_length=length,
                        associations=associations,
                        queries=config.task.queries,
                        lag_bucket=lag,
                    )
                    for index in range(start, min(start + batch_size, count))
                ]
                batch = collate_examples(examples)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(input_ids)
                    logits = output.logits if hasattr(output, "logits") else output
                metric_rows = _batch_metrics(logits, labels)
                for example, metrics in zip(examples, metric_rows, strict=True):
                    row = {
                        "split": split,
                        "grid": grid,
                        "sequence_length": length,
                        "associations": associations,
                        "lag_bucket": str(lag),
                        "seed": example.metadata.seed,
                        "exact_lags": list(example.metadata.exact_lags),
                        "token_sha256": example.manifest()["token_sha256"],
                        **metrics,
                    }
                    prediction_file.write(json.dumps(row, sort_keys=True) + "\n")
                    per_example.append(row)
                progress.update(len(examples))
            median_lag = statistics.median(
                lag_value for row in per_example for lag_value in row["exact_lags"]
            )
            cell_rows.append(
                {
                    "sequence_length": length,
                    "associations": associations,
                    "lag_bucket": str(lag),
                    "examples": count,
                    **{
                        key: _mean_metric(per_example, key)
                        for key in (
                            "exact_match",
                            "answer_token_accuracy",
                            "cross_entropy",
                            "mean_margin",
                        )
                    },
                    "median_exact_lag": median_lag,
                    "easy": length <= 256
                    and associations <= 16
                    and (lag in (LagBucket.NEAR, LagBucket.MIDDLE)),
                }
            )
    fieldnames = list(cell_rows[0])
    with (output_dir / "stress_grid.csv").open("w", newline="") as cell_file:
        writer = csv.DictWriter(cell_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cell_rows)
    load80: dict[str, int | None] = {}
    for length in sorted({int(row["sequence_length"]) for row in cell_rows}):
        passing = []
        for associations in sorted({int(row["associations"]) for row in cell_rows}):
            relevant = [
                row
                for row in cell_rows
                if row["sequence_length"] == length
                and row["associations"] == associations
            ]
            if relevant and _mean_metric(relevant, "exact_match") >= 0.8:
                passing.append(associations)
        load80[str(length)] = max(passing) if passing else None
    lag80: dict[str, dict[str, object] | None] = {}
    for length, associations in sorted(
        {(int(row["sequence_length"]), int(row["associations"])) for row in cell_rows}
    ):
        passing = [
            row
            for row in cell_rows
            if row["sequence_length"] == length
            and row["associations"] == associations
            and (float(row["exact_match"]) >= 0.8)
        ]
        key = f"length={length},associations={associations}"
        if passing:
            best = max(passing, key=lambda row: float(row["median_exact_lag"]))
            lag80[key] = {
                "bucket": best["lag_bucket"],
                "median_exact_lag": best["median_exact_lag"],
            }
        else:
            lag80[key] = None
    easy_rows = [row for row in cell_rows if row["easy"]]
    train_cell_rows = [
        row
        for row in cell_rows
        if row["sequence_length"] == config.task.train_sequence_length
        and row["associations"] == config.task.train_associations
    ]
    summary: dict[str, object] = {
        "split": split,
        "grid": grid,
        "examples": total_examples,
        "cells": len(cell_rows),
        "stress_auc": _mean_metric(cell_rows, "exact_match"),
        **{
            key: _mean_metric(cell_rows, key)
            for key in ("answer_token_accuracy", "cross_entropy", "mean_margin")
        },
        "easy_stress_auc": _mean_metric(easy_rows, "exact_match"),
        "training_distribution_accuracy": _mean_metric(train_cell_rows, "exact_match"),
        "load80": load80,
        "lag80": lag80,
        "artifacts": {
            "predictions": str(predictions_path),
            "stress_grid": str(output_dir / "stress_grid.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
