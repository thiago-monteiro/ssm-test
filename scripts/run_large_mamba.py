#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.large_mamba.adapter import (
    instrument_model,
    make_remaining_mixers_lora_compatible,
    set_projection_strength,
)
from src.large_mamba.calibration import calibrate_radii
from src.large_mamba.config import Condition, ExperimentConfig, LagBucket
from src.large_mamba.data import (
    RecallBatchFactory,
    TokenPools,
    derived_example_seed,
    generate_recall_example,
)
from src.large_mamba.evaluation import evaluate_stress_grid
from src.large_mamba.evaluation import extended_stress_grid_cells
from src.large_mamba.faithfulness import (
    answer_score,
    install_state_override,
    random_directions,
    summarize_faithfulness_rows,
    write_faithfulness_artifacts,
)
from src.large_mamba.mechanisms import perturb_state
from src.large_mamba.manifest import write_environment_manifest
from src.large_mamba.modeling import (
    apply_lora,
    enable_activation_checkpointing,
    enable_recurrence_parameters,
    load_official_model,
    load_trainable_checkpoint,
    parameter_manifest,
)
from src.large_mamba.training import run_training
from src.large_mamba.validation import validate_norm_invariant


def validate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    config.write_resolved(output / "resolved_config.json")
    write_environment_manifest(output / "environment_manifest.json", _git_commit())
    norm_error = validate_norm_invariant()
    pools = TokenPools(
        keys=tuple(range(100, 200)), values=tuple(range(300, 400)),
        distractors=tuple(range(500, 600)), association_marker=2, query_marker=3,
    )
    cells = 0
    for length in config.task.sequence_lengths:
        for associations in config.task.association_counts:
            for lag in LagBucket:
                try:
                    generate_recall_example(
                        pools, seed=123 + cells, sequence_length=length,
                        associations=associations, lag_bucket=lag,
                    )
                except ValueError as error:
                    if "cannot fit syntactically" not in str(error):
                        raise
                else:
                    cells += 1
    report = {"status": "PASS", "norm_relative_error": norm_error, "valid_stress_cells": cells}
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _selected_layers(config: ExperimentConfig, scope: str) -> tuple[int, ...]:
    if scope == "all":
        return tuple(range(64))
    return config.model.fallback_layers


def _calibration_batches(
    pools: TokenPools,
    config: ExperimentConfig,
    device: torch.device,
    batch_size: int,
):
    seed = config.task.split_seeds["calibration"]
    starts = range(0, config.task.calibration_examples, batch_size)
    for start in tqdm(starts, desc="calibrating radii", unit="batch", dynamic_ncols=True):
        rows = []
        for index in range(start, min(start + batch_size, config.task.calibration_examples)):
            example = generate_recall_example(
                pools,
                seed=derived_example_seed(seed, index),
                sequence_length=config.task.train_sequence_length,
                associations=config.task.train_associations,
                queries=config.task.queries,
                lag_bucket=tuple(LagBucket)[index % len(LagBucket)],
            )
            rows.append(example.input_ids)
        yield torch.stack(rows).to(device)


def train(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    config = replace(
        config,
        training=replace(config.training, micro_batch_size=args.micro_batch_size),
    )
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "training state-spaces/mamba-2.8b requires a CUDA-enabled PyTorch install; "
            "install requirements.txt in the ssm-test-plan environment"
        )
    if args.scan_chunk_size <= 0:
        raise ValueError("--scan-chunk-size must be positive")
    if args.calibration_batch_size <= 0:
        raise ValueError("--calibration-batch-size must be positive")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    condition = Condition(args.condition)
    layers = _selected_layers(config, args.scope)
    output = args.output or Path(config.output_dir) / f"seed{args.seed}" / str(condition)
    output.mkdir(parents=True, exist_ok=True)
    config.write_resolved(output / "resolved_config.json")
    write_environment_manifest(output / "environment_manifest.json", _git_commit())
    runtime_config = {
        "condition": str(condition),
        "seed": args.seed,
        "scope": args.scope,
        "layers": list(layers),
        "scan_chunk_size": args.scan_chunk_size,
        "compile_projected_scan": args.compile_projected_scan,
        "activation_checkpointing": args.activation_checkpointing,
        "calibration_batch_size": args.calibration_batch_size,
        "effective_batch_size": config.training.effective_batch_size,
        "micro_batch_sizes": list(config.training.micro_batch_sizes),
        "device": str(device),
    }
    (output / "runtime_config.json").write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True) + "\n"
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install requirements.txt before training") from error
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_repository,
        revision=config.model.tokenizer_revision,
    )
    pools = TokenPools.from_tokenizer(tokenizer)
    model, snapshot = load_official_model(
        config.model.repository,
        revision=config.model.revision,
        dtype=torch.bfloat16,
        device=device,
    )
    (output / "checkpoint_snapshot.txt").write_text(snapshot + "\n")

    radii_path = args.radii or Path(config.output_dir) / "calibration" / f"radii-{args.scope}.json"
    if condition in (Condition.SPHERE, Condition.READ) and radii_path.exists():
        radii = {int(key): float(value) for key, value in json.loads(radii_path.read_text()).items()}
        adapters = instrument_model(
            model, layers, radii, condition, epsilon=config.model.epsilon,
            scan_chunk_size=args.scan_chunk_size, compile_scan=args.compile_projected_scan,
        )
    else:
        adapters = instrument_model(
            model, layers, {index: 1.0 for index in layers}, Condition.ORD,
            epsilon=config.model.epsilon,
            scan_chunk_size=args.scan_chunk_size, compile_scan=args.compile_projected_scan,
        )
        if condition in (Condition.SPHERE, Condition.READ):
            print(f"Calibrating radii from {config.task.calibration_examples} examples...")
            calibration_batch_size = 1 if args.scope == "all" else args.calibration_batch_size
            radii = calibrate_radii(
                model,
                _calibration_batches(pools, config, device, calibration_batch_size),
                adapters,
                max_examples=config.task.calibration_examples,
            )
            radii_path.parent.mkdir(parents=True, exist_ok=True)
            radii_path.write_text(json.dumps(radii, indent=2, sort_keys=True) + "\n")
            for adapter in adapters:
                adapter.set_condition(condition)
                adapter.set_radius(radii[int(adapter.layer_idx)])

    make_remaining_mixers_lora_compatible(model)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = apply_lora(
        model,
        rank=config.training.lora_rank,
        alpha=config.training.lora_alpha,
        dropout=config.training.lora_dropout,
    )
    enable_recurrence_parameters(model)
    if args.activation_checkpointing:
        enable_activation_checkpointing(model)
    manifest = parameter_manifest(model)
    manifest.write(output / "trainable_parameters.json")
    factory = RecallBatchFactory(
        pools,
        stream_seed=derived_example_seed(config.task.split_seeds["train"], args.seed),
        sequence_length=config.task.train_sequence_length,
        associations=config.task.train_associations,
        queries=config.task.queries,
        micro_batch_sizes=config.training.micro_batch_sizes,
    )
    summary = run_training(
        model,
        condition=condition,
        seed=args.seed,
        config=config,
        batch_factory=factory,
        output_dir=output,
        device=device,
    )
    print(json.dumps({key: summary[key] for key in (
        "condition", "seed", "optimizer_steps", "processed_tokens",
        "wall_seconds", "tokens_per_second",
        "peak_allocated_gib",
    )}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation of the 2.8B checkpoint requires CUDA")
    run_dir = args.run_dir
    runtime_path = run_dir / "runtime_config.json"
    if not runtime_path.exists():
        raise FileNotFoundError(f"missing training runtime config: {runtime_path}")
    runtime = json.loads(runtime_path.read_text())
    config = ExperimentConfig()
    config.validate()
    condition = Condition(runtime["condition"])
    seed = int(runtime["seed"])
    layers = tuple(int(index) for index in runtime["layers"])
    scope = str(runtime["scope"])
    checkpoint = args.checkpoint or run_dir / "checkpoint-1000.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing trainable checkpoint: {checkpoint}")
    evaluation_name = args.split if args.grid == "standard" else f"extended-{args.split}"
    output = args.output or run_dir / "evaluation" / evaluation_name / checkpoint.stem
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install requirements.txt before evaluation") from error
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_repository,
        revision=config.model.tokenizer_revision,
    )
    pools = TokenPools.from_tokenizer(tokenizer)
    model, snapshot = load_official_model(
        config.model.repository,
        revision=config.model.revision,
        dtype=torch.bfloat16,
        device=device,
    )
    radii_path = args.radii or Path(config.output_dir) / "calibration" / f"radii-{scope}.json"
    if condition in (Condition.SPHERE, Condition.READ):
        if not radii_path.exists():
            raise FileNotFoundError(f"missing calibrated radii: {radii_path}")
        radii = {
            int(key): float(value)
            for key, value in json.loads(radii_path.read_text()).items()
        }
    else:
        radii = {index: 1.0 for index in layers}
    instrument_model(
        model,
        layers,
        radii,
        condition,
        epsilon=config.model.epsilon,
        scan_chunk_size=int(runtime["scan_chunk_size"]),
        compile_scan=bool(runtime["compile_projected_scan"]),
    )
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
    
    if bool(runtime.get("activation_checkpointing", False)):
        enable_activation_checkpointing(model)
    load_trainable_checkpoint(model, checkpoint)
    set_projection_strength(model, 1.0)

    summary = evaluate_stress_grid(
        model,
        pools,
        config,
        split=args.split,
        grid=args.grid,
        output_dir=output,
        device=device,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        examples_per_cell=args.examples_per_cell,
        sequence_lengths=args.sequence_lengths,
        association_counts=args.association_counts,
        lag_buckets=args.lag_buckets,
    )
    evaluation_config = {
        "condition": str(condition),
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_snapshot": snapshot,
        "split": args.split,
        "grid": args.grid,
        "scope": scope,
        "layers": list(layers),
        "max_batch_size": args.max_batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "examples_per_cell_override": args.examples_per_cell,
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: summary[key] for key in (
        "split", "grid", "examples", "cells", "stress_auc", "answer_token_accuracy",
        "cross_entropy", "mean_margin", "easy_stress_auc",
        "training_distribution_accuracy",
    )}, indent=2))


def _load_run_model(run_dir: Path, checkpoint: Path | None = None):
    runtime = json.loads((run_dir / "runtime_config.json").read_text())
    config = ExperimentConfig()
    condition = Condition(runtime["condition"])
    layers = tuple(int(index) for index in runtime["layers"])
    checkpoint = checkpoint or run_dir / "checkpoint-1000.pt"
    device = torch.device("cuda", 0)
    model, _ = load_official_model(
        config.model.repository,
        revision=config.model.revision,
        dtype=torch.bfloat16,
        device=device,
    )
    radii_path = Path(config.output_dir) / "calibration" / f"radii-{runtime['scope']}.json"
    radii = (
        {int(key): float(value) for key, value in json.loads(radii_path.read_text()).items()}
        if condition in (Condition.SPHERE, Condition.READ)
        else {index: 1.0 for index in layers}
    )
    adapters = instrument_model(
        model,
        layers,
        radii,
        condition,
        epsilon=config.model.epsilon,
        scan_chunk_size=int(runtime["scan_chunk_size"]),
        
        
        compile_scan=bool(runtime["compile_projected_scan"]),
    )
    make_remaining_mixers_lora_compatible(model)
    torch.manual_seed(int(runtime["seed"]))
    torch.cuda.manual_seed_all(int(runtime["seed"]))
    apply_lora(
        model,
        rank=config.training.lora_rank,
        alpha=config.training.lora_alpha,
        dropout=config.training.lora_dropout,
    )
    enable_recurrence_parameters(model)
    if bool(runtime.get("activation_checkpointing", False)):
        enable_activation_checkpointing(model)
    load_trainable_checkpoint(model, checkpoint)
    set_projection_strength(model, 1.0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, adapters, condition, config, device


def _faithfulness_one_run(args: argparse.Namespace, run_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    model, adapters, condition, config, device = _load_run_model(run_dir)
    adapter_by_layer = {int(adapter.layer_idx): adapter for adapter in adapters}
    layer = args.layer if args.layer is not None else max(adapter_by_layer)
    if layer not in adapter_by_layer:
        raise ValueError(f"layer {layer} is not instrumented; choose from {sorted(adapter_by_layer)}")
    adapter = adapter_by_layer[layer]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_repository,
        revision=config.model.tokenizer_revision,
    )
    pools = TokenPools.from_tokenizer(tokenizer)
    cells = extended_stress_grid_cells(config)
    cell = (args.sequence_length, args.associations, LagBucket(args.lag_bucket))
    if cell not in cells:
        raise ValueError(f"requested cell {cell} is not in the extended grid")
    cell_offset = cells.index(cell) * 1_000_000
    split_seed = config.task.split_seeds["extended_test"]
    rows: list[dict[str, object]] = []
    progress = tqdm(range(args.examples), desc=f"faithfulness {condition}", unit="example", dynamic_ncols=True)

    for example_index in progress:
        seed = derived_example_seed(split_seed, cell_offset + example_index)
        example = generate_recall_example(
            pools,
            seed=seed,
            sequence_length=args.sequence_length,
            associations=args.associations,
            queries=config.task.queries,
            lag_bucket=args.lag_bucket,
        )
        input_ids = example.input_ids.unsqueeze(0).to(device)
        association_position = example.metadata.association_value_positions[0]
        answer_position = example.metadata.answer_positions[0]
        prediction_position = answer_position - 1
        intervention_step = round((association_position + prediction_position) / 2)
        target = int(example.input_ids[answer_position])

        adapter.capture_steps = {intervention_step}
        adapter.state_transform = None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            clean_hidden = model.backbone(input_ids)
        
        
        clean_row = clean_hidden[0, prediction_position].float()
        clean_logits = F.linear(
            clean_row,
            model.lm_head.weight.float(),
            model.lm_head.bias.float() if model.lm_head.bias is not None else None,
        ).view(1, 1, -1)
        clean_score_tensor, alternative = answer_score(
            clean_logits,
            prediction_position=0,
            target=target,
        )
        clean_score = float(clean_score_tensor)
        clean_prediction = int(clean_logits[0, 0].argmax())
        state = adapter.captured_states[intervention_step]
        adapter.capture_steps.clear()

        directions = random_directions(
            state[0], args.directions, seed=derived_example_seed(args.direction_seed, example_index)
        )
        perturbed, displacements = perturb_state(
            state[0], directions, args.theta, spherical=condition is Condition.SPHERE
        )

        leaf = state.detach().clone().requires_grad_(True)
        install_state_override(adapter, step=intervention_step, replacement=leaf)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gradient_hidden = model.backbone(input_ids)[0, prediction_position]
        score_weight = (
            model.lm_head.weight[target].float() - model.lm_head.weight[alternative].float()
        )
        gradient_score = gradient_hidden.float() @ score_weight
        if model.lm_head.bias is not None:
            gradient_score = gradient_score + (
                model.lm_head.bias[target].float() - model.lm_head.bias[alternative].float()
            )
        gradient = torch.autograd.grad(gradient_score, leaf)[0].detach()[0]
        predicted_effects = (displacements.float() * gradient.float()).flatten(start_dim=1).sum(dim=1)

        for direction_index in range(args.directions):
            replacement = perturbed[direction_index : direction_index + 1]
            install_state_override(adapter, step=intervention_step, replacement=replacement)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                changed_hidden = model.backbone(input_ids)[0, prediction_position]
            changed_score = changed_hidden.float() @ score_weight
            if model.lm_head.bias is not None:
                changed_score = changed_score + (
                    model.lm_head.bias[target].float() - model.lm_head.bias[alternative].float()
                )
            rows.append({
                "condition": str(condition),
                "example_index": example_index,
                "example_seed": seed,
                "direction_index": direction_index,
                "theta": args.theta,
                "layer": layer,
                "association_position": association_position,
                "intervention_step": intervention_step,
                "prediction_position": prediction_position,
                "exact_lag": example.metadata.exact_lags[0],
                "target": target,
                "alternative": alternative,
                "clean_correct": clean_prediction == target,
                "clean_score": clean_score,
                "state_norm": float(state.float().norm()),
                "predicted_effect": float(predicted_effects[direction_index]),
                "actual_effect": float(changed_score) - clean_score,
            })
        adapter.state_transform = None

    summary = summarize_faithfulness_rows(rows)
    summary.update({
        "condition": str(condition),
        "layer": layer,
        "theta": args.theta,
        "sequence_length": args.sequence_length,
        "associations": args.associations,
        "lag_bucket": args.lag_bucket,
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows, summary


def faithfulness(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("large-model faithfulness evaluation requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[list[dict[str, object]], dict[str, object]]] = {}
    for label, run_dir in (("ord", args.ord_run_dir), ("sphere", args.sphere_run_dir)):
        rows, summary = _faithfulness_one_run(args, run_dir)
        write_faithfulness_artifacts(args.output / label, rows, summary)
        results[label] = rows, summary

    common_correct = {
        index for index in range(args.examples)
        if all(any(int(row["example_index"]) == index and bool(row["clean_correct"])
                   for row in results[label][0]) for label in ("ord", "sphere"))
    }
    combined: dict[str, object] = {
        "design": {
            "examples": args.examples,
            "directions_per_example": args.directions,
            "theta": args.theta,
            "layer": results["ord"][1]["layer"],
            "intervention": "midpoint from first queried association value to answer prediction",
        },
        "all_examples": {label: result[1] for label, result in results.items()},
        "both_correct_examples": len(common_correct),
    }
    if common_correct:
        combined["both_correct"] = {
            label: summarize_faithfulness_rows([
                row for row in result[0] if int(row["example_index"]) in common_correct
            ])
            for label, result in results.items()
        }
    (args.output / "comparison.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(combined, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-Mamba spherical recurrence experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="run import-safe implementation preflight")
    validate_parser.add_argument("--output", type=Path, default=ROOT / "results/large_mamba/preflight")
    train_parser = subparsers.add_parser("train", help="run one continuation-training condition")
    train_parser.add_argument("--condition", required=True, choices=[
        str(Condition.ORD), str(Condition.SPHERE), str(Condition.READ)
    ])
    train_parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    train_parser.add_argument(
        "--scope", choices=("fallback", "all"), default="fallback",
        help="use the fixed eight-layer fallback or every SSM layer",
    )
    train_parser.add_argument("--radii", type=Path, help="reuse a radius calibration JSON file")
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument("--scan-chunk-size", type=int, default=4)
    train_parser.add_argument("--calibration-batch-size", type=int, default=2)
    train_parser.add_argument(
        "--micro-batch-size", type=int, default=8,
        help="largest physical batch; the final accumulation batch may be smaller",
    )
    train_parser.add_argument(
        "--compile-projected-scan", action=argparse.BooleanOptionalAction, default=True,
        help="compile static recurrence chunks with TorchInductor",
    )
    train_parser.add_argument(
        "--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False,
        help="recompute Mamba blocks during backward to reduce activation memory",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a trained checkpoint on fixed held-out recall examples"
    )
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--radii", type=Path)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate_parser.add_argument(
        "--grid", choices=("standard", "extended"), default="standard",
        help="use the original grid or the frozen harder post-saturation grid",
    )
    evaluate_parser.add_argument("--output", type=Path)
    evaluate_parser.add_argument("--max-batch-size", type=int, default=8)
    evaluate_parser.add_argument("--max-batch-tokens", type=int, default=2048)
    evaluate_parser.add_argument(
        "--examples-per-cell", type=int,
        help="override the fixed count for smoke runs only",
    )
    evaluate_parser.add_argument("--sequence-lengths", nargs="+", type=int)
    evaluate_parser.add_argument("--association-counts", nargs="+", type=int)
    evaluate_parser.add_argument(
        "--lag-buckets", nargs="+", choices=tuple(str(item) for item in LagBucket)
    )
    faith_parser = subparsers.add_parser(
        "faithfulness", help="run the bounded large-model mid-state faithfulness spot-check"
    )
    faith_parser.add_argument("--ord-run-dir", type=Path, required=True)
    faith_parser.add_argument("--sphere-run-dir", type=Path, required=True)
    faith_parser.add_argument(
        "--output", type=Path, default=ROOT / "results/large_mamba/faithfulness-spot-check"
    )
    faith_parser.add_argument("--examples", type=int, default=32)
    faith_parser.add_argument("--directions", type=int, default=4)
    faith_parser.add_argument("--theta", type=float, default=0.1)
    faith_parser.add_argument("--layer", type=int)
    faith_parser.add_argument("--sequence-length", type=int, default=1024)
    faith_parser.add_argument("--associations", type=int, default=128)
    faith_parser.add_argument(
        "--lag-bucket", choices=tuple(str(item) for item in LagBucket), default="middle"
    )
    faith_parser.add_argument("--direction-seed", type=int, default=271828)
    args = parser.parse_args()
    if args.command == "validate":
        validate(args.output)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "faithfulness":
        faithfulness(args)


if __name__ == "__main__":
    main()
