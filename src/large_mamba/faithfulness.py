from __future__ import annotations

import json
from pathlib import Path

import torch

from .adapter import ProjectedMambaMixer
from .mechanisms import faithfulness_metrics


def answer_score(
    logits: torch.Tensor,
    *,
    prediction_position: int,
    target: int,
    alternative: int | None = None,
) -> tuple[torch.Tensor, int]:
    pass
    row = logits[0, prediction_position].float()
    if alternative is None:
        competitors = row.detach().clone()
        competitors[target] = -torch.inf
        alternative = int(competitors.argmax())
    return row[target] - row[alternative], alternative


def random_directions(
    state: torch.Tensor,
    count: int,
    *,
    seed: int,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("direction count must be positive")
    generator = torch.Generator(device=state.device).manual_seed(seed)
    return torch.randn(
        (count,) + tuple(state.shape),
        generator=generator,
        device=state.device,
        dtype=state.dtype,
    )


def summarize_faithfulness_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("faithfulness summary needs at least one perturbation")
    examples = sorted({int(row["example_index"]) for row in rows})
    directions = sorted({int(row["direction_index"]) for row in rows})
    predicted = torch.tensor(
        [[next(float(row["predicted_effect"]) for row in rows
               if int(row["example_index"]) == example
               and int(row["direction_index"]) == direction)
          for direction in directions] for example in examples]
    )
    actual = torch.tensor(
        [[next(float(row["actual_effect"]) for row in rows
               if int(row["example_index"]) == example
               and int(row["direction_index"]) == direction)
          for direction in directions] for example in examples]
    )
    metrics = faithfulness_metrics(predicted, actual)
    return {
        "examples": len(examples),
        "directions_per_example": len(directions),
        "perturbations": len(rows),
        "e_all": metrics.e_all,
        "e_eff": metrics.e_eff,
        "rank_correlation": metrics.rank_correlation,
        "sign_accuracy": metrics.sign_accuracy,
        "false_positive_rate": metrics.false_positive_rate,
        "false_negative_rate": metrics.false_negative_rate,
        "clean_first_answer_accuracy": sum(bool(row["clean_correct"]) for row in rows[::len(directions)]) / len(examples),
        "mean_state_norm": sum(float(row["state_norm"]) for row in rows[::len(directions)]) / len(examples),
    }


def write_faithfulness_artifacts(
    output_dir: str | Path,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "perturbations.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def install_state_override(
    adapter: ProjectedMambaMixer,
    *,
    step: int,
    replacement: torch.Tensor,
) -> None:
    pass
    def transform(current_step: int, state: torch.Tensor) -> torch.Tensor:
        if current_step != step:
            return state
        if state.shape != replacement.shape:
            raise ValueError("replacement state shape does not match scan state")
        return replacement

    adapter.state_transform = transform
