from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Protocol

import numpy as np
import torch

from .config import LagBucket


class TokenizerLike(Protocol):
    def get_vocab(self) -> dict[str, int]: ...
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class TokenPools:
    keys: tuple[int, ...]
    values: tuple[int, ...]
    distractors: tuple[int, ...]
    association_marker: int
    query_marker: int

    def validate(self, associations: int, queries: int) -> None:
        if len(self.keys) < associations or len(self.values) < associations:
            raise ValueError("token pool is too small for unique key/value assignments")
        if queries > associations:
            raise ValueError("queries cannot exceed associations")
        groups = [set(self.keys), set(self.values), set(self.distractors)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("key, value, and distractor pools must be disjoint")
        if self.association_marker == self.query_marker:
            raise ValueError("association and query markers must differ")

    @classmethod
    def from_tokenizer(cls, tokenizer: TokenizerLike, pool_size: int = 512) -> "TokenPools":
        stable: list[int] = []
        for token_id in sorted(set(tokenizer.get_vocab().values())):
            text = tokenizer.decode([token_id])
            if text and tokenizer.encode(text, add_special_tokens=False) == [token_id]:
                stable.append(token_id)
            if len(stable) >= pool_size * 3 + 2:
                break
        if len(stable) < pool_size * 3 + 2:
            raise ValueError("tokenizer does not expose enough reversible single-token IDs")
        return cls(
            keys=tuple(stable[:pool_size]),
            values=tuple(stable[pool_size : 2 * pool_size]),
            distractors=tuple(stable[2 * pool_size : 3 * pool_size]),
            association_marker=stable[-2],
            query_marker=stable[-1],
        )


@dataclass(frozen=True)
class ExampleMetadata:
    seed: int
    sequence_length: int
    associations: int
    queries: int
    lag_bucket: str
    association_value_positions: tuple[int, ...]
    answer_positions: tuple[int, ...]
    exact_lags: tuple[int, ...]


@dataclass
class RecallExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    metadata: ExampleMetadata

    def manifest(self) -> dict[str, object]:
        result = asdict(self.metadata)
        result["token_sha256"] = hashlib.sha256(
            self.input_ids.detach().cpu().numpy().astype("<i8", copy=False).tobytes()
        ).hexdigest()
        return result


LAG_FRACTIONS: dict[LagBucket, tuple[float, float]] = {
    LagBucket.NEAR: (0.05, 0.25),
    LagBucket.MIDDLE: (0.35, 0.55),
    LagBucket.FAR: (0.70, 0.90),
}


def lag_bounds(sequence_length: int, bucket: LagBucket | str) -> tuple[int, int]:
    low, high = LAG_FRACTIONS[LagBucket(bucket)]
    return max(1, round(low * sequence_length)), max(1, round(high * sequence_length))


def cell_is_valid(sequence_length: int, associations: int, queries: int = 4) -> bool:
    return associations >= queries and 3 * (associations + queries) <= sequence_length


def generate_recall_example(
    pools: TokenPools,
    *,
    seed: int,
    sequence_length: int,
    associations: int,
    queries: int = 4,
    lag_bucket: LagBucket | str = LagBucket.MIDDLE,
) -> RecallExample:
    lag_bucket = LagBucket(lag_bucket)
    pools.validate(associations, queries)
    if not cell_is_valid(sequence_length, associations, queries):
        raise ValueError("stress-grid cell cannot fit syntactically")
    rng = np.random.default_rng(seed)
    keys = rng.choice(pools.keys, size=associations, replace=False).tolist()
    values = rng.choice(pools.values, size=associations, replace=False).tolist()
    order = rng.permutation(associations).tolist()
    keys, values = [keys[i] for i in order], [values[i] for i in order]

    query_start = sequence_length - 3 * queries
    slots = list(range(0, query_start - 2, 3))
    low_lag, high_lag = lag_bounds(sequence_length, lag_bucket)
    answer_positions = [query_start + 3 * i + 2 for i in range(queries)]
    chosen_slots = _match_query_slots(answer_positions, slots, low_lag, high_lag, rng)
    if chosen_slots is None:
        raise ValueError(f"no placement realizes {lag_bucket} lags for this cell")

    free_slots = [slot for slot in slots if slot not in chosen_slots]
    if len(free_slots) < associations - queries:
        raise ValueError("not enough non-overlapping association slots")
    other_slots = rng.choice(free_slots, size=associations - queries, replace=False).tolist()
    association_slots = chosen_slots + other_slots

    tokens = rng.choice(pools.distractors, size=sequence_length, replace=True).astype(np.int64)
    for idx, slot in enumerate(association_slots):
        tokens[slot : slot + 3] = (pools.association_marker, keys[idx], values[idx])
    for idx, start in enumerate(range(query_start, sequence_length, 3)):
        tokens[start : start + 3] = (pools.query_marker, keys[idx], values[idx])

    labels = np.full(sequence_length, -100, dtype=np.int64)
    labels[answer_positions] = np.asarray(values[:queries], dtype=np.int64)
    value_positions = [slot + 2 for slot in chosen_slots]
    exact_lags = [answer - value for answer, value in zip(answer_positions, value_positions)]
    metadata = ExampleMetadata(
        seed=seed,
        sequence_length=sequence_length,
        associations=associations,
        queries=queries,
        lag_bucket=str(lag_bucket),
        association_value_positions=tuple(value_positions),
        answer_positions=tuple(answer_positions),
        exact_lags=tuple(exact_lags),
    )
    return RecallExample(torch.from_numpy(tokens), torch.from_numpy(labels), metadata)


def _match_query_slots(
    answers: list[int], slots: list[int], low: int, high: int, rng: np.random.Generator
) -> list[int] | None:
    candidates = []
    for answer in answers:
        valid = [slot for slot in slots if low <= answer - (slot + 2) <= high]
        rng.shuffle(valid)
        candidates.append(valid)

    def search(index: int, used: set[int], result: list[int]) -> list[int] | None:
        if index == len(answers):
            return result.copy()
        for slot in candidates[index]:
            if slot not in used:
                used.add(slot)
                result.append(slot)
                match = search(index + 1, used, result)
                if match is not None:
                    return match
                result.pop()
                used.remove(slot)
        return None

    return search(0, set(), [])


def collate_examples(examples: Iterable[RecallExample]) -> dict[str, torch.Tensor | list[ExampleMetadata]]:
    examples = list(examples)
    return {
        "input_ids": torch.stack([item.input_ids for item in examples]),
        "labels": torch.stack([item.labels for item in examples]),
        "metadata": [item.metadata for item in examples],
    }


def batch_hash(input_ids: torch.Tensor) -> str:
    array = input_ids.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def assert_disjoint_manifests(manifests: dict[str, Iterable[dict[str, object]]]) -> None:
    seen: dict[int, str] = {}
    for split, rows in manifests.items():
        for row in rows:
            seed = int(row["seed"])
            if seed in seen:
                raise ValueError(f"seed {seed} occurs in both {seen[seed]} and {split}")
            seen[seed] = split


def manifest_jsonl(examples: Iterable[RecallExample]) -> str:
    return "".join(json.dumps(item.manifest(), sort_keys=True) + "\n" for item in examples)


def derived_example_seed(stream_seed: int, index: int) -> int:
    pass
    payload = f"large-mamba-recall-v1:{stream_seed}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


class RecallBatchFactory:
    pass

    def __init__(
        self,
        pools: TokenPools,
        *,
        stream_seed: int,
        sequence_length: int = 256,
        associations: int = 16,
        queries: int = 4,
        micro_batch_sizes: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1),
    ) -> None:
        self.pools = pools
        self.stream_seed = stream_seed
        self.sequence_length = sequence_length
        self.associations = associations
        self.queries = queries
        if not micro_batch_sizes or any(size <= 0 for size in micro_batch_sizes):
            raise ValueError("micro_batch_sizes must contain positive batch sizes")
        self.micro_batch_sizes = micro_batch_sizes

    def __call__(self, optimizer_step: int, micro_step: int) -> dict[str, torch.Tensor]:
        if not 0 <= micro_step < len(self.micro_batch_sizes):
            raise ValueError("micro_step is outside the configured accumulation window")
        global_batch = sum(self.micro_batch_sizes)
        micro_batch_size = self.micro_batch_sizes[micro_step]
        start = (
            optimizer_step * global_batch
            + sum(self.micro_batch_sizes[:micro_step])
        )
        examples = []
        for offset in range(micro_batch_size):
            index = start + offset
            examples.append(
                generate_recall_example(
                    self.pools,
                    seed=derived_example_seed(self.stream_seed, index),
                    sequence_length=self.sequence_length,
                    associations=self.associations,
                    queries=self.queries,
                    lag_bucket=tuple(LagBucket)[index % len(LagBucket)],
                )
            )
        batch = collate_examples(examples)
        return {"input_ids": batch["input_ids"], "labels": batch["labels"]}
