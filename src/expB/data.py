
from __future__ import annotations

import torch

V_DEFAULT = 16
QUERY_ID_OFFSET = 1


def make_batch(
    batch_size: int,
    L: int,
    V: int = V_DEFAULT,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
    with_replacement: bool = True,
    associative_cue_noise: float = 0.0,
) -> dict[str, torch.Tensor]:
    if device is None:
        device = "cpu"
    device = torch.device(device)

    effective_V = V
    if not with_replacement and L > V:
        effective_V = L * 2
    if with_replacement:
        tokens = torch.randint(0, effective_V, (batch_size, L), device=device, generator=generator)
    else:
        tokens_list = []
        for _ in range(batch_size):
            tokens_list.append(torch.randperm(effective_V, device=device, generator=generator)[:L])
        tokens = torch.stack(tokens_list, dim=0)

    query_pos = torch.randint(0, L, (batch_size,), device=device, generator=generator)
    target = tokens[torch.arange(batch_size, device=device), query_pos]

    if associative_cue_noise > 0.0:
        pass

    query_tok = torch.full((batch_size, 1), V, device=device, dtype=tokens.dtype)
    input_ids = torch.cat([tokens, query_tok], dim=1)

    return {
        "tokens": tokens,
        "query_pos": query_pos,
        "target": target,
        "input_ids": input_ids,
    }


def make_associative_batch(
    batch_size: int,
    L: int,
    V: int = V_DEFAULT,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if device is None:
        device = "cpu"
    device = torch.device(device)

    token_vocab = V // 2
    keys = torch.randint(0, token_vocab, (batch_size, L // 2), device=device, generator=generator)
    values = torch.randint(token_vocab, V, (batch_size, L // 2), device=device, generator=generator)

    tokens_list = []
    for i in range(L // 2):
        tokens_list.append(keys[:, i:i+1])
        tokens_list.append(values[:, i:i+1])
    tokens = torch.cat(tokens_list, dim=1)

    query_idx = torch.randint(0, L // 2, (batch_size,), device=device, generator=generator)
    query_key = keys[torch.arange(batch_size, device=device), query_idx]
    target = values[torch.arange(batch_size, device=device), query_idx]

    query_tok = torch.full((batch_size, 1), V, device=device, dtype=tokens.dtype)
    input_ids = torch.cat([tokens, query_tok], dim=1)

    return {
        "tokens": tokens,
        "query_pos": query_idx * 2 + 1,
        "target": target,
        "input_ids": input_ids,
        "query_key": query_key,
    }
