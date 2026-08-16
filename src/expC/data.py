from __future__ import annotations

import torch


def make_copy_batch(
    batch_size: int,
    L: int,
    V: int = 16,
    delay: int = 0,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if device is None:
        device = "cpu"
    device = torch.device(device)
    assert delay >= 0 and L > delay + 1

    tokens = torch.randint(0, V, (batch_size, L), device=device, generator=generator)

    query_pos = delay + torch.randint(0, L - delay, (batch_size,), device=device, generator=generator)
    target = tokens[torch.arange(batch_size, device=device), query_pos - delay]

    query_tok = torch.full((batch_size, 1), V, device=device, dtype=tokens.dtype)
    input_ids = torch.cat([tokens, query_tok], dim=1)

    return {
        "tokens": tokens,
        "query_pos": query_pos,
        "target": target,
        "input_ids": input_ids,
    }
