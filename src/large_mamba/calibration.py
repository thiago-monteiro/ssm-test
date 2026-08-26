from __future__ import annotations

from collections.abc import Iterable

import torch

from .adapter import ProjectedMambaMixer


@torch.no_grad()
def calibrate_radii(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    adapters: Iterable[ProjectedMambaMixer],
    *,
    max_examples: int = 1024,
) -> dict[int, float]:
    pass
    adapters = list(adapters)
    for adapter in adapters:
        adapter.capture_states = True
        adapter.set_projection_strength(0.0)
    norms: dict[int, list[torch.Tensor]] = {int(a.layer_idx): [] for a in adapters}
    observed = 0
    model.eval()
    try:
        for input_ids in batches:
            if observed >= max_examples:
                break
            keep = min(input_ids.shape[0], max_examples - observed)
            model(input_ids[:keep])
            observed += keep
            for adapter in adapters:
                if adapter.last_scan is None or adapter.last_scan.states is None:
                    raise RuntimeError("adapter did not capture recurrent states")
                states = adapter.last_scan.states.permute(0, 2, 1, 3).flatten(start_dim=2)
                norms[int(adapter.layer_idx)].append(torch.linalg.vector_norm(states.float(), dim=2).cpu())
                adapter.last_scan = None
    finally:
        for adapter in adapters:
            adapter.capture_states = False
    if observed != max_examples:
        raise ValueError(f"calibration supplied {observed} examples, expected {max_examples}")
    return {layer: float(torch.cat(rows).median()) for layer, rows in norms.items()}
