from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any


def run_parallel(
    worker_fn: Callable,
    tasks: list[Any],
    n_workers: int | None = None,
    timeout: float | None = None,
) -> list[Any]:
    if n_workers is None or n_workers <= 1:
        return [worker_fn(t) for t in tasks]
    n_workers = min(n_workers, len(tasks), os.cpu_count() or 4)
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        return list(pool.imap_unordered(worker_fn, tasks))
