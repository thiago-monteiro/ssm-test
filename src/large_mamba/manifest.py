from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from .modeling import package_versions


def environment_manifest(repository_commit: str | None = None) -> dict[str, object]:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repository_commit": repository_commit,
        "gpu": gpu,
        "packages": package_versions(
            ("torch", "mamba-ssm", "causal-conv1d", "triton", "peft", "transformers", "datasets")
        ),
    }


def write_environment_manifest(path: str | Path, repository_commit: str | None = None) -> None:
    Path(path).write_text(json.dumps(environment_manifest(repository_commit), indent=2) + "\n")
