from __future__ import annotations

import json
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path


def wheel_url(
    repo: str, tag: str, package: str, version: str, python_tag: str, abi: str
) -> str:
    asset = f"{package}-{version}+cu12torch2.6cxx11abi{abi}-{python_tag}-{python_tag}-linux_x86_64.whl"
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
        headers={"User-Agent": "ssm-test-colab"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    matches = [
        entry["browser_download_url"]
        for entry in release["assets"]
        if entry["name"] == asset
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"No pinned prebuilt wheel for {asset}; refusing a source build"
        )
    return matches[0]


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This installer is for Linux x86_64 Colab GPU runtimes")
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError("Pinned CUDA wheels require Python 3.11, 3.12 or 3.13")
    pip = [sys.executable, "-m", "pip"]
    subprocess.run([*pip, "uninstall", "-y", "torchvision", "torchaudio"], check=True)
    subprocess.run(
        [
            *pip,
            "install",
            "torch==2.6.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        check=True,
    )
    subprocess.run(
        [
            *pip,
            "install",
            "-r",
            str(Path(__file__).resolve().parents[1] / "requirements-t4.txt"),
        ],
        check=True,
    )
    import torch

    if (
        torch.__version__.split("+")[0] != "2.6.0"
        or not torch.version.cuda
        or (not torch.version.cuda.startswith("12."))
    ):
        raise RuntimeError(
            "Expected PyTorch 2.6.0 with CUDA 12; restart the session and retry"
        )
    abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    url = wheel_url(
        "Dao-AILab/causal-conv1d",
        "v1.5.0.post8",
        "causal_conv1d",
        "1.5.0.post8",
        python_tag,
        abi,
    )
    print(f"Installing convolution kernel: {url}", flush=True)
    subprocess.run([*pip, "install", "--no-deps", url], check=True)
    subprocess.run(
        [*pip, "install", "--no-build-isolation", "mamba-ssm==2.2.4"], check=True
    )
    print(
        "Installation complete. Restart the Colab session before running training.",
        flush=True,
    )


if __name__ == "__main__":
    main()
