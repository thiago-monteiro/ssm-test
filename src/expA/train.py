
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.expA.models import Autoencoder
from src.quantize import (
    calibrate_ranges,
    entropy_estimate,
    matched_noise_inject,
    quantize_fixed_bitwidth,
    quantize_uniform,
)
from src.seed import seed_everything
from src.snr import effective_snr, signal_noise_rho, variance_fraction


def train_ae(
    seed: int,
    normalized: bool = False,
    d: int = 64,
    k: int = 8,
    h: int = 128,
    steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str | torch.device | None = None,
    log_every: int = 500,
    sphere_on_z_only: bool = False,
    probe_width: int | None = None,
) -> tuple[Autoencoder, dict[str, Any]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)
    device = torch.device(device)
    model = Autoencoder(
        d=d, k=k, h=h, normalized=normalized,
        sphere_on_z_only=sphere_on_z_only,
        probe_width=probe_width,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[float] = []
    model.train()
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, d, device=device)
        x_hat, _ = model(x)
        loss = nn.functional.mse_loss(x_hat, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.row_normalize_weights_()
        history.append(float(loss.item()))
        if log_every and step % log_every == 0:
            print(
                f"  [A seed={seed} norm={normalized}] step {step}/{steps} "
                f"mse={loss.item():.6f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        x = torch.randn(2048, d, device=device)
        x_hat, _ = model(x)
        final_mse = float(nn.functional.mse_loss(x_hat, x).item())

    meta = {"seed": seed, "normalized": normalized, "final_mse": final_mse, "history": history}
    return model, meta


@torch.no_grad()
def eval_quant_sweep(
    model: Autoencoder,
    seed: int,
    d: int = 64,
    n_eval: int = 2048,
    n_cal: int = 2048,
    levels_list: list[int | None] | None = None,
    device: str | torch.device | None = None,
    also_matched_noise: bool = True,
) -> list[dict[str, Any]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if levels_list is None:
        levels_list = [None, 16, 8, 4, 2]

    seed_everything(seed + 10_000)
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    x_cal = torch.randn(n_cal, d, device=device)
    z_cal = model.encode(x_cal)
    mode = "global" if model.normalized else "per_coord"
    lo, hi = calibrate_ranges(z_cal, mode=mode)

    x = torch.randn(n_eval, d, device=device)
    z_clean = model.encode(x)
    p_clean = model.probe_products(z_clean)

    z_entropy = entropy_estimate(z_clean)

    rows: list[dict[str, Any]] = []
    for levels in levels_list:
        if levels is None:
            z_q = z_clean
            label = "fp32"
            nlev = 0
        else:
            z_q = quantize_uniform(z_clean, levels=levels, lo=lo, hi=hi)
            label = f"{levels}-level"
            nlev = levels

        x_hat = model.decode(z_q)
        mse = float(nn.functional.mse_loss(x_hat, x).item())
        p_obs = model.probe_products(z_q)
        snr = signal_noise_rho(p_clean, p_obs, max_coords=min(64, z_clean.shape[1]))
        esnr = effective_snr(p_clean, p_obs)
        p_noise = p_obs - p_clean
        vf = variance_fraction(p_clean, p_noise, n_components=min(4, p_clean.shape[1]))

        row = {
            "seed": seed,
            "normalized": model.normalized,
            "quant_label": label,
            "levels": nlev,
            "mse": mse,
            "z_entropy": z_entropy,
            **snr,
            **esnr,
            **vf,
        }

        if also_matched_noise and levels is not None and levels >= 2:
            z_mn = matched_noise_inject(z_clean, lo, hi, levels)
            x_hat_mn = model.decode(z_mn)
            mse_mn = float(nn.functional.mse_loss(x_hat_mn, x).item())
            p_mn = model.probe_products(z_mn)
            snr_mn = signal_noise_rho(p_clean, p_mn, max_coords=min(64, z_clean.shape[1]))
            esnr_mn = effective_snr(p_clean, p_mn)
            row["mse_matched_noise"] = mse_mn
            row["matched_noise_rho_signal"] = snr_mn["rho_signal"]
            row["matched_noise_delta_rho"] = snr_mn["delta_rho"]
            row["matched_noise_snr"] = esnr_mn["snr_effective"]
            row["matched_noise_alignment"] = esnr_mn["alignment"]

        if levels is not None and levels in (4, 8, 16):
            bits = int(round(np.log2(levels)))
            if 2 ** bits == levels:
                z_ste = quantize_fixed_bitwidth(z_clean, bits, lo, hi)
                x_hat_ste = model.decode(z_ste)
                mse_ste = float(nn.functional.mse_loss(x_hat_ste, x).item())
                row["mse_fixedbit"] = mse_ste

        rows.append(row)

    return rows
