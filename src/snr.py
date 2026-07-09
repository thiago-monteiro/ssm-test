
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def mean_pairwise_corr(P: np.ndarray | torch.Tensor) -> float:
    if isinstance(P, torch.Tensor):
        P = P.detach().cpu().numpy()
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"expected (N, m), got {P.shape}")
    n, m = P.shape
    if m < 2 or n < 3:
        return 0.0
    std = P.std(axis=0)
    keep = std > 1e-12
    if keep.sum() < 2:
        return 0.0
    P = P[:, keep]
    P = P - P.mean(axis=0, keepdims=True)
    C = np.corrcoef(P, rowvar=False)
    if not np.isfinite(C).all():
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.triu_indices(C.shape[0], k=1)
    off = C[iu]
    if off.size == 0:
        return 0.0
    return float(np.mean(off))


def signal_noise_rho(
    p_clean: torch.Tensor | np.ndarray,
    p_obs: torch.Tensor | np.ndarray,
    max_coords: int | None = 64,
    probe_seed: int = 0,
) -> dict[str, float]:
    if isinstance(p_clean, torch.Tensor):
        p_clean = p_clean.detach().cpu().float()
    else:
        p_clean = torch.as_tensor(p_clean, dtype=torch.float32)
    if isinstance(p_obs, torch.Tensor):
        p_obs = p_obs.detach().cpu().float()
    else:
        p_obs = torch.as_tensor(p_obs, dtype=torch.float32)

    p_noise = p_obs - p_clean
    m = p_clean.shape[1]
    if max_coords is not None and m > max_coords:
        g = torch.Generator()
        g.manual_seed(probe_seed)
        idx = torch.randperm(m, generator=g)[:max_coords]
        p_clean = p_clean[:, idx]
        p_noise = p_noise[:, idx]

    rho_s = mean_pairwise_corr(p_clean)
    rho_n = mean_pairwise_corr(p_noise)
    return {
        "rho_signal": rho_s,
        "rho_noise": rho_n,
        "delta_rho": rho_s - rho_n,
    }


def effective_snr(
    p_clean: torch.Tensor | np.ndarray,
    p_obs: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    if isinstance(p_clean, torch.Tensor):
        p_clean = p_clean.detach().cpu().float()
    else:
        p_clean = torch.as_tensor(p_clean, dtype=torch.float32)
    if isinstance(p_obs, torch.Tensor):
        p_obs = p_obs.detach().cpu().float()
    else:
        p_obs = torch.as_tensor(p_obs, dtype=torch.float32)

    p_noise = p_obs - p_clean
    signal_pow = (p_clean ** 2).sum(dim=1).clamp_min(1e-12)
    noise_pow = (p_noise ** 2).sum(dim=1).clamp_min(1e-12)
    snr_per_sample = (signal_pow / noise_pow).mean().item()

    align = F.cosine_similarity(
        p_clean.mean(dim=0, keepdim=True),
        p_obs.mean(dim=0, keepdim=True),
    ).item()

    return {"snr_effective": snr_per_sample, "alignment": align}


def variance_fraction(
    p_clean: torch.Tensor | np.ndarray,
    p_noise: torch.Tensor | np.ndarray,
    n_components: int = 4,
) -> dict[str, float]:
    if isinstance(p_clean, torch.Tensor):
        p_clean = p_clean.detach().cpu().numpy()
    else:
        p_clean = np.asarray(p_clean, dtype=np.float64)
    if isinstance(p_noise, torch.Tensor):
        p_noise = p_noise.detach().cpu().numpy()
    else:
        p_noise = np.asarray(p_noise, dtype=np.float64)
    def _top_var_fraction(X, k):
        if not np.isfinite(X).all():
            return 0.0
        Xc = X - X.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0
        total_var = (S ** 2).sum()
        if total_var < 1e-12:
            return 0.0
        top_var = (S[:k] ** 2).sum()
        return float(top_var / total_var)
    n_components = min(n_components, p_clean.shape[1], p_clean.shape[0] - 1)
    signal_frac = _top_var_fraction(p_clean, n_components)
    noise_frac = _top_var_fraction(p_noise, n_components)
    return {
        "signal_topvar_frac": signal_frac,
        "noise_topvar_frac": noise_frac,
        "var_ratio": signal_frac / max(noise_frac, 1e-12),
    }
def train_decode_probe(
    h: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    device: torch.device,
    steps: int = 200,
    lr: float = 1e-2,
) -> tuple[float, nn.Linear]:
    h = h.detach().to(device).float()
    labels = labels.detach().to(device)
    B, D = h.shape
    probe = nn.Linear(D, n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randperm(B, device=device)[:min(256, B)]
        logits = probe(h[idx])
        loss = F.cross_entropy(logits, labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    probe.eval()
    logits = probe(h)
    acc = (logits.argmax(-1) == labels).float().mean().item()
    return acc, probe


def hypersphere(v: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    return v / (v.norm(dim=dim, keepdim=True) + eps)


def row_normalize_(W: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    if W.ndim != 2:
        return W
    W.data.div_(W.data.norm(dim=1, keepdim=True) + eps)
    return W
def row_normalize(W: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    if W.ndim != 2:
        return W
    return W / (W.norm(dim=1, keepdim=True) + eps)
