
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.snr import hypersphere, row_normalize_


class Autoencoder(nn.Module):
    def __init__(
        self,
        d: int = 64,
        k: int = 8,
        h: int = 128,
        normalized: bool = False,
        sphere_on_z_only: bool = False,
        probe_width: int | None = None,
        n_classes: int = 64,
    ):
        super().__init__()
        self.d = d
        self.k = k
        self.h = h
        self.normalized = normalized
        self.sphere_on_z_only = sphere_on_z_only

        self.enc1 = nn.Linear(d, h)
        self.enc2 = nn.Linear(h, k)
        self.dec1 = nn.Linear(k, h)
        self.dec2 = nn.Linear(h, d)
        self.act = nn.GELU()

        self.probe_head: nn.Linear | None = None
        if probe_width is not None:
            self.probe_encoder = nn.Linear(k, probe_width, bias=False)
            self._probe_width = probe_width
            with torch.no_grad():
                self.probe_encoder.weight.copy_(
                    torch.randn(probe_width, k) / (k ** 0.5)
                )
        else:
            self.probe_encoder = None
            self._probe_width = 0

        if normalized and not sphere_on_z_only:
            with torch.no_grad():
                for layer in (self.enc1, self.enc2, self.dec1, self.dec2):
                    row_normalize_(layer.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc2(self.act(self.enc1(x)))
        if self.normalized:
            z = hypersphere(z, dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec2(self.act(self.dec1(z)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def row_normalize_weights_(self) -> None:
        if not self.normalized or self.sphere_on_z_only:
            return
        for layer in (self.enc1, self.enc2, self.dec1, self.dec2):
            row_normalize_(layer.weight)

    def channel_gains(self) -> torch.Tensor:
        return self.dec1.weight.norm(dim=0)

    def probe_products(self, z: torch.Tensor) -> torch.Tensor:
        gains = self.channel_gains().detach()
        return z * gains.view(1, -1)

    def wide_probe_products(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.probe_encoder is None:
            return None
        p = self.probe_encoder(z)
        return z * p.norm(dim=0, keepdim=True)

    def train_probe_head(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
        steps: int = 200,
        lr: float = 1e-2,
    ) -> float:
        h = h.detach().to(device).float()
        labels = labels.detach().to(device)
        B, D = h.shape
        self.probe_head = nn.Linear(D, self.d if not hasattr(self, '_probe_width') else self._probe_width, bias=False).to(device)
        opt = torch.optim.AdamW(self.probe_head.parameters(), lr=lr)
        for _ in range(steps):
            idx = torch.randperm(B, device=device)[:min(256, B)]
            logits = self.probe_head(h[idx])
            loss = F.cross_entropy(logits, labels[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        self.probe_head.eval()
        with torch.no_grad():
            logits = self.probe_head(h)
            acc = (logits.argmax(-1) == labels).float().mean().item()
        return acc
