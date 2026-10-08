from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diagonal import initialize_layers

EPS = 1e-08


class CopySSM(nn.Module):
    def __init__(
        self,
        V: int = 16,
        L: int = 32,
        d_model: int = 64,
        k: int = 128,
        n_layers: int = 2,
        sphere: bool = False,
    ):
        super().__init__()
        assert n_layers == 2, "expC analysis assumes exactly 2 layers"
        self.V = V
        self.L = L
        self.d_model = d_model
        self.k = k
        self.n_layers = n_layers
        self.sphere = bool(sphere)
        self.embed = nn.Embedding(V + 1, d_model)
        self.pos_embed = nn.Embedding(L, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        initialize_layers(self, n_layers=n_layers, k=k, d_model=d_model, V=V)

    def A_bar(self, layer: int) -> torch.Tensor:
        return torch.exp(-F.softplus(self.a_raw[layer]))

    def _project(self, h: torch.Tensor) -> torch.Tensor:
        if self.sphere:
            return h / (h.norm(dim=-1, keepdim=True) + EPS)
        return h

    def scan(
        self,
        x_in: torch.Tensor,
        layer: int,
        t0: int = 0,
        h_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        A = self.A_bar(layer)
        Bu = F.linear(x_in, self.B[layer])
        if h_init is None:
            h_prev = torch.zeros(Bu.shape[0], self.k, device=Bu.device, dtype=Bu.dtype)
            start = 0
        else:
            h_prev = h_init
            start = t0 + 1
        steps = []
        for t in range(start, Bu.shape[-2]):
            step = Bu[:, t] if Bu.dim() == 3 else Bu[t]
            h_prev = self._project(A * h_prev + step)
            steps.append(h_prev.unsqueeze(-2))
        return torch.cat(steps, dim=-2)

    def _body(
        self, x0: torch.Tensor, query_pos: torch.Tensor, return_all: bool = False
    ):
        Bsz, L, _ = x0.shape
        h_last = None
        layer_inputs = []
        for i in range(self.n_layers):
            xi = self.layer_norm[i](x0)
            if return_all:
                layer_inputs.append(xi)
            h = self.scan(xi, i)
            y = F.linear(h, self.C[i])
            y = self.out_proj[i](y)
            x0 = x0 + y
            h_last = h
        assert h_last is not None
        idx = query_pos.view(Bsz, 1, 1).expand(Bsz, 1, self.k)
        h_q = h_last.gather(1, idx).squeeze(1)
        y_q = F.linear(h_q, self.C[self.n_layers - 1])
        idx_d = query_pos.view(Bsz, 1, 1).expand(Bsz, 1, self.d_model)
        x_q = x0.gather(1, idx_d).squeeze(1)
        feat = y_q + x_q
        logits = self.head(feat)
        out: dict[str, torch.Tensor] = {"logits": logits, "h_q": h_q, "x_q": x_q}
        if return_all:
            out["h_last"] = h_last
            out["layer_inputs"] = layer_inputs
        return out

    def forward(
        self, input_ids: torch.Tensor, query_pos: torch.Tensor, return_all: bool = False
    ) -> dict[str, torch.Tensor]:
        tokens = input_ids[:, :-1]
        Bsz, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(Bsz, L)
        x0 = self.embed(tokens) + self.pos_embed(pos)
        return self._body(x0, query_pos, return_all=return_all)

    def forward_x0(
        self, x0: torch.Tensor, query_pos: torch.Tensor, return_all: bool = False
    ):
        return self._body(x0, query_pos, return_all=return_all)

    def logits_from_final_state(
        self, h_prime: torch.Tensor, x_q: torch.Tensor
    ) -> torch.Tensor:
        y_q = F.linear(h_prime, self.C[self.n_layers - 1])
        return self.head(y_q + x_q)

    def tail_scan_to_q(
        self, x1_row: torch.Tensor, t0: int, h_init: torch.Tensor, q: int
    ) -> torch.Tensor:
        assert q > t0 >= 0
        states = self.scan(x1_row.unsqueeze(0), self.n_layers - 1, t0=t0, h_init=h_init)
        return states[..., q - t0 - 1, :]

    def mean_token_embed(self) -> torch.Tensor:
        with torch.no_grad():
            return self.embed.weight[: self.V].mean(dim=0)
