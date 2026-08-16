from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


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

        self.a_raw = nn.ParameterList()
        self.B = nn.ParameterList()
        self.C = nn.ParameterList()
        self.layer_norm = nn.ModuleList()
        self.out_proj = nn.ModuleList()
        for _ in range(n_layers):
            a = nn.Parameter(torch.zeros(k))
            with torch.no_grad():
                target = -math.log(0.995)
                a.fill_(math.log(math.expm1(max(target, 1e-4))))
            self.a_raw.append(a)
            Bm = nn.Parameter(torch.empty(k, d_model))
            Cm = nn.Parameter(torch.empty(d_model, k))
            nn.init.xavier_uniform_(Bm)
            nn.init.xavier_uniform_(Cm)
            self.B.append(Bm)
            self.C.append(Cm)
            self.layer_norm.append(nn.LayerNorm(d_model))
            self.out_proj.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
            )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, V),
        )


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
            Bsz, L, _ = Bu.shape
            h_prev = torch.zeros(Bsz, self.k, device=Bu.device, dtype=Bu.dtype)
            steps = []
            for t in range(L):
                h_prev = A * h_prev + Bu[:, t]
                h_prev = self._project(h_prev)
                steps.append(h_prev.unsqueeze(1))
            return torch.cat(steps, dim=1)

        h_prev = h_init
        out_steps = []
        for t in range(t0 + 1, Bu.shape[-2]):
            step = Bu[:, t] if Bu.dim() == 3 else Bu[t]
            h_prev = A * h_prev + step
            h_prev = self._project(h_prev)
            out_steps.append(h_prev.unsqueeze(-2))
        return torch.cat(out_steps, dim=-2)


    def _body(self, x0: torch.Tensor, query_pos: torch.Tensor, return_all: bool = False):
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
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "h_q": h_q,
            "x_q": x_q,
        }
        if return_all:
            out["h_last"] = h_last
            out["layer_inputs"] = layer_inputs
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        query_pos: torch.Tensor,
        return_all: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokens = input_ids[:, :-1]
        Bsz, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(Bsz, L)
        x0 = self.embed(tokens) + self.pos_embed(pos)
        return self._body(x0, query_pos, return_all=return_all)

    def forward_x0(self, x0: torch.Tensor, query_pos: torch.Tensor, return_all: bool = False):
        return self._body(x0, query_pos, return_all=return_all)


    def logits_from_final_state(self, h_prime: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
        y_q = F.linear(h_prime, self.C[self.n_layers - 1])
        return self.head(y_q + x_q)

    def tail_scan_to_q(
        self,
        x1_row: torch.Tensor,
        t0: int,
        h_init: torch.Tensor,
        q: int,
    ) -> torch.Tensor:
        assert q > t0 >= 0
        states = self.scan(x1_row.unsqueeze(0), self.n_layers - 1, t0=t0, h_init=h_init)
        return states[..., q - t0 - 1, :]

    def mean_token_embed(self) -> torch.Tensor:
        with torch.no_grad():
            return self.embed.weight[: self.V].mean(dim=0)
