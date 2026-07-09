
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.snr import hypersphere, row_normalize, row_normalize_


VALID_MODES = ("B0", "BW", "BR", "BX", "BW_BR", "B0_noshort", "BR_noshort", "sphere_on_z")


def diagonal_scan(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    B, T, k = x.shape
    A = A.clamp(1e-4, 1 - 1e-5)
    t = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1)
    A_b = A.view(1, 1, k)
    logA = torch.log(A_b)
    A_pow = torch.exp(logA * t)
    A_inv_pow = torch.exp(-logA * t)
    cum = torch.cumsum(x * A_inv_pow, dim=1)
    return cum * A_pow


class DiagonalSSM(nn.Module):

    def __init__(
        self,
        V: int = 16,
        L_max: int = 128,
        d_model: int = 64,
        k: int = 128,
        mode: str = "B0",
        n_layers: int = 2,
        no_pos_embed: bool = False,
    ):
        super().__init__()
        assert mode in VALID_MODES, f"Unknown mode {mode}, valid: {VALID_MODES}"
        self.V = V
        self.d_model = d_model
        self.k = k
        self.mode = mode
        self.L_max = L_max
        self.n_layers = n_layers
        self.no_pos_embed = no_pos_embed

        self.embed = nn.Embedding(V, d_model)
        if not no_pos_embed:
            self.pos_embed = nn.Embedding(L_max, d_model)
            nn.init.normal_(self.pos_embed.weight, std=0.02)
        else:
            self.pos_embed = None
        nn.init.normal_(self.embed.weight, std=0.02)

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
            B = nn.Parameter(torch.empty(k, d_model))
            C = nn.Parameter(torch.empty(d_model, k))
            nn.init.xavier_uniform_(B)
            nn.init.xavier_uniform_(C)
            self.B.append(B)
            self.C.append(C)
            self.layer_norm.append(nn.LayerNorm(d_model))
            self.out_proj.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            ))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, V),
        )

        if mode in ("BW", "BW_BR"):
            self.row_normalize_weights_()

    def A_bar(self, layer: int) -> torch.Tensor:
        return torch.exp(-F.softplus(self.a_raw[layer]))

    def effective_tau(self) -> torch.Tensor:
        taus = []
        for i in range(self.n_layers):
            ab = self.A_bar(i).clamp(1e-6, 1 - 1e-6)
            taus.append(-1.0 / torch.log(ab))
        return torch.cat(taus, dim=0)

    def row_normalize_weights_(self) -> None:
        if self.mode not in ("BW", "BW_BR"):
            return
        for i in range(self.n_layers):
            row_normalize_(self.B[i])
            row_normalize_(self.C[i])
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                row_normalize_(m.weight)
        for proj in self.out_proj:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    row_normalize_(m.weight)

    def _readout_h(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        C = self.C[layer]
        if self.mode in ("BR", "BR_noshort", "BW_BR"):
            h = hypersphere(h, dim=-1)
            C = row_normalize(C)
        return F.linear(h, C)

    def _state_step(self, h_prev: torch.Tensor, x_step: torch.Tensor, layer: int) -> torch.Tensor:
        A = self.A_bar(layer)
        Bu = F.linear(x_step, self.B[layer])
        h = A.view(1, -1) * h_prev + Bu
        if self.mode == "BX":
            h = hypersphere(h, dim=-1)
        return h
    def forward(
        self,
        input_ids: torch.Tensor,
        query_pos: torch.Tensor,
        return_states: bool = False,
        return_h_norms: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids.shape[1] >= 2 and int(input_ids[0, -1].item()) == self.V:
            tokens = input_ids[:, :-1]
        else:
            tokens = input_ids[:, :-1] if input_ids.shape[1] > 1 and input_ids[:, -1].max() >= self.V else input_ids
            if input_ids.shape[1] > 1 and (input_ids[:, -1] == self.V).all():
                tokens = input_ids[:, :-1]
        if input_ids.size(1) > 1 and (input_ids[:, -1] == self.V).all():
            tokens = input_ids[:, :-1]
        else:
            tokens = input_ids
        Bsz, L = tokens.shape
        device = tokens.device
        if self.pos_embed is not None:
            pos = torch.arange(L, device=device).unsqueeze(0).expand(Bsz, L)
            x = self.embed(tokens) + self.pos_embed(pos.clamp(0, self.L_max - 1))
        else:
            x = self.embed(tokens)
        states = None
        h_all = None
        h_norms: list[torch.Tensor] = []
        for i in range(self.n_layers):
            A = self.A_bar(i)
            Bu = F.linear(self.layer_norm[i](x), self.B[i])
            if self.mode == "sphere_on_z":
                Bu = hypersphere(Bu, dim=-1)
            if self.mode == "BX":
                h_steps = []
                h_prev = torch.zeros(Bsz, self.k, device=device)
                for t in range(L):
                    h_prev = self._state_step(h_prev, x[:, t, :], i)
                    h_steps.append(h_prev.unsqueeze(1))
                    h_norms.append(h_prev.norm(dim=-1, keepdim=True))
                h = torch.cat(h_steps, dim=1)
            else:
                h = diagonal_scan(A, Bu)
            y = self._readout_h(h, i)
            y = self.out_proj[i](y)
            noshort = self.mode in ("B0_noshort", "BR_noshort")
            if not noshort:
                x = x + y
            else:
                x = y
            h_all = h
            if return_states and i == self.n_layers - 1:
                states = h
        assert h_all is not None
        idx = query_pos.view(Bsz, 1, 1).expand(Bsz, 1, self.k)
        h_q = h_all.gather(1, idx).squeeze(1)
        y_q = self._readout_h(h_q, self.n_layers - 1)
        noshort = self.mode in ("B0_noshort", "BR_noshort")
        if not noshort:
            idx_d = query_pos.view(Bsz, 1, 1).expand(Bsz, 1, self.d_model)
            x_q = x.gather(1, idx_d).squeeze(1)
            feat = y_q + x_q
        else:
            feat = y_q
        logits = self.head(feat)
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "h_final": h_q,
            "y_last": feat,
        }
        if return_states and states is not None:
            out["states"] = states
        if return_h_norms and h_norms:
            out["h_norms"] = torch.stack(h_norms, dim=1).squeeze(-1)
        return out
    def probe_products(self, h: torch.Tensor, use_br_path: bool | None = None) -> torch.Tensor:
        layer = self.n_layers - 1
        C = self.C[layer]
        if use_br_path is None:
            use_br_path = self.mode in ("BR", "BR_noshort", "BW_BR")
        if h.shape[-1] != self.k:
            return h.new_zeros(h.shape[0], self.k)
        if use_br_path:
            h = hypersphere(h, dim=-1)
            C = row_normalize(C)
        gains = C.norm(dim=0)
        return h * gains.view(1, -1)
