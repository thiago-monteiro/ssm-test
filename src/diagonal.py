from __future__ import annotations

import math

import torch
import torch.nn as nn


def initialize_layers(model, *, n_layers, k, d_model, V):
    model.a_raw = nn.ParameterList()
    model.B = nn.ParameterList()
    model.C = nn.ParameterList()
    model.layer_norm = nn.ModuleList()
    model.out_proj = nn.ModuleList()
    for _ in range(n_layers):
        a = nn.Parameter(torch.zeros(k))
        with torch.no_grad():
            target = -math.log(0.995)
            a.fill_(math.log(math.expm1(max(target, 0.0001))))
        model.a_raw.append(a)
        B = nn.Parameter(torch.empty(k, d_model))
        C = nn.Parameter(torch.empty(d_model, k))
        nn.init.xavier_uniform_(B)
        nn.init.xavier_uniform_(C)
        model.B.append(B)
        model.C.append(C)
        model.layer_norm.append(nn.LayerNorm(d_model))
        model.out_proj.append(
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
        )
    model.head = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, V),
    )
