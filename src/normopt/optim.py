from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .normalize import tangentialize

EPS = 1e-08
OPTIMIZER_NAMES = ("sgdm", "adamw", "muon", "rmo", "gtmuon")
SPECTRAL_NAMES = ("muon", "rmo")


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-07,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    a, b, c = (3.4445, -4.775, 2.0315)
    if dtype is None:
        dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def _momentum_update(g, group, state, update_state):
    momentum = group["momentum"]
    buf = state.get("momentum_buffer")
    if buf is None:
        buf = state["momentum_buffer"] = torch.zeros_like(g)
    new_buf = buf.mul(momentum).add_(g)
    if update_state:
        state["momentum_buffer"] = new_buf
    raw = g.add(new_buf, alpha=momentum) if group["nesterov"] else new_buf.clone()
    return raw


class MatrixUpdateRule(torch.optim.Optimizer):
    def __init__(self, params, defaults: dict) -> None:
        super().__init__(params, defaults)

    def compute_update(
        self, p: torch.Tensor, update_state: bool = True
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def peek_update(self, p: torch.Tensor) -> torch.Tensor:
        return self.compute_update(p, update_state=False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(self.compute_update(p), alpha=-lr)
        return loss


class MuonMomentum(MatrixUpdateRule):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_dtype: torch.dtype | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
            ),
        )
        self.ns_dtype = ns_dtype

    @torch.no_grad()
    def compute_update(
        self, p: torch.Tensor, update_state: bool = True
    ) -> torch.Tensor:
        g = p.grad
        assert g is not None
        group = self.param_groups[0]
        raw = _momentum_update(g, group, self.state[p], update_state)
        ortho = zeropower_via_newtonschulz5(
            raw, steps=int(group["ns_steps"]), dtype=self.ns_dtype
        )
        fan_out, fan_in = p.shape
        return ortho * math.sqrt(max(1.0, fan_out / fan_in))


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    lr: float
    lr_rest: float = 0.003
    momentum: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    wd_rest: float = 0.01
    nesterov: bool = True
    ns_steps: int = 5
    ns_dtype: str = "auto"

    def __post_init__(self) -> None:
        if self.name not in OPTIMIZER_NAMES:
            raise ValueError(
                f"unknown optimizer {self.name!r}, expected {OPTIMIZER_NAMES}"
            )

    def resolved_ns_dtype(self, device: torch.device) -> torch.dtype | None:
        if self.name not in ("muon", "gtmuon"):
            return None
        use_bf16 = self.ns_dtype == "auto" and device.type == "cuda"
        return torch.bfloat16 if use_bf16 else torch.float32

    def with_lr(self, lr: float) -> OptimizerSpec:
        return replace(self, lr=lr)


class HybridOptimizer:
    def __init__(
        self,
        main: MatrixUpdateRule,
        rest: torch.optim.Optimizer | None,
        names: list[str],
        spec: OptimizerSpec | None = None,
    ) -> None:
        self.main = main
        self.rest = rest
        self.names = names
        self.spec = spec
        self._base_lrs = {
            id(g): g["lr"] for o in self.optimizers for g in o.param_groups
        }

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.main] if self.rest is None else [self.main, self.rest]

    @property
    def param_groups(self) -> list[dict]:
        return [g for o in self.optimizers for g in o.param_groups]

    @property
    def lr(self) -> float:
        return self.param_groups[0]["lr"]

    def set_scale(self, scale: float) -> None:
        for o in self.optimizers:
            for group in o.param_groups:
                group["lr"] = self._base_lrs[id(group)] * scale

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "main": self.main.state_dict(),
            "rest": None if self.rest is None else self.rest.state_dict(),
        }

    @torch.no_grad()
    def matrix_updates(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        params = self.main.param_groups[0]["params"]
        for name, p in zip(self.names, params, strict=True):
            if p.grad is None:
                continue
            out[name] = self.main.peek_update(p).detach().clone()
        return out

    def step(self) -> None:
        self.main.step()
        if self.rest is not None:
            self.rest.step()


class TangentRowMomentum(MatrixUpdateRule):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        unit_rows: bool = True,
        tangential: bool = True,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                unit_rows=unit_rows,
                tangential=tangential,
                weight_decay=weight_decay,
            ),
        )

    @torch.no_grad()
    def compute_update(
        self, p: torch.Tensor, update_state: bool = True
    ) -> torch.Tensor:
        g = p.grad
        assert g is not None
        group = self.param_groups[0]
        raw = _momentum_update(g, group, self.state[p], update_state)
        if group["tangential"]:
            raw = tangentialize(raw, p.detach())
        if group["unit_rows"]:
            raw = raw / (raw.norm(dim=1, keepdim=True) + EPS)
        return raw


class GraftedTangentMuon(MatrixUpdateRule):
    def __init__(
        self,
        params,
        lr: float = 0.003,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 3,
        ns_dtype: torch.dtype | None = None,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-08,
        tangential: bool = True,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                betas=betas,
                eps=eps,
                tangential=tangential,
                weight_decay=weight_decay,
            ),
        )
        self.ns_dtype = ns_dtype
        self.betas = betas
        self.eps = eps

    @torch.no_grad()
    def compute_update(
        self, p: torch.Tensor, update_state: bool = True
    ) -> torch.Tensor:
        g = p.grad
        assert g is not None
        group = self.param_groups[0]
        state = self.state[p]
        raw = _momentum_update(g, group, state, update_state)
        ortho = zeropower_via_newtonschulz5(
            raw, steps=int(group["ns_steps"]), dtype=self.ns_dtype
        )
        if group["tangential"]:
            ortho = tangentialize(ortho, p.detach())
        m = state.get("adam_m")
        v = state.get("adam_v")
        if m is None:
            m = state["adam_m"] = torch.zeros_like(g)
            v = state["adam_v"] = torch.zeros_like(g)
        b1, b2 = self.betas
        if update_state:
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            state["adam_step"] = state.get("adam_step", 0) + 1
        else:
            m = m * b1 + g * (1.0 - b1)
            v = v * b2 + g * g * (1.0 - b2)
        step_no = state.get("adam_step", 1)
        m_hat = m / (1.0 - b1 ** max(1, step_no))
        v_hat = v / (1.0 - b2 ** max(1, step_no))
        adam_step = m_hat / (v_hat.sqrt() + self.eps)
        graft = adam_step.norm(dim=1, keepdim=True).clamp_min(1e-12) / ortho.norm(
            dim=1, keepdim=True
        ).clamp_min(EPS)
        return ortho * graft
