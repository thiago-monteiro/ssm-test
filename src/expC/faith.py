from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from src.expC.data import make_copy_batch
from src.expC.model import CopySSM

EPS_E = 1e-3


@dataclass
class FaithResult:
    variant: str
    seed: int
    n_seq: int
    thetas: list[float]

    dS_pred: dict[float, np.ndarray] = field(default_factory=dict)
    dS_act: dict[float, np.ndarray] = field(default_factory=dict)

    h_norms: list[float] = field(default_factory=list)
    margins_clean: list[float] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)

    def metrics(self, theta: float) -> dict[str, float]:
        p = self.dS_pred[theta]
        a = self.dS_act[theta]
        n = len(a)
        E_all = float(np.mean(np.abs(a - p) / (np.abs(a) + EPS_E)))
        med = np.median(np.abs(a))
        eff = np.abs(a) >= max(med, 1e-9)
        if eff.sum() >= 8:
            ae, pe = a[eff], p[eff]
            E_eff = float(np.mean(np.abs(ae - pe) / (np.abs(ae) + EPS_E)))
            sign_acc = float(np.mean(np.sign(ae) == np.sign(pe))) if np.any(ae != 0) else float("nan")
        else:
            E_eff, sign_acc = float("nan"), float("nan")

        if eff.sum() >= 16:
            ae, pe = a[eff], p[eff]
            qa = np.quantile(np.abs(ae), [0.25, 0.75])
            qp = np.quantile(np.abs(pe), [0.25, 0.75])
            fn_rate = float(np.mean((np.abs(pe) <= qp[0]) & (np.abs(ae) >= qa[1])))
            fp_rate = float(np.mean((np.abs(pe) >= qp[1]) & (np.abs(ae) <= qa[0])))
        else:
            fn_rate, fp_rate = float("nan"), float("nan")
        if n >= 8 and np.std(p) > 1e-12 and np.std(a) > 1e-12:
            rho_rank = float(spearmanr(np.abs(p), np.abs(a)).statistic)
            calib_r = float(pearsonr(p, a).statistic)
            slope = float(np.cov(p, a)[0, 1] / np.var(p)) if np.var(p) > 1e-12 else float("nan")
        else:
            rho_rank, calib_r, slope = float("nan"), float("nan"), float("nan")
        return {
            "theta": theta,
            "n_pairs": n,
            "E_all": E_all,
            "E_eff": E_eff,
            "rho_rank": rho_rank,
            "calib_r": calib_r,
            "slope": slope,
            "sign_acc": sign_acc,
            "fn_rate": fn_rate,
            "fp_rate": fp_rate,
        }


def _runner_up(logits_row: torch.Tensor, target: int) -> int:
    z = logits_row.clone()
    z[target] = -1e9
    return int(z.argmax().item())


def _tangent_project(u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    c = u @ h
    ut = u - c.unsqueeze(1) * h.unsqueeze(0)
    nrm = ut.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return ut / nrm


def _unit_random(M: int, k: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    v = torch.randn(M, k, generator=gen, device=device)
    return F.normalize(v, dim=-1)


class FaithfulnessEvaluator:
    def __init__(self, model: CopySSM, device: str | torch.device = "cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)


    def eval_final_state(
        self,
        n_seq: int,
        L: int,
        V: int,
        seed: int,
        thetas: list[float],
        n_random_dirs: int = 64,
        use_basis: bool = True,
        delay: int = 0,
    ) -> FaithResult:
        model = self.model
        gen = torch.Generator(device="cpu").manual_seed(seed)
        res = FaithResult(variant="sphere" if model.sphere else "ordinary", seed=seed, n_seq=n_seq, thetas=list(thetas))

        for s in range(n_seq):
            with torch.no_grad():
                batch = make_copy_batch(1, L, V=V, delay=delay, device=self.device, generator=gen)
                input_ids = batch["input_ids"].to(self.device)
                query_pos = batch["query_pos"].to(self.device)
                target = int(batch["target"][0].item())

                out = model(input_ids, query_pos, return_all=True)
                logits = out["logits"][0]
                h_q = out["h_q"][0]
                x_q = out["x_q"][0]
            alt = _runner_up(logits, target)
            S_clean = float(logits[target].item() - logits[alt].item())

            res.h_norms.append(float(h_q.norm().item()))
            res.margins_clean.append(S_clean)
            res.correct.append(bool(int(logits.argmax().item()) == target))


            h_leaf = h_q.detach().clone().requires_grad_(True)
            lp = model.logits_from_final_state(h_leaf.unsqueeze(0), x_q.unsqueeze(0))[0]
            S_p = lp[target] - lp[alt]
            g = torch.autograd.grad(S_p, h_leaf)[0].detach()
            if model.sphere:
                g_tan = g - float(g @ h_q) * h_q
            else:
                g_tan = g

            k = model.k
            dirs_list = []
            if use_basis:
                dirs_list.append(torch.eye(k, device=self.device))
            dirs_list.append(_unit_random(n_random_dirs, k, gen, self.device).cpu())
            U = torch.cat(dirs_list, dim=0)

            for theta in thetas:
                if model.sphere:
                    Ut = _tangent_project(U.to(self.device), h_q)
                    Hp = math.cos(theta) * h_q.unsqueeze(0).expand_as(Ut) + math.sin(theta) * Ut
                    delta = (Hp - h_q.unsqueeze(0)).cpu()
                else:
                    R = float(h_q.norm().item())
                    U_d = U.to(self.device)
                    Hp = h_q.unsqueeze(0).expand_as(U_d) + theta * R * U_d
                    delta = (theta * R * U_d).cpu()
                dS_pred = (delta @ g_tan.cpu()).numpy()

                with torch.no_grad():
                    lp2 = model.logits_from_final_state(Hp, x_q.unsqueeze(0).expand(len(Hp), -1))

                S_act = (lp2[:, target] - lp2[:, alt]).cpu().numpy() - S_clean
                res.dS_pred[theta] = np.concatenate([res.dS_pred.get(theta, []), dS_pred]) if theta in res.dS_pred else dS_pred.copy()
                res.dS_act[theta] = np.concatenate([res.dS_act.get(theta, []), S_act]) if theta in res.dS_act else S_act.copy()

        return res

    def eval_mid_state(
        self,
        n_seq: int,
        L: int,
        V: int,
        seed: int,
        thetas: list[float],
        n_random_dirs: int = 32,
        delay: int = 0,
        mid_mode: str = "upstream",
        t0_fracs: list[float] | None = None,
    ) -> dict[str, FaithResult]:
        assert mid_mode in ("upstream", "downstream")
        model = self.model
        gen = torch.Generator(device="cpu").manual_seed(seed + 777)
        if t0_fracs is not None:
            labels: list[tuple[str, float | None]] = [(f"t0={f:.2f}q", f) for f in t0_fracs]
        else:
            labels = [("downstream" if mid_mode == "downstream" else "upstream", None)]
        results: dict[str, FaithResult] = {
            lab: FaithResult(variant="sphere" if model.sphere else "ordinary",
                             seed=seed, n_seq=n_seq, thetas=list(thetas))
            for lab, _ in labels
        }

        for s in range(n_seq):
            with torch.no_grad():
                batch = make_copy_batch(1, L, V=V, delay=delay, device=self.device, generator=gen)
                input_ids = batch["input_ids"].to(self.device)
                query_pos = batch["query_pos"].to(self.device)
                q = int(query_pos[0].item())
                target = int(batch["target"][0].item())

            t0_by_label: dict[str, int] = {}
            for lab, frac in labels:
                if mid_mode == "downstream":
                    t0c = max(q - delay, 1)
                    if t0c >= q:
                        continue
                else:
                    if q <= 1:
                        continue
                    t0c = int(round(frac * q)) if frac is not None else q // 2
                    t0c = min(max(1, t0c), q - 1)
                t0_by_label[lab] = t0c
            if not t0_by_label:
                continue

            with torch.no_grad():
                out = model(input_ids, query_pos, return_all=True)
                logits = out["logits"][0]
                x_q = out["x_q"][0]
                h_last = out["h_last"][0]
                x1_row = out["layer_inputs"][1][0]
            alt = _runner_up(logits, target)
            S_clean = float(logits[target].item() - logits[alt].item())

            for lab, t0 in t0_by_label.items():
                res = results[lab]
                h_mid = h_last[t0].clone()
                res.h_norms.append(float(h_mid.norm().item()))
                res.margins_clean.append(S_clean)
                res.correct.append(bool(int(logits.argmax().item()) == target))


                h_leaf = h_mid.detach().clone().requires_grad_(True)
                hq_p = model.tail_scan_to_q(x1_row, t0, h_leaf.unsqueeze(0), q)
                lp = model.logits_from_final_state(hq_p, x_q.unsqueeze(0))[0]
                S_p = lp[target] - lp[alt]
                g = torch.autograd.grad(S_p, h_leaf)[0].detach()
                if model.sphere:
                    g_tan = g - float(g @ h_mid) * h_mid
                else:
                    g_tan = g

                U = _unit_random(n_random_dirs, model.k, gen, self.device).cpu()
                for theta in thetas:
                    if model.sphere:
                        Ut = _tangent_project(U.to(self.device), h_mid)
                        Hp = math.cos(theta) * h_mid.unsqueeze(0).expand_as(Ut) + math.sin(theta) * Ut
                        delta = (Hp - h_mid.unsqueeze(0)).cpu()
                    else:
                        R = float(h_mid.norm().item())
                        U_d = U.to(self.device)
                        Hp = h_mid.unsqueeze(0).expand_as(U_d) + theta * R * U_d
                        delta = (theta * R * U_d).cpu()
                    dS_pred = (delta @ g_tan.cpu()).numpy()

                    with torch.no_grad():
                        hq2 = model.tail_scan_to_q(x1_row, t0, Hp, q)
                        lp2 = model.logits_from_final_state(hq2, x_q.unsqueeze(0).expand(len(Hp), -1))
                    S_act = (lp2[:, target] - lp2[:, alt]).cpu().numpy() - S_clean
                    res.dS_pred[theta] = np.concatenate([res.dS_pred.get(theta, []), dS_pred]) if theta in res.dS_pred else dS_pred.copy()
                    res.dS_act[theta] = np.concatenate([res.dS_act.get(theta, []), S_act]) if theta in res.dS_act else S_act.copy()

        return results


def radius_of_validity(res: FaithResult, threshold: float = 0.5) -> dict[float, bool]:
    out = {}
    for th in sorted(res.thetas):
        m = res.metrics(th)
        out[th] = bool(m["E_all"] <= threshold) if np.isfinite(m["E_all"]) else False
    return out
