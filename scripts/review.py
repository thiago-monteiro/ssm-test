from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification import classification_step
from src.expA.train import train_ae
from src.expB.data import make_batch
from src.expB.ssm import DiagonalSSM
from src.expB.train import eval_position, train_ssm
from src.expC.data import make_copy_batch
from src.expC.faith import FaithfulnessEvaluator, _runner_up
from src.expC.model import CopySSM
from src.expC.train import quick_accuracy
from src.normopt.normalize import tangentialize
from src.normopt.optim import (
    GraftedTangentMuon,
    MuonMomentum,
    TangentRowMomentum,
)
from src.quantize import calibrate_ranges, quantize_uniform
from src.seed import seed_everything
from src.snr import train_decode_probe
from src.stats import paired_stats

CPU = torch.device("cpu")
M3 = ("B0", "BR", "BX_iso")
OUT = ROOT / "results" / "review_controls"


def gtm4(lrs=(0.02, 0.003), ns=5, steps=10, seeds=(0, 1, 2)):
    def make(name, p, lr):
        if name == "tangent-only":
            return TangentRowMomentum([p], lr=lr)
        if name == "graft-only":
            return GraftedTangentMuon([p], lr=lr, ns_steps=ns, tangential=False)
        if name == "full":
            return GraftedTangentMuon([p], lr=lr, ns_steps=ns, tangential=True)
        return MuonMomentum([p], lr=lr, ns_steps=ns)

    out: dict = {"matched_ns_steps": ns, "by_lr": {}}
    for lr in lrs:
        arms = {}
        for name in ("baseline", "tangent-only", "graft-only", "full"):
            drops, tfracs, errs = [], [], []
            for seed in seeds:
                w0 = F.normalize(torch.randn(16, 32, generator=torch.Generator().manual_seed(1000 + seed)), dim=1)
                wt = torch.randn(16, 32, generator=torch.Generator().manual_seed(2000 + seed))
                x = torch.randn(256, 32, generator=torch.Generator().manual_seed(3000 + seed))
                y = x @ wt.T
                w = w0.clone().requires_grad_(True)
                opt = make(name, w, lr)
                l0 = float(F.mse_loss(x @ w.detach().T, y))
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    F.mse_loss(x @ w.T, y).backward()
                    before = w.detach().clone()
                    opt.step()
                    with torch.no_grad():
                        u = (before - w.detach().clone()) / lr
                        wht = before / (before.norm(dim=1, keepdim=True) + 1e-8)
                        rad = ((u - tangentialize(u, before)) * wht).sum(dim=1)
                        tan = tangentialize(u, before)
                        tfracs.append(float(tan.norm() ** 2 / max(u.norm() ** 2, 1e-24)))
                        meas = torch.acos((F.normalize(before, dim=1) * F.normalize(w.detach(), dim=1)).sum(dim=1).clamp(-1, 1))
                        errs.append(float((meas - torch.atan2(lr * tan.norm(dim=1), 1.0 - lr * rad)).abs().max()))
                        w.copy_(F.normalize(w.detach(), dim=1))
                drops.append(l0 - float(F.mse_loss(x @ w.detach().T, y)))
            arms[name] = {"mean_drop": float(np.mean(drops)), "drops": drops,
                          "tan_frac_mean": float(np.mean(tfracs)), "theta_err_max": float(np.max(errs))}
        arms["verdict"] = {"tangent_effect": arms["tangent-only"]["mean_drop"] - arms["baseline"]["mean_drop"],
                           "graft_effect": arms["graft-only"]["mean_drop"] - arms["baseline"]["mean_drop"],
                           "full_vs_best_single": arms["full"]["mean_drop"] - max(arms["tangent-only"]["mean_drop"], arms["graft-only"]["mean_drop"])}
        out["by_lr"][str(lr)] = arms
    return out


def bottleneck(seeds=(0, 1), steps=300):
    conds = {"plain": dict(normalized=False),
             "latent-only": dict(normalized=True, sphere_on_z_only=True),
             "weight-only": dict(normalized=True, latent_sphere=False),
             "full": dict(normalized=True)}
    rows = []
    for cname, kw in conds.items():
        for seed in seeds:
            m, meta = train_ae(seed=seed, steps=steps, device="cpu", log_every=0, **kw)
            m.eval()
            with torch.no_grad():
                z_cal = m.encode(torch.randn(1024, 64))
                x = torch.randn(1024, 64)
                z = m.encode(x)
                r = {"cond": cname, "seed": seed, "fp32_mse": meta["final_mse"]}
                for cal in ("per_coord", "global"):
                    lo, hi = calibrate_ranges(z_cal, mode=cal)
                    r[f"mse_2level_{cal}"] = float(F.mse_loss(m.decode(quantize_uniform(z, levels=2, lo=lo, hi=hi)), x).item())
                rows.append(r)
    out = {"rows": rows, "ci": {}}
    for col in ("fp32_mse", "mse_2level_per_coord", "mse_2level_global"):
        g = {c: [r[col] for r in rows if r["cond"] == c] for c in conds}
        out["ci"][col] = {f"full_vs_{c}": paired_stats(g["full"], g[c]) for c in conds if c != "full"}
    return out


def heldout(m, L):
    with torch.no_grad():
        b = make_batch(512, L, device="cpu")
        h = m(b["input_ids"], b["query_pos"])["h_final"]
    n = h.shape[0] // 2
    _, pr = train_decode_probe(h[:n], b["target"][:n], n_classes=m.V, device=CPU, steps=100)
    with torch.no_grad():
        return float((pr(h[n:].float()).argmax(-1) == b["target"][n:]).float().mean().item())


def bx_states(seeds, steps, L=16, k=32):
    mets, probes = {}, {}
    for mode in ("B0", "BX", "BX_iso"):
        ms, accs, hops = [], [], []
        for seed in seeds:
            m, meta = train_ssm(seed=seed, mode=mode, L=L, k=k, steps=steps, d_model=32, device="cpu", log_every=0)
            ev = eval_position(m, seed=seed, L=L, queries_per_pos=20, device="cpu",
                               do_intervention=False, do_decode_probe=False, do_task_os=False)
            ms.append(meta["final_acc"])
            accs.append(ev["overall_acc"])
            hops.append(heldout(m, L))
        mets[mode] = ms
        probes[mode] = {"overall_acc": accs, "heldout_probe": hops}
    return {"train_acc": mets, "eval": probes,
            "ci_overall_BX_vs_B0": paired_stats(probes["BX"]["overall_acc"], probes["B0"]["overall_acc"]),
            "ci_overall_BXiso_vs_B0": paired_stats(probes["BX_iso"]["overall_acc"], probes["B0"]["overall_acc"]),
            "ci_overall_BXiso_vs_BX": paired_stats(probes["BX_iso"]["overall_acc"], probes["BX"]["overall_acc"]),
            "ci_probe_BXiso_vs_BX": paired_stats(probes["BX_iso"]["heldout_probe"], probes["BX"]["heldout_probe"])}


def multiquery(models, L=16, n_seq=256, queries=(1, 2, 4)):
    out = {}
    for mode, m in models.items():
        m.eval()
        row = {}
        for Mm in queries:
            with torch.no_grad():
                toks = torch.randint(0, 16, (n_seq, L))
                qs = torch.stack([torch.randperm(L)[:Mm] for _ in range(n_seq)])
                ids = torch.cat([toks, torch.full((n_seq, 1), 16)], dim=1)
                hits = []
                for j in range(Mm):
                    pred = m(ids, qs[:, j])["logits"].argmax(-1)
                    hits.append(int((pred == toks[torch.arange(n_seq), qs[:, j]]).sum()))
                row[str(Mm)] = sum(hits) / (Mm * n_seq)
        out[mode] = row
    return out


def ckpt(seed, variant):
    m = CopySSM(V=16, L=16, d_model=64, k=64, sphere=(variant == "sphere"))
    m.load_state_dict(torch.load(ROOT / "results" / "tangent_geometry" / f"ckpt_seed{seed}_{variant}.pt", map_location="cpu", weights_only=True))
    return m.eval()


def curve(thetas=(0.05, 0.1, 0.2, 0.4), seeds=(0, 1, 2), n_seq=32):
    out = {}
    for variant in ("ordinary", "sphere"):
        for th in thetas:
            e0 = []
            for seed in seeds:
                fr = FaithfulnessEvaluator(ckpt(seed, variant), device="cpu").eval_final_state(
                    n_seq=n_seq, L=16, V=16, seed=seed, thetas=[th], n_random_dirs=16, use_basis=False)
                e0.append(fr.metrics(th)["E_all"])
            out[f"{variant}@{th}"] = {"E_all_per_seed": e0, "mean": float(np.mean(e0))}
    prev = json.loads((ROOT / "results" / "tangent_geometry" / "method.json").read_text())
    return {"E_all": out, "method_json_complete": {v: all(str(t) in prev["T2 prediction"][v] for t in thetas) for v in ("ordinary", "sphere")},
            "method_json_T2": prev["T2 prediction"]}


def fit(rows):
    rows = [(d, g) for d, g in rows if g > 0]
    if len(rows) < 4:
        return float("nan")
    d = np.array([r[0] for r in rows])
    return float(np.exp(np.polyfit(d, np.log(np.array([r[1] for r in rows])), 1)[0]))


def damping(seeds=(0, 1, 2), n_seq=8):
    out: dict = {}
    for variant in ("ordinary", "sphere"):
        dall, dmatch, cfro, marg, acc, nbc = [], [], [], [], [], []
        for seed in seeds:
            mo = ckpt(seed, "ordinary").to(CPU)
            ms = ckpt(seed, "sphere").to(CPU)
            mm = mo if variant == "ordinary" else ms
            acc.append(quick_accuracy(mm, L=16, V=16, device=CPU, n=512))
            with torch.no_grad():
                cfro.append(float(mm.C[mm.n_layers - 1].detach().norm().item()))
            gen = torch.Generator().manual_seed(seed + 99_000)
            batches, tries = [], 0
            while len(batches) < n_seq and tries < n_seq * 10:
                tries += 1
                b = make_copy_batch(1, 16, V=16, device=CPU, generator=gen)
                if int(b["query_pos"][0].item()) >= 4:
                    batches.append(b)
            ctxs: dict[str, list] = {}
            oks: dict[str, list] = {}
            for tag, m2 in (("o", mo), ("s", ms)):
                for b in batches:
                    with torch.no_grad():
                        o2 = m2(b["input_ids"], b["query_pos"], return_all=True)
                        lg = o2["logits"][0]
                        tg = int(b["target"][0].item())
                    oks.setdefault(tag, []).append(int(lg.argmax().item()) == tg)
                    ctxs.setdefault(tag, []).append((o2, lg, tg, b))
            both = [i for i in range(len(batches)) if oks["o"][i] and oks["s"][i]]
            nbc.append(len(both))
            tag = "s" if variant == "sphere" else "o"
            for idxset, dest in ((list(range(len(batches))), dall), (both, dmatch)):
                rows = []
                for i in idxset:
                    o2, lg, tg, b = ctxs[tag][i]
                    q = int(b["query_pos"][0].item())
                    ai = _runner_up(lg, tg)
                    if idxset is not both:
                        marg.append(float(lg[tg].item() - lg[ai].item()))
                    for t0 in range(1, q):
                        leaf = o2["h_last"][0][t0].detach().clone().requires_grad_(True)
                        hq = mm.tail_scan_to_q(o2["layer_inputs"][1][0], t0, leaf.unsqueeze(0), q)
                        lp = mm.logits_from_final_state(hq, o2["x_q"][0].unsqueeze(0))[0]
                        g = torch.autograd.grad(lp[tg] - lp[ai], leaf)[0]
                        rows.append((q - t0, float(g.norm().item())))
                dest.append(fit(rows))
        out[variant] = {"acc": acc, "C_fro": cfro, "margin_med": float(np.median(marg)),
                        "damping_all": dall, "damping_both_correct": dmatch, "n_both_correct_per_seed": nbc}
    out["ci_damping_matched"] = paired_stats(out["ordinary"]["damping_both_correct"], out["sphere"]["damping_both_correct"])
    return out


def logs():
    r = ROOT / "results"
    ids = ["t4_mamba_cpu_smoke/seed0/muon/summary.json", "modal_mamba_rmo/muon/summary.json",
           "tangent_geometry/summary.json", "tangent_geometry/method.json"]
    pts = sorted(str(p.relative_to(ROOT)) for p in r.rglob("checkpoint-*.pt"))
    out: dict = {"large_mamba_dir_exists": (r / "large_mamba").exists(),
                 "t4_dirs": sorted(p.name for p in r.glob("t4_*") if p.is_dir()),
                 "modal_dirs": sorted(p.name for p in r.glob("modal_*") if p.is_dir()),
                 "imported_dirs": sorted(p.name for p in r.glob("imported_*") if p.is_dir()),
                 "checkpoint_pts": pts[:20],
                 "n_checkpoint_pts": len(pts)}
    for c in ids:
        p = r / c
        if not p.exists():
            out[c] = "missing"
            continue
        try:
            out[c] = str(json.loads(p.read_text()))[:600]
        except Exception as e:
            out[c] = f"unparseable: {e}"
    return out


def bind_batch(n, K, device, generator=None):
    keys = torch.randint(0, 8, (n, K), device=device, generator=generator)
    vals = torch.randint(8, 16, (n, K), device=device, generator=generator)
    toks = torch.empty(n, 2 * K + 1, device=device, dtype=torch.long)
    toks[:, 0:2 * K:2] = keys
    toks[:, 1:2 * K + 1:2] = vals
    qi = torch.randint(0, K, (n,), device=device, generator=generator)
    toks[:, 2 * K] = keys[torch.arange(n, device=device), qi]
    return {"input_ids": toks, "query_pos": torch.full((n,), 2 * K, device=device),
            "target": vals[torch.arange(n, device=device), qi]}


def bind_train(seed, mode, K, steps=800):
    seed_everything(seed)
    m = DiagonalSSM(V=16, L_max=256, d_model=32, k=32, mode=mode).to(CPU)
    opt = torch.optim.AdamW(m.parameters(), lr=0.002, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    m.train()
    for _ in range(steps):
        classification_step(m, bind_batch(64, K, CPU), opt, sched, 1.0)
        m.row_normalize_weights_()
    m.eval()
    with torch.no_grad():
        b = bind_batch(512, K, CPU)
        return float((m(b["input_ids"], b["query_pos"])["logits"].argmax(-1) == b["target"]).float().mean())


def cmd_controls(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    q = {} if a.full else {"opt_steps": 6, "steps": 200, "ssm_steps": 800}
    s1 = gtm4(opt_steps=q.get("opt_steps", 10))
    print("1:", {lr: {k: round(v["mean_drop"], 4) for k, v in arms.items() if k != "verdict"} for lr, arms in s1["by_lr"].items()}, flush=True)
    s2 = bottleneck(steps=q.get("steps", 300))
    print("2:", {r["cond"]: round(r["fp32_mse"], 4) for r in s2["rows"] if r["seed"] == 0}, flush=True)
    s3 = bx_states((0, 1), q.get("ssm_steps", q.get("steps", 300)))
    print("3:", {m: [round(x, 3) for x in v["overall_acc"]] for m, v in s3["eval"].items()}, flush=True)
    models = {}
    for mode in ("B0", "BR", "BX_iso"):
        m, _ = train_ssm(seed=0, mode=mode, L=16, k=32, steps=q.get("ssm_steps", q.get("steps", 300)), d_model=32, device="cpu", log_every=0)
        models[mode] = m
    s4 = multiquery(models)
    print("4:", s4, flush=True)
    s5 = curve(n_seq=16 if not a.full else 32)
    print("5:", s5["method_json_complete"], flush=True)
    s6 = damping(n_seq=6 if not a.full else 8)
    print("6:", {v: s6[v]["damping_both_correct"] for v in ("ordinary", "sphere")}, flush=True)
    s7 = logs()
    print("7:", s7["large_mamba_dir_exists"], s7["n_checkpoint_pts"], flush=True)
    s = {"s1_gmuon4": s1, "s2_bottleneck": s2, "s3_bx": s3, "s4_multiquery": s4,
         "s5_curvature": s5, "s6_damping": s6, "s7_logs": s7, "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(f"Wrote {OUT / 'summary.json'} ({s['runtime_seconds']}s)", flush=True)


def cmd_followups(_a):
    t0 = time.time()
    bind = {}
    for mode in M3:
        for K in (2, 4, 8):
            accs = [bind_train(s, mode, K) for s in (0, 1)]
            bind[f"{mode}/K={K}"] = accs
            print(f"bind {mode} K={K}: {accs}", flush=True)
    bind["ci_BR_vs_BXiso"] = {f"K={K}": paired_stats(bind[f"BR/K={K}"], bind[f"BX_iso/K={K}"]) for K in (2, 4, 8)}
    prev = json.loads((OUT / "summary.json").read_text())["s3_bx"]
    ext: dict = {"overall_acc": {}, "heldout_probe": {}, "train_acc": {}}
    for mode in ("B0", "BX", "BX_iso"):
        for seed in (2, 3):
            m, meta = train_ssm(seed=seed, mode=mode, L=16, k=32, steps=800, d_model=32, device="cpu", log_every=0)
            ev = eval_position(m, seed=seed, L=16, queries_per_pos=20, device="cpu",
                               do_intervention=False, do_decode_probe=False, do_task_os=False)
            ext["overall_acc"].setdefault(mode, []).append(ev["overall_acc"])
            ext["heldout_probe"].setdefault(mode, []).append(heldout(m, 16))
            ext["train_acc"].setdefault(mode, []).append(meta["final_acc"])
            print(f"ext {mode} seed={seed} acc={ev['overall_acc']:.3f}", flush=True)
    n4 = {k: {m: prev["eval"][m][k] + ext[k][m] for m in ("B0", "BX", "BX_iso")} for k in ("overall_acc", "heldout_probe")}
    n4["ci_probe_BXiso_vs_BX_n4"] = paired_stats(n4["heldout_probe"]["BX_iso"], n4["heldout_probe"]["BX"])
    n4["ci_overall_BXiso_vs_BX_n4"] = paired_stats(n4["overall_acc"]["BX_iso"], n4["overall_acc"]["BX"])
    b0long = {}
    for seed in (0, 1, 2, 3):
        m, meta = train_ssm(seed=seed, mode="B0", L=16, k=32, steps=4000, d_model=32, device="cpu", log_every=0)
        ev = eval_position(m, seed=seed, L=16, queries_per_pos=20, device="cpu",
                           do_intervention=False, do_decode_probe=False, do_task_os=False)
        b0long[str(seed)] = {"train": meta["final_acc"], "overall": ev["overall_acc"]}
        print(f"B0-long seed={seed}: {meta['final_acc']:.3f} / {ev['overall_acc']:.3f}", flush=True)
    out = {"binding": bind, "probes_n4": n4, "B0_long": b0long,
           "bottleneck_n4": bottleneck(seeds=(0, 1, 2, 3), steps=200), "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / "followups.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"Wrote {OUT / 'followups.json'} ({out['runtime_seconds']}s)", flush=True)


def cmd_recall(a):
    t0 = time.time()
    res: dict = {}
    for mode in M3:
        acc, mq = [], []
        for seed in range(a.seeds):
            m, meta = train_ssm(seed=seed, mode=mode, L=a.L, k=a.k, steps=a.steps, d_model=a.d, device="cpu", log_every=0)
            acc.append(round(meta["final_acc"], 4))
            mq.append({k: round(v, 4) for k, v in multiquery({mode: m}, L=a.L, queries=(1, 2, 4, 8))[mode].items()})
            print(f"{mode} seed={seed} train={acc[-1]} mq={mq[-1]}", flush=True)
        res[mode] = {"train_acc": acc, "multiquery": mq}
    res["settings"] = {"L": a.L, "k": a.k, "d_model": a.d, "steps": a.steps,
                       "queries": [1, 2, 4, 8], "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / a.out).write_text(json.dumps(res, indent=1))
    print(f"Wrote {OUT / a.out} ({res['settings']['runtime_seconds']}s)", flush=True)


def cmd_settle(_a):
    out: dict = {"acc": {}}
    for mode in M3:
        for K in (8, 12, 16):
            accs = []
            for s in (0, 1, 2, 3):
                x = bind_train(s, mode, K)
                accs.append(x)
                print(f"bind {mode} K={K} seed={s}: {x:.4f}", flush=True)
            out["acc"][f"{mode}/K={K}"] = accs
    out["ci"] = {}
    for K in (8, 12, 16):
        o = out["acc"][f"BX_iso/K={K}"]
        out["ci"][f"K={K}/BXiso_vs_BR"] = paired_stats(o, out["acc"][f"BR/K={K}"])
        out["ci"][f"K={K}/BXiso_vs_B0"] = paired_stats(o, out["acc"][f"B0/K={K}"])
    (OUT / "binding_settle.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"Wrote {OUT / 'binding_settle.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("controls")
    p.add_argument("--full", action="store_true")
    sub.add_parser("followups")
    p = sub.add_parser("recall")
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--out", default="hard_recall.json")
    sub.add_parser("settle")
    a = ap.parse_args()
    {"controls": cmd_controls, "followups": cmd_followups, "recall": cmd_recall, "settle": cmd_settle}[a.cmd](a)


if __name__ == "__main__":
    main()
