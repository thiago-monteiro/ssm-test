
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expA.models import Autoencoder
from src.expA.train import train_ae
from src.expB.data import make_associative_batch, make_batch
from src.expB.ssm import VALID_MODES, DiagonalSSM
from src.quantize import calibrate_ranges, quantize_uniform
from src.snr import (
    effective_snr,
    mean_pairwise_corr,
    row_normalize_,
    train_decode_probe,
    variance_fraction,
)


def test_expA_trains() -> None:
    model, meta = train_ae(seed=0, normalized=False, steps=100, batch_size=64, log_every=0)
    assert meta["final_mse"] < 2.0, meta["final_mse"]
    hist = meta["history"]
    assert hist[-1] < hist[0] * 1.5 or hist[-1] < 1.0
    print("PASS: ExpA trains and MSE decreases/reasonable")


def test_expA_normalized_trains() -> None:
    model, meta = train_ae(seed=0, normalized=True, steps=100, batch_size=64, log_every=0)
    assert meta["final_mse"] < 2.0, meta["final_mse"]
    print("PASS: ExpA Normalized trains")


def test_expA_sphere_on_z() -> None:
    model, meta = train_ae(seed=0, normalized=True, steps=100, batch_size=64, log_every=0,
                           sphere_on_z_only=True)
    assert meta["final_mse"] < 2.0, meta["final_mse"]
    print("PASS: ExpA sphere_on_z trains")


def test_quantizer() -> None:
    z = torch.linspace(-1, 1, 100).view(50, 2)
    lo, hi = calibrate_ranges(z, mode="per_coord")
    zq = quantize_uniform(z, levels=2, lo=lo, hi=hi)
    for j in range(2):
        nuniq = len(torch.unique(zq[:, j]))
        assert nuniq == 2, nuniq
    print("PASS: Quantizer 2-level has exactly 2 values per dim")


def test_ssm_shapes() -> None:
    for mode in ("B0", "BW", "BR", "BX", "B0_noshort", "BR_noshort", "sphere_on_z"):
        model = DiagonalSSM(V=16, d_model=64, k=128, mode=mode)
        batch = make_batch(8, L=32, V=16)
        out = model(batch["input_ids"], batch["query_pos"], return_states=True)
        assert out["logits"].shape == (8, 16), f"{mode}: {out['logits'].shape}"
        if "states" in out:
            assert out["states"].shape == (8, 32, 128), f"{mode}: {out['states'].shape}"
        print(f"  {mode}: shapes OK", end="")
    print("\nPASS: All SSM modes forward shapes")


def test_bw_row_norm() -> None:
    for mode in ("BW", "BW_BR"):
        model = DiagonalSSM(V=16, d_model=32, k=32, mode=mode)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = make_batch(4, L=16)
        out = model(batch["input_ids"], batch["query_pos"])
        loss = torch.nn.functional.cross_entropy(out["logits"], batch["target"])
        loss.backward()
        opt.step()
        model.row_normalize_weights_()
        for i, W in enumerate(model.B):
            norms = W.norm(dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (mode, "B", i, norms[:3])
        for i, W in enumerate(model.C):
            norms = W.norm(dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (mode, "C", i, norms[:3])
        print(f"PASS: {mode} row norms == 1 after step")


def test_bx_force_h() -> None:
    model = DiagonalSSM(V=16, d_model=32, k=32, mode="BX")
    batch = make_batch(4, L=16)
    out = model(batch["input_ids"], batch["query_pos"], return_states=True)
    states = out["states"]
    norms = states.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"BX norms: {norms.mean().item()}"
    print("PASS: BX forces unit norm every step")
def test_br_no_force_h() -> None:
    model = DiagonalSSM(V=16, d_model=32, k=32, mode="BR")
    batch = make_batch(4, L=16)
    out = model(batch["input_ids"], batch["query_pos"], return_states=True)
    norms = out["states"].norm(dim=-1)
    assert norms.std() > 1e-6 or (norms - 1.0).abs().mean() > 1e-3
    p = model.probe_products(out["h_final"])
    assert p.shape == out["h_final"].shape
    print("PASS: BR leaves recurrent h unconstrained")
def test_noshort_modes() -> None:
    for mode in ("B0_noshort", "BR_noshort"):
        model = DiagonalSSM(V=16, d_model=32, k=32, mode=mode)
        batch = make_batch(4, L=16)
        out = model(batch["input_ids"], batch["query_pos"])
        assert out["logits"].shape == (4, 16), mode
        print(f"PASS: {mode} forward OK")


def test_snr_api() -> None:
    P = torch.randn(256, 8)
    r = mean_pairwise_corr(P)
    assert -1.0 <= r <= 1.0
    P_noise = P + torch.randn_like(P) * 0.1
    esnr = effective_snr(P, P_noise)
    assert "snr_effective" in esnr and "alignment" in esnr
    vf = variance_fraction(P, P_noise, n_components=2)
    assert "signal_topvar_frac" in vf
    print("PASS: Enhanced SNR API")


def test_decode_probe() -> None:
    h = torch.randn(128, 32)
    labels = torch.randint(0, 8, (128,))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc, probe = train_decode_probe(h, labels, n_classes=8, device=device, steps=50)
    assert 0.0 <= acc <= 1.0
    print(f"PASS: Decode probe trains (acc={acc:.3f})")


def test_associative_batch() -> None:
    batch = make_associative_batch(8, L=16, V=16)
    assert batch["tokens"].shape == (8, 16)
    assert batch["query_key"].shape == (8,)
    assert batch["target"].shape == (8,)
    print("PASS: Associative batch shapes")


def test_no_replacement_batch() -> None:
    batch = make_batch(8, L=8, V=16, with_replacement=False)
    tokens = batch["tokens"]
    for i in range(8):
        assert len(torch.unique(tokens[i])) == 8, f"Row {i} has duplicates"
    print("PASS: No-replacement batch has unique tokens per row")


def test_enhanced_metrics() -> None:
    from src.expB.train import eval_position, train_ssm
    model, _ = train_ssm(seed=0, mode="B0", L=16, k=64, steps=100, batch_size=32,
                         log_every=0, eval_every=50)
    m = eval_position(model, seed=0, L=16, queries_per_pos=20,
                      do_intervention=True, do_decode_probe=True, do_task_os=True)
    assert len(m["acc_curve"]) == 16
    assert 0.0 <= m["overall_acc"] <= 1.0
    for key in ("decode_probe_acc", "intervention_drop", "task_conditioned_os",
                "snr_effective", "alignment", "signal_topvar_frac", "over_smoothing_readout"):
        assert key in m, f"Missing key: {key}"
    print(f"PASS: Enhanced metrics (overall={m['overall_acc']:.3f}, "
          f"probe={m.get('decode_probe_acc', 0):.3f})")
def test_device_auto() -> None:
    from src.expA.train import train_ae as train_ae_auto
    model, _ = train_ae_auto(seed=0, steps=10, batch_size=16, log_every=0)
    dev = next(model.parameters()).device
    expected = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert str(dev) == expected, f"Expected {expected}, got {dev}"
    print(f"PASS: Device auto-select ({dev})")


def main() -> None:
    test_quantizer()
    test_snr_api()
    test_decode_probe()
    test_ssm_shapes()
    test_bw_row_norm()
    test_br_no_force_h()
    test_bx_force_h()
    test_noshort_modes()
    test_associative_batch()
    test_no_replacement_batch()
    test_expA_trains()
    test_expA_normalized_trains()
    test_expA_sphere_on_z()
    test_enhanced_metrics()
    test_device_auto()
    print("\nAll acceptance tests passed.")


if __name__ == "__main__":
    main()
