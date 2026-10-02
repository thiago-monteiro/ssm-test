from argparse import Namespace

import pytest
import torch

from scripts.run_t4_mamba_normopt import batch, load_model, loss_and_accuracy, prepare_model
from src.large_mamba.data import TokenPools


@pytest.mark.parametrize("geometry", ["plain", "normW"])
def test_fp16_backbone_fp32_matrices_and_answer_only_loss(geometry):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = load_model(smoke=True).to(dtype=torch.float16)
    frozen = model.backbone.layers[0].mixer.out_proj.weight.detach().clone()
    names, params = prepare_model(model, layers=1, geometry=geometry)
    assert names == ["backbone.layers.1.mixer.out_proj.weight"]
    assert params[0].dtype == torch.float32
    pools = TokenPools(tuple(range(4, 36)), tuple(range(36, 68)), tuple(range(68, 100)), 2, 3)
    args = Namespace(micro_batch_size=1, sequence_length=128, associations=4)
    inputs, labels = batch(pools, args, 30007, 0, torch.device("cpu"))
    loss, _ = loss_and_accuracy(model, inputs, labels)
    reference = model(inputs, labels=labels, use_cache=False).loss
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert params[0].grad is not None
    assert torch.isfinite(params[0].grad).all()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    torch.testing.assert_close(model.backbone.layers[0].mixer.out_proj.weight, frozen)
    if geometry == "normW":
        torch.testing.assert_close(params[0].norm(dim=1), torch.ones(16))
