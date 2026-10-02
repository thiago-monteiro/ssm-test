import pytest
import torch

from src.normopt.optim import HybridOptimizer, MuonMomentum, OptimizerSpec, TangentRowMomentum


@pytest.mark.parametrize("optimizer_type", [MuonMomentum, TangentRowMomentum])
def test_momentum_accumulates_and_peek_preserves_buffer(optimizer_type):
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    optimizer = optimizer_type([parameter], momentum=0.5)
    parameter.grad = torch.ones_like(parameter)
    optimizer.compute_update(parameter)
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], torch.ones_like(parameter))
    parameter.grad = torch.full_like(parameter, 2.0)
    expected_peek = optimizer.peek_update(parameter)
    hybrid = HybridOptimizer(optimizer, None, ["matrix"], OptimizerSpec("rmo", 0.02))
    torch.testing.assert_close(hybrid.matrix_updates()["matrix"], expected_peek)
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], torch.ones_like(parameter))
    actual = optimizer.compute_update(parameter)
    torch.testing.assert_close(actual, expected_peek)
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], torch.full_like(parameter, 2.5))
