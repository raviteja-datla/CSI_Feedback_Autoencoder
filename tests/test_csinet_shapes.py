import pytest
import torch

from csinet.models.csinet import CsiNet

CR_TO_M = {0.25: 512, 0.0625: 128, 0.03125: 64, 0.015625: 32}


@pytest.mark.parametrize("m", list(CR_TO_M.values()))
def test_forward_shapes(m):
    model = CsiNet(m)
    x = torch.randn(4, 2, 32, 32)
    x_hat, codeword = model(x)
    assert x_hat.shape == (4, 2, 32, 32)
    assert codeword.shape == (4, m)
    assert torch.isfinite(x_hat).all()


@pytest.mark.parametrize("m", list(CR_TO_M.values()))
def test_gradients_flow_through_encoder_and_decoder(m):
    model = CsiNet(m)
    x = torch.rand(4, 2, 32, 32)
    x_hat, _ = model(x)
    loss = x_hat.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.any(p.grad != 0), f"all-zero grad for {name}"


def test_param_count_scales_with_m():
    counts = {m: sum(p.numel() for p in CsiNet(m).parameters()) for m in CR_TO_M.values()}
    ms = sorted(counts)
    for smaller, larger in zip(ms, ms[1:]):
        assert counts[smaller] < counts[larger]
