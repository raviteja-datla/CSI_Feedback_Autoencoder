import numpy as np
import torch

from csinet.quantize import codeword_clip_scale, quantize_ste, uniform_quantize


def test_clip_scale_is_positive_and_scales_with_spread():
    rng = np.random.default_rng(0)
    narrow = rng.standard_normal((500, 64)).astype(np.float32) * 0.1
    wide = rng.standard_normal((500, 64)).astype(np.float32) * 10.0
    assert codeword_clip_scale(narrow) > 0
    assert codeword_clip_scale(wide) > codeword_clip_scale(narrow)


def test_high_bitwidth_is_near_lossless():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 64)).astype(np.float32)
    scale = codeword_clip_scale(x)
    q = uniform_quantize(x, scale, n_bits=16)
    np.testing.assert_allclose(q, x, atol=1e-2)


def test_more_bits_reduces_quantization_error():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((200, 64)).astype(np.float32)
    scale = codeword_clip_scale(x)
    err_1bit = np.mean((x - uniform_quantize(x, scale, n_bits=1)) ** 2)
    err_4bit = np.mean((x - uniform_quantize(x, scale, n_bits=4)) ** 2)
    err_8bit = np.mean((x - uniform_quantize(x, scale, n_bits=8)) ** 2)
    assert err_1bit > err_4bit > err_8bit


def test_clipping_bounds_output():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((200, 64)).astype(np.float32) * 20.0  # deliberately wide vs scale
    scale = 1.0
    q = uniform_quantize(x, scale, n_bits=4)
    assert q.min() >= -scale - 1e-6
    assert q.max() <= scale + 1e-6


def test_quantize_ste_forward_matches_numpy_uniform_quantize():
    rng = np.random.default_rng(4)
    x_np = rng.standard_normal((50, 32)).astype(np.float32)
    scale = codeword_clip_scale(x_np)
    q_np = uniform_quantize(x_np, scale, n_bits=3)

    q_torch = quantize_ste(torch.from_numpy(x_np), scale, n_bits=3)
    np.testing.assert_allclose(q_torch.numpy(), q_np, atol=1e-5)


def test_quantize_ste_backward_is_straight_through():
    x = torch.randn(20, 16, requires_grad=True)
    q = quantize_ste(x, clip_scale=2.0, n_bits=3)
    loss = q.sum()
    loss.backward()
    # straight-through: gradient of a sum w.r.t. each input element is exactly 1,
    # as if the quantization op were the identity function on the backward pass.
    assert torch.allclose(x.grad, torch.ones_like(x))
