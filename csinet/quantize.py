"""Uniform scalar quantization of the encoder's codeword, simulating digital CSI feedback.

Real feedback is a bitstring, not a continuous vector -- the UE must quantize the
codeword before transmission. This applies fixed-range uniform quantization (the
standard, simplest scheme) post-hoc to an already-trained model's codeword, to
measure how much reconstruction accuracy is lost as bit-width shrinks.
"""

import numpy as np
import torch


def codeword_clip_scale(codewords: np.ndarray, k: float = 4.0) -> float:
    """Global symmetric clip range for the codeword, derived from training-set statistics.

    A real system fixes its quantizer range in advance from known/expected codeword
    statistics, not per-sample -- so this is one scalar (k standard deviations) over
    the whole training set's codewords, not a per-sample or per-dimension range.
    """
    return float(k * np.std(codewords))


def uniform_quantize(codeword: np.ndarray, clip_scale: float, n_bits: int) -> np.ndarray:
    """Quantize-then-dequantize codeword values to 2**n_bits uniform levels within
    [-clip_scale, clip_scale], clipping outliers.

    Returns an array the same shape/dtype as the input that the decoder can consume
    directly -- simulates ideal, error-free digital transmission of the quantized
    levels (no channel bit errors modeled).
    """
    levels = 2**n_bits
    x = np.clip(codeword, -clip_scale, clip_scale)
    step = (2.0 * clip_scale) / (levels - 1)
    q = np.round((x + clip_scale) / step) * step - clip_scale
    return q.astype(codeword.dtype)


class _STEQuantize(torch.autograd.Function):
    """Forward: uniform_quantize's math in torch. Backward: straight-through estimator
    (gradient passes through unchanged) -- the quantization step function has zero
    gradient almost everywhere, so a real gradient can't flow through it; the standard
    trick used for quantization-aware training is to pretend it's the identity function
    on the backward pass, which lets the encoder/decoder still receive a training
    signal despite the forward pass being non-differentiable.
    """

    @staticmethod
    def forward(ctx, codeword: torch.Tensor, clip_scale: float, n_bits: int) -> torch.Tensor:
        levels = 2**n_bits
        x = torch.clamp(codeword, -clip_scale, clip_scale)
        step = (2.0 * clip_scale) / (levels - 1)
        return torch.round((x + clip_scale) / step) * step - clip_scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def quantize_ste(codeword: torch.Tensor, clip_scale: float, n_bits: int) -> torch.Tensor:
    """Differentiable (straight-through estimator) uniform quantization, for QAT."""
    return _STEQuantize.apply(codeword, clip_scale, n_bits)
