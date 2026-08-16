"""CsiNet encoder/decoder architecture (Wen, Shi, et al., IEEE WCL 2018)."""

import torch
from torch import nn


class RefineNetUnit(nn.Module):
    """Residual refinement block used in the CsiNet decoder."""

    def __init__(self, channels: int = 2, leaky_slope: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(leaky_slope),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(leaky_slope),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(leaky_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + x)


class CsiNetEncoder(nn.Module):
    def __init__(self, m: int, leaky_slope: float = 0.3, width: tuple[int, int] = (8, 16)):
        super().__init__()
        # Deeper than the original paper's single conv layer: a linear PCA/SVD
        # baseline at the same M was found to beat this encoder at low compression
        # (CR=1/4, see README section 4) -- a single 2->2 conv layer gives the
        # encoder very little nonlinear capacity before the linear M-dim projection,
        # so it struggled to even match what a plain linear method could do. This
        # mirrors RefineNetUnit's channel widths (2->8->16->2 by default) to give the
        # encoder comparable feature-extraction depth to the decoder before its
        # bottleneck. `width` is exposed (rather than hardcoded) to let a specific
        # hard operating point (e.g. outdoor CR=1/4, see README) trade more compute
        # for more capacity without touching the default used everywhere else.
        w1, w2 = width
        self.conv = nn.Sequential(
            nn.Conv2d(2, w1, kernel_size=3, padding=1),
            nn.BatchNorm2d(w1),
            nn.LeakyReLU(leaky_slope),
            nn.Conv2d(w1, w2, kernel_size=3, padding=1),
            nn.BatchNorm2d(w2),
            nn.LeakyReLU(leaky_slope),
            nn.Conv2d(w2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(leaky_slope),
        )
        self.fc = nn.Linear(2 * 32 * 32, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class CsiNetDecoder(nn.Module):
    def __init__(self, m: int, n_refine: int = 2, leaky_slope: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(m, 2 * 32 * 32)
        self.refine = nn.Sequential(*[RefineNetUnit(2, leaky_slope) for _ in range(n_refine)])
        # No final Sigmoid: the paper's decoder ends in [0,1]-bounded Sigmoid because
        # its (COST2100) targets are pre-scaled into that range. Our synthetic
        # clustered channel is heavy-tailed (peak/RMS ~6-7x), so an RMS-based scale
        # (see transform.py) is used instead and left unbounded here -- a Sigmoid
        # would saturate/clip exactly the large, informative path entries.
        self.out_conv = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        h = h.view(-1, 2, 32, 32)
        h = self.refine(h)
        return self.out_conv(h)


class CsiNet(nn.Module):
    def __init__(self, m: int, n_refine: int = 2, leaky_slope: float = 0.3, encoder_width: tuple[int, int] = (8, 16)):
        super().__init__()
        self.m = m
        self.encoder = CsiNetEncoder(m, leaky_slope, width=encoder_width)
        self.decoder = CsiNetDecoder(m, n_refine, leaky_slope)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codeword = self.encoder(x)
        x_hat = self.decoder(codeword)
        return x_hat, codeword
