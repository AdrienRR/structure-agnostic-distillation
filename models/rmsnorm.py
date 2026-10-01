import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019), computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight
