import math

import einx
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt((1 / self.d_model) * einx.sum("... [d_model] -> ...", x.square()) + self.eps)
        rmsnorm_x = einx.divide("... d_model, ... -> ... d_model", x, rms_x) * self.weight

        return rmsnorm_x.to(in_dtype)
