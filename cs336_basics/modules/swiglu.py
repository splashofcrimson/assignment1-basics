import math

import einx
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn

class SwiGLU(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        d_ff = d_ff if d_ff else math.round((8.0 / 3.0) / 64.0) * 64.0 * d_model

        INIT_MEAN = 0
        INIT_STD = math.sqrt(2 / (d_model + d_ff))
        
        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.w1_weight, mean=INIT_MEAN, std=INIT_STD, a=(-3.0 * INIT_STD), b=(3.0 * INIT_STD))
        nn.init.trunc_normal_(self.w2_weight, mean=INIT_MEAN, std=INIT_STD, a=(-3.0 * INIT_STD), b=(3.0 * INIT_STD))
        nn.init.trunc_normal_(self.w3_weight, mean=INIT_MEAN, std=INIT_STD, a=(-3.0 * INIT_STD), b=(3.0 * INIT_STD))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_one = einx.dot("d_ff [d_model], ... [d_model] -> ... d_ff", self.w1_weight, x)
        silu_out = nn.Sigmoid().forward(linear_one) * linear_one

        linear_three = einx.dot("d_ff [d_model], ... [d_model] -> ... d_ff", self.w3_weight, x)

        return einx.dot("d_model [d_ff], ... [d_ff] -> ... d_model", self.w2_weight, silu_out * linear_three)
