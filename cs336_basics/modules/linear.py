import math

import einx
import torch
from jaxtyping import Float
from torch import nn

class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        INIT_MEAN = 0
        INIT_STD = math.sqrt(2 / (self.in_features + self.out_features))

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=INIT_MEAN, std=INIT_STD, a=(-3.0 * INIT_STD), b=(3.0 * INIT_STD))

    def forward(self, x: Float[torch.Tensor, "*batch d_in"]) -> Float[torch.Tensor, "*batch d_out"]:
        assert isinstance(self.weight, Float[torch.Tensor, "d_out d_in"])  # ty: ignore[invalid-argument-type]
        return einx.dot("out_feat [in_feat], ... [in_feat] -> ... out_feat", self.weight, x)
