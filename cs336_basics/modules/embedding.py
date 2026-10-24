import einx
import torch
from jaxtyping import Float, Int
from torch import nn

class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        
        INIT_MEAN = 0
        INIT_STD = 1

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=INIT_MEAN, std=INIT_STD, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[torch.Tensor, "batch seq"]) -> Float[torch.Tensor, "batch seq d_model"]:
        return einx.get_at("[vocab] d_model, batch seq -> batch seq d_model", self.weight, token_ids)
