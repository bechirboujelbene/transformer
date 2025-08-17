from torch import nn
import torch
from ..network import SCORE_SAVER

class ScaledDotAttention(nn.Module):

    def __init__(self,
                 d_k):
        """

        Args:
            d_k: Dimension of Keys and Queries
            dropout: Dropout probability
        """
        super().__init__()
        self.d_k = d_k

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the scaled dot attention given query, key and value inputs. Stores the scores in SCORE_SAVER for
        visualization

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional boolean mask where True denotes allowed positions. Shape should be broadcastable to
                  the attention scores shape (..., sequence_length_queries, sequence_length_keys).

        Shape:
            - q: (*, sequence_length_queries, d_model)
            - k: (*, sequence_length_keys, d_model)
            - v: (*, sequence_length_keys, d_model)
            - outputs: (*, sequence_length_queries, d_v)
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=q.dtype, device=q.device))
        # Apply mask (if provided) BEFORE softmax. Mask should be True for valid positions.
        if mask is not None:
            if mask.dim() == scores.dim() - 1:  # e.g., (B, Lq, Lk)
                mask = mask.unsqueeze(1)        # -> (B, 1, Lq, Lk)
            scores = scores.masked_fill(~mask, float('-inf'))
        scores = self.softmax(scores)
        outputs = torch.matmul(scores, v)

        SCORE_SAVER.save(scores)

        return outputs
