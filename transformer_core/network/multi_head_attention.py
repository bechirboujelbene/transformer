from torch import nn
import torch    

from ..network import ScaledDotAttention

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            dropout: Dropout probability
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.weights_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.attention = ScaledDotAttention(d_k)
        self.project = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional causal/padding mask (batch, Lq, Lk) or broadcastable

        Shape:
            - q: (batch_size, sequence_length_queries, d_model)
            - k: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_keys, d_model)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """

        # You will need these here!
        batch_size, sequence_length_queries, _ = q.size()
        _, sequence_length_keys, _ = k.size()

        q = self.weights_q(q)
        k = self.weights_k(k)
        v = self.weights_v(v)

        q = q.reshape(batch_size, sequence_length_queries, self.n_heads, self.d_k)
        k = k.reshape(batch_size, sequence_length_keys, self.n_heads, self.d_k)
        v = v.reshape(batch_size, sequence_length_keys, self.n_heads, self.d_v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)


        # Prepare mask for multi-head shape: (B, 1, Lq, Lk) or (B, H, Lq, Lk)
        attn_mask = None
        if mask is not None:
            if mask.dim() == 3:
                attn_mask = mask.unsqueeze(1)  # (B,1,Lq,Lk)
            else:
                attn_mask = mask
        outputs = self.attention(q, k, v, mask=attn_mask)
        outputs = outputs.transpose(1, 2)
        outputs = outputs.reshape(batch_size, sequence_length_queries, self.n_heads * self.d_v)

        outputs = self.project(outputs)



        return outputs
    