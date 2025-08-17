from torch import nn
import torch
from .multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, d_model), mask: (B, L, L)
        attn_out = self.self_attn(x, x, x, mask=attn_mask)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class MiniDecoderLM(nn.Module):
    """A minimal single-block decoder LM with tied embeddings option.
    Requires external embedding module for input IDs -> embeddings.
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int, vocab_size: int):
        super().__init__()
        self.block = DecoderBlock(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, d_model)
        h = self.block(x, attn_mask=attn_mask)
        logits = self.lm_head(h)  # (B, L, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, embed_fn, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int, device=None):
        """
        Greedy decoding.
        - embed_fn: function(ids: (B, L) -> (B, L, d_model))
        - input_ids: starting ids (B, L)
        - returns: ids (B, L + T)
        """
        device = device or input_ids.device
        ids = input_ids.to(device)
        for _ in range(max_new_tokens):
            B, L = ids.shape
            # Build causal mask (B, L, L): True where allowed
            mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device)).unsqueeze(0).expand(B, -1, -1)
            x = embed_fn(ids)
            logits = self.forward(x, attn_mask=mask)
            next_token_logits = logits[:, -1, :]  # (B, vocab)
            next_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B,1)
            ids = torch.cat([ids, next_ids], dim=1)
            if eos_id is not None:
                # stop early if all batches hit EOS at the last position
                if torch.all(next_ids.squeeze(-1) == eos_id):
                    break
        return ids
