"""Minimal demonstration of the Transformer building blocks in this repository.

This script showcases a complete end-to-end pipeline:
1. Tokenization (BPE) → 2. Embeddings + Positional Encoding → 3. Multi-Head Attention
4. Causal Masking → 5. Mini Decoder → 6. Greedy Generation

All components are implemented from scratch in PyTorch. The generated text will be
garbled since this is an untrained model - the goal is to demonstrate that the
Transformer mechanics work correctly.

Run via: `python demo_transformer.py`
"""
from transformer_core.data import BytePairTokenizer, load_pretrained_fast
from transformer_core.network import Embedding, MultiHeadAttention, MiniDecoderLM
from transformer_core.util import create_causal_mask
import torch


def main():
    # ---------------------------------------------------------------------
    # 1. Tokenisation ------------------------------------------------------
    # ---------------------------------------------------------------------
    sample_sentence = "Transformers are changing the world of NLP!"

    # Try different tokenizer options in order of preference:
    # 1. Pretrained fast tokenizer (if available)
    # 2. Our from-scratch BytePairTokenizer (if pretrained data exists)
    # 3. Simple character-level fallback
    
    try:
        tokenizer = load_pretrained_fast()
        ids = tokenizer.encode(sample_sentence)
        vocab_size = tokenizer.vocab_size
        print("✓ Loaded pretrained fast tokenizer.")
    except Exception:
        try:
            # Try our from-scratch BPE implementation
            tokenizer = BytePairTokenizer.get_from_pretrained("models/pretrainedModels/")
            ids = tokenizer.encode(sample_sentence)
            vocab_size = len(tokenizer.vocab_encode)
            print("✓ Loaded from-scratch BytePairTokenizer.")
        except Exception:
            # Robust fallback – a minimal character-level tokenizer just for the demo
            class CharTokenizer:
                def __init__(self, text: str):
                    self.unk, self.sos, self.eos, self.pad = "<[UNK]>", "<[SOS]>", "<[EOS]>", "<[PAD]>"
                    chars = sorted(set(ch for ch in text))
                    self.vocab = [self.unk, self.sos, self.eos, self.pad] + chars
                    self.stoi = {t: i for i, t in enumerate(self.vocab)}
                    self.itos = {i: t for i, t in enumerate(self.vocab)}
                    self.vocab_size = len(self.vocab)
                    self.eos_token_id = self.stoi[self.eos]
                    self.sos_token_id = self.stoi[self.sos]
                def encode(self, s: str):
                    ids = [self.sos_token_id] + [self.stoi.get(ch, self.stoi[self.unk]) for ch in s] + [self.eos_token_id]
                    return ids
                def decode(self, ids):
                    # drop special tokens
                    toks = [self.itos[i] for i in ids if i in self.itos]
                    toks = [t for t in toks if t not in {self.unk, self.sos, self.eos, self.pad}]
                    return "".join(toks)
            tokenizer = CharTokenizer(sample_sentence)
            ids = tokenizer.encode(sample_sentence)
            vocab_size = tokenizer.vocab_size
            print("→ Using CharTokenizer fallback for the demo.")

    token_tensor = torch.tensor([ids])  # (batch=1, seq_len)
    seq_len = token_tensor.shape[-1]

    # ---------------------------------------------------------------------
    # 2. Embeddings --------------------------------------------------------
    # ---------------------------------------------------------------------
    d_model = 64
    embed_layer = Embedding(vocab_size=vocab_size, d_model=d_model, max_length=512)
    embeddings = embed_layer(token_tensor)  # (1, seq_len, d_model)
    print("Embeddings shape:", embeddings.shape)

    # ---------------------------------------------------------------------
    # 3. Self-Attention ----------------------------------------------------
    # ---------------------------------------------------------------------
    n_heads = 8
    d_k = d_model // n_heads
    d_v = d_model // n_heads

    mha = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)

    # causal mask for autoregressive decoding (not used further here)
    mask = create_causal_mask(seq_len)

    attn_output = mha(embeddings, embeddings, embeddings, mask=mask)  # masked self-attention
    print("Multi-Head Attention output shape:", attn_output.shape)
    print("✓ Causal masking applied - tokens can only attend to previous positions")

    # ---------------------------------------------------------------------
    # 4. Mini Decoder LM + Greedy Generation ------------------------------
    # ---------------------------------------------------------------------
    d_ff = 4 * d_model
    lm = MiniDecoderLM(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, vocab_size=vocab_size)

    # convenience embedding function for the LM generate loop
    def embed_fn(ids: torch.Tensor) -> torch.Tensor:
        return embed_layer(ids)

    # start from the same prompt and generate a few tokens
    max_new_tokens = 10
    try:
        eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
    except Exception:
        eos_id = None

    generated = lm.generate(embed_fn, token_tensor, max_new_tokens=max_new_tokens, eos_id=eos_id, device=embeddings.device)
    print("Generated token ids:", generated[0].tolist())

    # Try to decode
    decoded = None
    try:
        if hasattr(tokenizer, 'decode'):
            decoded = tokenizer.decode(generated[0].tolist())
    except Exception:
        pass
    if decoded:
        print("Decoded text:", decoded)
        print("\n💡 Note: The generated text after the original input is garbled because")
        print("   this is an untrained model - it demonstrates the generation mechanics work!")
    else:
        print("(No decoder available for these ids; provide a pretrained tokenizer to view text)")


if __name__ == "__main__":
    main()
