# Minimal Transformer Building Blocks (PyTorch)

This repository contains a **minimal yet fully-functional implementation of the fundamental components that make up the Transformer architecture** (Vaswani et al., 2017), built from scratch in PyTorch. It is designed to be compact, readable, and easy to extend.

All modules are implemented and covered by unit tests. You can reuse the components as a lightweight reference implementation.

---

## 1. Repository structure

```
├── transformer_core         # portfolio-friendly package namespace
│   ├── data                 # tokenizer utilities (BPE)
│   ├── network              # Transformer building blocks
│   └── util                 # helper functions (masks, parameter counting)
├── models/                  # (optional) folder for pretrained artefacts
├── README.md                # **<– you are here**
└── demo_transformer.py      # quick end-to-end usage example
```

### Workflow in a nutshell

1. **Tokenisation** – `BytePairTokenizer` implements Byte-Pair-Encoding (BPE) from scratch. It can either be
   • trained on raw text *(train from file)* or  
   • loaded from a pretrained JSON via `transformers.PreTrainedTokenizerFast`.
2. **Embedding layer** – `Embedding` maps token IDs to dense vectors and adds sinusoidal *positional encodings*.
3. **Scaled-Dot Attention** – computes attention weights between queries **Q** and keys **K** and applies them to values **V**.
4. **Multi-Head Attention** – performs the above operation in `n_heads` parallel sub-spaces and projects back to model dimensionality.
5. **Minimal Decoder LM** – `MiniDecoderLM` stacks a decoder block and a projection head for toy generation.
6. **Utilities & Tests** – helpers for causal masks, parameter counting, and an extensive pytest test-suite.

Together these parts form the core of the *Transformer* encoder/decoder stack; you can build higher-level components (feed-forward blocks, layer-norm, encoder-decoder bridge, etc.) on top.

---

## 2. What is included?

The repository implements the following components:

- Sinusoidal positional encoding and token embeddings
- Scaled-Dot Product Attention with optional masking
- Multi-Head Attention (projection-in, projection-out)
- A minimal Transformer Decoder block (self-attn + FFN + layer norms)
- A tiny language model head (`MiniDecoderLM`) with greedy generation

If you need a full, pretrained model you can plug these blocks into an existing architecture or directly rely on `torch.nn.Transformer` or HuggingFace `transformers`.

---

## 3. Installation & quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # minimal set to run code and tests
pytest -q                         # run unit tests 

python demo_transformer.py        # tiny demo script
```

---

## 4. Demo

The `demo_transformer.py` script shows a micro-workflow:

1. Load / train a BPE tokenizer
2. Convert a sample sentence → token IDs
3. Obtain embeddings incl. positional encoding
4. Run them through a single `MultiHeadAttention` layer with an explicit causal mask
5. Optionally, try the tiny `MiniDecoderLM` for toy next-token generation

Take a look and play with the hyper-parameters!

---

## 6. Repo hygiene and keeping the repo light

To keep the repository small and fast to clone, large or optional artefacts are ignored via `.gitignore`:

- `models/` (pretrained artefacts, checkpoints)
- `datasets/`
- `outputs/`, `checkpoints/`, `wandb/`
- build caches (`__pycache__/`, `.pytest_cache/`), notebook checkpoints

Unit tests depend on small tensor fixtures — these are kept so that `pytest` runs out-of-the-box. If you add large models or datasets, please store them outside the repo or use releases/LFS.

