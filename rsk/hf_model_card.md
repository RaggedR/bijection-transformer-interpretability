---
language: en
license: mit
tags:
  - algebraic-combinatorics
  - rsk-correspondence
  - young-tableaux
  - permutations
  - semistandard-tableaux
  - reverse-plane-partitions
  - hillman-grassl
  - cylindric-plane-partitions
  - growth-diagrams
  - transformer
  - pytorch
datasets:
  - ACDRepo/robinson_schensted_knuth_correspondence_8
  - ACDRepo/robinson_schensted_knuth_correspondence_10
pipeline_tag: other
---

# RSK Transformer

A 1.2M-parameter transformer that learns **inverse combinatorial bijections** — the Robinson-Schensted-Knuth correspondence (permutations and matrices), the Hillman-Grassl correspondence (reverse plane partitions), and the cylindric growth diagram bijection (cylindric plane partitions). The same architecture handles all tasks without modification.

Achieves **100% exact-match accuracy** on held-out test data for permutations at n=10, **99.99%** at n=15 (1.3 trillion permutations), **100%** on reverse plane partitions, and **100%** on cylindric plane partitions — significantly improving on the [PNNL ML4AlgComb benchmark](https://github.com/pnnl/ML4AlgComb/tree/master/rsk).

**Embedding ablation** confirms 2D structure is essential: replacing row+column with 1D sequential position drops accuracy from 100% to 72%, and removing P-vs-Q tableau identity collapses it to 6%. **Mechanistic interpretability via sparse autoencoders** reveals two families of features for permutation RSK (insertion-order detectors and step-specific Q-entry locators) and, for cylindric plane partitions, features that fire on the specific partition triples where the Burge local rule operates — direct evidence the model has learned Fomin's growth diagram framework.

📄 **Paper**: [paper.pdf](paper.pdf)
💻 **Code**: [github.com/RaggedR/rsk-transformer](https://github.com/RaggedR/rsk-transformer)
📘 **Thesis**: [Langer (2013) — Cylindric plane partitions, Lambda determinants, Commutants in semicircular systems](https://arxiv.org/abs/2110.12629) — the mathematical foundation for the cylindric growth diagram bijection (§4.2–4.3) and generalized RSK via Fomin growth diagrams (§2.1–2.2)

## Results

### Experiment 1: Permutation RSK

Given a pair of standard Young tableaux (P, Q), predict the permutation σ.

| n | \|S_n\| | Training data | Test exact match | Best epoch |
|---|---------|--------------|-----------------|------------|
| 8 | 40,320 | 29,031 (72% of S_n) | 99.95% | 23 |
| 10 | 3,628,800 | 500,000 (14% of S_n) | **100.00%** | 28 |
| 15 | 1.3 × 10¹² | 500,000 (0.00004%) | **99.99%** | 52 |

The n=10 result rules out memorisation: a 1.2M-parameter model trained on 14% of the permutation space achieves perfect accuracy on 50,000 unseen test permutations. At n=15 (1.3 trillion permutations), training on 0.00004% of the space yields 99.99% — unambiguous algorithmic generalisation.

### Experiment 2: Full Matrix RSK

Given a pair of semistandard Young tableaux (P, Q) from Knuth's full RSK on non-negative integer matrices, recover the biword. **Same model architecture, same embedding** — only the task flag changes.

| Experiment | \|λ\| | Classes | Training data | Test exact match | Per-position | Best epoch |
|-----------|-------|---------|--------------|-----------------|-------------|------------|
| 3×3, N=10 | 10 | 3 | 500,000 | **100.00%** | **100.00%** | 18 |
| 4×4, N=20 | 20 | 4 | 500,000 | **99.32%** | **99.96%** | 20 |
| 5×5, N=30 | 30 | 5 | 2,000,000 | **96.79%** | **99.87%** | 16 |

Results are data-limited, not architecture-limited: per-position accuracy is 99.87%+ and exact-match gaps are explained by independent errors compounding across positions ((0.9987)^30 ≈ 96.2%). The space of 5×5 matrices with entry sum 30 is ~10¹⁴; 2M training samples covers ~10⁻⁸ of it. More data would likely improve results, but with limited computational resources (single Apple M4 Max laptop) we prioritised moving on to qualitatively new experiments (reverse plane partitions via Fomin growth diagrams).

### Ablation: Transformer vs MLP (Permutations)

| n | Model | Parameters | Greedy exact | Argmax exact | Per-position |
|---|-------|-----------|-------------|-------------|-------------|
| 10 | **RSKEncoder** (transformer) | 1,207,012 | **100.00%** | **100.00%** | **100.00%** |
| 10 | BaselineMLP (flat) | 133,604 | 95.67% | 90.31% | 98.85% |
| 15 | **RSKEncoder** (transformer) | 1,225,057 | **99.99%** | **99.98%** | **100.00%** |
| 15 | BaselineMLP (flat) | 133,604 | 3.07% | 0.04% | 62.02% |

The MLP collapses from 95.67% to 3.07% as n increases from 10 to 15, while the transformer barely notices (100% → 99.99%). Without spatial structure, the MLP cannot coordinate predictions across positions.

### Ablation: Embedding Components (n=10)

| Ablation | What changes | Greedy exact | Per-position |
|----------|-------------|-------------|-------------|
| Full model | — | **100.00%** | 100.00% |
| Drop row | No row embedding | **100.00%** | 100.00% |
| Drop column | No column embedding | **99.99%** | 100.00% |
| Concatenate | 4×32d concat, not 4×128d sum | **100.00%** | 100.00% |
| 1D position | Sequential pos replaces row+col | 72.38% | 91.34% |
| Drop tableau | No P-vs-Q identity | 5.79% | 58.49% |
| Drop row+col | No spatial info | 0.00% | 9.99% |

Tableau identity (P vs Q) is the single most critical component. Row and column are individually redundant (either alone suffices given the constrained tableau shape), but *some* spatial embedding is essential. 1D positional encoding recovers only 72% — 2D structure specifically is what enables learning.

### Experiment 3: Reverse Plane Partitions (Hillman-Grassl)

Given a reverse plane partition (RPP) of shape λ, recover the arbitrary filling via the inverse [Hillman-Grassl correspondence](https://en.wikipedia.org/wiki/Hillman%E2%80%93Grassl_correspondence). **Same model architecture** — the only change is that the input is a single filling (not a pair), so `tableau_emb(0)` is used for all tokens.

| Shape λ | Type | \|λ\| | Classes | Training data | Test exact match | Per-position | Best epoch |
|---------|------|-------|---------|--------------|-----------------|-------------|------------|
| (4,3,2,1) | Staircase | 10 | 5 | 500,000 | **100.00%** | **100.00%** | 23 |
| (6,4,2) | Wide | 12 | 5 | 500,000 | **99.99%** | **100.00%** | 17 |
| (2,2,2,2,2,1) | Tall | 11 | 5 | 500,000 | **99.99%** | **100.00%** | 36 |

The Hillman-Grassl bijection is fundamentally different from RSK — it involves zigzag paths through the Young diagram rather than Schensted insertion — yet the same transformer architecture learns it to near-perfect accuracy. Tall shapes converge slower (36 epochs vs 17-23) because longer zigzag paths create longer-range dependencies.

### Experiment 4: Cylindric Plane Partitions (Growth Diagrams)

Given a cylindric plane partition (CPP) with binary profile π, recover the base partition γ and the ALCD face labels via the inverse cylindric growth diagram bijection. This uses the **Burge local rule** applied recursively through a cylindric growth diagram, as described in [Langer (2013), §4.2–4.3](https://arxiv.org/abs/2110.12629). **Same model architecture**.

| Profile π | T | ALCD labels | Training data | Test exact match | Per-position | Best epoch |
|-----------|---|-------------|--------------|-----------------|-------------|------------|
| (1,0,1,0) | 4 | 3 | 500,000 | **100.00%** | **100.00%** | 2 |
| (1,0,1,0,0) | 5 | 5 | 500,000 | **100.00%** | **100.00%** | 7 |
| (1,0,1,0,1,0) | 6 | 6 | 500,000 | **100.00%** | **100.00%** | 3 |
| (1,0,1,0,1,0,1,0) | 8 | 10 | 500,000 | **99.98%** | **100.00%** | 9 |

The cylindric bijection is qualitatively different from all previous experiments: there is no direct closed-form algorithm. The bijection is defined implicitly by the Burge local rule applied at each face of the cylindric growth diagram. The model must learn to invert a recursive process (the 𝔏_i composition from [Langer 2013, §4.2](https://arxiv.org/abs/2110.12629)) that peels off one ALCD label at each step by solving a local Burge equation. Despite this complexity, the transformer achieves 100% on all tested profiles.

## Key Idea: Structured 2D Token Embeddings

Previous work encoded tableaux as flat bracket strings, destroying 2D geometry. We encode each tableau entry as a token with four learned embeddings:

```
embedding(entry) = value_emb(v) + row_emb(r) + col_emb(c) + tableau_emb(P or Q)
```

## Architecture

```
Input: (P, Q) as 2n structured tokens
  → TokenEmbedding (value + row + col + tableau_id)
  → 6-layer TransformerEncoder (d=128, 8 heads, pre-norm, GELU)
  → Mean pool over all 2n tokens
  → n parallel classification heads → logits (batch, n, n)
  → Masked greedy decode → predicted σ
```

~1.2M parameters. Encoder-only (no autoregressive decoding needed).

## Checkpoints

### Experiment 1: Permutation RSK

| File | Description | Parameters |
|------|-------------|-----------|
| `checkpoints/encoder_n8/best.pt` | RSKEncoder trained on S₈ (HuggingFace data) | 1,202,368 |
| `checkpoints/encoder_n10/best.pt` | RSKEncoder trained on S₁₀ (sampled) | 1,207,012 |
| `checkpoints/encoder_n15/best.pt` | RSKEncoder trained on S₁₅ (sampled) | 1,225,057 |
| `checkpoints/mlp_n10/best.pt` | Baseline MLP on S₁₀ (for ablation) | 133,604 |
| `checkpoints/mlp_n15/best.pt` | Baseline MLP on S₁₅ (for ablation) | 133,604 |

### Experiment 2: Full Matrix RSK

| File | Description | Parameters |
|------|-------------|-----------|
| `checkpoints/encoder_matrix_a3_b3_N10/best.pt` | RSKEncoder on 3×3 matrices, N=10 | ~1.2M |
| `checkpoints/encoder_matrix_a4_b4_N20/best.pt` | RSKEncoder on 4×4 matrices, N=20 | ~1.2M |
| `checkpoints/encoder_matrix_a5_b5_N30/best.pt` | RSKEncoder on 5×5 matrices, N=30 | ~1.2M |

### Experiment 3: Reverse Plane Partitions (Hillman-Grassl)

| File | Description | Parameters |
|------|-------------|-----------|
| `checkpoints/encoder_rpp_4x3x2x1_m4/best.pt` | RSKEncoder on RPP shape (4,3,2,1), max_entry=4 | ~1.2M |
| `checkpoints/encoder_rpp_6x4x2_m4/best.pt` | RSKEncoder on RPP shape (6,4,2), max_entry=4 | ~1.2M |
| `checkpoints/encoder_rpp_2x2x2x2x2x1_m4/best.pt` | RSKEncoder on RPP shape (2,2,2,2,2,1), max_entry=4 | ~1.2M |

### Experiment 4: Cylindric Plane Partitions

| File | Description | Parameters |
|------|-------------|-----------|
| `checkpoints/encoder_cyl_1010_m3/best.pt` | RSKEncoder on CPP profile (1,0,1,0), max_label=3 | ~1.2M |
| `checkpoints/encoder_cyl_10100_m3/best.pt` | RSKEncoder on CPP profile (1,0,1,0,0), max_label=3 | ~1.2M |
| `checkpoints/encoder_cyl_101010_m3/best.pt` | RSKEncoder on CPP profile (1,0,1,0,1,0), max_label=3 | ~1.2M |

### Loading a checkpoint

```python
import torch
from model import RSKEncoder
from config import ModelConfig

# Load n=10 model
ckpt = torch.load("checkpoints/encoder_n10/best.pt", map_location="cpu", weights_only=False)
config = ckpt["model_config"]
model = RSKEncoder(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Training

```bash
pip install torch datasets

# --- Experiment 1: Permutation RSK ---
python train.py --model encoder --n 10 --source sample --train-size 500000 --batch-size 512
python train.py --model encoder --n 8 --source hf

# --- Experiment 2: Full Matrix RSK ---
python train.py --model encoder --task matrix --a-dim 3 --b-dim 3 --total-n 10 \
    --source sample --train-size 500000
python train.py --model encoder --task matrix --a-dim 4 --b-dim 4 --total-n 20 \
    --source sample --train-size 500000
python train.py --model encoder --task matrix --a-dim 5 --b-dim 5 --total-n 30 \
    --source sample --train-size 2000000 --batch-size 512

# --- Experiment 3: Reverse Plane Partitions ---
python train.py --model encoder --task rpp --shape 4,3,2,1 --max-entry 4 \
    --source sample --train-size 500000
python train.py --model encoder --task rpp --shape 6,4,2 --max-entry 4 \
    --source sample --train-size 500000

# --- Experiment 4: Cylindric Plane Partitions ---
python train.py --model encoder --task cylindric --profile 1010 --max-label 3 \
    --source sample --train-size 500000
python train.py --model encoder --task cylindric --profile 101010 --max-label 3 \
    --source sample --train-size 500000
```

## Citation

```bibtex
@software{rsk_transformer,
  author={Langer, Robin},
  title={Learning Combinatorial Bijections with Transformers: Inverse RSK, Growth Diagrams, and Interpretable Features},
  year={2026},
  url={https://github.com/RaggedR/rsk-transformer}
}
```

## Acknowledgements

- [PNNL ML4AlgComb](https://github.com/pnnl/ML4AlgComb/tree/master/rsk) for the original benchmark
- [ACDRepo](https://huggingface.co/ACDRepo) for pre-computed RSK datasets
