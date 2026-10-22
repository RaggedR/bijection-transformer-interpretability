# Learning the RSK Correspondence with Transformers

A transformer that learns **inverse combinatorial bijections** — the Robinson-Schensted-Knuth correspondence (permutations and matrices), the Hillman-Grassl correspondence (reverse plane partitions), and the cylindric growth diagram bijection (cylindric plane partitions). The same architecture handles all tasks without modification.

Achieves **100% exact-match accuracy** on held-out test data for permutations at n=10, **99.99%** at n=15 (1.3 trillion permutations), **100%** on 3×3 matrix RSK, **100%** on reverse plane partitions of shape (4,3,2,1), and **100%** on cylindric plane partitions — significantly improving on the [PNNL ML4AlgComb benchmark](https://github.com/pnnl/ML4AlgComb/tree/master/rsk) which only achieved weak baselines.

For a detailed writeup of the architecture, training, and results, see [SUMMARY.md](SUMMARY.md).

**Links**: [Trained models (HuggingFace)](https://huggingface.co/RobBobin/rsk-transformer) | [Thesis (arXiv:2110.12629)](https://arxiv.org/abs/2110.12629) | HuggingFace datasets: [n=8](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_8), [n=9](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_9), [n=10](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_10)

## Results

### Experiment 1: Permutation RSK (σ → SYT pair)

Given a pair of standard Young tableaux (P, Q), predict the permutation σ.

| n | \|S_n\| | Training data | Source | Model params | Test exact match | Best epoch |
|---|---------|--------------|--------|-------------|-----------------|------------|
| 8 | 40,320 | 29,031 (72% of S_n) | [HuggingFace](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_8) | 1,202,368 | 99.95% | 23 |
| 10 | 3,628,800 | 500,000 (14% of S_n) | sampling | 1,207,012 | **100.00%** | 28 |
| 15 | 1.3 × 10¹² | 500,000 (0.00004%) | sampling | 1,225,057 | **99.99%** | 52 |

The n=10 result rules out memorisation: a 1.2M-parameter model trained on 14% of the permutation space achieves perfect accuracy on 50,000 unseen test permutations. At n=15 (1.3 trillion permutations), the same architecture trained on 0.00004% of the space gets 49,995 out of 50,000 held-out permutations exactly right — unambiguous algorithmic generalisation.

### Experiment 2: Full Matrix RSK (matrix → SSYT pair)

Given a pair of semistandard Young tableaux (P, Q) from Knuth's full RSK, recover the biword (and hence the matrix). **Same model, same embedding, same architecture** — only the task flag changes.

| Experiment | \|λ\| | Classes | Training data | Test exact match | Per-position | Best epoch |
|-----------|-------|---------|--------------|-----------------|-------------|------------|
| 3×3, N=10 | 10 | 3 | 500,000 | **100.00%** | **100.00%** | 18 |
| 4×4, N=20 | 20 | 4 | 500,000 | **99.32%** | **99.96%** | 20 |
| 5×5, N=30 | 30 | 5 | 2,000,000 | **96.79%** | **99.87%** | 16 |

The results are data-limited, not architecture-limited: per-position accuracy is 99.87%+ across all experiments, and the exact-match gaps are explained by independent errors compounding across positions ((0.9987)^30 ≈ 96.2%, matching the observed 96.79%).

### Experiment 3: Reverse Plane Partitions (Hillman-Grassl)

Given a reverse plane partition (RPP) of shape λ, recover the arbitrary filling via the inverse [Hillman-Grassl correspondence](https://en.wikipedia.org/wiki/Hillman%E2%80%93Grassl_correspondence). **Same model architecture** — the only change is that the input is a single filling (not a pair), so `tableau_emb(0)` is used for all tokens.

| Shape λ | Type | \|λ\| | Classes | Training data | Test exact match | Per-position | Best epoch |
|---------|------|-------|---------|--------------|-----------------|-------------|------------|
| (4,3,2,1) | Staircase | 10 | 5 | 500,000 | **100.00%** | **100.00%** | 23 |
| (6,4,2) | Wide | 12 | 5 | 500,000 | **99.99%** | **100.00%** | 17 |
| (2,2,2,2,2,1) | Tall | 11 | 5 | 500,000 | **99.99%** | **100.00%** | 36 |

The Hillman-Grassl bijection maps non-negative integer fillings of shape λ to reverse plane partitions (weakly increasing rows and columns) of the same shape, with weight preservation: Σ RPP[r][c] = Σ filling[r][c] × hook_length(r,c). This is a fundamentally different bijection from RSK — it involves zigzag paths through the Young diagram rather than Schensted insertion — yet the same transformer architecture learns it to near-perfect accuracy.

Tall shapes converge slower (36 epochs vs 17-23) because the Hillman-Grassl zigzag paths are longer, creating longer-range dependencies for attention to capture.

### Experiment 4: Cylindric Plane Partitions (Growth Diagrams)

Given a cylindric plane partition (CPP) with binary profile π, recover the base partition γ and the ALCD face labels via the inverse cylindric growth diagram bijection. This experiment uses the **Burge local rule** applied recursively through a cylindric growth diagram, as described in [Langer (2013), §4.2–4.3](https://arxiv.org/abs/2110.12629). **Same model architecture** — the input is a CPP encoded as (value, partition_index, part_index, 0) tokens.

| Profile π | T | ALCD labels | Training data | Test exact match | Per-position | Best epoch |
|-----------|---|-------------|--------------|-----------------|-------------|------------|
| (1,0,1,0) | 4 | 3 | 500,000 | **100.00%** | **100.00%** | 2 |
| (1,0,1,0,0) | 5 | 5 | 500,000 | **100.00%** | **100.00%** | 7 |
| (1,0,1,0,1,0) | 6 | 6 | 500,000 | **100.00%** | **100.00%** | 3 |
| (1,0,1,0,1,0,1,0) | 8 | 10 | 500,000 | **99.98%** | **100.00%** | 9 |

The cylindric growth diagram bijection is qualitatively different from all previous experiments: there is no direct closed-form algorithm — the bijection is defined implicitly by the Burge local rule applied at each face of the cylindric growth diagram. The model must learn to invert a recursive process (the 𝔏_i composition from [Langer 2013, §4.2]) that peels off one ALCD label at each step by solving a local Burge equation. Despite this complexity, the transformer achieves 100% on all tested profiles.

## Key Idea: Structured 2D Token Embeddings

Previous work (PNNL) encoded tableaux as flat bracket strings like `[[1,3,5],[2,4]]`, destroying the 2D geometric structure that makes RSK work. We instead encode each tableau entry as a token with four learned embeddings:

```
embedding(entry) = value_emb(v) + row_emb(r) + col_emb(c) + tableau_emb(P or Q)
```

This preserves the spatial relationships that govern Schensted insertion — the model can learn that bumping paths move down rows, that column-strictness constrains which values can appear where, and that P and Q share the same shape.

## Architecture

```
Input: (P, Q) as 2n structured tokens
  → TokenEmbedding (value + row + col + tableau_id)
  → 6-layer TransformerEncoder (d=128, 8 heads, pre-norm, GELU)
  → Mean pool over all 2n tokens
  → n parallel classification heads → logits (batch, n, n)
  → Masked greedy decode → predicted σ
```

**Encoder-only**: since (P, Q) fully determines σ, all information is in the input. No autoregressive decoding needed — all 2n tokens attend to each other simultaneously.

**Masked greedy decoding**: at inference, we enforce the permutation constraint by iteratively picking the highest-confidence (position, value) pair globally, locking it in, and masking that value from all other heads.

## Quick Start

```bash
pip install torch datasets

# --- Experiment 1: Permutation RSK ---

# Train on n=10 with random sampling
python train.py --model encoder --n 10 --source sample --train-size 500000 --batch-size 512

# Train on n=8 with HuggingFace data
python train.py --model encoder --n 8 --source hf

# Train on any n (sampling scales to arbitrary size)
python train.py --model encoder --n 20 --source sample --train-size 1000000

# --- Experiment 2: Full Matrix RSK ---

# 3×3 matrices with entry sum 10
python train.py --model encoder --task matrix --a-dim 3 --b-dim 3 --total-n 10 \
    --source sample --train-size 500000

# 4×4 matrices with entry sum 20
python train.py --model encoder --task matrix --a-dim 4 --b-dim 4 --total-n 20 \
    --source sample --train-size 500000

# 5×5 matrices with entry sum 30 (2M training samples)
python train.py --model encoder --task matrix --a-dim 5 --b-dim 5 --total-n 30 \
    --source sample --train-size 2000000 --batch-size 512

# --- Experiment 3: Reverse Plane Partitions ---

# Staircase shape (4,3,2,1) with max filling value 4
python train.py --model encoder --task rpp --shape 4,3,2,1 --max-entry 4 \
    --source sample --train-size 500000

# Wide shape (6,4,2)
python train.py --model encoder --task rpp --shape 6,4,2 --max-entry 4 \
    --source sample --train-size 500000

# --- Experiment 4: Cylindric Plane Partitions ---

# Profile (1,0,1,0) — simplest non-trivial cylindric case
python train.py --model encoder --task cylindric --profile 1010 --max-label 3 \
    --source sample --train-size 500000

# Profile (1,0,1,0,1,0) — 6 ALCD labels
python train.py --model encoder --task cylindric --profile 101010 --max-label 3 \
    --source sample --train-size 500000

# --- Verification ---

# Verify RSK + Hillman-Grassl + cylindric growth diagram round-trip bijections
python rsk.py
```

### Data Sources

| Source | Flag | Description |
|--------|------|-------------|
| HuggingFace | `--source hf` | [ACDRepo](https://huggingface.co/ACDRepo) datasets for [n=8](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_8), [n=9](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_9), [n=10](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_10). Full enumeration with 80/20 split. |
| Sampling | `--source sample` | Random permutations with on-the-fly RSK computation. Works for **any n**. Use `--train-size` to control dataset size. |
| Generate | `--source generate` | Enumerates all n! permutations locally. Only feasible for n ≤ ~8. |

### CLI Options

```
--model {encoder,mlp}     Model architecture (default: encoder)
--task {permutation,matrix,rpp,cylindric}  Task type (default: permutation)
--n N                     Permutation size (default: 8)
--source {hf,generate,sample}  Data source (default: hf)
--train-size N            Training samples for --source sample (default: 500000)
--val-size N              Validation samples (default: 50000)
--test-size N             Test samples (default: 50000)
--epochs N                Max epochs (default: 100)
--batch-size N            Batch size (default: 256)
--lr FLOAT                Learning rate (default: 3e-4)
--device DEVICE           Device (default: mps)
--d-model N               Transformer dimension (default: 128)
--num-layers N            Transformer layers (default: 6)
--nhead N                 Attention heads (default: 8)
--resume                  Resume training from existing checkpoint

# Matrix-specific options:
--a-dim N                 Matrix rows (for --task matrix)
--b-dim N                 Matrix cols (for --task matrix)
--total-n N               Total matrix entries |λ| (for --task matrix)

# RPP-specific options:
--shape 4,3,2,1           Partition shape (for --task rpp)
--max-entry N             Max filling value (for --task rpp)

# Cylindric-specific options:
--profile 1010            Binary profile (for --task cylindric)
--max-label N             Max ALCD face label (for --task cylindric)
--max-gamma-parts N       Max parts in base partition γ (default: 3)
--max-gamma-size N        Max part size in γ (default: 4)
```

## Project Structure

```
rsk.py      RSK forward/inverse, Hillman-Grassl, Burge local rule, cylindric growth diagrams
data.py     Data pipeline: HuggingFace loading, sampling, structured encoding
model.py    RSKEncoder (transformer) and BaselineMLP (flat comparison)
config.py   ModelConfig and TrainConfig dataclasses
train.py    Training loop, masked greedy decoding, evaluation
```

## Interpretability (Work in Progress)

Early mechanistic interpretability experiments on the trained models, investigating whether the transformer learned [Fomin's growth diagram local rule](https://arxiv.org/abs/2110.12629) (a single parallel operation applied at every cell) or Schensted's sequential bumping algorithm.

**What we've found so far:**

- On **cylindric plane partitions** (where the local rule is the *only* known algorithm), attention in the critical computation layer (Layer 2) concentrates on the partition triples where the Burge local rule operates, at 2.5–3.7× above baseline. The big/small vertex structure of the profile is visible in the attention patterns.
- On **permutation RSK** (where both algorithms apply), the signal is weaker and inconclusive — neither growth diagrams nor bumping dominates clearly.
- The model does not map transformer layers to recursion depth sequentially. It parallelises across heads within each layer, using different heads for different local rule applications.

**What we can't say yet:** attention concentration on local rule triples is necessary but not sufficient evidence. We haven't shown the *content* of attention implements the Burge rule (the specific partition arithmetic). The model may have found a different algorithm with similar attention patterns.

**Next steps:**

- Probing: train linear probes to predict intermediate partition states in the growth diagram from the residual stream
- Content analysis: examine what information flows through the attention value vectors
- Cross-model comparison: check if attention patterns transfer between the permutation, Hillman-Grassl, and cylindric models (which all use the same Burge local rule with different boundary conditions)
- Larger cylindric profiles (T=10, 12) for more structure to test against

Interpretability code and full report: [octopus-streams/rsk](https://github.com/RaggedR/octopus-streams/tree/main/rsk)

## Background

The [RSK correspondence](https://en.wikipedia.org/wiki/Robinson%E2%80%93Schensted%E2%80%93Knuth_correspondence) is a bijection between permutations σ ∈ S_n and pairs (P, Q) of standard Young tableaux of the same shape λ ⊢ n. It is central to algebraic combinatorics, connecting representation theory, symmetric functions, and enumerative combinatorics.

The **forward** direction (σ → P, Q) uses Schensted row insertion (the bumping algorithm). The **inverse** (P, Q → σ) uses reverse bumping, processing entries in decreasing order. For a detailed treatment, see [Langer (2013), Chapter 2, §2.1–2.2, pp. 25–43](https://arxiv.org/abs/2110.12629). This project trains a neural network to learn the inverse direction.

## Why Not Memorisation?

At n=10, there are 3,628,800 permutations. The model has 1.2M parameters and was trained on 500,000 samples (14% of the space). A lookup table would need at least 3.6M × 10 = 36M values to store the full mapping — 30× more than the model's parameter count. Yet the model achieves 100% accuracy on held-out samples.

At n=15, there are 1.3 trillion permutations. The model sees 0.00004% of the space during training and gets 99.99% exact match — unambiguously algorithmic generalisation. A baseline MLP on the same data achieves only 3.07%, showing the transformer's structured 2D embedding and attention are essential.

## Citation

If you use this work, please cite:

```
@software{rsk_transformer,
  author={Langer, Robin},
  title={Learning the RSK Correspondence with Transformers},
  year={2026},
  url={https://github.com/RaggedR/rsk-transformer}
}
```

## Acknowledgements

- [PNNL ML4AlgComb](https://github.com/pnnl/ML4AlgComb/tree/master/rsk) for the original benchmark and HuggingFace datasets
- [ACDRepo](https://huggingface.co/ACDRepo) for pre-computed RSK datasets
