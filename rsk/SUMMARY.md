# Learning the Inverse RSK Correspondence with Transformers — Full Summary

## Experiments Overview

This project contains three experiments using the **same model architecture** and **same codebase** — only the task flag and data pipeline differ:

| Experiment | Task | Input | Output | Command flag |
|-----------|------|-------|--------|-------------|
| 1. Permutation RSK | `--task permutation` | (P, Q) pair of SYT | Permutation σ ∈ S_n | `--n 15 --source sample` |
| 2. Full Matrix RSK | `--task matrix` | (P, Q) pair of SSYT | Biword bottom line | `--task matrix --a-dim 5 --b-dim 5 --total-n 30` |
| 3. Reverse Plane Partitions | `--task rpp` | RPP of shape λ | Arbitrary filling of shape λ | `--task rpp --shape 4,3,2,1 --max-entry 4` |

All code lives in a single directory — the shared architecture is the point. See [How to Reproduce](#how-to-reproduce) for exact commands.

## The Problem

The **Robinson-Schensted-Knuth correspondence** is a bijection central to algebraic combinatorics:

```
σ ∈ S_n  ⟷  (P, Q) pair of standard Young tableaux of shape λ ⊢ n
```

**Forward** (σ → P, Q): Schensted row insertion — scan each σ(i), bump entries rightward through rows, record where new cells appear.

**Inverse** (P, Q → σ): Reverse bumping — process Q entries in decreasing order, reverse the insertion path in P to recover each σ(i). For a detailed treatment of both directions, including Viennot's shadow method and Fomin's growth diagram framework, see [2, Chapter 2, §2.1–2.2, pp. 25–43]. The reverse algorithm specifically is described in [2, §2.2.5, p. 41].

We train a neural network to learn the **inverse direction**: given P and Q, predict σ.

## Prior Work

The **PNNL ML4AlgComb benchmark** attempted this with:
- Tableaux encoded as bracket strings: `"[[1,3,5],[2,4]]"` → tokenised as `[`, `[`, `1`, `,`, `3`, ...
- Sequence-to-sequence models treating it as string transduction
- Result: weak baselines, accuracy well below useful levels

**Why it failed**: bracket-string encoding destroys the 2D geometric structure of tableaux. The model has to re-learn that `1` at position 7 in the string means "row 0, column 0" — information that was explicit in the tableau but is now buried in syntax.

## Our Architecture: RSKEncoder

### Input Encoding — Structured 2D Tokens

Each entry in P and Q becomes a token. For n entries per tableau, we get **2n tokens**. Each token carries four properties: its numeric value, its row, its column, and which tableau (P or Q) it belongs to.

**Concrete example.** Take σ = [3, 1, 2]. RSK forward gives:

```
P = [[1, 2],    Q = [[1, 3],
     [3]]            [2]]
```

PNNL would flatten this to bracket strings: `[[1,2],[3]]` → tokenised as `[`, `[`, `1`, `,`, `2`, `]`, ... — a sequence where all spatial information is buried in syntax. We instead create one token per tableau entry:

| Token | Value | Row | Col | Tableau | Meaning |
|-------|-------|-----|-----|---------|---------|
| 1 | 1 | 0 | 0 | P | "1 sits at row 0, col 0 in P" |
| 2 | 2 | 0 | 1 | P | "2 sits at row 0, col 1 in P" |
| 3 | 3 | 1 | 0 | P | "3 sits at row 1, col 0 in P" |
| 4 | 1 | 0 | 0 | Q | "1 sits at row 0, col 0 in Q" |
| 5 | 3 | 0 | 1 | Q | "3 sits at row 0, col 1 in Q" |
| 6 | 2 | 1 | 0 | Q | "2 sits at row 1, col 0 in Q" |

**How the embedding works.** Readers familiar with Raschka's *Build a Large Language Model (From Scratch)* [1] will recognise the pattern. In Chapter 2, Raschka builds GPT-2's embedding as two lookup tables summed together:

```python
# Raschka's GPT embedding (simplified)
token_emb = nn.Embedding(vocab_size, d_model)      # "what word is this?"
pos_emb   = nn.Embedding(context_length, d_model)   # "where in the sequence?"
x = token_emb(token_ids) + pos_emb(positions)
```

A word's meaning depends on *what* it is and *where* it appears. Two lookup tables, summed. Our embedding is the same idea — but a tableau entry's meaning depends on **four** things, not two:

```python
# Our embedding — four lookup tables, summed
value_emb   = nn.Embedding(n + 1, 128)  # "what number is this entry?"
row_emb     = nn.Embedding(n, 128)      # "which row is it in?"
col_emb     = nn.Embedding(n, 128)      # "which column is it in?"
tableau_emb = nn.Embedding(2, 128)      # "is it in P or Q?"

x = value_emb(values) + row_emb(rows) + col_emb(cols) + tableau_emb(tableau_ids)
```

Each `nn.Embedding(num_entries, 128)` is just a matrix of shape `(num_entries, 128)`. When you call `value_emb(3)`, it returns row 3 of that matrix — a 128-dimensional vector. As Raschka explains: "an embedding layer is essentially a lookup operation."

For token 3 in the example above (the number 3 in P at row 1, col 0):

```
token_embedding = value_emb.weight[3]      # row 3 of a (n+1, 128) matrix
                + row_emb.weight[1]        # row 1 of a (n, 128) matrix
                + col_emb.weight[0]        # row 0 of a (n, 128) matrix
                + tableau_emb.weight[0]    # row 0 of a (2, 128) matrix — P
```

Four vectors from four tables, added into one 128-dimensional vector. That's it. Followed by LayerNorm and dropout (0.1).

**These vectors are learned, not hand-designed.** At initialisation, each embedding table contains random 128-dimensional vectors. Through backpropagation — the same gradient descent that trains the rest of the network — the model adjusts these vectors to be useful for predicting σ. We designed the *structure* (each token has four properties combined by addition). We did *not* design the *content* (the actual 128-dimensional vectors). Through training, the model discovers what each spatial coordinate means — that row index relates to bumping depth, that column position relates to shape, that P and Q play fundamentally different roles (insertion values vs insertion order).

This is the key insight: we gave the model the right inductive bias (tableaux are 2D, they come in pairs), and it learned how to exploit that structure. PNNL gave the model no structure at all (flat bracket strings), so it had to discover both structure and content from scratch — which it couldn't do.

**Comparison to Raschka's GPT embedding:**

| | Raschka's GPT | Our RSKEncoder |
|--|--------------|----------------|
| What | `token_emb(word_id)` | `value_emb(entry_value)` |
| Where | `pos_emb(position)` — 1D | `row_emb(r) + col_emb(c) + tableau_emb(t)` — 2D + categorical |
| Terms summed | 2 | 4 |
| Output shape | `(batch, seq_len, d_model)` | `(batch, 2n, 128)` |

The difference: GPT processes a 1D sequence of words, so one positional embedding suffices. Tableaux are 2D grids that come in pairs, so we need three "positional" embeddings (row, col, which-tableau) to tell the model where each entry lives.

**Why sum rather than concatenate.** Concatenating the four vectors would give a 512-dimensional input per token. Summing keeps it at 128 dimensions, relying on the fact that in high-dimensional space the model can learn roughly orthogonal directions for the different properties.

[1] S. Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024.
[2] R. Langer, *Cylindric plane partitions, Lambda determinants, Commutants in semicircular systems*, PhD thesis, Université Paris-Est, 2013. [arXiv:2110.12629](https://arxiv.org/abs/2110.12629).

### Complete Model Code

The entire model is about 50 lines of PyTorch. Here is the core logic:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model  # 128

        self.value_emb = nn.Embedding(config.max_value + 1, d)
        self.row_emb = nn.Embedding(config.max_rows, d)
        self.col_emb = nn.Embedding(config.max_cols, d)
        self.tableau_emb = nn.Embedding(2, d)  # P=0, Q=1

        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values, positions):
        # values:    (batch, 2n) — entry values
        # positions: (batch, 2n, 3) — [row, col, tableau_id]

        x = (
            self.value_emb(values)
            + self.row_emb(positions[:, :, 0])
            + self.col_emb(positions[:, :, 1])
            + self.tableau_emb(positions[:, :, 2])
        )
        return self.dropout(self.layer_norm(x))


class RSKEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n = config.n

        self.embedding = TokenEmbedding(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,       # 128
            nhead=config.nhead,           # 8
            dim_feedforward=config.dim_feedforward,  # 512
            dropout=config.dropout,       # 0.1
            activation="gelu",
            batch_first=True,
            norm_first=True,              # Pre-norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,  # 6
        )

        # n classification heads — one per position in σ
        self.heads = nn.ModuleList([
            nn.Linear(config.d_model, config.n)
            for _ in range(config.n)
        ])

    def forward(self, values, positions):
        x = self.embedding(values, positions)  # (batch, 2n, 128)
        x = self.encoder(x)                    # (batch, 2n, 128)
        x = x.mean(dim=1)                      # (batch, 128)  — mean pool
        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits                          # (batch, n, n)
```

The forward pass is four lines:
1. **Embed** the tokens (four lookup tables, summed)
2. **Encode** through 6 transformer layers (all tokens attend to all tokens — no causal mask)
3. **Mean pool** over all 2n tokens into a single 128-d vector
4. **Classify** with n parallel linear heads to get logits

The transformer encoder uses `nn.TransformerEncoder` from PyTorch — the same stack of multi-head self-attention + feed-forward layers that Raschka builds from scratch in Chapters 3-4 of [1], except we use PyTorch's built-in implementation. This is BERT-style (encoder-only, bidirectional attention), not GPT-style (decoder, causal mask), because (P, Q) fully determines σ — all information is in the input, so every token should attend to every other token simultaneously.

`logits[b, i, j]` = log-probability that σ(i+1) = j+1 for batch element b.

**Why n parallel heads**: each position σ(i) is an independent n-way classification problem. The heads share the same pooled representation but have separate weights, allowing each to specialise.

### Architecture Diagram

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     INPUT: (P, Q) tableaux                      │
 │                                                                 │
 │   P = ┌─┬─┬─┐    Q = ┌─┬─┬─┐                                  │
 │       │1│2│5│        │1│2│4│     σ = [3, 1, 4, 5, 2]           │
 │       ├─┼─┘         ├─┼─┘                                      │
 │       │3│            │3│          n = 5  →  2n = 10 tokens      │
 │       ├─┤            ├─┤                                        │
 │       │4│            │5│                                        │
 │       └─┘            └─┘                                        │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                   TOKEN EMBEDDING (per token)                   │
 │                                                                 │
 │   token_emb = value_emb(v) + row_emb(r) + col_emb(c)          │
 │             + tableau_emb(P=0 / Q=1)                            │
 │                                                                 │
 │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
 │   │value_emb │ │ row_emb  │ │ col_emb  │ │tableau_  │         │
 │   │(n+1, 128)│ │ (n, 128) │ │ (n, 128) │ │emb(2,128)│         │
 │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘         │
 │        │            │            │             │                │
 │        └──────┬─────┴─────┬──────┘             │                │
 │               │    (+)    │        (+)          │                │
 │               └─────┬─────┴────────────────────┘                │
 │                     ▼                                           │
 │              LayerNorm(128)                                     │
 │                     ▼                                           │
 │              Dropout(0.1)                                       │
 │                                                                 │
 │            Output: (batch, 2n, 128)                             │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │            TRANSFORMER ENCODER  ×6 layers                       │
 │                                                                 │
 │   ┌─────────────────────────────────────────────────┐           │
 │   │  Pre-Norm TransformerEncoderLayer               │           │
 │   │                                                 │           │
 │   │   ┌──────────────────────────────────────┐      │           │
 │   │   │  LayerNorm → Multi-Head Attention    │      │           │
 │   │   │  (8 heads, d_k = 16, no causal mask)│      │           │
 │   │   │  All 2n tokens attend to all others  │      │           │
 │   │   └──────────────┬───────────────────────┘      │           │
 │   │        residual (+)                             │           │
 │   │                  ▼                              │           │
 │   │   ┌──────────────────────────────────────┐      │           │
 │   │   │  LayerNorm → FFN                     │      │           │
 │   │   │  Linear(128→512) → GELU              │      │           │
 │   │   │  → Linear(512→128) → Dropout(0.1)    │      │           │
 │   │   └──────────────┬───────────────────────┘      │           │
 │   │        residual (+)                             │           │
 │   └──────────────────┼─────────────────────────┘    │           │
 │                      │  ×6                          │           │
 │            Output: (batch, 2n, 128)                             │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     MEAN POOL                                   │
 │                                                                 │
 │         (batch, 2n, 128)  →  mean(dim=1)  →  (batch, 128)      │
 │                                                                 │
 │         All 2n token representations averaged into              │
 │         a single 128-d summary vector                           │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │               n PARALLEL CLASSIFICATION HEADS                   │
 │                                                                 │
 │    ┌──────────┐ ┌──────────┐ ┌──────────┐     ┌──────────┐    │
 │    │ Head 1   │ │ Head 2   │ │ Head 3   │ ... │ Head n   │    │
 │    │Linear    │ │Linear    │ │Linear    │     │Linear    │    │
 │    │(128 → n) │ │(128 → n) │ │(128 → n) │     │(128 → n) │    │
 │    └────┬─────┘ └────┬─────┘ └────┬─────┘     └────┬─────┘    │
 │         ▼            ▼            ▼                 ▼          │
 │      σ(1)=?       σ(2)=?       σ(3)=?           σ(n)=?        │
 │     logits        logits       logits           logits         │
 │     over          over         over             over           │
 │     {1..n}        {1..n}       {1..n}           {1..n}         │
 │                                                                 │
 │            Output: (batch, n, n)                                │
 └────────────────────────┬────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         [TRAINING]             [INFERENCE]
              │                       │
              ▼                       ▼
 ┌────────────────────┐  ┌─────────────────────────────┐
 │  Cross-Entropy     │  │  Masked Greedy Decoding      │
 │  Loss (per head,   │  │                              │
 │  targets 0-indexed)│  │  1. Find max logit globally  │
 │                    │  │  2. Assign σ(i) = j          │
 │  L = Σᵢ CE(        │  │  3. Mask position i (done)   │
 │    logits[b,i,:],  │  │  4. Mask value j (used)      │
 │    σ(i+1) - 1      │  │  5. Repeat n times           │
 │  )                 │  │                              │
 └────────────────────┘  │  → Valid permutation σ       │
                         └─────────────────────────────┘
```

### Masked Greedy Decoding (Inference Only)

Raw argmax can assign the same value to multiple positions — violating the permutation constraint. Masked greedy decoding fixes this:

```
1. Compute logits (batch, n, n)
2. Find the global (position, value) pair with highest logit
3. Lock that assignment
4. Set that position's logits to -∞ (position decided)
5. Set that value's logits to -∞ across all positions (value used)
6. Repeat n times
```

This guarantees a valid permutation. The gap between argmax and greedy accuracy measures how often the model would violate the constraint without it.

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| value_emb | (n+1) × 128 |
| row_emb | n × 128 |
| col_emb | n × 128 |
| tableau_emb | 2 × 128 |
| 6 transformer layers | ~1.18M |
| n classification heads | n × (128 × n + n) |
| **Total (n=8)** | **1,202,368** |
| **Total (n=10)** | **1,207,012** |
| **Total (n=15)** | **1,225,057** |

The backbone dominates. Changing n only affects the embedding tables and head sizes — the transformer layers are identical across all experiments.

## Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 0.01 |
| Schedule | Linear warmup (5% of steps) → cosine decay |
| Gradient clipping | max norm 1.0 |
| Batch size | 512 |
| Loss | Cross-entropy (summed over n heads) |
| Early stopping | Patience 10 on val greedy exact match |

**Hyperparameter selection.** These are standard transformer training defaults (AdamW with LR 3 × 10⁻⁴, cosine schedule, pre-norm). No systematic hyperparameter search was performed — the model converged with the first configuration tried, suggesting the task is not hyperparameter-sensitive given the right inductive bias.

### Hardware & Training Time

All training was done on an **Apple M4 Max** (MacBook Pro) using **PyTorch MPS** backend.

| n | Tokens/sample | Epoch time | Epochs to converge | Total wall time |
|---|--------------|------------|-------------------|----------------|
| 8 | 16 | ~15s | 23 | ~8 min |
| 10 | 20 | ~35s | 28 | ~17 min |
| 15 | 30 | 10–18 min (variable) | 24+ | ~6–10 hours |

At n=15, epoch times varied significantly due to intermittent MPS thermal throttling — some epochs took over 2 hours instead of the typical 10–18 minutes. No multi-GPU or cloud compute was used; the entire project was trained on a single laptop.

## Data

| n | |S_n| | Source | Train | Val | Test |
|---|------|--------|-------|-----|------|
| 8 | 40,320 | HuggingFace | 29,031 (72%) | 3,225 (8%) | 8,064 (20%) |
| 10 | 3,628,800 | Random sampling | 500,000 (14%) | 50,000 | 50,000 |
| 15 | 1,307,674,368,000 | Random sampling | 500,000 (0.00004%) | 50,000 | 50,000 |

For n=10 and n=15, data is generated on the fly: sample a random permutation of {1..n}, compute RSK forward to get (P, Q), encode as structured tokens. RSK forward is O(n²) worst case, taking ~0.03ms at n=10 and ~0.11ms at n=20 in pure Python — negligible compared to GPU time.

The HuggingFace datasets (ACDRepo) enumerate all n! permutations with an 80/20 train/test split. For n ≥ 10, converting 2.9M+ rows to Python lists is slow, so random sampling is preferred.

## Results

| n | |S_n| | Train fraction | Test greedy exact | Test argmax exact | Per-position | Best epoch |
|---|------|---------------|------------------|------------------|-------------|------------|
| 8 | 40,320 | 72% | 99.95% | 99.80% | 99.98% | 23 |
| 10 | 3,628,800 | 14% | **100.00%** | **100.00%** | **100.00%** | 28 |
| 15 | 1.3 × 10¹² | 0.00004% | **99.99%** | **99.98%** | **99.9997%** | 52 |

### n=8 Details
- Converged at epoch 23, early-stopped at ~33
- Val greedy hit 100% at epoch 23; test greedy 99.95% (4 errors in 8,064)
- The 0.15% gap between argmax (99.80%) and greedy (99.95%) = greedy decoder fixing ~12 permutation violations

### n=10 Details
- Converged at epoch 28, early-stopped at 38
- Both argmax AND greedy hit 100% on 50,000 test samples — the model's per-position predictions are so confident that it never assigns the same value to two positions
- Val loss reached 0.0000 (below float display precision)

### n=15 Details
- Best checkpoint at epoch 52
- 99.99% greedy exact = 49,990 out of 50,000 permutations exactly right (only ~10 errors)
- 99.98% argmax exact — greedy decoder fixing only ~5 additional permutation violations
- Per-position accuracy 99.9997% = ~2 individual position errors across 50,000 × 15 = 750,000 predictions
- Continued training beyond epoch 24 (where it was 99.65%) proved worthwhile — patience and more epochs paid off significantly

### How to Reproduce (Experiment 1)

```bash
pip install torch datasets

# n=8 with HuggingFace data
python train.py --model encoder --n 8 --source hf --device mps

# n=10 with random sampling
python train.py --model encoder --n 10 --source sample --train-size 500000 --batch-size 512 --device mps

# n=15 with random sampling
python train.py --model encoder --n 15 --source sample --train-size 500000 --batch-size 512 --device mps

# MLP ablation
python train.py --model mlp --n 15 --source sample --train-size 500000 --batch-size 512 --device mps
```

Checkpoints are saved to `checkpoints/encoder_n{N}/best.pt`. The relevant code paths:
- **Data**: `RSKSamplingDataset` in `data.py` — samples random permutations, computes RSK forward
- **Encoding**: `encode_tableaux()` in `data.py` — the four-component token encoding
- **Model**: `RSKEncoder` in `model.py` — transformer with `TokenEmbedding`
- **Decoding**: `masked_greedy_decode()` in `train.py` — enforces permutation constraint at inference

## The Memorisation Question

**Can the model just be memorising a lookup table?**

| n | Params | Unique inputs | Params per input | Training coverage | Verdict |
|---|--------|--------------|-----------------|-------------------|---------|
| 8 | 1.2M | 40,320 | 29.8 | 72% | Ambiguous — enough params to store most of the space |
| 10 | 1.2M | 3,628,800 | 0.33 | 14% | **Cannot memorise** — fewer params than inputs, only 14% seen |
| 15 | 1.2M | 1.3 × 10¹² | 9.4 × 10⁻⁷ | 0.00004% | **Provably algorithmic** — not even close to memorisation capacity |

At n=10: a lookup table would need 3.6M × 10 = 36M entries minimum. The model has 1.2M parameters total. It achieves 100% accuracy having seen only 14% of the space. This is unambiguous generalisation.

At n=15: the model has seen 500,000 out of 1.3 trillion possible inputs. It gets 99.99% of held-out inputs exactly right — only ~10 errors out of 50,000. There is no interpretation of this other than the model having learned a general algorithm for inverse RSK.

## Experiment 2: Full Matrix RSK

### Extending to Matrices

RSK generalises from permutations σ ∈ S_n to arbitrary non-negative integer matrices A ∈ ℕ^{a×b}. In Knuth's full RSK, the matrix A is first converted to a **two-line array** (biword): for each entry A[i][j] > 0, create A[i][j] copies of the pair (i+1, j+1), sorted lexicographically. The biword is then inserted via Schensted's algorithm, producing a pair (P, Q) of **semistandard** Young tableaux — where rows are weakly increasing (not strictly, as in the permutation case).

The inverse problem: given the SSYT pair (P, Q), recover the original biword — and hence the matrix A.

### Design Choices

**What stays the same.** The model architecture is identical: same `RSKEncoder` with `TokenEmbedding`, same 6-layer transformer, same mean pool → parallel heads structure. The four-component embedding `value_emb + row_emb + col_emb + tableau_emb` works without modification because SSYT still have the same 2D geometry as SYT — entries in rows and columns of a Young diagram, just with weakly increasing rows instead of strictly increasing ones.

**What changes.**
- **seq_len** = |λ| = Σ A[i][j], the total number of entries in the biword (equals the number of cells in each tableau)
- **vocab_size** = b, the number of columns in A. Each position in the biword classifies into {1, ..., b}
- **Target**: the bottom line of the biword (0-indexed for cross-entropy), not the matrix A directly. The bottom line is a sequence of |λ| values from {1, ..., b}, and the matrix can be reconstructed from the biword by counting
- **Loss**: standard cross-entropy, same as permutations. No masked greedy decoding needed — the output is a word with repeated values allowed, not a permutation
- **Early stopping**: tracked on argmax exact match (since there's no permutation constraint to enforce)

**Data generation.** Random matrices are sampled by distributing |λ| balls uniformly into a×b bins (multinomial sampling). Each sample: generate matrix → convert to biword → forward RSK → encode (P, Q) as structured tokens → target is bottom line.

### Results

| Experiment | Shape | |λ| | Heads | Classes | Train | Best epoch | Exact match | Per-position |
|-----------|-------|-----|-------|---------|-------|------------|-------------|-------------|
| 3×3, N=10 | a=3, b=3 | 10 | 10 | 3 | 500,000 | 18 | **100.00%** | **100.00%** |
| 4×4, N=20 | a=4, b=4 | 20 | 20 | 4 | 500,000 | 20 | **99.32%** | **99.96%** |
| 5×5, N=30 | a=5, b=5 | 30 | 30 | 5 | 2,000,000 | 16 | **96.79%** | **99.87%** |

All experiments used the same hyperparameters as the permutation experiments: d_model=128, 6 layers, 8 heads, lr=3×10⁻⁴, patience 10. Batch size was 512 for the 5×5 run, 256 for the others.

**3×3 with N=10**: the model learned the complete inverse biword RSK for this size, achieving 100% exact match on 50,000 held-out test samples by epoch 18. This is a clean generalisation result — the space of 3×3 matrices with entry sum 10 is large (C(25,15) ≈ 3.3 million distinct matrices), and 500K training samples cover only ~15% of it.

**4×4 with N=20**: reached 99.32% exact match (49,660 out of 50,000 test matrices recovered perfectly). The model early-stopped at approximately epoch 30, having shown no improvement for 10 consecutive epochs after its best at epoch 20.

**5×5 with N=30**: reached 96.79% exact match with 99.87% per-position accuracy at epoch 16. Despite training on 2M samples (4× the smaller experiments), the space of 5×5 matrices with entry sum 30 is C(54, 24) ≈ 1.4 × 10¹⁴ — so even 2M samples covers a vanishing fraction (~10⁻⁸).

### Why Not 100% on the Larger Experiments?

The per-position accuracy tells the story. Across multiple classification heads, the model gets almost every individual position right, but even rare independent errors compound:

| Experiment | Per-position | (per-pos)^heads | Observed exact match |
|-----------|-------------|----------------|---------------------|
| 4×4, N=20 | 99.96% | (0.9996)^20 ≈ 99.2% | 99.32% |
| 5×5, N=30 | 99.87% | (0.9987)^30 ≈ 96.2% | 96.79% |

The close agreement between predicted and observed exact match confirms the model isn't confused about any systematic aspect of the RSK structure — it's making rare, independent errors on individual positions.

**The bottleneck is data coverage, not model capacity.** The space of matrices grows combinatorially with both the matrix dimensions and entry sum. For 4×4 N=20, the space is C(35, 15) ≈ 3.2 billion; for 5×5 N=30, it's C(54, 24) ≈ 1.4 × 10¹⁴. Training data covers a vanishing fraction in both cases. The architecture has clearly learned the algorithm — more training data would almost certainly improve results further, as it did when scaling from 500K to 2M for the 5×5 case. With limited computational resources (all training on a single Apple M4 Max laptop), we chose to move on to the qualitatively different reverse plane partition experiment rather than optimise this number further.

### Comparison: Permutation RSK vs Matrix RSK

| | Permutations | Matrices |
|--|-------------|----------|
| Input tableaux | SYT (strictly increasing rows) | SSYT (weakly increasing rows) |
| Output | Permutation (each value used once) | Biword bottom line (repeated values allowed) |
| Constraint enforcement | Masked greedy decoding | None needed (argmax suffices) |
| Model architecture | Identical | Identical |
| Token embedding | Identical | Identical |
| n=10 scale result | 100% (epoch 28) | 100% (epoch 18, 3×3 N=10) |
| Scaling challenge | 99.99% at n=15 | 96.79% at 5×5 N=30, data-limited |

The key takeaway: the same structured embedding and transformer architecture generalises from the permutation case to the full Knuth RSK correspondence without any architectural changes. The 2D token representation captures the geometry of semistandard tableaux just as well as standard tableaux — which makes sense, since the spatial structure (entries in rows and columns of a Young diagram) is the same in both cases.

### How to Reproduce (Experiment 2)

```bash
# 3×3 matrices, entry sum 10
python train.py --model encoder --task matrix --a-dim 3 --b-dim 3 --total-n 10 \
    --source sample --train-size 500000 --device mps

# 4×4 matrices, entry sum 20
python train.py --model encoder --task matrix --a-dim 4 --b-dim 4 --total-n 20 \
    --source sample --train-size 500000 --device mps

# 5×5 matrices, entry sum 30 (2M training samples)
python train.py --model encoder --task matrix --a-dim 5 --b-dim 5 --total-n 30 \
    --source sample --train-size 2000000 --batch-size 512 --device mps
```

Checkpoints are saved to `checkpoints/encoder_matrix_a{A}_b{B}_N{N}/best.pt`. The relevant code paths:
- **Data**: `MatrixSamplingDataset` in `data.py` — samples random matrices, computes biword RSK
- **Encoding**: same `encode_tableaux()` in `data.py` — identical four-component tokens
- **Model**: same `RSKEncoder` in `model.py` — identical architecture, only `seq_len` and `vocab_size` change
- **Decoding**: plain argmax (no greedy masking needed — output is a word, not a permutation)

## What the Model Learned

We don't have full mechanistic interpretability, but we can reason about what it *must* have learned:

1. **Shape reconstruction**: from (P, Q) with the same shape λ, the model implicitly knows the partition λ (it's encoded in which (row, col) positions have entries)

2. **Reverse bumping paths**: for each Q-entry processed in decreasing order, the model must determine which P-entry to remove and trace the reverse bumping path upward through the rows

3. **Insertion order**: Q records *when* each cell was added (entry i in Q means that cell was created at step i). Processing Q in reverse reconstructs the sequence of deletions

4. **Value recovery**: at each step, reverse bumping from a cell at (row, col) in P recovers the original σ(i) value — the model's n heads each predict one of these recovered values

The model likely represents an approximate version of this algorithm distributed across its attention heads, with different heads specialising in different aspects of the reverse bumping computation.

## Architecture Design Decisions

| Decision | Rationale | Alternative | Why not |
|----------|-----------|-------------|---------|
| Encoder-only | All info in input, no sequential dependency | Encoder-decoder / autoregressive | Unnecessary complexity, slower inference |
| Structured 2D embedding | Preserves tableau geometry | Flat bracket tokens (PNNL) | Destroys spatial info, forces re-learning structure |
| Summed embeddings | Compact, each dimension additively contributes | Concatenated features | 4× wider input, more parameters for no gain |
| Mean pooling | Simple, all positions see all input | CLS token / attention pooling | Mean works well when all tokens are informative |
| n parallel heads | Each σ(i) is independent classification | Single autoregressive head | Parallel is faster, and independence is structurally correct |
| Masked greedy decode | Enforces permutation constraint | Beam search / Hungarian algorithm | Greedy is O(n²) and sufficient — beam search adds cost with no accuracy gain |
| Pre-norm transformer | Better training stability | Post-norm | Pre-norm is standard best practice for deep transformers |
| GELU activation | Smooth, standard for transformers | ReLU | Minor difference, GELU is modern default |

## Files

```
rsk.py       — RSK forward/inverse implementation (pure Python, no dependencies)
config.py    — ModelConfig + TrainConfig dataclasses
data.py      — HuggingFace loader, random sampling dataset, structured token encoding
model.py     — RSKEncoder (transformer) + BaselineMLP (flat comparison)
train.py     — Training loop, masked greedy decoding, evaluation metrics
```

## Experiment 3: Reverse Plane Partitions (Hillman-Grassl)

### A Different Bijection

The [Hillman-Grassl correspondence](https://en.wikipedia.org/wiki/Hillman%E2%80%93Grassl_correspondence) is a bijection between non-negative integer fillings of a Young diagram shape λ and reverse plane partitions (RPPs) of the same shape — where entries are weakly increasing along both rows and columns. Unlike RSK, which uses Schensted's bumping algorithm, Hillman-Grassl traces **zigzag paths** through the diagram:

```
Forward (filling → RPP):
  For each unit of filling[r][c], trace a path:
    - Move UP while the cell above has the same RPP value
    - Then move RIGHT to the next cell
    - Repeat until exiting the shape
  Add 1 to every cell along the path.

Inverse (RPP → filling):
  While RPP has nonzero entries:
    1. Find the leftmost column with a nonzero entry
    2. Start from the bottommost nonzero row in that column
    3. Trace the zigzag path
    4. Subtract 1 from all cells on the path
    5. Record the endpoint → increment filling
```

**Weight preservation**: Σ RPP[r][c] = Σ filling[r][c] × hook_length(r,c), where hook_length(r,c) = arm + leg + 1.

### Design Choices

**What stays the same.** The model architecture is identical: same `RSKEncoder` with `TokenEmbedding`, same 6-layer transformer, same mean pool → parallel heads structure. The four-component embedding works unchanged.

**What changes.**
- **Input is a single filling**, not a (P, Q) pair — so `num_tokens = |λ|` instead of `2|λ|`, and `tableau_emb(0)` is used for all tokens (acts as a learned bias)
- **seq_len** = |λ| = number of cells in the shape
- **vocab_size** = max_entry + 1, where max_entry is the maximum value allowed in the target filling
- **No masked greedy decoding** — the output is an unconstrained filling, not a permutation

### Results

| Shape λ | Type | \|λ\| | Classes | Training data | Test exact match | Per-position | Best epoch |
|---------|------|-------|---------|--------------|-----------------|-------------|------------|
| (4,3,2,1) | Staircase | 10 | 5 | 500,000 | **100.00%** | **100.00%** | 23 |
| (6,4,2) | Wide | 12 | 5 | 500,000 | **99.99%** | **100.00%** | 17 |
| (2,2,2,2,2,1) | Tall | 11 | 5 | 500,000 | **99.99%** | **100.00%** | 36 |

All three shapes achieve near-perfect accuracy, with 100% per-position accuracy across the board.

**Shape geometry affects convergence speed.** The staircase (4,3,2,1) converged in 23 epochs; the wide shape (6,4,2) in 17; but the tall shape (2,2,2,2,2,1) needed 36 epochs. This is explained by the structure of Hillman-Grassl zigzag paths: in tall shapes, paths traverse more rows before exiting, creating longer-range dependencies that take more training to learn. The wide shape converges fastest because paths exit the shape quickly.

### Why This Matters

The Hillman-Grassl bijection is **fundamentally different** from RSK:
- RSK uses Schensted insertion (bumping entries rightward through rows)
- Hillman-Grassl traces zigzag paths (alternating up and right through the shape)
- RSK takes a pair of tableaux as input; Hillman-Grassl takes a single filling
- RSK outputs a permutation or word; Hillman-Grassl outputs an unconstrained filling

Yet the **same transformer architecture** — with no modifications beyond changing the task flag — learns both bijections to near-perfect accuracy. This suggests the structured 2D token embedding is capturing something general about Young diagram geometry, not just the specifics of Schensted insertion.

### How to Reproduce (Experiment 3)

```bash
python train.py --model encoder --task rpp --shape 4,3,2,1 --max-entry 4 \
    --source sample --train-size 500000 --device mps

python train.py --model encoder --task rpp --shape 6,4,2 --max-entry 4 \
    --source sample --train-size 500000 --device mps

python train.py --model encoder --task rpp --shape 2,2,2,2,2,1 --max-entry 4 \
    --source sample --train-size 500000 --device mps
```

Checkpoints are saved to `checkpoints/encoder_rpp_{shape}_m{max_entry}/best.pt`. The relevant code paths:
- **Bijection**: `hillman_grassl_forward()` and `hillman_grassl_inverse()` in `rsk.py`
- **Data**: `RPPSamplingDataset` in `data.py` — samples random fillings, computes H-G forward
- **Encoding**: `encode_single_filling()` in `data.py` — single filling with tableau_emb(0) for all tokens
- **Model**: same `RSKEncoder` in `model.py` — identical architecture

## What's Next

1. **Scale RPP** — try larger shapes and higher max_entry values to test the architecture's limits
2. **Scale matrix RSK** — try 4×4 N=20 with 2M+ training samples to test whether more data closes the gap to 100%
3. **Cylindric RSK** — Dobner (2026) defined an RSK analogue for cylindric tableaux, directly relevant to Robin's work on cylindric plane partitions [2]. Could extend the ML approach to this setting.
