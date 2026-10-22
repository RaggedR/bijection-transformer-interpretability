# Phase 5: Cylindric Plane Partitions via Growth Diagrams

## Overview

Learn the cylindric RSK bijection: **CPP(π) ↔ (γ, ALCD(π))** using a growth diagram built from the Burge local rule. This is the punchline of the research plan — the local rule is the *only* way to compute this bijection (no direct algorithm like Hillman-Grassl exists for the cylindric case).

**Reference**: Robin Langer's thesis (arXiv:2110.12629), Chapter 4 (§4.2–4.3).

## Mathematical Objects

### Cylindric plane partition (CPP)

A CPP with profile π ∈ Bin(n,m) is:
- An interlacing sequence of partitions (μ⁰, μ¹, ..., μᵀ) with T = n+m and μ⁰ = μᵀ (periodic boundary)
- Equivalently: a **filling of the cylindric diagram** λ̂(π) with non-negative integers, weakly decreasing along rows and columns
- An ordinary RPP is the special case μ⁰ = μᵀ = ∅

The profile π is a binary string: π_k = 0 means μᵏ ⊃ μᵏ⁻¹ (add horizontal strip), π_k = 1 means μᵏ ⊂ μᵏ⁻¹ (remove horizontal strip).

### Arbitrarily labelled cylindric diagram (ALCD)

An ALCD with profile π is a non-negative integer labelling of the faces (boxes) of the cylindric poset 𝔊(π), with only finitely many nonzero labels.

**Depth** d: the smallest k such that all boxes at depth ≥ k have label zero.

### The bijection

```
ψ_π : (γ, ALCD(π)) → CPP(π)
```

where γ is an integer partition (the "base partition").

**Cylindric growth diagram** (Definition 4.3.4):
- Vertices labelled by integer partitions
- Faces labelled by non-negative integers
- Upper boundary → CPP (interlacing sequence)
- Faces → ALCD labels
- Lower boundary → γ repeated (below depth d, everything is γ)
- **At each face**: 𝔘_{α,β}(m, μ) = λ (Burge local rule)

**Key propositions**:
- Prop 4.3.1: Every CPP uniquely determines a growth diagram (apply 𝔇 top-down)
- Prop 4.3.2: Every (γ, ALCD) uniquely determines a growth diagram (apply 𝔘 bottom-up)

### Cylindric inversion coordinates

Each box of the cylindric diagram has coordinates (i, j, k) where:
- i = position in the profile (0 to T-1, periodic)
- j = cylindric column
- k = depth / winding number

Cylindric hook length: h = j - i + kT (thesis §1.3.4, Lemma 1.3.2)

### Weight

|𝔡| = Σ label(b) × |hook(b)| summed over all boxes b in the cylindric diagram (eq 1.10).

The bijection is weight-preserving: Σ|μᵏ| tracks the CPP weight, while T|γ| + |𝔡| gives the (γ, ALCD) weight.

## ML Task

**Input**: CPP filling — structured tokens (weakly decreasing rows/cols)
**Target**: ALCD face labels — unconstrained non-negative integers

This parallels Phase 4 exactly: structured input → unstructured target.

### Data generation

1. Fix profile π ∈ Bin(n,m)
2. Sample γ (random partition, bounded parts/size)
3. Sample 𝔡 (random ALCD: bounded depth d, bounded max label value)
4. Build growth diagram bottom-up using 𝔘 at each face → read CPP from upper boundary
5. Training pair: (CPP, ALCD face labels) + γ

### Token encoding

Same 4-component structure as RPPs:
```
embedding = value_emb(v) + pos_i_emb(i) + pos_j_emb(j) + depth_emb(k)
```

- value_emb: entry value in the CPP (for input) or ALCD (for target)
- pos_i_emb: position in profile (0 to T-1) — periodicity is learned, not hardcoded
- pos_j_emb: cylindric column coordinate
- depth_emb: winding number k — replaces tableau_emb, which was constant (=0) for RPPs

Architecture unchanged: same TokenEmbedding, RSKEncoder, parallel heads.

## Implementation Plan

### Step 1: `rsk.py` — Growth diagram engine

**Cylindric diagram utilities:**
- `cylindric_diagram_cells(profile, max_depth)` → list of (i, j, k) coordinates for all cells
- `cylindric_hook_length(i, j, k, T)` → j - i + kT

**Growth diagram forward** (bottom-up, for data generation):
```python
def growth_diagram_forward(profile, gamma, alcd, max_depth):
    """
    (γ, ALCD) → CPP via growth diagram.

    Build from bottom: assign γ to all vertices below depth d.
    At each face, apply 𝔘_{α,β}(m, μ) = λ to propagate upward.
    Return the CPP (upper boundary vertex labels).
    """
```

**Growth diagram inverse** (top-down, for verification):
```python
def growth_diagram_inverse(profile, cpp):
    """
    CPP → (γ, ALCD) via growth diagram.

    Start from upper boundary (the CPP interlacing sequence).
    At each face, apply 𝔇_{α,β}(λ) = (m, μ) to propagate downward.
    Return (γ, face labels).
    """
```

**Sampling:**
- `sample_alcd(profile, max_depth, max_label, rng)` → random ALCD
- `sample_gamma(max_parts, max_part_size, rng)` → random partition
- `verify_cylindric_bijection(profile, num_samples)` → round-trip test

Uses existing `burge_forward_rule` and `burge_inverse_rule` from rsk.py.

### Step 2: `config.py` — Cylindric task config

New task `"cylindric"` with parameters:
- `profile`: binary string/tuple (e.g., (0,0,1,0,1))
- `max_depth`: maximum ALCD depth
- `max_label`: maximum face label value
- `max_gamma_parts` / `max_gamma_size`: bounds on base partition γ

Derived:
- `seq_len` = number of cells in cylindric diagram at given depth
- `vocab_size` = max_label + 1 (or max CPP entry value + 1 for input)
- `num_tokens` = number of CPP cells (input tokens)

### Step 3: `data.py` — CylindricSamplingDataset

```python
class CylindricSamplingDataset(Dataset):
    def __getitem__(self, idx):
        gamma = sample_gamma(...)
        alcd = sample_alcd(profile, max_depth, max_label, rng)
        cpp = growth_diagram_forward(profile, gamma, alcd)

        # Input: CPP as structured tokens
        values, positions = encode_cylindric_filling(cpp, profile)

        # Target: ALCD face labels in reading order
        target = [alcd[i][j][k] for (i,j,k) in cell_order]

        return values, positions, target
```

### Step 4: `model.py` — Minimal change

Same RSKEncoder. The depth_emb slot is already tableau_emb(0) for RPPs — now it uses depth_emb(k) with a larger embedding table. May need to increase `nn.Embedding(2, d)` to `nn.Embedding(max_depth + 1, d)`.

### Step 5: `train.py` — CLI

```
--task cylindric
--profile 00101        # binary string
--max-depth 2
--max-label 3
--max-gamma-parts 4
--max-gamma-size 6
```

## Experiment Plan

### Start small
| Profile | T | n | m | Description | max_depth | max_label |
|---------|---|---|---|-------------|-----------|-----------|
| 010     | 3 | 2 | 1 | Smallest non-trivial | 2 | 3 |
| 0101    | 4 | 2 | 2 | Square-ish | 2 | 3 |
| 00101   | 5 | 3 | 2 | Thesis example | 2 | 3 |

### Then scale
- Increase max_depth (longer-range dependencies through the cylinder)
- Increase max_label (larger target space)
- Vary the profile (affects which local rule applications are needed)
- Compare profiles with different numbers of inversions

## Key Considerations

### Growth diagram computation order

The growth diagram is filled face by face. The order matters:
- **Forward (bottom-up)**: process anti-diagonals from bottom to top
- **Inverse (top-down)**: process anti-diagonals from top to bottom
- **Commutativity** (Lemma 4.3.6): 𝔏_i ∘ 𝔏_j = 𝔏_j ∘ 𝔏_i, so within each anti-diagonal the order doesn't matter

### Rotation operator

When building the bijection recursively, sometimes the **rotation operator** σ (Definition 1.3.5) is needed:
σ(π)_i = π_{(i+1) mod T}

This rotates the cylinder. If the profile has "maximal" inversions, you rotate to get to a profile where a local rule can be applied. The growth diagram handles this automatically by its structure — the rotation corresponds to the cylindric equivalence (x,y) ≡ (x+n, y-m).

### What makes this different from Phase 4

1. **No direct algorithm**: Must use growth diagram with local rules
2. **Periodic boundary**: μ⁰ = μᵀ introduces global structure
3. **Base partition γ**: RPPs had γ = ∅; CPPs have nontrivial γ
4. **Depth dimension**: RPPs lived on a flat shape; CPPs wind around the cylinder
5. **The punchline**: If SAE analysis (Phase 3) extracts a local-rule feature from both RPP and CPP models, that's strong evidence the transformer discovered Fomin's framework

## Files to Modify

- `rsk.py` — Growth diagram engine (largest change, ~200 lines)
- `config.py` — Cylindric task parameters
- `data.py` — CylindricSamplingDataset
- `model.py` — Increase tableau_emb capacity (or add depth_emb)
- `train.py` — CLI args, task routing

## Existing Code to Reuse

- `burge_forward_rule(alpha, beta, m, mu)` → 𝔘_{α,β}(m, μ) = λ (rsk.py:731)
- `burge_inverse_rule(alpha, beta, lam)` → 𝔇_{α,β}(λ) = (m, μ) (rsk.py:792)
- `partition_conjugate()`, `hook_lengths()`, `is_rpp()` (rsk.py)
- `encode_single_filling()` (data.py:224) — adapt for cylindric coordinates
- `RPPSamplingDataset` pattern (data.py:245) — template for CylindricSamplingDataset

## Implementation Notes (from thesis study)

### Growth diagram structure (Def 4.3.4, pp.68-69)

The growth diagram lives on the **cylindric poset** 𝔊(n,m) = ℤ²/((x,y)≡(x+n,y-m)).

- **Vertices** labelled by integer partitions
- **Faces** (u; v1, v2; w) labelled by non-negative integers
- At each face: if u,v1,v2,w have labels μ,α,β,λ then 𝔘_{α,β}(m, μ) = λ
- **Upper boundary** = the CPP (interlacing sequence of T partitions)
- Below some depth, all vertices = γ, all faces = 0

### Faces = boxes of the cylindric diagram λ̂(π)

The ALCD face labels live on the **boxes of the cylindric diagram**, NOT just at "inversion positions per depth level." The cylindric diagram λ̂(π) has boxes indexed by cylindric inversion coordinates (i, j, k) where:
- i = which "1" in the profile (the 1 of the inversion pair)
- j = which "0" in the profile (the 0 of the inversion pair)
- k = winding number (depth around the cylinder)
- Hook length h = j - i + kT (Lemma 1.3.2, p.23)

For π = 10100 (T=5, n=3, m=2), the fundamental domain has boxes at inversions of the infinite periodic profile ...10100|10100|10100|...

### Worked example (§4.3.3, p.69): π = 00101, γ = (3,2)

The growth diagram is a diamond lattice. Reading the figure:

**Upper boundary (CPP)**: (6,4,3,2), (6,4,3,2), (4,3,2,1), (5,3,2), (6,4,3,2), (4,3,2)
Note: profile is read right to left in the thesis figure. The interlacing is:
- μ⁰ = (6,4,3,2)
- Then π₁=0: μ⁰ ⊂ μ¹ (add)
- Then π₂=0: μ¹ ⊂ μ² (add)
- Then π₃=1: μ² ⊃ μ³ (remove)
- Then π₄=0: μ³ ⊂ μ⁴ (add)
- Then π₅=1: μ⁴ ⊃ μ⁵ = μ⁰ (remove, wraps)

**ALCD face labels** (reading from the diamond): 5, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 5
Total 15 faces in the growth diagram.

**Bottom**: all vertices = (3,2) = γ.

### Forward algorithm ((γ, ALCD) → CPP via Prop 4.3.2)

1. Assign label 0 to faces with depth > d (depth of ALCD)
2. Assign γ to all vertices lying below a face with depth > d
3. Property (3) of Def 4.3.4 (the local rule 𝔘) uniquely determines remaining vertices, bottom-up
4. Read the CPP from the upper boundary

### Inverse algorithm (CPP → (γ, ALCD) via Prop 4.3.1)

1. Start from upper boundary (the CPP interlacing sequence)
2. At each face, apply 𝔇_{α,β}(λ) = (m, μ) to propagate downward
3. Stop when all vertices are equal (= γ) and all faces are 0

### Key lemmas

- **Lemma 4.3.3**: 𝔇_{γ,γ}(γ) = (0, γ) — base case: when all neighbors are γ, inverse gives m=0 and μ=γ
- **Lemma 4.3.6**: 𝔏_i ∘ 𝔏_j = 𝔏_j ∘ 𝔏_i — local rules at different inversions commute (because inversions are never adjacent: a "10" pattern prevents the next position from starting another "10")
- **Prop 4.4.2**: The bijection is strongly weight-preserving: |μᵏ| = |γ| + |diag(k)|_𝔡

### Processing order for the diamond lattice

The diamond lattice has faces arranged on anti-diagonals. Each anti-diagonal can be processed independently (commutativity). Bottom-up: start from the lowest anti-diagonal (all vertices = γ), apply 𝔘 at each face, propagate upward. The face (u; v1, v2; w) has u=bottom, w=top, v1=left, v2=right.

### Concrete data structure

For implementation, represent the growth diagram as a 2D grid of vertices where:
- Each "row" is a path with profile π (T vertices, periodic)
- Row 0 = upper boundary (CPP), row D = lower boundary (all γ)
- Between rows r and r+1, faces exist at specific positions determined by the cylindric poset structure
- At each face: apply 𝔘 (forward) or 𝔇 (inverse) with the four surrounding vertices
