# Research Plan: From Permutations to Local Rules via the RSK Hierarchy

## Vision

Train neural networks to learn the inverse RSK correspondence at increasing levels of generality — from permutations (done) through words, matrices, reverse plane partitions, and cylindric plane partitions — then use **sparse autoencoders** (SAEs) to extract the internal features the networks learn. The central hypothesis: if the transformer discovers the mathematical structure of RSK, **Fomin's local rule should emerge as an identifiable feature** in the network's activations, invariant across all levels of the hierarchy.

This connects mechanistic interpretability to algebraic combinatorics in a setting with provable ground truth.

## Background

### The RSK hierarchy

The RSK correspondence operates at several levels of generality, each strictly containing the previous:

| Level | Input | Output (P, Q) | Constraint |
|-------|-------|---------------|------------|
| **Permutations** | σ ∈ S_n | (SYT, SYT) of same shape λ ⊢ n | Bijection |
| **Words** | w ∈ {1,...,k}^m | (SSYT over {1,...,k}, SYT with m boxes) | Bijection |
| **Two-line arrays** | Generalized permutation | (SSYT, SSYT) of same shape | Bijection |
| **Integer matrices** | A ∈ N^{k×m}, row/col sums fixed | (SSYT, SSYT) of same shape | Bijection (Knuth) |
| **Reverse plane partitions** | RPP of shape λ/μ | Growth diagram on a grid | Via local rules |
| **Cylindric plane partitions** | CPP on cylindric shape | Growth diagram with periodic boundary | Via local rules |

Key: SYT = standard Young tableau, SSYT = semistandard Young tableau.

### Fomin's growth diagrams and local rules

All levels of RSK can be understood through a single framework: **Fomin growth diagrams** (Fomin, 1986/1994). The idea:

1. Arrange the input data on the boundary of a rectangular grid
2. At each vertex of the grid, place a partition (or more generally, an element of a differential poset)
3. Fill the grid using a **local rule**: given three adjacent partitions (SW, S, W), the fourth (NE) is determined by a rule that depends only on these three neighbors
4. The output tableaux P, Q appear on the remaining two boundaries

The local rule for RSK is:

> Given partitions (λ_SW, λ_S, λ_W) where λ_S and λ_W each cover or equal λ_SW, determine λ_NE — the unique partition covering or equaling both λ_S and λ_W such that |λ_NE| - |λ_W| = |λ_S| - |λ_SW|.

For permutations (0/1 matrix on boundary), the partitions at vertices differ by at most one box, reducing to a simple case analysis. For integer matrices, the rule generalizes naturally. For RPPs and CPPs, the objects at vertices become partitions rather than shapes, but the local rule remains structurally identical.

**The key insight**: the local rule barely changes across the hierarchy. What changes is the boundary conditions and the objects at grid vertices. This is why the hierarchy is the right experimental scaffold — if the network learns the local rule, it should transfer.

### Connection to Langer (2013)

Robin Langer's thesis (arXiv:2110.12629) uses Fomin growth diagrams for generalized RSK correspondences to prove Borodin's identity for cylindric plane partitions. Part I gives a bijective proof using exactly the local-rule framework described above, extended to cylindric (periodic) boundary conditions. The final phase of this research plan — cylindric plane partitions — directly builds on this work.

## Phase 1: Words (inverse RSK for words)

**Goal**: Given (P, Q) where P is SSYT over {1,...,k} and Q is SYT with m boxes, predict the word w ∈ {1,...,k}^m.

### Mathematical setup

- Input: word w = w_1 w_2 ... w_m with each w_i ∈ {1,...,k}
- Forward RSK: insert w_1, w_2, ..., w_m into P (Schensted insertion produces SSYT); Q records insertion order (remains SYT)
- Inverse: given (P_ssyt, Q_syt), recover w
- The bijection is between words of length m over alphabet {1,...,k} and pairs (P, Q) of same shape, P semistandard with entries in {1,...,k}, Q standard with entries in {1,...,m}

### What changes in the code

**`rsk.py`**:
- `rsk_forward` already works for words (Schensted insertion handles repeated values)
- Need `rsk_inverse_word`: process Q entries in decreasing order as before, but use column-insertion (not row-insertion) for the reverse bump when handling semistandard P. Actually, the same reverse-bumping works — the difference is that recovered values may repeat
- Add `is_semistandard_young_tableau` validator (weakly increasing rows, strictly increasing columns)
- Add `verify_word_bijection` for small cases

**`config.py`**:
- Split `n` into `seq_len` (m = word length) and `vocab_size` (k = alphabet size)
- `num_tokens = 2 * seq_len` (P has m entries, Q has m entries)
- `num_heads = seq_len` (m output positions)
- Each head outputs `vocab_size` logits (k classes)

**`data.py`**:
- `WordSamplingDataset`: generate random words by sampling m values from {1,...,k} with replacement
- Target encoding: `[w_i - 1 for w_i in word]` (0-indexed, values in {0,...,k-1})
- `encode_tableaux` already handles SSYT (values need not be distinct)

**`model.py`**:
- Output heads: `seq_len` heads, each `Linear(d_model, vocab_size)` (was n heads × n classes)
- Remove/make optional `masked_greedy_decode` — words can have repeated values, so no permutation constraint

**`train.py`**:
- Add `--task {permutation,word}` flag (or infer from config)
- For words: use argmax decoding (no masking needed)
- Metrics: exact match (full word correct), per-position accuracy

### Experimental design

**Controlled comparison with permutations**:
- Fix m = n = 15 (same sequence length as our best permutation result)
- Vary k: alphabet size 2, 5, 10, 15, 20, 50
- k = m = 15 with distinct values is equivalent to permutations (sanity check)
- k = 2 (binary words) should be easiest
- k >> m (large alphabet, most values unused) tests sparse regimes

**Key questions**:
- Does accuracy degrade when the permutation constraint is removed? (Masked greedy decode was doing real work)
- How does the transformer vs MLP gap change for words? (Hypothesis: still large, since 2D structure of SSYT matters regardless)
- Does difficulty scale with k, m, or the ratio k/m?

### Success criteria

- Exact match > 99% on held-out words for m = 15, k = 15
- Clear demonstration that the model generalizes (trained on vanishing fraction of k^m space)

### Lifting permutation RSK to words: the anti-diagonal expansion

The fundamental reason the RSKEncoder (mean pool → m independent heads) fails on words is **computational**: it allocates m classification heads for m positions, but the growth diagram requires more work than that. The following explains why, and motivates the level decoder architecture.

**Permutation RSK in the growth diagram.** For σ ∈ S_n, the growth diagram is an n×n grid with a 0/1 filling (one 1 per row and column). At each cell, the Fomin local rule takes three partitions (SW, S, W) and produces the fourth (NE). The filling entry determines whether a box is added:

- **Entry 0** (no insertion): NE is the join of S and W over SW — partitions propagate but no new box appears.
- **Entry 1** (insertion): NE gains one box beyond what propagation would give.

Total non-trivial local rule applications: n (one per 1 in the filling). Each application is a binary decision: the local rule is determined by the three neighboring partitions and the single bit "is there an insertion here?"

**Word RSK naively.** For w ∈ {1,...,k}^m, the growth diagram is an m×k grid with a 0/1 filling: M_{i,j} = δ(w_i = j). Again one 1 per row (each word position inserts exactly one letter), but now columns can have multiple 1's (letters repeat). Total non-trivial cells: m. So far this looks just like permutation RSK — same number of local rule applications.

**The Burge expansion reveals the true complexity.** But the 0/1 picture is deceptive. In the Burge correspondence for integer matrices, a cell with entry a > 0 can be **expanded** into an a×a block with 1's on the anti-diagonal:

```
Cell (i,j) with entry a=3:

  ┌─────────┐         ┌───┬───┬───┐
  │         │         │   │   │ 1 │  ← anti-diagonal
  │  a = 3  │   →     │   │ 1 │   │     of 1's
  │         │         │ 1 │   │   │
  └─────────┘         └───┴───┴───┘
```

Each 1 on the anti-diagonal triggers one elementary permutation RSK local rule application. The anti-diagonal ordering ensures causal dependency: each application sees the result of the previous one.

For word RSK, the 0/1 matrix has m non-zero cells, each with value 1. No expansion needed. **But** — and this is the key — the *value* j = w_i tells us which column the insertion occurs in. Column j of the growth diagram represents "alphabet letter j", and by the time we reach column j, the diagram has accumulated j columns of partition evolution. The information needed to determine the local rule outcome at column j depends on all j preceding columns.

More precisely: we can reinterpret word RSK as a **refined** growth diagram where column j is expanded into j sub-columns (representing the j elementary steps of a Burge-expanded insertion of value j). In this refined picture:

- Position i with value w_i = j contributes **j** elementary local rule applications along an anti-diagonal
- Position i with value w_i = 1 contributes **1** application (like permutation RSK)
- The total number of elementary applications across all positions is **Σᵢ wᵢ**
- For random words over {1,...,k}^m, the expected total is m(k+1)/2

**The capacity mismatch.** The RSKEncoder allocates m independent heads — one per output position. Each head makes a single k-way classification independently, with no inter-head communication. But the growth diagram tells us:

1. The m positions are **not independent**: the local rule at position i depends on the cumulative state of all previous insertions.
2. A single k-way classification per position is too coarse: predicting w_i = j requires j ordered binary decisions, each depending on the previous.
3. Total computational budget should scale as **Σ wᵢ** ≈ m(k+1)/2, not m.

For m=15, k=10: the encoder allocates 15 heads. The growth diagram needs ~82.5 elementary operations on average. The encoder is 5.5× underprovisioned.

**The level decoder resolves this.** The RSKLevelDecoder has k × m = 150 decoder tokens, one per (level, position) pair. Token (l, i) represents the binary decision "does position i grow at level l?" The block-causal mask ensures:

- Within a level: all m positions see each other (capturing interactions between concurrent insertions)
- Across levels: level l sees levels 0..l-1 but not l+1..k-1 (causal dependency along the anti-diagonal)

The ordinal encoding — w_i = j maps to (1,1,...,1,0,...,0) with j ones — directly mirrors the anti-diagonal expansion: j "grow" decisions followed by k-j "don't grow" decisions.

**Why ordinal, not classification.** The ordinal structure imposes a monotonicity constraint: if the model predicts "grow at level l", it must also predict "grow at level l-1". This is not just a useful inductive bias — it is **mathematically necessary**. In the anti-diagonal expansion, the grow decisions are ordered: the level-0 decision happens before level-1, which happens before level-2, and so on. Predicting "don't grow at level 2 but grow at level 3" would correspond to a non-monotone partition sequence, violating the fundamental interlacing property of the growth diagram.

**Connection to Phase 2.** For integer matrices (Phase 2), a cell (i,j) in the matrix can have value a_{i,j} > 1. The Burge expansion replaces this with an a_{i,j} × a_{i,j} anti-diagonal block. The total work becomes Σ_{i,j} a_{i,j} = |λ| (the total number of boxes in the tableau). The level decoder architecture generalizes naturally: the number of "grow" decisions at position (i,j) is exactly a_{i,j}.

## Phase 2: Full RSK (integer matrices)

**Goal**: Given (P, Q) both SSYT of the same shape, predict the integer matrix A ∈ N^{k×m} (or equivalently the two-line array / generalized permutation) that produced them.

### Mathematical setup

- The full RSK correspondence (Knuth's generalization) maps non-negative integer matrices to pairs of SSYT
- A matrix A with row sums r_1,...,r_k and column sums c_1,...,c_m maps to (P, Q) where P has entries in {1,...,k}, Q has entries in {1,...,m}, both semistandard
- The trace tr(A) = |λ| = number of boxes in each tableau

### What changes vs Phase 1

- Q is now SSYT (not SYT) — entries can repeat
- The input to `rsk_forward` is a two-line array (pairs (i,j) sorted lexicographically)
- Output is no longer a sequence but a **matrix** — this is the big structural change
- Need to decide: predict the matrix entries directly, or predict the two-line array?

### Architecture options

**Option A: Predict the two-line array** (sequence of pairs)
- Variable-length output (length = tr(A) = number of boxes)
- Each output token is a pair (i, j) — could factor as two separate predictions
- Requires autoregressive decoding or fixed-length padding

**Option B: Predict the matrix entries**
- Fixed k×m grid of non-negative integers
- Each entry is a count (regression or bounded classification)
- More natural for the SAE analysis (grid structure mirrors growth diagram)

**Option C: Predict the matrix as k×m independent classifications**
- Each A_{i,j} ∈ {0, 1, ..., max_entry}
- Bounded by the number of boxes: sum of entries = |λ|
- k×m heads, each with max_entry+1 classes

Recommendation: **Option C**. It preserves the grid structure that maps onto the growth diagram, and the bounded classification is a natural extension of our existing head architecture.

### Key questions

- How does the model handle the constraint that matrix entries must sum to |λ|?
- Does attention pattern in the trained model reflect the growth diagram grid structure?
- Is the model harder to train when both P and Q are semistandard?

## Phase 3: Sparse autoencoder analysis

**Goal**: Apply SAEs to the internal activations of models trained in Phases 0–2 to extract interpretable features, looking for the emergence of the local rule.

### Methodology

**Training SAEs**:
1. For each trained model (permutation, word, matrix), collect residual stream activations from all 6 transformer layers on a large evaluation set (100K+ examples)
2. Train overcomplete sparse autoencoders on each layer's activations
   - Dictionary size: 4× to 16× overcomplete (512 to 2048 features for d_model=128)
   - L1 sparsity penalty, tuned to achieve ~10-20 active features per input
   - Standard architecture: encoder (linear + ReLU), decoder (linear), reconstruction + sparsity loss
3. Analyze learned features

**Feature analysis**:
For each SAE feature f:
1. **Max-activating examples**: find the inputs that most strongly activate f
2. **Positional analysis**: which token positions activate f? Do they form patterns related to the growth diagram?
3. **Cross-model comparison**: train SAEs on permutation, word, and matrix models separately. Do similar features appear? Measure cosine similarity between feature dictionaries
4. **Local rule test**: construct synthetic inputs that isolate specific local-rule configurations. Test whether any feature's activation correlates with specific local-rule cases
5. **Causal intervention**: ablate individual features and measure which aspects of the output degrade. If a feature implements part of the local rule, ablating it should cause systematic errors

### What would constitute evidence for the local rule

**Strong evidence**:
- A feature (or small set of features) that activates on tokens whose (row, col) positions correspond to specific growth-diagram cells
- The same feature appearing in models trained on permutations, words, and matrices (invariance across RSK levels)
- Feature activation pattern matching the case analysis of the local rule: different activation for the "do nothing" case vs the "add a box" case vs the "bump" case
- Ablating the feature causes errors that are **systematic** (not random) — e.g., the model gets the local-rule case wrong at specific grid positions

**Interesting negative result**:
- The network uses a **different** decomposition of RSK — not the local rule, but some other algorithm. This would be mathematically interesting: what decomposition did it find?

### Timeline

SAE analysis should begin after at least Phases 0 and 1 are complete (two points of comparison). Phase 2 provides a third point. The analysis becomes more powerful with each additional RSK level.

## Phase 4: Reverse plane partitions

**Goal**: Learn the RSK-type correspondence for reverse plane partitions via growth diagrams.

### Mathematical setup

A reverse plane partition (RPP) of shape λ/μ is a filling of the skew shape with non-negative integers that are:
- Weakly increasing along rows (left to right)
- Weakly increasing down columns (top to bottom)

The RSK-type correspondence for RPPs is most naturally expressed via growth diagrams. The local rule operates on **partitions** (not just shapes/diagrams) at grid vertices.

### Why this phase matters

RPPs are the natural bridge between "classical RSK with matrices" and "cylindric plane partitions." The growth diagram formulation is **essential** here — there is no clean Schensted-insertion description. This forces us to implement the growth diagram framework explicitly, which sets up Phase 5.

### Architecture considerations

- Input: a pair of tableaux-like objects arising from the RPP growth diagram
- The objects at grid vertices are now partitions (sequences of non-negative integers), not just shapes
- Token embedding needs to encode partition-valued data
- The local rule becomes more complex but retains the same 3→1 structure

### Implementation approach

- Implement growth diagrams as the primary RSK engine (not Schensted insertion)
- The local rule module should be **swappable** — same framework, different rule
- This modularity pays off in Phase 5

## Phase 5: Cylindric plane partitions

**Goal**: Learn the cylindric RSK correspondence from Langer (2013).

### Mathematical setup

A cylindric plane partition is a reverse plane partition on a cylindric shape — the shape "wraps around" with periodic boundary conditions. The RSK correspondence for CPPs was studied in Langer's thesis using Fomin growth diagrams with cylindric boundary conditions.

The key difference from RPPs:
- The growth diagram grid is **periodic** in one direction
- The local rule is the **same** — only the boundary conditions change
- The correspondence connects to Borodin's identity and cylindric Schur functions

### Why this is the punchline

If the SAE analysis in Phase 3 extracts a local-rule feature from models trained on permutations, words, and matrices — and that **same feature** appears in a model trained on cylindric plane partitions (where only the boundary topology changed) — that is strong evidence that:

1. The transformer discovered Fomin's framework
2. The local rule is a genuine computational primitive, not an artifact of the specific combinatorial setting
3. Neural networks can discover deep mathematical structure

This is Robin Langer's own theorem being rediscovered by a neural network.

### Practical notes

- Cylindric shapes require careful handling of periodicity in the token embedding (row/col positions wrap)
- May need positional embeddings that encode cylindric topology (e.g., sinusoidal with period matching the cylinder circumference)
- Data generation: sample CPPs uniformly? Use the transfer matrix method? TBD based on what's efficient

## Architecture Evolution

Summary of how the model architecture changes across phases:

```
Phase 0 (Permutations — DONE):
  Input:  2n tokens (P_syt, Q_syt)
  Output: n heads × n classes
  Decode: masked greedy (permutation constraint)

Phase 1 (Words):
  Input:  2m tokens (P_ssyt, Q_syt)
  Output: m heads × k classes
  Decode: argmax (no constraint, values repeat)

Phase 2 (Matrices):
  Input:  2|λ| tokens (P_ssyt, Q_ssyt)
  Output: k×m heads × (max_entry+1) classes  [or regression]
  Decode: argmax with sum constraint (entries sum to |λ|)

Phase 4 (RPP):
  Input:  tokens from growth diagram boundary
  Output: RPP entries (weakly increasing constraint)
  Decode: constrained decoding respecting monotonicity

Phase 5 (Cylindric):
  Input:  tokens with periodic positional encoding
  Output: CPP entries (periodic monotonicity)
  Decode: constrained decoding with wraparound
```

## SAE Architecture (Phase 3, applied to all models)

```
For each trained model and each layer l ∈ {0,...,5}:
  Collect activations h ∈ R^d_model on eval set
  Train SAE:
    encode: z = ReLU(W_enc · h + b_enc)     z ∈ R^(d_dict)
    decode: h_hat = W_dec · z + b_dec        h_hat ∈ R^(d_model)
    loss = ||h - h_hat||² + α · ||z||₁

  d_dict ∈ {512, 1024, 2048}  (4× to 16× overcomplete)
  α tuned for ~10-20 active features per input
```

## Dependencies and Tools

**Current**: `torch`, `datasets` (HuggingFace)

**New for SAE work**:
- SAE training: custom (simple enough to implement from scratch) or adapt from existing SAE libraries
- Visualization: `matplotlib` for feature analysis, activation heatmaps
- Potentially `transformer-lens` or similar for clean hook-based activation extraction (or just use PyTorch forward hooks)

## Open Questions

1. **Sequence length variability**: For matrices (Phase 2), the number of tokens depends on |λ| which varies. Do we pad to a fixed maximum, or use variable-length batching?

2. **What "the same feature" means across models**: Models trained on different RSK levels have different training data and potentially different d_model. How do we rigorously compare SAE features across models? Options: shared embedding space, representational similarity analysis, or behavioral comparison (same max-activating patterns).

3. **Autoregressive vs parallel decoding**: For RPPs and CPPs, the output has ordering constraints (weak increase). Should we switch to autoregressive decoding, or can parallel heads with constrained decoding still work?

4. **Scaling**: Our current model is small (1.2M params, d=128). For SAE interpretability, is this an advantage (simpler features, less superposition) or disadvantage (features might not cleanly separate at low dimension)?

5. **Growth diagram as architecture**: Instead of using SAEs to *find* the local rule post-hoc, could we build the growth diagram structure directly into the architecture? E.g., a 2D grid of small networks, each implementing the local rule, composed in the correct order. This would be a strong inductive bias — but comparing it against the unconstrained transformer is exactly the point.

## References

- Fomin, S. (1986). Generalized Robinson-Schensted-Knuth correspondence. *Zapiski Nauchnykh Seminarov LOMI*, 155, 156-175.
- Fomin, S. (1994). Duality of graded graphs. *Journal of Algebraic Combinatorics*, 3(4), 357-404.
- Fomin, S. (1995). Schensted algorithms for dual graded graphs. *Journal of Algebraic Combinatorics*, 4(1), 5-45.
- Knuth, D. E. (1970). Permutations, matrices, and generalized Young tableaux. *Pacific Journal of Mathematics*, 34(3), 709-727.
- Langer, R. (2013). *Cylindric plane partitions, Lambda determinants, Commutants in semicircular systems*. PhD thesis, Universite Paris-Est. arXiv:2110.12629.
- Krattenthaler, C. (2006). Growth diagrams, and increasing and decreasing chains in fillings of Ferrers shapes. *Advances in Applied Mathematics*, 37(3), 404-431.
- Dobner, S. (2026). RSK analogues for cylindric tableaux. arXiv:2603.09119.
- Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Anthropic Research*.
- Cunningham, H. et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *ICLR 2024*.
