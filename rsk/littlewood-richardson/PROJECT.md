# PROJECT.md — LR Bijection Transformer

## Goal

Train a small transformer to learn bijections between different combinatorial representations of Littlewood-Richardson coefficients c^ν_{λ,μ}. Extends the RSK transformer (parent directory `../`) to a new class of combinatorial bijection.

## Mathematical Setup

LR coefficients c^ν_{λ,μ} count several independent combinatorial objects:

| Model | Object counted | Geometry |
|-------|---------------|----------|
| **LR tableaux** | Skew SSYT of shape ν/λ, content μ, lattice word condition | Rectangular grid (skew shape) |
| **KTW puzzles** | Triangular tilings with 3 rhombus types | Triangular grid |
| **Hives** | Integer vertex assignments satisfying rhombus inequalities | Triangular grid |
| **GT patterns** | Interlacing triangular arrays | Triangular array |
| **Pipe dreams** | Grid configurations of crosses and bumps | Rectangular grid |

All models give the same count. Known bijections connect them. Clio's paper (github.com/clio-vega/integrability-hierarchy) argues these bijections arise from Yang-Baxter groupoid morphisms of an underlying integrable lattice model.

**The ML task**: given one representation (e.g., a KTW puzzle), predict the corresponding object in another representation (e.g., the LR tableau). Each puzzle UNIQUELY determines an LR tableau via the bijection, so this is a well-defined function learning task — structurally similar to inverse RSK, but with variable input/output sizes across different (λ,μ,ν) triples.

## Bijection Targets (in priority order)

1. **KTW puzzle → LR tableau** (Purbhoo mosaic bijection) — the primary target
2. **LR tableau → KTW puzzle** (inverse direction)
3. **LR tableau ↔ hive** (simpler warmup — validates the pipeline)

## Existing Code

### Parent RSK transformer (`../`)
The full training infrastructure. Key files:
- `model.py` — `TokenEmbedding` (value + row + col + tableau_id embeddings), `RSKEncoder` (6-layer, d=128, 8 heads, ~1.2M params), `BaselineMLP`
- `data.py` — dataset classes, token encoding, dataloaders. Pattern: each task has a `SamplingDataset` that generates data on the fly.
- `train.py` — training loop, masked greedy decode, evaluation, CLI with `--task`, `--source sample`
- `config.py` — `ModelConfig` (add new tasks by adding an `elif self.task == "..."` block)
- `rsk.py` — all bijection implementations (pure Python, no external deps)

### Robin's puzzle codebase (`~/git/puzzles/`)
- `lr_coefficients.py` — `lr_puzzles(lam, mu, nu)` returns `list[Puzzle]` where `len == c^ν_{λ,μ}`
  - `Puzzle.levels`: tuple of N+1 tuples — horizontal edge labels (from {0, 1, 10}) at each row, bottom to apex
  - `Puzzle.edge_labels()`: reconstructs all edge labels as dict `{((r1,j1),(r2,j2)): label}`
  - Edge labels: 0 = empty, 1 = green path, 10 = rhombus (both)
  - `partition_to_boundary(lam, n, m)` / `boundary_to_partition(bits)` — partition ↔ binary string encoding
- `transfer_operators.py` — Hopf coproduct computation via Kostka matrix inversion
- `solve_puzzle.py` — brute-force solver for cross-validation

### Clio's paper (`github.com/clio-vega/integrability-hierarchy`)
35-page expository paper: "LR Coefficients: From Puzzles to the Integrability Hierarchy." Key claims:
- All combinatorial models for c^ν_{λ,μ} are shadows of one integrable lattice model (§3)
- Bijections between models are Yang-Baxter groupoid morphisms (§4.1)
- Three-level hierarchy: solvable (braided, full YBE) → quasi-solvable (coboundary, column YBE) → combinatorial (symmetric, trivial YBE), governed by Hecke chain H_q → H_0 → C[S_n] (§5)
- Open question: does the Purbhoo puzzle↔tableau bijection decompose into R-matrix moves? (§7.1)

## Worked Example: c^{(3,2,1)}_{(2,1),(2,1)} = 2

This is the smallest case with c > 1 (used as running example in Clio's paper §2.2).

**Partitions**: λ = (2,1), μ = (2,1), ν = (3,2,1). Bounding box: n=3, m=3, so N=6.

**Puzzle 1 levels** (bottom to apex, R = rhombus = 10):
```
Level 0: 1 0 1 0 1 0    (= boundary of ν)
Level 1: 1 0 1 R 0
Level 2: 0 1 1 0
Level 3: R 1 0
Level 4: 0 1
Level 5: R
Level 6: (empty)
```

**Puzzle encoding as tokens** (6+5+4+3+2+1 = 21 tokens for the non-empty levels):
```
(val=1, row=0, col=0), (val=0, row=0, col=1), ..., (val=0, row=0, col=5),  # 6 tokens
(val=1, row=1, col=0), (val=0, row=1, col=1), ..., (val=0, row=1, col=4),  # 5 tokens
...
(val=2, row=5, col=0)                                                       # 1 token
```
where val ∈ {0, 1, 2} maps {0, 1, 10}.

**Corresponding LR tableaux** (skew shape (3,2,1)/(2,1)):
- Tableau 1: filling (1, 1, 2) → entries at boxes (0,2), (1,1), (2,0) with values 1, 1, 2
- Tableau 2: filling (1, 2, 1) → entries at boxes (0,2), (1,1), (2,0) with values 1, 2, 1

**Target encoding**: 3 classification heads, each predicting a value from {1, 2} (or from {0, 1} after 0-indexing).

## Data Format

### Puzzle encoding (INPUT)
The `Puzzle` class stores `levels`: N+1 tuples of horizontal edge labels.
- Level 0 (bottom): N labels — this IS the boundary encoding of ν
- Levels 1 to N−1: shrinking tuples of internal edge labels (these define the tiling)
- Level N (apex): empty tuple

One token per horizontal edge label across ALL levels, with structured embedding:
- `value`: the edge label (0, 1, or 10 → mapped to 0, 1, 2)
- `row`: which level (0 to N−1)
- `col`: position within the level (0 to N−r−1)
- `tableau_id`: 0 (single object, not a pair)

Total tokens: N + (N−1) + ... + 1 = N(N+1)/2. For N=5: 15 tokens. For N=7: 28.

**Note on boundary information**: level 0 encodes ν directly. The diagonal edges (encoding λ and μ) are NOT stored in levels — they're reconstructable via the deterministic △ rule (`Puzzle.edge_labels()`). So the model sees ν explicitly but must infer λ and μ from the internal tiling structure. This is analogous to how the RSK transformer sees P and Q but must infer σ — the "missing" information is what the model learns to extract. If this makes the task too hard, we can add boundary tokens with a distinct `tableau_id`.

### LR tableau encoding (OUTPUT / TARGET)
Skew SSYT of shape ν/λ with content μ. Number of boxes = |ν| − |λ|.
Target: one classification head per box position (in reading order), predicting the entry value.
The number of boxes varies across triples. Fix the head count to max possible |ν|−|λ| for the given N (which is at most N²/4). Pad unused positions with a PADDING class and mask them in the loss.

### Training strategy
Fix N = n + m (puzzle side length). Two options:

**Option A: Enumerate all triples.** For each valid (λ,μ,ν) with c > 0 fitting in the n×m box, compute all puzzle-tableau pairs via the bijection. Train on ALL pairs across ALL triples. The model learns a universal puzzle→tableau map.
- Pro: complete coverage, no sampling bias
- Con: limited data for small N; data volume is Σ c^ν_{λ,μ} over all triples

**Option B: Sample triples, enumerate within.** Randomly sample (λ,μ,ν) triples (possibly with repetition weighted by c), enumerate all pairs for each. Regenerate each epoch.
- Pro: effectively unlimited data via resampling
- Con: non-uniform coverage, may miss rare triples

Start with Option A. If data volume is insufficient, switch to B or train across multiple N values simultaneously.

## Implementation Plan

### Phase 1: Data pipeline
1. Implement LR tableau enumeration (skew SSYT with lattice word condition)
2. Implement the Purbhoo mosaic bijection (puzzle ↔ LR tableau)
3. Verify: for every (λ,μ,ν) with N ≤ 7, the bijection produces exactly c^ν_{λ,μ} pairs and is consistent with `lr_puzzles()` counts
4. Build `LRDataset` — pre-enumerate all (puzzle, tableau) pairs for given N. This is an enumeration dataset, NOT a sampling dataset (unlike RSK where you sample random permutations). The total data volume is finite and determined by N.

### Phase 2: Model adaptation
1. Add `--task lr --N 5` to `config.py` and `train.py`
2. Encode puzzles as structured tokens (levels representation)
3. Train on N=5, then N=7
4. Compare against MLP baseline

### Phase 3: Analysis
1. Error analysis: which triples are hardest? Does difficulty correlate with c^ν_{λ,μ}?
2. SAE interpretability: do features correspond to R-matrix moves or local rule steps?
3. Cross-task transfer: does a model trained on puzzle→tableau generalize to puzzle→hive?

### Future: Cross-level experiments
Train on classical (Schur) LR bijections, test transfer to K-theoretic (Grothendieck) — does the model generalize across the horizontal axis of Clio's phase diagram?

## Key References

- Purbhoo, "Puzzles, tableaux, and mosaics" (arXiv:0705.1184) — the bijection we implement
- Knutson-Tao-Woodward (2004) — KTW puzzle model
- Zinn-Justin (arXiv:0809.2392, 2008) — puzzles as integrable vertex model
- Bump-Naprienko (arXiv:2503.05960, 2025) — Yang-Baxter groupoid
- Clio (github.com/clio-vega/integrability-hierarchy) — integrability hierarchy framework
- Robin's thesis (arXiv:2110.12629) — growth diagrams, cylindric partitions, Burge local rule

## Open Design Questions

1. **Puzzle encoding granularity**: horizontal levels (N(N+1)/2 tokens) vs all edges (~3N² tokens) vs one token per unit triangle (N² tokens). Levels is most compact and matches the transfer matrix computation. Start with levels; try edges if accuracy plateaus.

2. **Multi-N training**: train a single model on N=4,5,6,7 simultaneously? This would require padding to max N but would give much more data and test generalization. The RSK project trains per-n, but the LR task may benefit from cross-N data.

3. **Data volume estimation**: need to actually count Σ c^ν_{λ,μ} for all triples at each N to know if we have enough data. Run `lr_coefficients.py` across all valid triples for N=5,6,7 and count.
