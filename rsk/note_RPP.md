# Phase 4: Inverse RSK for Reverse Plane Partitions

**Status**: Designing. Bijection identified: **Hillman-Grassl**.

## Next session: startup checklist

1. Read `BURGE_HIGHER_ORDER.md` — notes on the Burge local rule as higher-order function (from Robin's thesis §2.2.10 and §4.2)
2. Read the Hillman-Grassl paper: https://arxiv.org/abs/1710.09695 — this is the bijection we'll implement
3. Optionally re-read Robin's thesis §2.2.10 (Burge local rule) at `/Users/robin/git/thesis/phD/phD-files/thesis.pdf` pages 52-56
4. Then: implement the Hillman-Grassl bijection in Python, generate training data, choose shape and max_entry

## The Hillman-Grassl bijection

**Input**: Reverse plane partition (RPP) of shape λ — a filling of the Young diagram with non-negative integers, weakly increasing along rows and columns.

**Output**: Arbitrarily labelled Young diagram of the **same shape** λ — non-negative integers with NO monotonicity constraint.

**Weight**: For each box, its label × its hook length. The bijection is weight-preserving (the sum of entries of the RPP equals the sum of label × hook_length over all boxes of the output).

**Reference**: https://arxiv.org/abs/1710.09695

This is much simpler than the Fomin growth diagram approach — both input and output are fillings of the same shape λ, analogous to how our existing RSK experiments map between different encodings of the same combinatorial data.

## ML task

- **Input**: Hillman-Grassl image (arbitrary non-negative integer filling of shape λ)
- **Target**: RPP entries in reading order (row by row), each classified into {0, ..., max_entry}
- **Or vice versa** — TBD which direction is more natural for the model

## Scientific goal

The model must learn the Hillman-Grassl bijection (or its inverse) to generalize. We then use a **sparse autoencoder** to extract the learned algorithm — mechanistic interpretability for mathematical discovery.

## Token encoding

Since both input and output are fillings of the same shape λ, we can use a similar 2D embedding as RSKEncoder:

```
embedding(entry) = value_emb(v) + row_emb(r) + col_emb(c)
```

No need for tableau_id since there's only one filling as input (not a pair).

## Burge local rule (from Robin's thesis §2.2.10)

See BURGE_HIGHER_ORDER.md for full details of the Burge local rule and its higher-order function interpretation. The Hillman-Grassl bijection may or may not use the Burge rule directly — need to read the paper.

## Files to change

1. **rsk.py** — `hillman_grassl_forward()`, `hillman_grassl_inverse()`, `sample_rpp()`, round-trip verification
2. **config.py** — `task="rpp"`, shape-related fields
3. **data.py** — `RPPDataset`, encoding for single filling
4. **model.py** — Possibly reuse RSKEncoder with minor changes (single filling input instead of pair)
5. **train.py** — `--task rpp`, `--shape`, `--max-entry` CLI args

## First experiments

| Shape λ   | |λ| | max_entry | Description       |
|-----------|-----|-----------|-------------------|
| (3,2,1)   | 6   | 3         | Tiny sanity check |
| (4,3,2,1) | 10  | 4         | Staircase shape   |
| (5,5,5)   | 15  | 5         | Rectangle-ish     |
