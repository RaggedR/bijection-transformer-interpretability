# The Burge Local Rule as Higher-Order Function

## Key insight (from Robin's thesis, Chapter 4, Section 4.2)

The local rule is not just a function on partitions — it is a **higher-order function** that takes a bijection as input and returns a new bijection as output. This is the key to understanding the growth diagram for RPPs.

## Type signatures

### The local rule (forward)

```
𝔘_{α,β} : (ℤ≥0, D(α) ∩ D(β)) → U(α) ∩ U(β)
```

Given two "border" partitions α, β and input (m, μ), produces λ = 𝔘_{α,β}(m, μ).

Weight condition: |λ| + |μ| = |α| + |β| + m

Where:
- U(λ) = {partitions obtained from λ by adding a horizontal strip}
- D(λ) = {partitions obtained from λ by removing a horizontal strip}

### The local rule (reverse)

```
𝔇_{α,β} : U(α) ∩ U(β) → (ℤ≥0, D(α) ∩ D(β))
```

Given λ, produces (m, μ) = 𝔇_{α,β}(λ).

### As higher-order function (thesis eq. 4.43)

```
𝔏_i : ((P, ALCD(π)) → CPP(π)) → ((P, ALCD(π')) → CPP(π'))
```

In other words: 𝔏_i[ψ_π] = ψ_{π'}, where π' is obtained from π by adding an inversion at position i.

It transforms a bijection for profile π into a bijection for profile π'.

### The inverse higher-order function (thesis eq. 4.45)

```
𝔐_i : (CPP(π) → (P, ALCD(π))) → (CPP(π') → (P, ALCD(π')))
```

𝔐_i[φ_π] = φ_{π'}

## How it operates on the interlacing sequence

Given a CPP c = (μ⁰, μ¹, ..., μᵀ) with profile π:

**Forward local rule** at position i (where π has an inversion):
```
α = μ^{i-1}
γ = μ^i        (the partition being replaced)
β = μ^{i+1}
λ = 𝔘_{α,β}(γ, m)
```
New sequence: (μ⁰, ..., μ^{i-1}, λ, μ^{i+1}, ..., μᵀ)

**Inverse local rule** at position i:
```
α = μ^{i-1}
λ = μ^i
β = μ^{i+1}
(m, ν) = 𝔇_{α,β}(λ)
```
New sequence: (μ⁰, ..., μ^{i-1}, ν, μ^{i+1}, ..., μᵀ)

## The Burge local rule (forward): 𝔘_{α,β}(m, μ) = λ

From thesis Section 2.2.10. Works in terms of column lengths (conjugate partitions).

1. A = columns(α/μ) = {j : α'_j > μ'_j}  — columns where α exceeds μ
2. B = columns(β/μ) = {j : β'_j > μ'_j}  — columns where β exceeds μ
3. For each i ∈ A ∩ B, find ε(i) = smallest integer > i not in A ∪ B, with ε(i) ≠ ε(j) for j < i. Process A ∩ B in increasing order.
4. C = {ε(i) | i ∈ A ∩ B}
5. D = first m elements of the complement of A ∪ B ∪ C (smallest available columns)
6. λ = partition obtained from μ by adding one box to the bottom of each column in A ∪ B ∪ C ∪ D

## The Burge local rule (reverse): 𝔇_{α,β}(λ) = (m, μ)

1. Ā = columns(λ/α) = {j : λ'_j > α'_j}  — columns where λ exceeds α
2. B̄ = columns(λ/β) = {j : λ'_j > β'_j}  — columns where λ exceeds β
3. For each i ∈ Ā ∩ B̄, find δ(i) = largest integer < i not in Ā ∪ B̄, with δ(i) ≠ δ(j) for j > i. Process Ā ∩ B̄ in decreasing order.
4. C̄ = {δ(i) > 0 | i ∈ Ā ∩ B̄}  (only positive indices)
5. μ = partition obtained from λ by removing one box from the bottom of each column in Ā ∪ B̄ ∪ C̄
6. m = #{δ(i) ≤ 0 | i ∈ Ā ∩ B̄}  (count of unpaired elements)

## Worked example (from thesis)

𝔇_{(6,5,5,3),(6,6,5,2)}(7,6,5,3,1) = (1, (6,5,4,2))

```
λ' = (5, 4, 4, 3, 3, 2, 1)
Ā = columns(λ/α) = {1, 6, 7}
B̄ = columns(λ/β) = {1, 3, 7}

Ā ∩ B̄ = {1, 7}

Process in decreasing order:
  i=7: δ(7) = largest < 7 not in {1,6,7} ∪ {1,3,7} = not in {1,3,6,7} → δ(7) = 5
  i=1: δ(1) = largest < 1 not in {1,3,6,7} ∪ {5} → δ(1) = 0 (or negative)

C̄ = {5}  (δ(7)=5 > 0)
m = 1    (δ(1)=0 ≤ 0)

Remove boxes from columns Ā ∪ B̄ ∪ C̄ = {1,3,5,6,7}:
μ' = (4, 4, 3, 3, 2, 1, 0) → μ = (6, 5, 4, 2)
```

## Growth diagram (Definition 4.3.4)

A cylindric growth diagram with profile π has:
- **Vertices** labelled by integer partitions
- **Faces** labelled by non-negative integers
- At each face (u; v1, v2; w) with labels μ, α, β, λ: **𝔘_{α,β}(m, μ) = λ**

The upper boundary vertices give the CPP (= RPP) interlacing sequence.

**Proposition 4.3.1**: Every CPP determines a unique growth diagram.

**Proposition 4.3.2**: Every pair (γ, ALCD) determines a unique growth diagram.

The bijection is: CPP(π) ↔ (P, ALCD(π))

## For RPPs

An RPP is a CPP with μ⁰ = μᵀ = ∅. The profile π encodes the boundary of the shape λ.

The ML task: given boundary interlacing sequences (input), predict RPP entries (target). The model must learn the local rule to do this — making it a candidate for **mechanistic interpretability** via sparse autoencoders.

## Commutativity (Lemma 4.3.6)

If π has inversions at both positions i and j, then:
```
𝔏_i ∘ 𝔏_j[ψ_π] = 𝔏_j ∘ 𝔏_i[ψ_π]
```

The order of local rule application doesn't matter. This is why the growth diagram is well-defined.

## Column indexing

Column indices are **1-indexed** (column 1 = leftmost/tallest column). The δ and ε matching functions can reach index 0 or below, which contributes to m (the face label) rather than to the set C̄.
