"""
Littlewood-Richardson tableaux: enumeration and verification.

Enumerates all LR tableaux of skew shape ν/λ with content μ — semistandard
Young tableaux whose reading word (right-to-left, top-to-bottom) is a lattice
word. The algorithm builds a chain of horizontal strips, ported from the
Haskell implementation in ~/git/puzzles/haskell/src/QHask/Sym/LR.hs.
"""

from __future__ import annotations

from collections import Counter


def lr_tableaux(lam, mu, nu):
    """Enumerate all LR tableaux of skew shape ν/λ with content μ.

    An LR tableau is a semistandard Young tableau of skew shape ν/λ whose
    content is μ and whose reading word (right-to-left, top-to-bottom) is
    a lattice word.

    Algorithm: build all chains λ = ν_0 ⊂ ν_1 ⊂ ... ⊂ ν_r = ν where each
    strip ν_k/ν_{k-1} is a horizontal strip of size μ_k, and the incremental
    lattice condition holds at each step.

    Returns list of filling dicts {(row, col): value} with 1-indexed values.
    """
    n_rows = max(len(nu), len(lam), 1)
    lam_p = list(lam) + [0] * (n_rows - len(lam))
    nu_p = list(nu) + [0] * (n_rows - len(nu))
    mu_list = list(mu) if mu else []

    # Validate
    if sum(nu_p) != sum(lam_p) + sum(mu_list):
        return []
    if any(lam_p[i] > nu_p[i] for i in range(n_rows)):
        return []
    if not mu_list:
        return [{}] if lam_p == nu_p else []

    results = []

    def build_chain(k, inner, prev_inner, chain):
        if k == len(mu_list):
            if inner == nu_p:
                results.append(_chain_to_filling(chain))
            return
        for new_inner in _horizontal_strips(nu_p, inner, mu_list[k], n_rows):
            if k == 0 or _lattice_check(prev_inner, inner, new_inner, n_rows):
                build_chain(k + 1, new_inner, inner, chain + [new_inner])

    build_chain(0, lam_p, lam_p, [lam_p])
    return results


def _horizontal_strips(outer, inner, count, n_rows):
    """Yield partitions new where new/inner is a horizontal strip of size count, inside outer.

    Constraints per row i:
      inner[i] ≤ new[i] ≤ outer[i]
      new[i] ≤ inner[i-1] for i ≥ 1  (horizontal strip: at most one box per column)
      Σ(new[i] - inner[i]) = count
    """
    def gen(row, remaining, acc):
        if remaining == 0:
            yield acc + inner[row:]
            return
        if row >= n_rows:
            return
        lo = inner[row]
        hi = min(outer[row], lo + remaining)
        if row > 0:
            hi = min(hi, inner[row - 1])
        for v in range(hi, lo - 1, -1):
            yield from gen(row + 1, remaining - (v - lo), acc + [v])

    if count == 0:
        yield list(inner)
        return
    yield from gen(0, count, [])


def _lattice_check(prev, inner, new, n_rows):
    """Incremental lattice condition when placing value k (k ≥ 2).

    At the "dangerous point" of each row j — after reading all k-entries in
    rows 0..j but before reading k-1 entries in row j — the condition is:

        Σ_{i≤j} (new[i] - inner[i]) ≤ Σ_{i<j} (inner[i] - prev[i])

    The shift (i<j vs i≤j) accounts for the within-row reading order:
    k-entries sit RIGHT of (k-1)-entries, so they are read first in R-to-L.
    """
    cum_a = 0
    cum_b = 0
    for j in range(n_rows):
        cum_a += new[j] - inner[j]
        if cum_a > cum_b:
            return False
        cum_b += inner[j] - prev[j]
    return True


def _chain_to_filling(chain):
    """Convert partition chain [ν_0, ν_1, ..., ν_r] to filling dict."""
    filling = {}
    for k in range(len(chain) - 1):
        for row in range(len(chain[k])):
            for col in range(chain[k][row], chain[k + 1][row]):
                filling[(row, col)] = k + 1
    return filling


def filling_to_values(filling, lam, nu):
    """Extract filling values in row-major reading order (for ML targets).

    Iterates over boxes of skew shape ν/λ left-to-right, top-to-bottom.
    """
    values = []
    for row in range(len(nu)):
        lam_r = lam[row] if row < len(lam) else 0
        for col in range(lam_r, nu[row]):
            values.append(filling[(row, col)])
    return values


# ================================================================
# Verification helpers (independent checks, not used in enumeration)
# ================================================================

def _check_ssyt(filling, lam, nu):
    """Check weakly increasing rows, strictly increasing columns."""
    for row in range(len(nu)):
        lam_r = lam[row] if row < len(lam) else 0
        for col in range(lam_r, nu[row] - 1):
            if filling[(row, col)] > filling[(row, col + 1)]:
                return False
        if row + 1 < len(nu):
            lam_r1 = lam[row + 1] if row + 1 < len(lam) else 0
            for col in range(max(lam_r, lam_r1), min(nu[row], nu[row + 1])):
                if (row, col) in filling and (row + 1, col) in filling:
                    if filling[(row, col)] >= filling[(row + 1, col)]:
                        return False
    return True


def _check_content(filling, mu):
    """Check that filling has content μ."""
    counts = Counter(filling.values())
    for i, m in enumerate(mu):
        if counts.get(i + 1, 0) != m:
            return False
    return len(counts) == len(mu) or not filling


def _check_lattice_word(filling, lam, nu):
    """Check reading word (R-to-L, top-to-bottom) is a lattice word."""
    counts = {}
    for row in range(len(nu)):
        lam_r = lam[row] if row < len(lam) else 0
        for col in range(nu[row] - 1, lam_r - 1, -1):
            v = filling[(row, col)]
            counts[v] = counts.get(v, 0) + 1
            if v > 1 and counts[v] > counts.get(v - 1, 0):
                return False
    return True


# ================================================================
# Self-tests
# ================================================================

if __name__ == '__main__':
    import sys
    import time
    sys.path.insert(0, '/Users/robin/git/puzzles')
    from lr_coefficients import lr_count
    from count_data import partitions_in_box

    print("=" * 60)
    print("  LR Tableau Enumeration — Self-Tests")
    print("=" * 60)

    # 1. Worked example
    tabs = lr_tableaux((2, 1), (2, 1), (3, 2, 1))
    assert len(tabs) == 2, f"Expected 2, got {len(tabs)}"
    vals = [tuple(filling_to_values(t, (2, 1), (3, 2, 1))) for t in tabs]
    assert (1, 1, 2) in vals and (1, 2, 1) in vals, f"Wrong tableaux: {vals}"
    print(f"\n  Worked example c^(3,2,1)_{{(2,1),(2,1)}} = 2:")
    for i, t in enumerate(tabs):
        print(f"    Tableau {i+1}: {filling_to_values(t, (2,1), (3,2,1))}")
    print("    OK")

    # 2. Edge cases
    assert lr_tableaux((), (), ()) == [{}]
    assert lr_tableaux((2, 1), (), (2, 1)) == [{}]
    assert lr_tableaux((1,), (1,), (3,)) == []
    assert lr_tableaux((), (2, 1), (2, 1)) != []
    print("\n  Edge cases: OK")

    # 3. Exhaustive count check for N ≤ 7
    print("\n  Cross-validating counts against puzzle transfer matrix...")
    total_checked = 0
    t0 = time.time()
    for N in range(2, 8):
        n_checked = 0
        for n in range(N + 1):
            m = N - n
            parts = list(partitions_in_box(n, m))
            for lam in parts:
                for mu in parts:
                    s_lm = sum(lam) + sum(mu)
                    for nu in parts:
                        if sum(nu) != s_lm:
                            continue
                        if len(lam) > len(nu):
                            continue
                        if any(lam[i] > nu[i]
                               for i in range(min(len(lam), len(nu)))):
                            continue
                        c_puzzle = lr_count(lam, mu, nu)
                        c_tab = len(lr_tableaux(lam, mu, nu))
                        assert c_puzzle == c_tab, (
                            f"MISMATCH: c^{nu}_{{{lam},{mu}}}: "
                            f"puzzle={c_puzzle}, tableaux={c_tab}"
                        )
                        n_checked += 1
        total_checked += n_checked
        print(f"    N={N}: {n_checked} triples checked")
    dt = time.time() - t0
    print(f"    Total: {total_checked} triples, all match ({dt:.1f}s)")

    # 4. SSYT + content + lattice property check
    print("\n  Checking SSYT, content, and lattice word properties...")
    n_tableaux = 0
    errors = 0
    for N in range(2, 7):
        for n in range(N + 1):
            m = N - n
            parts = list(partitions_in_box(n, m))
            for lam in parts:
                for mu in parts:
                    for nu in parts:
                        if sum(nu) != sum(lam) + sum(mu):
                            continue
                        if len(lam) > len(nu):
                            continue
                        if any(lam[i] > nu[i]
                               for i in range(min(len(lam), len(nu)))):
                            continue
                        for filling in lr_tableaux(lam, mu, nu):
                            if not filling:
                                continue
                            n_tableaux += 1
                            if not _check_ssyt(filling, lam, nu):
                                print(f"    SSYT FAIL: {lam},{mu},{nu}")
                                errors += 1
                            if not _check_content(filling, mu):
                                print(f"    CONTENT FAIL: {lam},{mu},{nu}")
                                errors += 1
                            if not _check_lattice_word(filling, lam, nu):
                                print(f"    LATTICE FAIL: {lam},{mu},{nu}")
                                errors += 1
    status = 'OK' if errors == 0 else f'{errors} ERRORS'
    print(f"    {n_tableaux} tableaux checked: {status}")

    print()
    print("  All checks passed!" if errors == 0 else "  SOME CHECKS FAILED!")
