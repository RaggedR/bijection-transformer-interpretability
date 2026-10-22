#!/usr/bin/env python3
"""Count total (puzzle, LR-tableau) pairs for each puzzle side length N.

For each N, iterates all valid (λ, μ, ν) triples fitting in an n×m box
with n+m = N and sums c^ν_{λ,μ}. This gives the total data volume for
training at that N.
"""

import sys
import time
from itertools import product as cartprod

sys.path.insert(0, '/Users/robin/git/puzzles')
from lr_coefficients import lr_count, partition_to_boundary, boundary_to_partition


def partitions_in_box(n, m):
    """Generate all partitions fitting in an n×m bounding box."""
    if n == 0 or m == 0:
        yield ()
        return

    def _gen(row, max_val, parts):
        if row == n:
            yield tuple(parts) if parts else ()
            return
        # This row can have 0..max_val
        for v in range(max_val, -1, -1):
            if v == 0:
                yield tuple(parts) if parts else ()
            else:
                yield from _gen(row + 1, v, parts + [v])

    yield from _gen(0, m, [])


def count_for_N(N):
    """Count total puzzle-tableau pairs and distinct triples for given N."""
    total_pairs = 0
    total_triples = 0
    max_c = 0
    triple_details = []

    # For each valid (n, m) decomposition
    for n in range(N + 1):
        m = N - n
        # Generate all partitions in this box
        parts = list(partitions_in_box(n, m))

        for lam in parts:
            for mu in parts:
                size_lam_mu = sum(lam) + sum(mu)
                for nu in parts:
                    if sum(nu) != size_lam_mu:
                        continue
                    # Check containment: λ ⊆ ν (necessary for skew shape)
                    contained = True
                    for i in range(len(lam)):
                        if i < len(nu):
                            if lam[i] > nu[i]:
                                contained = False
                                break
                        else:
                            contained = False
                            break
                    if not contained:
                        continue

                    c = lr_count(lam, mu, nu)
                    if c > 0:
                        total_pairs += c
                        total_triples += 1
                        max_c = max(max_c, c)
                        if c > 1:
                            triple_details.append((lam, mu, nu, c))

    return total_pairs, total_triples, max_c, triple_details


if __name__ == '__main__':
    print("Data volume estimation for LR puzzle → tableau learning")
    print("=" * 65)
    print(f"{'N':>3}  {'Pairs':>10}  {'Triples':>10}  {'max c':>6}  {'Time':>8}")
    print("-" * 65)

    for N in range(2, 10):
        t0 = time.time()
        pairs, triples, mc, details = count_for_N(N)
        dt = time.time() - t0
        print(f"{N:3d}  {pairs:10d}  {triples:10d}  {mc:6d}  {dt:7.1f}s")

        # Show triples with c > 1 for small N
        if details and N <= 6:
            for lam, mu, nu, c in details[:5]:
                print(f"      c^{nu}_{{{lam},{mu}}} = {c}")
            if len(details) > 5:
                print(f"      ... and {len(details) - 5} more with c > 1")
