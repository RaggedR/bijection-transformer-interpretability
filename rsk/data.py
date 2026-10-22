"""
Data pipeline for RSK neural network experiments.

Loads (P, Q, σ) data from HuggingFace or generates it with our RSK implementation.
Encodes tableaux into structured tokens: each entry becomes a token with
(value, row, col, tableau_id) — preserving the 2D geometric structure that
PNNL's bracket-token encoding destroys.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from rsk import (
    rsk_forward, rsk_forward_biword, matrix_to_biword, tableau_positions, Tableau,
    hillman_grassl_forward, sample_filling, Filling,
    growth_diagram_forward, sample_gamma, sample_alcd, _num_alcd_labels,
)
from config import ModelConfig, TrainConfig


# HuggingFace dataset names
HF_DATASET_TEMPLATE = "ACDRepo/robinson_schensted_knuth_correspondence_{n}"


def load_hf_dataset(n: int) -> tuple[list, list]:
    """
    Load RSK dataset from HuggingFace.

    Returns (train_data, test_data) where each is a list of dicts with keys:
        'Permutation', 'Standard Young tableau 1', 'Standard Young tableau 2'
    """
    from datasets import load_dataset

    name = HF_DATASET_TEMPLATE.format(n=n)
    ds = load_dataset(name)
    return list(ds["train"]), list(ds["test"])


def verify_hf_against_rsk(data: list[dict], n: int, max_check: int = 100) -> tuple[int, int]:
    """
    Cross-check HuggingFace data against our RSK implementation.

    Returns (num_checked, num_matched).
    """
    checked = 0
    matched = 0

    for item in data[:max_check]:
        sigma = item["Permutation"]
        P_hf = item["Standard Young tableau 1"]
        Q_hf = item["Standard Young tableau 2"]

        P_ours, Q_ours = rsk_forward(sigma)

        if P_ours == P_hf and Q_ours == Q_hf:
            matched += 1
        checked += 1

    return checked, matched


def encode_tableaux(P: Tableau, Q: Tableau) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a pair of SYTs into token representations.

    Each tableau entry becomes a token. For n entries in each tableau, we get 2n tokens.

    Returns:
        values: LongTensor of shape (2n,) — entry values (1-indexed)
        positions: LongTensor of shape (2n, 3) — [row, col, tableau_id] per token
            tableau_id: 0 for P, 1 for Q
    """
    P_positions = tableau_positions(P)  # list of (value, row, col)
    Q_positions = tableau_positions(Q)

    values = []
    positions = []

    # P tokens (tableau_id = 0)
    for val, row, col in P_positions:
        values.append(val)
        positions.append([row, col, 0])

    # Q tokens (tableau_id = 1)
    for val, row, col in Q_positions:
        values.append(val)
        positions.append([row, col, 1])

    return torch.tensor(values, dtype=torch.long), torch.tensor(positions, dtype=torch.long)


class RSKDataset(Dataset):
    """
    PyTorch Dataset for RSK data.

    Each item returns:
        values: (2n,) LongTensor — token values
        positions: (2n, 3) LongTensor — [row, col, tableau_id] per token
        target: (n,) LongTensor — permutation σ (0-indexed for cross-entropy)
    """

    def __init__(self, data: list[dict]):
        """
        Args:
            data: list of dicts with keys 'Permutation',
                  'Standard Young tableau 1', 'Standard Young tableau 2'
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        P = item["Standard Young tableau 1"]
        Q = item["Standard Young tableau 2"]
        sigma = item["Permutation"]

        values, positions = encode_tableaux(P, Q)

        # Convert σ to 0-indexed for cross-entropy loss
        target = torch.tensor([v - 1 for v in sigma], dtype=torch.long)

        return values, positions, target


class RSKSamplingDataset(Dataset):
    """
    On-the-fly RSK dataset for arbitrary n.

    Samples random permutations and computes RSK at access time.
    No enumeration needed — works for any n where RSK is computable.
    """

    def __init__(self, n: int, size: int, seed: int | None = None):
        self.n = n
        self.size = size
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Random permutation of {1..n}
        sigma = (torch.randperm(self.n, generator=self.rng) + 1).tolist()
        P, Q = rsk_forward(sigma)
        values, positions = encode_tableaux(P, Q)
        target = torch.tensor([v - 1 for v in sigma], dtype=torch.long)
        return values, positions, target


class WordSamplingDataset(Dataset):
    """
    On-the-fly RSK dataset for words w ∈ {1,...,k}^m.

    Samples random words and computes RSK at access time.
    Target is the word itself (0-indexed), not a permutation.
    """

    def __init__(self, m: int, k: int, size: int, seed: int | None = None):
        self.m = m  # word length (= seq_len)
        self.k = k  # alphabet size (= vocab_size)
        self.size = size
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Random word: m values from {1,...,k}
        word = (torch.randint(1, self.k + 1, (self.m,), generator=self.rng)).tolist()
        P, Q = rsk_forward(word)
        values, positions = encode_tableaux(P, Q)
        # 0-indexed targets for cross-entropy
        target = torch.tensor([v - 1 for v in word], dtype=torch.long)
        return values, positions, target


class MatrixSamplingDataset(Dataset):
    """
    On-the-fly RSK dataset for non-negative integer matrices A ∈ ℕ^{a×b}.

    Samples random matrices with fixed entry sum total_n = |λ|,
    computes biword RSK, and returns (P, Q) tokens with bottom-line targets.

    Target is the bottom line of the two-line array (0-indexed), not the matrix.
    This gives |λ| classification heads over {1,...,b}, matching the growth diagram.
    """

    def __init__(self, a: int, b: int, total_n: int, size: int, seed: int | None = None):
        self.a = a
        self.b = b
        self.total_n = total_n
        self.size = size
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Random matrix: place total_n balls into a*b bins
        bins = torch.randint(0, self.a * self.b, (self.total_n,), generator=self.rng)
        flat = torch.zeros(self.a * self.b, dtype=torch.long)
        flat.scatter_add_(0, bins, torch.ones(self.total_n, dtype=torch.long))
        A = flat.reshape(self.a, self.b).tolist()

        top, bottom = matrix_to_biword(A)
        P, Q = rsk_forward_biword(top, bottom)
        values, positions = encode_tableaux(P, Q)

        # 0-indexed targets: bottom line values are in {1,...,b}
        target = torch.tensor([v - 1 for v in bottom], dtype=torch.long)
        return values, positions, target


def encode_single_filling(filling: Filling) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a single filling (RPP or arbitrary) into token representations.

    Each cell becomes a token with (value, row, col, tableau_id=0).
    Unlike encode_tableaux which handles a (P, Q) pair, this handles one filling.

    Returns:
        values: LongTensor of shape (|λ|,) — entry values
        positions: LongTensor of shape (|λ|, 3) — [row, col, tableau_id=0]
    """
    values = []
    positions = []
    for row_idx, row in enumerate(filling):
        for col_idx, val in enumerate(row):
            values.append(val)
            positions.append([row_idx, col_idx, 0])

    return torch.tensor(values, dtype=torch.long), torch.tensor(positions, dtype=torch.long)


class RPPSamplingDataset(Dataset):
    """
    On-the-fly Hillman-Grassl dataset for RPP task.

    Samples random fillings, computes forward HG to get RPP, returns:
    - Input: RPP tokens (structured, weakly increasing)
    - Target: filling values in reading order (unconstrained)

    This mirrors the RSK pattern: structured input → unstructured target.
    """

    def __init__(
        self, shape: tuple[int, ...], max_entry: int, size: int, seed: int | None = None,
    ):
        self.shape = list(shape)
        self.max_entry = max_entry
        self.size = size
        import random
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        filling = sample_filling(self.shape, self.max_entry, self.rng)
        rpp = hillman_grassl_forward(self.shape, filling)

        # Input: RPP as structured tokens
        values, positions = encode_single_filling(rpp)

        # Target: filling values in reading order (row by row, left to right)
        target = torch.tensor(
            [filling[r][c] for r in range(len(self.shape)) for c in range(self.shape[r])],
            dtype=torch.long,
        )

        return values, positions, target


def encode_cpp(
    cpp: list[list[int]], T: int, max_parts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a cylindric plane partition as tokens.

    Each partition part becomes a token: (value, partition_index, part_index, 0).
    Partitions are padded to max_parts with zeros.

    Returns:
        values: LongTensor of shape (T * max_parts,) — part values (0 = padding)
        positions: LongTensor of shape (T * max_parts, 3) — [partition_idx, part_idx, 0]
    """
    values = []
    positions = []
    for k in range(T):
        parts = cpp[k] if k < len(cpp) else []
        for j in range(max_parts):
            val = parts[j] if j < len(parts) else 0
            values.append(val)
            positions.append([k, j, 0])

    return torch.tensor(values, dtype=torch.long), torch.tensor(positions, dtype=torch.long)


class CylindricSamplingDataset(Dataset):
    """
    On-the-fly cylindric growth diagram dataset.

    Samples (γ, ALCD), computes forward growth diagram to get CPP, returns:
    - Input: CPP partition entries as structured tokens
    - Target: ALCD face labels (flat list)
    """

    def __init__(
        self,
        profile: tuple[int, ...],
        max_label: int,
        max_gamma_parts: int,
        max_gamma_size: int,
        size: int,
        seed: int | None = None,
    ):
        self.profile = profile
        self.max_label = max_label
        self.max_gamma_parts = max_gamma_parts
        self.max_gamma_size = max_gamma_size
        self.size = size
        self.T = len(profile)
        self.num_labels = _num_alcd_labels(profile)
        self.max_parts = max_gamma_parts + self.num_labels
        import random
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = sample_gamma(self.max_gamma_parts, self.max_gamma_size, self.rng)
        alcd = sample_alcd(self.profile, self.max_label, self.rng)

        # Forward: (γ, ALCD) → CPP
        cpp = growth_diagram_forward(self.profile, gamma, alcd)

        # Input: CPP as structured tokens
        values, positions = encode_cpp(cpp, self.T, self.max_parts)

        # Target: ALCD face labels (already 0-indexed, no -1 needed)
        target = torch.tensor(alcd, dtype=torch.long)

        return values, positions, target


def generate_our_dataset(n: int) -> list[dict]:
    """
    Generate all (P, Q, σ) triples for S_n using our RSK implementation.

    Returns data in the same format as HuggingFace for compatibility.
    """
    from itertools import permutations

    data = []
    for perm in permutations(range(1, n + 1)):
        sigma = list(perm)
        P, Q = rsk_forward(sigma)
        data.append({
            "Permutation": sigma,
            "Standard Young tableau 1": P,
            "Standard Young tableau 2": Q,
        })
    return data


def make_dataloaders(
    n: int,
    train_config: TrainConfig,
    source: str = "hf",
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    task: str = "permutation",
    vocab_size: int | None = None,
    a_dim: int | None = None,
    b_dim: int | None = None,
    shape: tuple[int, ...] | None = None,
    max_entry: int | None = None,
    profile: tuple[int, ...] | None = None,
    max_label: int | None = None,
    max_gamma_parts: int = 3,
    max_gamma_size: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        n: permutation size (also used as seq_len for permutations)
        train_config: training configuration
        source: "hf" for HuggingFace, "generate" for enumeration, "sample" for random sampling
        train_size: for source="sample", number of training examples
        val_size: for source="sample", number of validation examples
        test_size: for source="sample", number of test examples
        task: "permutation", "word", or "matrix"
        vocab_size: alphabet size k for words (ignored for permutations)
        a_dim: number of rows for matrix task
        b_dim: number of columns for matrix task

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if source == "sample":
        # On-the-fly sampling — works for any n
        if train_size is None:
            train_size = 500_000
        if val_size is None:
            val_size = 50_000
        if test_size is None:
            test_size = 50_000

        if task == "rpp":
            if shape is None or max_entry is None:
                raise ValueError("shape and max_entry are required for task='rpp'")
            train_ds = RPPSamplingDataset(shape, max_entry, train_size, seed=train_config.seed)
            val_ds = RPPSamplingDataset(shape, max_entry, val_size, seed=train_config.seed + 1)
            test_ds = RPPSamplingDataset(shape, max_entry, test_size, seed=train_config.seed + 2)
        elif task == "cylindric":
            if profile is None or max_label is None:
                raise ValueError("profile and max_label are required for task='cylindric'")
            train_ds = CylindricSamplingDataset(
                profile, max_label, max_gamma_parts, max_gamma_size,
                train_size, seed=train_config.seed,
            )
            val_ds = CylindricSamplingDataset(
                profile, max_label, max_gamma_parts, max_gamma_size,
                val_size, seed=train_config.seed + 1,
            )
            test_ds = CylindricSamplingDataset(
                profile, max_label, max_gamma_parts, max_gamma_size,
                test_size, seed=train_config.seed + 2,
            )
        elif task == "matrix":
            if a_dim is None or b_dim is None:
                raise ValueError("a_dim and b_dim are required for task='matrix'")
            train_ds = MatrixSamplingDataset(a_dim, b_dim, n, train_size, seed=train_config.seed)
            val_ds = MatrixSamplingDataset(a_dim, b_dim, n, val_size, seed=train_config.seed + 1)
            test_ds = MatrixSamplingDataset(a_dim, b_dim, n, test_size, seed=train_config.seed + 2)
        elif task == "word":
            if vocab_size is None:
                raise ValueError("vocab_size (k) is required for task='word'")
            train_ds = WordSamplingDataset(n, vocab_size, train_size, seed=train_config.seed)
            val_ds = WordSamplingDataset(n, vocab_size, val_size, seed=train_config.seed + 1)
            test_ds = WordSamplingDataset(n, vocab_size, test_size, seed=train_config.seed + 2)
        else:
            train_ds = RSKSamplingDataset(n, train_size, seed=train_config.seed)
            val_ds = RSKSamplingDataset(n, val_size, seed=train_config.seed + 1)
            test_ds = RSKSamplingDataset(n, test_size, seed=train_config.seed + 2)
    elif task in ("word", "matrix", "rpp", "cylindric"):
        raise ValueError(f"{task} task only supports source='sample'")
    else:
        if source == "hf":
            train_data, test_data = load_hf_dataset(n)
        else:
            all_data = generate_our_dataset(n)
            split = int(0.8 * len(all_data))
            import random
            random.seed(train_config.seed)
            random.shuffle(all_data)
            train_data = all_data[:split]
            test_data = all_data[split:]

        # Split train into train + val
        full_train = RSKDataset(train_data)
        v_size = int(train_config.val_fraction * len(full_train))
        t_size = len(full_train) - v_size

        generator = torch.Generator().manual_seed(train_config.seed)
        train_ds, val_ds = random_split(full_train, [t_size, v_size], generator=generator)
        test_ds = RSKDataset(test_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test encoding
    P = [[1, 3, 5, 8], [2, 6], [4, 7]]
    Q = [[1, 3, 5, 8], [2, 4], [6, 7]]
    values, positions = encode_tableaux(P, Q)
    print(f"Values shape: {values.shape}, Positions shape: {positions.shape}")
    print(f"Values: {values}")
    print(f"Positions:\n{positions}")

    # Test dataset generation for small n
    data = generate_our_dataset(4)
    ds = RSKDataset(data)
    vals, pos, target = ds[0]
    print(f"\nn=4, first item:")
    print(f"  values={vals}, positions={pos}, target={target}")
    print(f"  sigma (1-indexed) = {[t.item() + 1 for t in target]}")

    # Test WordSamplingDataset
    print("\nWord dataset test (m=8, k=5):")
    wds = WordSamplingDataset(m=8, k=5, size=10, seed=42)
    vals, pos, target = wds[0]
    print(f"  values shape={vals.shape}, positions shape={pos.shape}, target shape={target.shape}")
    print(f"  target (0-indexed) = {target.tolist()}")
    print(f"  word (1-indexed) = {[t.item() + 1 for t in target]}")
    # Verify target values are in range [0, k-1]
    assert target.min() >= 0 and target.max() < 5, f"Target out of range: {target}"
    print("  Target range OK")

    # Test MatrixSamplingDataset
    print("\nMatrix dataset test (a=3, b=3, N=10):")
    mds = MatrixSamplingDataset(a=3, b=3, total_n=10, size=10, seed=42)
    vals, pos, target = mds[0]
    print(f"  values shape={vals.shape}, positions shape={pos.shape}, target shape={target.shape}")
    print(f"  target (0-indexed) = {target.tolist()}")
    print(f"  bottom line (1-indexed) = {[t.item() + 1 for t in target]}")
    # Verify: 2*N tokens, N target positions, target values in [0, b-1]
    assert vals.shape == (20,), f"Expected values shape (20,), got {vals.shape}"
    assert pos.shape == (20, 3), f"Expected positions shape (20, 3), got {pos.shape}"
    assert target.shape == (10,), f"Expected target shape (10,), got {target.shape}"
    assert target.min() >= 0 and target.max() < 3, f"Target out of range: {target}"
    print("  Shapes and target range OK")

    # Test RPPSamplingDataset
    print("\nRPP dataset test (shape=(3,2,1), max_entry=3):")
    shape_rpp = (3, 2, 1)
    size_rpp = sum(shape_rpp)  # = 6
    rds = RPPSamplingDataset(shape=shape_rpp, max_entry=3, size=10, seed=42)
    vals, pos, target = rds[0]
    print(f"  values shape={vals.shape}, positions shape={pos.shape}, target shape={target.shape}")
    print(f"  values (RPP entries) = {vals.tolist()}")
    print(f"  target (filling entries) = {target.tolist()}")
    # Verify shapes: |λ| tokens, |λ| target positions, target in [0, max_entry]
    assert vals.shape == (size_rpp,), f"Expected values shape ({size_rpp},), got {vals.shape}"
    assert pos.shape == (size_rpp, 3), f"Expected positions shape ({size_rpp}, 3), got {pos.shape}"
    assert target.shape == (size_rpp,), f"Expected target shape ({size_rpp},), got {target.shape}"
    assert target.min() >= 0 and target.max() <= 3, f"Target out of range: {target}"
    # All tableau_ids should be 0 (single filling, not a pair)
    assert (pos[:, 2] == 0).all(), "All tableau_ids should be 0 for RPP"
    print("  Shapes and target range OK")
