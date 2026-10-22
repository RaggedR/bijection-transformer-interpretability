"""
Hyperparameter configuration for RSK neural network experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    n: int = 8  # permutation size (S_n) or default for seq_len/vocab_size
    task: str = "permutation"  # "permutation", "word", "matrix", "rpp", or "cylindric"
    seq_len: int | None = None  # output sequence length (m for words, defaults to n)
    vocab_size: int | None = None  # output classes per head (k for words, defaults to n)

    # RPP task config
    shape: tuple[int, ...] | None = None  # partition shape for RPP task
    max_entry: int | None = None  # max value in filling (target classes = max_entry + 1)

    # Cylindric task config
    profile: tuple[int, ...] | None = None  # binary profile (e.g., (0,1,0))
    max_label: int | None = None  # max ALCD face label value
    max_gamma_parts: int = 3  # max parts in base partition γ
    max_gamma_size: int = 4  # max part size in base partition γ

    d_model: int = 128  # transformer embedding dimension
    nhead: int = 8  # number of attention heads
    num_layers: int = 6  # transformer encoder layers
    dim_feedforward: int = 512  # FFN hidden dimension (4× d_model)
    dropout: float = 0.1

    # Embedding table sizes (generous upper bounds for n ≤ 10)
    max_value: int = 10  # max entry value in tableaux (= n)
    max_rows: int = 10  # max number of rows in a tableau
    max_cols: int = 10  # max number of columns in a tableau

    # Ablation config
    ablation: str | None = None  # "drop-row", "drop-col", "drop-tab", "drop-row-col", "1d-pos", "concat"

    # Level decoder config
    num_decoder_layers: int = 2  # transformer decoder layers for RSKLevelDecoder

    # Baseline MLP config
    mlp_hidden: list[int] = field(default_factory=lambda: [256, 256, 128])

    VALID_ABLATIONS = {None, "drop-row", "drop-col", "drop-tab", "drop-row-col", "1d-pos", "concat"}

    def __post_init__(self):
        if self.ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"ablation must be one of {self.VALID_ABLATIONS}, got {self.ablation!r}")
        if self.ablation == "concat" and self.d_model % 4 != 0:
            raise ValueError(f"concat ablation requires d_model divisible by 4, got {self.d_model}")
        if self.task == "rpp":
            if self.shape is None:
                raise ValueError("shape is required for task='rpp'")
            if self.max_entry is None:
                raise ValueError("max_entry is required for task='rpp'")
            self.seq_len = sum(self.shape)
            self.vocab_size = self.max_entry + 1
            # RPP values can be large: max_entry × max_hook + padding
            from rsk import hook_lengths
            max_hook = max(h for row in hook_lengths(list(self.shape)) for h in row)
            self.max_value = max(self.max_value, self.max_entry * max_hook + 1)
            self.max_rows = max(self.max_rows, len(self.shape))
            self.max_cols = max(self.max_cols, self.shape[0])
        elif self.task == "cylindric":
            if self.profile is None:
                raise ValueError("profile is required for task='cylindric'")
            if self.max_label is None:
                raise ValueError("max_label is required for task='cylindric'")
            from rsk import _num_alcd_labels
            T = len(self.profile)
            num_labels = _num_alcd_labels(self.profile)
            self.seq_len = num_labels
            self.vocab_size = self.max_label + 1
            # Upper bounds for CPP partition entries
            max_parts = self.max_gamma_parts + num_labels
            max_cpp_val = self.max_gamma_size + self.max_label * num_labels
            self.max_value = max(self.max_value, max_cpp_val + 1)
            self.max_rows = max(self.max_rows, T)  # partition index in profile
            self.max_cols = max(self.max_cols, max_parts)  # part index
        else:
            # Resolve seq_len and vocab_size: default to n for permutations
            if self.seq_len is None:
                self.seq_len = self.n
            if self.vocab_size is None:
                self.vocab_size = self.n
            self.max_value = max(self.max_value, self.vocab_size)
            self.max_rows = max(self.max_rows, self.seq_len)
            self.max_cols = max(self.max_cols, self.seq_len)

    @property
    def num_tokens(self) -> int:
        """Number of input tokens.

        - RPP: seq_len (single filling, not a pair)
        - Cylindric: T × max_parts (padded partition entries)
        - Others: 2 × seq_len (P, Q pair)
        """
        if self.task == "rpp":
            return self.seq_len  # single filling, not a pair
        if self.task == "cylindric":
            T = len(self.profile)
            max_parts = self.max_gamma_parts + self.seq_len
            return T * max_parts
        return 2 * self.seq_len

    @property
    def num_heads(self) -> int:
        """Number of classification heads (= seq_len, one per position in output)."""
        return self.seq_len


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_fraction: float = 0.05  # fraction of total steps for warmup
    patience: int = 10  # early stopping patience (epochs)
    device: str = "mps"  # Apple Silicon GPU

    # Data
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    num_workers: int = 0  # DataLoader workers (0 = main process)
    seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    log_every: int = 50  # log every N batches
