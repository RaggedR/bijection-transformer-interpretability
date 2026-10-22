"""
Neural network architectures for learning the RSK correspondence.

Two models:
1. RSKEncoder — Encoder-only transformer with structured 2D token embeddings
   and n parallel classification heads. Our main model.
2. BaselineMLP — Flat MLP on flattened input for comparison with PNNL approach.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Structured embedding for tableau tokens.

    Each token represents one entry in a tableau. The embedding is:
        value_emb(entry) + row_emb(row) + col_emb(col) + tableau_emb(P_or_Q)

    This preserves the 2D geometric structure of the tableaux — the row/col
    embeddings let the model learn that "row 0, col 3" in P relates to
    "row 1, col 0" in Q through the RSK structure.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        self.ablation = config.ablation

        if self.ablation == "concat":
            # Each component gets d_model//4 dims; concat back to d_model
            d_sub = d // 4
            self.value_emb = nn.Embedding(config.max_value + 1, d_sub)
            self.row_emb = nn.Embedding(config.max_rows, d_sub)
            self.col_emb = nn.Embedding(config.max_cols, d_sub)
            self.tableau_emb = nn.Embedding(2, d_sub)
        elif self.ablation == "1d-pos":
            # Replace 2D (row, col) with single sequential position
            self.value_emb = nn.Embedding(config.max_value + 1, d)
            self.pos_emb = nn.Embedding(config.num_tokens, d)
            self.tableau_emb = nn.Embedding(2, d)
        else:
            # Standard or drop-* variants: only create what we need
            self.value_emb = nn.Embedding(config.max_value + 1, d)
            if self.ablation not in ("drop-row", "drop-row-col"):
                self.row_emb = nn.Embedding(config.max_rows, d)
            if self.ablation not in ("drop-col", "drop-row-col"):
                self.col_emb = nn.Embedding(config.max_cols, d)
            if self.ablation != "drop-tab":
                self.tableau_emb = nn.Embedding(2, d)

        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: (batch, 2n) — entry values (1..n)
            positions: (batch, 2n, 3) — [row, col, tableau_id] per token

        Returns:
            (batch, 2n, d_model) — token embeddings
        """
        rows = positions[:, :, 0]
        cols = positions[:, :, 1]
        tableau_ids = positions[:, :, 2]

        if self.ablation == "concat":
            x = torch.cat([
                self.value_emb(values),
                self.row_emb(rows),
                self.col_emb(cols),
                self.tableau_emb(tableau_ids),
            ], dim=-1)
        elif self.ablation == "1d-pos":
            seq_len = values.shape[1]
            pos_ids = torch.arange(seq_len, device=values.device).unsqueeze(0)
            x = self.value_emb(values) + self.pos_emb(pos_ids) + self.tableau_emb(tableau_ids)
        else:
            x = self.value_emb(values)
            if hasattr(self, "row_emb"):
                x = x + self.row_emb(rows)
            if hasattr(self, "col_emb"):
                x = x + self.col_emb(cols)
            if hasattr(self, "tableau_emb"):
                x = x + self.tableau_emb(tableau_ids)

        return self.dropout(self.layer_norm(x))


class RSKEncoder(nn.Module):
    """
    Encoder-only transformer for inverse RSK: (P, Q) → σ.

    Architecture:
    - TokenEmbedding: structured 2D embeddings for each tableau entry
    - TransformerEncoder: standard multi-head self-attention
    - n classification heads: each predicts σ(i) ∈ {1..n}

    The key insight: since (P, Q) fully determines σ, all information is present
    in the input — no autoregressive decoding needed. The encoder can attend to
    all 2n tokens simultaneously, and n parallel heads predict each position.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n = config.n

        self.embedding = TokenEmbedding(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # seq_len classification heads, each predicting distribution over {0..vocab_size-1}
        # Each head sees the pooled encoder output
        self.heads = nn.ModuleList([
            nn.Linear(config.d_model, config.vocab_size)
            for _ in range(config.seq_len)
        ])

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            values: (batch, 2n) — entry values
            positions: (batch, 2n, 3) — [row, col, tableau_id]

        Returns:
            logits: (batch, seq_len, vocab_size) — logits[b, i, j] = log-prob that output(i+1) = j+1
        """
        # Embed tokens: (batch, 2n, d_model)
        x = self.embedding(values, positions)

        # Encode: (batch, 2n, d_model)
        x = self.encoder(x)

        # Pool over all tokens: mean pooling → (batch, d_model)
        x = x.mean(dim=1)

        # n classification heads → (batch, n, n)
        logits = torch.stack([head(x) for head in self.heads], dim=1)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RSKLevelDecoder(nn.Module):
    """
    Encoder + level decoder for inverse word RSK via growth diagram.

    Architecture:
    - Same encoder as RSKEncoder (TokenEmbedding + TransformerEncoder)
    - Transformer decoder with k × m query tokens and block-causal self-attention
    - Each query token (level l, position i) produces a binary logit (grow/don't grow)
    - Ordinal reconstruction: w_i = sum of "grow" decisions across levels

    The block-causal mask mirrors the growth diagram: inserting value j into an SSYT
    decomposes into j elementary local rule applications along an anti-diagonal.
    Level l can see levels 0..l but not l+1..k-1.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.m = config.seq_len       # word length (number of positions)
        self.k = config.vocab_size    # alphabet size (number of levels)

        # Encoder (same as RSKEncoder)
        self.embedding = TokenEmbedding(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Decoder query embeddings
        self.level_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.position_emb = nn.Embedding(config.seq_len, config.d_model)

        # Transformer decoder with block-causal self-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )

        # Shared binary head: each decoder token → 1 logit
        self.binary_head = nn.Linear(config.d_model, 1)

        # Pre-compute and register block-causal mask
        mask = self._make_block_causal_mask(self.k, self.m)
        self.register_buffer("causal_mask", mask)

    @staticmethod
    def _make_block_causal_mask(k: int, m: int) -> torch.Tensor:
        """
        Block-causal mask for k levels × m positions.

        Token ordering: (level 0, pos 0), (level 0, pos 1), ..., (level 0, pos m-1),
                        (level 1, pos 0), ..., (level k-1, pos m-1)

        Token (l, i) can attend to all tokens at levels ≤ l.
        Within a level, all m positions see each other.

        Returns:
            mask: (k*m, k*m) boolean — True means BLOCKED (PyTorch convention)
        """
        total = k * m
        mask = torch.zeros(total, total, dtype=torch.bool)
        for q_level in range(k):
            for kv_level in range(k):
                if kv_level > q_level:
                    # Block attention from level q to future level kv
                    q_start = q_level * m
                    q_end = (q_level + 1) * m
                    kv_start = kv_level * m
                    kv_end = (kv_level + 1) * m
                    mask[q_start:q_end, kv_start:kv_end] = True
        return mask

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            values: (batch, 2m) — entry values
            positions: (batch, 2m, 3) — [row, col, tableau_id]

        Returns:
            logits: (batch, m, k) — binary logits per (position, level)
                    Same shape as RSKEncoder output (batch, seq_len, vocab_size)
                    but semantics are binary (ordinal) not classification.
        """
        batch = values.shape[0]

        # Encode input tableaux: (batch, 2m, d_model)
        x = self.embedding(values, positions)
        memory = self.encoder(x)

        # Build decoder queries: k × m tokens
        # Token ordering: level 0 positions, level 1 positions, ...
        levels = torch.arange(self.k, device=values.device)
        pos = torch.arange(self.m, device=values.device)

        # (k, m, d_model) via broadcasting
        queries = self.level_emb(levels).unsqueeze(1) + self.position_emb(pos).unsqueeze(0)
        queries = queries.reshape(self.k * self.m, self.config.d_model)  # (k*m, d_model)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)  # (batch, k*m, d_model)

        # Decode with block-causal self-attention
        decoded = self.decoder(
            queries,
            memory,
            tgt_mask=self.causal_mask,
        )  # (batch, k*m, d_model)

        # Binary head → (batch, k*m, 1) → (batch, k*m)
        logits = self.binary_head(decoded).squeeze(-1)

        # Reshape to (batch, k, m) then transpose to (batch, m, k)
        logits = logits.reshape(batch, self.k, self.m).permute(0, 2, 1)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineMLP(nn.Module):
    """
    Flat MLP baseline for comparison with PNNL approach.

    Flattens both tableaux into a single vector and predicts each σ(i)
    independently via shared hidden layers + n heads.

    This destroys the 2D structure — comparing its performance against
    RSKEncoder isolates the contribution of our structured encoding.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n = config.n

        # Flatten: num_tokens × 4 features (value, row, col, tableau_id)
        input_dim = config.num_tokens * 4

        layers = []
        prev_dim = input_dim
        for hidden_dim in config.mlp_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(config.dropout),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # seq_len classification heads, each predicting over vocab_size classes
        self.heads = nn.ModuleList([
            nn.Linear(prev_dim, config.vocab_size)
            for _ in range(config.seq_len)
        ])

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Same interface as RSKEncoder for drop-in comparison."""
        batch = values.shape[0]

        # Flatten: concatenate values and positions into a single vector
        # values: (batch, 2n) → (batch, 2n, 1)
        # positions: (batch, 2n, 3)
        # combined: (batch, 2n, 4) → (batch, 2n * 4)
        v = values.unsqueeze(-1).float()
        p = positions.float()
        x = torch.cat([v, p], dim=-1)  # (batch, 2n, 4)
        x = x.view(batch, -1)  # (batch, 2n * 4)

        x = self.shared(x)  # (batch, hidden)

        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test permutation config (backward compatibility)
    config = ModelConfig(n=8)
    assert config.seq_len == 8 and config.vocab_size == 8

    model = RSKEncoder(config)
    print(f"RSKEncoder (perm n=8): {model.count_parameters():,} parameters")

    batch = 4
    values = torch.randint(1, 9, (batch, 16))
    positions = torch.zeros(batch, 16, 3, dtype=torch.long)
    positions[:, :, 2] = torch.cat([torch.zeros(8), torch.ones(8)]).long()

    logits = model(values, positions)
    print(f"Input: values {values.shape}, positions {positions.shape}")
    print(f"Output: logits {logits.shape}")  # (4, 8, 8)
    assert logits.shape == (4, 8, 8)

    mlp = BaselineMLP(config)
    print(f"BaselineMLP (perm n=8): {mlp.count_parameters():,} parameters")
    logits_mlp = mlp(values, positions)
    print(f"Output: logits {logits_mlp.shape}")
    assert logits_mlp.shape == (4, 8, 8)

    # Test word config (seq_len ≠ vocab_size)
    print("\n--- Word config test (m=15, k=10) ---")
    word_config = ModelConfig(n=15, task="word", seq_len=15, vocab_size=10)
    assert word_config.seq_len == 15 and word_config.vocab_size == 10
    assert word_config.num_tokens == 30
    assert word_config.num_heads == 15

    word_model = RSKEncoder(word_config)
    print(f"RSKEncoder (word m=15, k=10): {word_model.count_parameters():,} parameters")

    wv = torch.randint(1, 11, (batch, 30))
    wp = torch.zeros(batch, 30, 3, dtype=torch.long)
    wp[:, :, 2] = torch.cat([torch.zeros(15), torch.ones(15)]).long()

    wl = word_model(wv, wp)
    print(f"Output: logits {wl.shape}")  # (4, 15, 10)
    assert wl.shape == (4, 15, 10), f"Expected (4, 15, 10), got {wl.shape}"

    word_mlp = BaselineMLP(word_config)
    print(f"BaselineMLP (word m=15, k=10): {word_mlp.count_parameters():,} parameters")
    wl_mlp = word_mlp(wv, wp)
    print(f"Output: logits {wl_mlp.shape}")
    assert wl_mlp.shape == (4, 15, 10)

    # Test RSKLevelDecoder
    print("\n--- RSKLevelDecoder test (m=15, k=10) ---")
    level_config = ModelConfig(n=15, task="word", seq_len=15, vocab_size=10)
    level_model = RSKLevelDecoder(level_config)
    print(f"RSKLevelDecoder (word m=15, k=10): {level_model.count_parameters():,} parameters")

    # Verify block-causal mask shape and structure
    mask = level_model.causal_mask
    print(f"Block-causal mask shape: {mask.shape}")  # (150, 150)
    assert mask.shape == (150, 150), f"Expected (150, 150), got {mask.shape}"
    # Level 0 cannot see level 1+: mask[0:15, 15:150] should be True
    assert mask[0, 15].item() == True, "Level 0 should not attend to level 1"
    # Level 0 sees itself: mask[0:15, 0:15] should be False
    assert mask[0, 0].item() == False, "Level 0 should attend to itself"
    # Last level sees everything: mask[135:150, :] should be all False
    assert mask[140, 0].item() == False, "Last level should attend to level 0"

    wl_level = level_model(wv, wp)
    print(f"Output: logits {wl_level.shape}")  # (4, 15, 10)
    assert wl_level.shape == (4, 15, 10), f"Expected (4, 15, 10), got {wl_level.shape}"

    # Test ordinal reconstruction
    binary_preds = (wl_level > 0).long().sum(dim=-1) - 1  # (batch, m) 0-indexed
    print(f"Reconstructed predictions shape: {binary_preds.shape}")  # (4, 15)
    assert binary_preds.shape == (4, 15)

    # Test ablation variants
    print("\n--- Ablation tests (n=8) ---")
    for abl in ["drop-row", "drop-col", "drop-tab", "drop-row-col", "1d-pos", "concat"]:
        abl_config = ModelConfig(n=8, ablation=abl)
        abl_model = RSKEncoder(abl_config)
        abl_logits = abl_model(values, positions)
        assert abl_logits.shape == (4, 8, 8), f"Ablation {abl} failed: {abl_logits.shape}"
        print(f"  {abl}: {abl_model.count_parameters():,} params, OK")

    print("\nAll tests passed!")
