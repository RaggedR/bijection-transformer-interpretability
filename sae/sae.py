"""
Sparse Autoencoder (SAE) for interpreting RSK transformer internals.

Trains SAEs on residual-stream activations at each transformer layer to
decompose learned representations into interpretable features. The core
question: do SAE features correspond to Schensted reverse bumping /
Fomin's local rule steps, or has the transformer found a different algorithm?

Usage (from repo root):
    python3 sae/sae.py --checkpoint encoder_n10 --top-k 32
    python3 sae/sae.py --analyze --checkpoint encoder_n10 --layer layer_3
    python3 sae/sae.py --analyze --checkpoint encoder_n10 --layer layer_3 --feature 47
    python3 sae/sae.py --recon-test --checkpoint encoder_n10 --layer layer_3
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path so we can import config, model, data, rsk
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from model import RSKEncoder
from data import RSKSamplingDataset


# ─── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class SAEConfig:
    """Sparse autoencoder hyperparameters."""

    d_input: int = 128       # residual stream dimension (= d_model)
    d_dict: int = 1024       # dictionary size (8× overcomplete)
    l1_coeff: float = 3e-4   # L1 sparsity penalty (ignored if top_k is set)
    top_k: int | None = None # if set, keep only top-k features (TopK SAE)
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 4096   # per-token mini-batch
    seed: int = 42


# ─── Sparse Autoencoder ────────────────────────────────────────────────────────


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder (Bricken et al. 2023, "Towards Monosemanticity").

    Encoder:  z = ReLU(W_enc (h − b_dec) + b_enc)     z ∈ ℝ^d_dict
    Decoder:  ĥ = W_dec z + b_dec                      ĥ ∈ ℝ^d_input
    Loss:     ‖h − ĥ‖² + α ‖z‖₁

    Decoder columns are unit-normalized after each optimizer step to prevent
    the encoder from shrinking activations to dodge the L1 penalty.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d_in, d_dict = config.d_input, config.d_dict

        self.W_enc = nn.Linear(d_in, d_dict)        # weight: (d_dict, d_in), bias: b_enc
        self.W_dec = nn.Linear(d_dict, d_in)         # weight: (d_in, d_dict), bias: b_dec

        # Per-dimension normalization params (set by train_sae)
        self.register_buffer("act_mean", torch.zeros(d_in))
        self.register_buffer("act_std", torch.ones(d_in))

        # Initialize decoder columns (dictionary vectors) to unit norm
        with torch.no_grad():
            self.W_dec.weight.data = nn.functional.normalize(
                self.W_dec.weight.data, dim=0
            )

    def forward(
        self, h: torch.Tensor, normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (..., d_input) — activation vectors (raw, unnormalized)
            normalize: if True, apply stored normalization before encoding
                       and unnormalize the reconstruction. Set False if input
                       is already normalized (used internally by train_sae).

        Returns:
            h_hat: reconstructed activations (same space as input h)
            z:     sparse feature activations (d_dict)
            loss:  scalar (reconstruction MSE + L1 sparsity, in normalized space)
        """
        if normalize:
            h_norm = (h - self.act_mean) / self.act_std
        else:
            h_norm = h

        z = torch.relu(self.W_enc(h_norm - self.W_dec.bias))

        # TopK: zero out all but the top-k activations per input
        if self.config.top_k is not None:
            topk_vals, topk_idx = z.topk(self.config.top_k, dim=-1)
            z = torch.zeros_like(z).scatter(-1, topk_idx, topk_vals)

        h_hat_norm = self.W_dec(z)

        # Loss in normalized space (scale-invariant across layers)
        recon = (h_norm - h_hat_norm).pow(2).sum(dim=-1).mean()
        if self.config.top_k is not None:
            loss = recon  # TopK enforces sparsity directly, no L1 needed
        else:
            sparsity = z.abs().sum(dim=-1).mean()
            loss = recon + self.config.l1_coeff * sparsity

        if normalize:
            h_hat = h_hat_norm * self.act_std + self.act_mean
        else:
            h_hat = h_hat_norm

        return h_hat, z, loss

    @torch.no_grad()
    def normalize_decoder(self):
        """Unit-normalize decoder weight columns after each optimizer step."""
        self.W_dec.weight.data = nn.functional.normalize(
            self.W_dec.weight.data, dim=0
        )


# ─── Activation Collection ──────────────────────────────────────────────────────


def _hook_names(num_layers: int) -> list[str]:
    """Hook point names: embed + layer_0 .. layer_{n-1}."""
    return ["embed"] + [f"layer_{i}" for i in range(num_layers)]


class ActivationCollector:
    """
    Collects residual-stream activations from RSKEncoder at each layer.

    Hook points (all producing (batch, num_tokens, d_model)):
      embed     — post-embedding (value + row + col + tableau, after LayerNorm)
      layer_i   — residual stream after transformer layer i
    """

    def __init__(self, model: RSKEncoder, device: torch.device):
        self.model = model
        self.device = device
        self._hooks: list = []
        self._captured: dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def fn(module, inp, out):
            self._captured[name] = out.detach().cpu()
        return fn

    def _register(self):
        self._hooks.append(
            self.model.embedding.register_forward_hook(self._make_hook("embed"))
        )
        for i, layer in enumerate(self.model.encoder.layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_hook(f"layer_{i}"))
            )

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def collect(
        self, dataloader: DataLoader, max_samples: int = 50_000,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Run model and collect activations at all hook points.

        Returns:
            activations: {hook_name: (N, num_tokens, d_model)}
            metadata:    {values: (N, T), positions: (N, T, 3), targets: (N, seq_len)}
        """
        self.model.eval()
        num_layers = len(self.model.encoder.layers)
        names = _hook_names(num_layers)

        self._register()
        acts = {n: [] for n in names}
        mv, mp, mt = [], [], []
        collected = 0

        try:
            for values, positions, targets in dataloader:
                if collected >= max_samples:
                    break
                rem = max_samples - collected
                if values.shape[0] > rem:
                    values, positions, targets = (
                        values[:rem], positions[:rem], targets[:rem],
                    )

                self.model(values.to(self.device), positions.to(self.device))
                for n in names:
                    acts[n].append(self._captured[n])
                mv.append(values)
                mp.append(positions)
                mt.append(targets)
                collected += values.shape[0]
        finally:
            self._remove()

        return (
            {n: torch.cat(acts[n]) for n in names},
            {
                "values": torch.cat(mv),
                "positions": torch.cat(mp),
                "targets": torch.cat(mt),
            },
        )


# ─── SAE Training ───────────────────────────────────────────────────────────────


def train_sae(
    activations: torch.Tensor,
    config: SAEConfig,
    device: torch.device,
) -> tuple[SparseAutoencoder, dict]:
    """
    Train a sparse autoencoder on activation vectors.

    Args:
        activations: (N, num_tokens, d) or (N, d) — flattened to per-token vectors
        config: SAE hyperparameters
        device: compute device

    Returns:
        (trained SAE, training metrics dict)
    """
    flat = activations.reshape(-1, activations.shape[-1])
    N = flat.shape[0]
    print(f"  Training on {N:,} vectors (d={config.d_input} → dict={config.d_dict})")

    # Normalize to zero mean, unit variance per dimension
    act_mean = flat.mean(dim=0)
    act_std = flat.std(dim=0).clamp(min=1e-8)
    flat_norm = (flat - act_mean) / act_std
    print(f"  Activation norm: mean_norm={flat.norm(dim=-1).mean().item():.1f}, "
          f"after norm={flat_norm.norm(dim=-1).mean().item():.1f}")

    sae = SparseAutoencoder(config).to(device)
    sae.act_mean.copy_(act_mean)
    sae.act_std.copy_(act_std)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
    torch.manual_seed(config.seed)

    for epoch in range(1, config.epochs + 1):
        perm = torch.randperm(N)
        total_loss = total_recon = total_l0 = 0.0
        nb = 0

        for start in range(0, N, config.batch_size):
            h = flat_norm[perm[start : start + config.batch_size]].to(device)
            h_hat, z, loss = sae(h, normalize=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            with torch.no_grad():
                total_loss += loss.item()
                total_recon += (h - h_hat).pow(2).sum(dim=-1).mean().item()
                total_l0 += (z > 0).float().sum(dim=-1).mean().item()
                nb += 1

        avg_loss = total_loss / nb
        avg_recon = total_recon / nb
        avg_l0 = total_l0 / nb

        # Dead feature count (sampled)
        with torch.no_grad():
            si = torch.randperm(N)[: min(50_000, N)]
            _, zs, _ = sae(flat_norm[si].to(device), normalize=False)
            alive = (zs > 0).any(dim=0).sum().item()
            dead = config.d_dict - alive

        print(
            f"    Epoch {epoch}/{config.epochs}: "
            f"loss={avg_loss:.4f} recon={avg_recon:.4f} "
            f"L0={avg_l0:.1f} dead={dead}/{config.d_dict}"
        )

    return sae, {
        "final_loss": avg_loss,
        "final_recon": avg_recon,
        "avg_l0": avg_l0,
        "dead_features": dead,
        "active_features": alive,
        "n_tokens": N,
    }


def _compat_model_config(mc: ModelConfig) -> ModelConfig:
    """Handle checkpoints saved before seq_len/vocab_size/task were added."""
    if getattr(mc, "seq_len", None) is not None:
        return mc
    # Reconstruct with current defaults (triggers __post_init__)
    kwargs = {}
    for field in ("n", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout"):
        if hasattr(mc, field):
            kwargs[field] = getattr(mc, field)
    return ModelConfig(**kwargs)


def train_all_layers(
    checkpoint_dir: str,
    sae_config: SAEConfig,
    device: torch.device,
    num_samples: int = 50_000,
    layers: list[str] | None = None,
):
    """Load model checkpoint, collect activations, train SAEs, save everything."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_path = ckpt_dir / "best.pt"

    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = _compat_model_config(ckpt["model_config"])
    sae_config.d_input = mc.d_model

    if mc.task != "permutation":
        print(f"Note: task={mc.task}, bumping-path analysis will be skipped")

    model = RSKEncoder(mc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Epoch {ckpt['epoch']}, val={ckpt['val_metrics']}")
    print(f"  n={mc.n}, d_model={mc.d_model}, layers={mc.num_layers}")

    # Output directories
    sae_base = ckpt_dir.parent / f"sae_{ckpt_dir.name}"
    act_dir = sae_base / "activations"
    sae_dir = sae_base / "saes"
    act_dir.mkdir(parents=True, exist_ok=True)
    sae_dir.mkdir(parents=True, exist_ok=True)

    names = _hook_names(mc.num_layers)

    # Collect or load cached activations
    meta_path = act_dir / "metadata.pt"
    if meta_path.exists() and all((act_dir / f"{n}.pt").exists() for n in names):
        print(f"\nUsing cached activations from {act_dir}/")
    else:
        print(f"\nCollecting activations from {num_samples:,} samples...")
        dataset = RSKSamplingDataset(mc.n, num_samples, seed=sae_config.seed + 100)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        collector = ActivationCollector(model, device)
        activations, metadata = collector.collect(loader, num_samples)

        for n in names:
            torch.save(activations[n], act_dir / f"{n}.pt")
        torch.save(metadata, meta_path)
        print(f"  Saved {len(names)} activation files + metadata")

    # Train SAEs for requested layers (default: all + pool)
    targets = layers if layers else names + ["pool"]

    for name in targets:
        print(f"\n{'='*60}")
        print(f"SAE for: {name}")

        if name == "pool":
            last = f"layer_{mc.num_layers - 1}"
            acts = torch.load(act_dir / f"{last}.pt", weights_only=False)
            acts = acts.mean(dim=1, keepdim=True)  # (N, 1, d)
        else:
            acts = torch.load(act_dir / f"{name}.pt", weights_only=False)

        sae, metrics = train_sae(acts, sae_config, device)

        torch.save(
            {
                "sae_state_dict": sae.state_dict(),
                "sae_config": sae_config,
                "metrics": metrics,
            },
            sae_dir / f"{name}_d{sae_config.d_dict}.pt",
        )
        print(f"  Saved to {sae_dir}/{name}_d{sae_config.d_dict}.pt")


# ─── RSK Annotation ─────────────────────────────────────────────────────────────


def _reverse_bump_traced(
    tableau: list[list[int]], row_idx: int, col_idx: int,
) -> tuple[list[list[int]], int, list[tuple[int, int]]]:
    """
    Reverse bumping with path tracing.

    Same as rsk.reverse_bump but also returns the list of (row, col) cells
    visited along the reverse bumping path — from the removed cell up to row 0.

    Note: corner cells are always rightmost in their row, so popping them never
    shifts column indices of remaining cells. This means (row, col) positions
    in the path map exactly back to original input token positions.
    """
    tableau = copy.deepcopy(tableau)
    path = [(row_idx, col_idx)]

    value = tableau[row_idx].pop(col_idx)
    if not tableau[row_idx]:
        tableau.pop(row_idx)

    r = row_idx - 1
    while r >= 0:
        row = tableau[r]
        pos = len(row) - 1
        while pos >= 0 and row[pos] >= value:
            pos -= 1
        if pos < 0:
            raise ValueError(f"Reverse bump failed at row {r}")
        path.append((r, pos))
        value, row[pos] = row[pos], value
        r -= 1

    return tableau, value, path


def annotate_inverse_rsk(
    P: list[list[int]], Q: list[list[int]],
) -> list[dict]:
    """
    Run inverse RSK step by step, recording reverse-bumping path annotations.

    Processes Q entries in decreasing order (n, n-1, ..., 1). For each step,
    records which P cells are traversed during reverse bumping. These paths
    are the "local rule applications" we want SAE features to detect.

    Returns one dict per step:
        q_entry:      which Q value was processed
        removed_cell: (row, col) of the cell removed from P
        bumping_path: list of (row, col) traversed during reverse bump
        sigma_value:  the recovered σ(i) value (1-indexed)
    """
    P, Q = copy.deepcopy(P), copy.deepcopy(Q)
    n = sum(len(row) for row in Q)
    annotations = []

    for i in range(n, 0, -1):
        found = False
        for ri, row in enumerate(Q):
            for ci, val in enumerate(row):
                if val == i:
                    Q[ri].pop(ci)
                    if not Q[ri]:
                        Q.pop(ri)
                    P, sigma_val, path = _reverse_bump_traced(P, ri, ci)
                    annotations.append({
                        "q_entry": i,
                        "removed_cell": (ri, ci),
                        "bumping_path": path,
                        "sigma_value": sigma_val,
                    })
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Entry {i} not found in Q")

    return annotations


def _tokens_to_tableaux(
    values: torch.Tensor, positions: torch.Tensor,
) -> tuple[list[list[int]], list[list[int]]]:
    """Reconstruct (P, Q) tableaux from flat token representation."""
    P: list[list[int]] = []
    Q: list[list[int]] = []
    for v, (r, c, t) in zip(values.tolist(), positions.tolist()):
        tab = P if t == 0 else Q
        while len(tab) <= r:
            tab.append([])
        tab[r].append(v)
    return P, Q


# ─── Analysis ────────────────────────────────────────────────────────────────────


class SAEAnalyzer:
    """
    Analyzes SAE features for interpretability.

    Key analyses:
    - Feature frequencies and L0 statistics
    - Max-activating examples with positional/value attributes
    - Positional heatmaps (which rows/cols/values/tableaux does a feature prefer?)
    - Bumping path correlation (does a feature fire on tokens involved in
      reverse-bumping paths? permutation task only)
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        activations: torch.Tensor,
        metadata: dict[str, torch.Tensor],
        task: str = "permutation",
    ):
        self.sae = sae
        self.activations = activations  # (N, T, d)
        self.metadata = metadata
        self.task = task
        self.N, self.T, self.d = activations.shape
        self._z: torch.Tensor | None = None
        self._annotations: list | None = None

    @torch.no_grad()
    def _get_z(self) -> torch.Tensor:
        """Compute and cache feature activations: (N, T, d_dict)."""
        if self._z is None:
            flat = self.activations.reshape(-1, self.d)
            parts = []
            for s in range(0, flat.shape[0], 8192):
                _, z, _ = self.sae(flat[s : s + 8192])
                parts.append(z.cpu())
            self._z = torch.cat(parts).reshape(self.N, self.T, -1)
        return self._z

    def feature_frequencies(self) -> torch.Tensor:
        """(d_dict,) — fraction of tokens where each feature fires."""
        z = self._get_z()
        return (z > 0).float().reshape(-1, z.shape[-1]).mean(dim=0)

    def dead_features(self) -> list[int]:
        """Indices of features that never activate."""
        return (self.feature_frequencies() == 0).nonzero(as_tuple=True)[0].tolist()

    def max_activating_examples(self, fi: int, top_k: int = 10) -> list[dict]:
        """Top-k tokens by activation strength for feature fi."""
        z = self._get_z()[:, :, fi].reshape(-1)
        k = min(top_k, z.shape[0])
        vals, idxs = z.topk(k)
        results = []
        for a, idx in zip(vals, idxs):
            si, ti = divmod(idx.item(), self.T)
            v = self.metadata["values"][si, ti].item()
            r, c, t = self.metadata["positions"][si, ti].tolist()
            results.append(dict(
                sample_idx=si, token_idx=ti, activation=a.item(),
                value=v, row=r, col=c, tableau_id=t,
            ))
        return results

    def positional_heatmap(self, fi: int) -> dict:
        """Mean activation of feature fi broken down by value/row/col/tableau."""
        z = self._get_z()[:, :, fi]
        vals = self.metadata["values"]
        pos = self.metadata["positions"]

        def by_attr(attr):
            d = {}
            for v in attr.unique().tolist():
                m = attr == v
                if m.any():
                    d[v] = z[m].mean().item()
            return dict(sorted(d.items()))

        tabs = pos[:, :, 2]
        return {
            "by_value": by_attr(vals),
            "by_row": by_attr(pos[:, :, 0]),
            "by_col": by_attr(pos[:, :, 1]),
            "by_tableau": {
                "P": z[tabs == 0].mean().item() if (tabs == 0).any() else 0,
                "Q": z[tabs == 1].mean().item() if (tabs == 1).any() else 0,
            },
        }

    def _get_annotations(self) -> list[list[dict]] | None:
        """Compute inverse RSK annotations (permutation task only). Cached."""
        if self.task != "permutation":
            return None
        if self._annotations is None:
            print("  Computing inverse RSK annotations...")
            self._annotations = []
            for i in range(self.N):
                P, Q = _tokens_to_tableaux(
                    self.metadata["values"][i], self.metadata["positions"][i],
                )
                self._annotations.append(annotate_inverse_rsk(P, Q))
        return self._annotations

    def bumping_path_correlation(self, fi: int) -> dict | None:
        """
        How much more likely is feature fi to fire on P-tokens that lie
        on a reverse-bumping path, compared to the base rate?

        Returns None for non-permutation tasks. Otherwise returns:
            feature_on_path:  P(feature active | token on bumping path)
            feature_base_rate: P(feature active | P-token)
            path_base_rate:   P(token on bumping path | P-token)
            lift:             feature_on_path / feature_base_rate
        """
        anns = self._get_annotations()
        if anns is None:
            return None

        z = self._get_z()[:, :, fi]
        pos = self.metadata["positions"]

        # Build per-sample bumping path membership mask
        on_path = torch.zeros(self.N, self.T, dtype=torch.bool)
        for i, ann_list in enumerate(anns):
            # Map (row, col) in P → token index
            p_cells = {}
            for j in range(self.T):
                r, c, t = pos[i, j].tolist()
                if t == 0:  # P token
                    p_cells[(r, c)] = j
            for ann in ann_list:
                for rc in ann["bumping_path"]:
                    if rc in p_cells:
                        on_path[i, p_cells[rc]] = True

        is_p = pos[:, :, 2] == 0
        active = z > 0

        n_p = is_p.sum().item()
        n_act_p = (active & is_p).sum().item()
        n_path_p = (on_path & is_p).sum().item()
        n_act_path = (active & on_path & is_p).sum().item()

        base = n_act_p / max(n_p, 1)
        path_rate = n_act_path / max(n_path_p, 1) if n_path_p else 0
        return {
            "feature_on_path": path_rate,
            "feature_base_rate": base,
            "path_base_rate": n_path_p / max(n_p, 1),
            "lift": path_rate / max(base, 1e-10),
        }

    def print_feature_summary(self, fi: int, top_k: int = 10):
        """Print detailed summary for one feature."""
        freq = self.feature_frequencies()
        z = self._get_z()[:, :, fi]
        active = z > 0

        print(f"\n{'─'*60}")
        print(f"Feature {fi}")
        print(f"  Frequency: {freq[fi].item():.1%}")
        if active.any():
            print(f"  Mean activation (when active): {z[active].mean().item():.3f}")
            print(f"  Max activation: {z.max().item():.3f}")

        hm = self.positional_heatmap(fi)
        print(f"  Tableau: P={hm['by_tableau']['P']:.3f}  Q={hm['by_tableau']['Q']:.3f}")

        by_row = hm["by_row"]
        if by_row:
            print(f"  By row: {', '.join(f'r{r}:{a:.3f}' for r, a in by_row.items())}")

        by_val = hm["by_value"]
        if by_val:
            top_vals = sorted(by_val.items(), key=lambda x: -x[1])[:5]
            print(f"  Top values: {', '.join(f'v={v}:{a:.3f}' for v, a in top_vals)}")

        examples = self.max_activating_examples(fi, top_k)
        print(f"\n  Top-{len(examples)} activating tokens:")
        for rank, ex in enumerate(examples, 1):
            tab = "P" if ex["tableau_id"] == 0 else "Q"
            print(
                f"    #{rank:2d}  act={ex['activation']:.2f}  "
                f"val={ex['value']} r={ex['row']} c={ex['col']} {tab}  "
                f"sample={ex['sample_idx']}"
            )

        bp = self.bumping_path_correlation(fi)
        if bp:
            print(f"\n  Bumping path correlation:")
            print(f"    On-path fire rate: {bp['feature_on_path']:.1%}")
            print(f"    Base fire rate:    {bp['feature_base_rate']:.1%}")
            print(f"    Path membership:   {bp['path_base_rate']:.1%}")
            print(f"    Lift:              {bp['lift']:.2f}×")

    def print_dashboard(self, top_n: int = 20):
        """Print overview of the SAE: stats, reconstruction quality, top features."""
        freq = self.feature_frequencies()
        z = self._get_z()
        d_dict = z.shape[-1]

        dead = (freq == 0).sum().item()
        avg_l0 = (z > 0).float().sum(dim=-1).mean().item()

        # Reconstruction quality (explained variance)
        with torch.no_grad():
            flat = self.activations.reshape(-1, self.d)
            sample = flat[: min(10_000, flat.shape[0])]
            h_hat, _, _ = self.sae(sample)
            recon = (sample - h_hat).pow(2).sum(dim=-1).mean().item()
            var = sample.var(dim=0).sum().item()
            ev = 1.0 - recon / max(var, 1e-10)

        print(f"\n{'='*60}")
        print(f"SAE Dashboard  (d_dict={d_dict})")
        print(f"  Active features: {d_dict - dead}/{d_dict}  ({dead} dead)")
        print(f"  Avg L0 (features per token): {avg_l0:.1f}")
        print(f"  Explained variance: {ev:.1%}")

        live = (freq > 0).nonzero(as_tuple=True)[0]
        order = freq[live].argsort(descending=True)
        top = live[order[:top_n]]

        header = f"  {'Idx':>5}  {'Freq':>7}  {'Mean':>6}  {'Max':>6}  {'P/Q':>3}  {'Row':>3}  {'Val':>3}"
        print(f"\n  Top-{top_n} features by frequency:")
        print(header)
        for idx in top:
            fi = idx.item()
            f = freq[fi].item()
            zf = z[:, :, fi]
            a = zf > 0
            ma = zf[a].mean().item() if a.any() else 0
            mx = zf.max().item()
            hm = self.positional_heatmap(fi)
            pq = "P" if hm["by_tableau"]["P"] > hm["by_tableau"]["Q"] else "Q"
            tr = max(hm["by_row"].items(), key=lambda x: x[1])[0] if hm["by_row"] else -1
            tv = max(hm["by_value"].items(), key=lambda x: x[1])[0] if hm["by_value"] else -1
            print(
                f"  {fi:5d}  {f:7.1%}  {ma:6.3f}  {mx:6.3f}  {pq:>3}  r{tr:>2}  v{tv:>2}"
            )


# ─── Reconstruction Accuracy Test ───────────────────────────────────────────────


@torch.no_grad()
def test_reconstruction_accuracy(
    ckpt_dir: str,
    sae_dir_str: str,
    layer: str,
    device: torch.device,
    d_dict: int = 1024,
    num_samples: int = 10_000,
):
    """
    Replace one layer's activations with SAE reconstructions and measure
    the downstream accuracy drop. A good SAE preserves >95% accuracy.
    """
    from train import compute_metrics

    ckpt = torch.load(
        Path(ckpt_dir) / "best.pt", map_location=device, weights_only=False,
    )
    mc = _compat_model_config(ckpt["model_config"])
    model = RSKEncoder(mc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sp = Path(sae_dir_str) / "saes" / f"{layer}_d{d_dict}.pt"
    sc = torch.load(sp, map_location=device, weights_only=False)
    sae = SparseAutoencoder(sc["sae_config"]).to(device)
    sae.load_state_dict(sc["sae_state_dict"])
    sae.eval()

    ds = RSKSamplingDataset(mc.n, num_samples, seed=12345)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    # Collect all data first (RSKSamplingDataset RNG is stateful — can't iterate twice)
    all_v, all_p, all_t = [], [], []
    for v, p, t in loader:
        all_v.append(v)
        all_p.append(p)
        all_t.append(t)
    all_v, all_p, all_t = torch.cat(all_v), torch.cat(all_p), torch.cat(all_t)

    # Original accuracy (no intervention)
    lo = []
    for i in range(0, len(all_v), 256):
        v = all_v[i : i + 256].to(device)
        p = all_p[i : i + 256].to(device)
        lo.append(model(v, p).cpu())
    orig = compute_metrics(torch.cat(lo), all_t, task=mc.task)

    # With SAE intervention: replace layer output with SAE(output)
    if layer == "embed":
        target_module = model.embedding
    else:
        target_module = model.encoder.layers[int(layer.split("_")[1])]

    def hook(module, inp, out):
        b, t, d = out.shape
        h_hat, _, _ = sae(out.reshape(-1, d))
        return h_hat.reshape(b, t, d)

    handle = target_module.register_forward_hook(hook)
    ls = []
    for i in range(0, len(all_v), 256):
        v = all_v[i : i + 256].to(device)
        p = all_p[i : i + 256].to(device)
        ls.append(model(v, p).cpu())
    handle.remove()

    sae_m = compute_metrics(torch.cat(ls), all_t, task=mc.task)

    print(f"\nReconstruction accuracy test ({layer}):")
    for k in orig:
        print(f"  {k:25s}  original={orig[k]:.4f}  with_sae={sae_m[k]:.4f}")


# ─── CLI ─────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Sparse Autoencoder for RSK transformer interpretability",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint name under checkpoints/ (e.g. encoder_n10)",
    )
    parser.add_argument("--dict-size", type=int, default=1024)
    parser.add_argument("--l1-coeff", type=float, default=3e-4)
    parser.add_argument("--top-k", type=int, default=None,
                        help="TopK SAE: keep only top-k features (overrides --l1-coeff)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--layer", type=str, default=None,
                        help="Specific layer (e.g. layer_3, embed, pool)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze a trained SAE instead of training")
    parser.add_argument("--feature", type=int, default=None,
                        help="Analyze a specific feature index")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top features to show in dashboard")
    parser.add_argument("--recon-test", action="store_true",
                        help="Test reconstruction accuracy")
    args = parser.parse_args()

    if args.checkpoint is None:
        _self_test()
        return

    device = torch.device(args.device)
    ckpt_dir = f"checkpoints/{args.checkpoint}"
    sae_dir = f"checkpoints/sae_{args.checkpoint}"

    sae_config = SAEConfig(
        d_dict=args.dict_size,
        l1_coeff=args.l1_coeff,
        top_k=args.top_k,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if args.recon_test:
        test_reconstruction_accuracy(
            ckpt_dir, sae_dir, args.layer or "layer_5",
            device, d_dict=args.dict_size,
        )
    elif args.analyze:
        _analyze(
            ckpt_dir, sae_dir, args.layer or "layer_5",
            device, sae_config, args.feature, args.top_n,
        )
    else:
        layers = [args.layer] if args.layer else None
        train_all_layers(ckpt_dir, sae_config, device, args.num_samples, layers)


def _analyze(ckpt_dir, sae_dir, layer, device, sae_config, feature_idx, top_n):
    """Load SAE + activations and run analysis."""
    sb = Path(sae_dir)
    sp = sb / "saes" / f"{layer}_d{sae_config.d_dict}.pt"
    print(f"Loading SAE from {sp}...")
    sc = torch.load(sp, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(sc["sae_config"])
    sae.load_state_dict(sc["sae_state_dict"])
    sae.eval()

    ad = sb / "activations"
    if layer == "pool":
        # Find the last layer's activations and mean-pool them
        layer_files = sorted(ad.glob("layer_*.pt"))
        last = layer_files[-1]
        acts = torch.load(last, weights_only=False).mean(dim=1, keepdim=True)
    else:
        acts = torch.load(ad / f"{layer}.pt", weights_only=False)

    metadata = torch.load(ad / "metadata.pt", weights_only=False)

    ckpt = torch.load(
        Path(ckpt_dir) / "best.pt", map_location="cpu", weights_only=False,
    )
    task = _compat_model_config(ckpt["model_config"]).task

    analyzer = SAEAnalyzer(sae, acts, metadata, task=task)
    if feature_idx is not None:
        analyzer.print_feature_summary(feature_idx)
    else:
        analyzer.print_dashboard(top_n)


def _self_test():
    """Self-test with synthetic data (runs when no checkpoint is given)."""
    from rsk import rsk_forward

    print("Self-test: SAE on synthetic data\n")

    config = SAEConfig(
        d_input=128, d_dict=256, l1_coeff=5e-4, epochs=5, batch_size=512,
    )

    # Synthetic data: 5 sparse ground-truth directions + noise
    torch.manual_seed(42)
    true_dirs = nn.functional.normalize(torch.randn(5, 128), dim=1)
    h = torch.zeros(5000, 128)
    for i in range(5000):
        n_active = torch.randint(1, 4, (1,)).item()
        active = torch.randperm(5)[:n_active]
        for j in active:
            h[i] += (torch.rand(1) * 2 + 0.5) * true_dirs[j]
    h += torch.randn_like(h) * 0.1

    sae, metrics = train_sae(h.unsqueeze(1), config, torch.device("cpu"))

    with torch.no_grad():
        h_hat, z, _ = sae(h[:100])
        ev = 1 - (h[:100] - h_hat).pow(2).mean().item() / h[:100].var().item()
    print(f"\n  Explained variance: {ev:.1%}")
    assert ev > 0.6, f"Poor reconstruction: {ev:.1%}"
    assert metrics["active_features"] > 3, f"Too few active features: {metrics['active_features']}"

    # Test annotate_inverse_rsk
    print("\nTesting annotate_inverse_rsk...")
    sigma = [3, 1, 4, 2, 5]
    P, Q = rsk_forward(sigma)
    anns = annotate_inverse_rsk(P, Q)

    # Verify recovery
    recovered = [0] * len(sigma)
    for a in anns:
        recovered[a["q_entry"] - 1] = a["sigma_value"]
    assert recovered == sigma, f"Recovery failed: {recovered} ≠ {sigma}"

    print(f"  σ = {sigma}")
    print(f"  P = {P}")
    print(f"  Q = {Q}")
    for a in anns:
        print(
            f"  Step q={a['q_entry']}: remove {a['removed_cell']}, "
            f"path={a['bumping_path']}, σ({a['q_entry']})={a['sigma_value']}"
        )

    print("\nAll self-tests passed!")


if __name__ == "__main__":
    main()
