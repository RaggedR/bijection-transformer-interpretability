"""
Systematic search for interpretable SAE features in the RSK transformer.

Key insight: every P-token lies on SOME bumping path (n steps remove all cells),
so aggregate path correlation is trivially 1.0. Instead we test PER-STEP:
for each inverse RSK step k, which features fire disproportionately on the
specific tokens involved in step k's reverse bumping path?

Usage:
    python3 sae/find_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# sae.py expects config/model/data from rsk/
rsk_dir = str(Path(__file__).resolve().parent.parent / "rsk")
sae_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, rsk_dir)
sys.path.insert(0, sae_dir)

import torch
from sae import SparseAutoencoder, SAEConfig, SAEAnalyzer, annotate_inverse_rsk, _tokens_to_tableaux

BASE = Path(__file__).resolve().parent.parent / "rsk" / "checkpoints" / "sae_encoder_n10"
ACTS_DIR = BASE / "activations"
SAES_DIR = BASE / "saes"
N_SAMPLES = 2000


def load_sae_and_data(layer: str):
    """Load pre-trained SAE, activations, and metadata."""
    acts = torch.load(ACTS_DIR / f"{layer}.pt", map_location="cpu", weights_only=True)[:N_SAMPLES]
    metadata = torch.load(ACTS_DIR / "metadata.pt", map_location="cpu", weights_only=True)
    meta = {k: v[:N_SAMPLES] for k, v in metadata.items()}
    sae_data = torch.load(SAES_DIR / f"{layer}_d1024.pt", map_location="cpu", weights_only=False)

    sae = SparseAutoencoder(sae_data["sae_config"])
    sae.load_state_dict(sae_data["sae_state_dict"])
    sae.eval()

    print(f"  Loaded {N_SAMPLES} samples, {acts.shape[1]} tokens/sample, {acts.shape[2]}d")
    return sae, acts, meta


@torch.no_grad()
def compute_feature_activations(sae, acts):
    """Run SAE encoder on all activations → (N, T, d_dict)."""
    N, T, d = acts.shape
    flat = acts.reshape(-1, d)
    parts = []
    for s in range(0, flat.shape[0], 8192):
        _, z, _ = sae(flat[s:s + 8192])
        parts.append(z.cpu())
    return torch.cat(parts).reshape(N, T, -1)


def compute_annotations(meta):
    """Compute inverse RSK annotations for all samples."""
    print("  Computing inverse RSK annotations...")
    annotations = []
    for i in range(N_SAMPLES):
        P, Q = _tokens_to_tableaux(meta["values"][i], meta["positions"][i])
        annotations.append(annotate_inverse_rsk(P, Q))
    return annotations


def build_step_masks(annotations, meta):
    """Build per-step bumping path masks.

    Returns: dict mapping step_k → (N, T) bool tensor
    where step_k is the Q-entry being processed (n, n-1, ..., 1).
    Also builds a P-token mask.
    """
    N = len(annotations)
    T = meta["positions"].shape[1]
    pos = meta["positions"]

    # Build (row, col) → token_idx mapping for P-tokens in each sample
    p_cell_maps = []
    for i in range(N):
        cells = {}
        for j in range(T):
            r, c, t = pos[i, j].tolist()
            if t == 0:  # P token
                cells[(r, c)] = j
        p_cell_maps.append(cells)

    # P-token mask
    is_p = pos[:, :, 2] == 0

    # Per-step masks
    n = len(annotations[0])  # number of steps = n
    step_masks = {}
    for step_idx in range(n):
        mask = torch.zeros(N, T, dtype=torch.bool)
        for i, ann_list in enumerate(annotations):
            ann = ann_list[step_idx]
            for rc in ann["bumping_path"]:
                if rc in p_cell_maps[i]:
                    mask[i, p_cell_maps[i][rc]] = True
        step_masks[step_idx] = mask

    return step_masks, is_p


def per_step_feature_scan(z, step_masks, is_p):
    """For each feature, find which step's bumping path it correlates with most.

    Returns list of dicts sorted by max per-step lift.
    """
    N, T, d_dict = z.shape
    active = z > 0  # (N, T, d_dict)
    freqs = active.float().reshape(-1, d_dict).mean(dim=0)
    alive = (freqs > 0.001).nonzero(as_tuple=True)[0].tolist()  # >0.1% frequency
    print(f"  Scanning {len(alive)} features with freq > 0.1%...")

    n_steps = len(step_masks)
    results = []

    for fi in alive:
        fi_active = active[:, :, fi]  # (N, T)

        # Base rate: P(feature active | P-token)
        n_p = is_p.sum().item()
        base_rate = (fi_active & is_p).sum().item() / max(n_p, 1)

        if base_rate < 1e-6:
            continue

        # Per-step lift
        best_step = -1
        best_lift = 0
        step_lifts = []
        for step_idx, mask in step_masks.items():
            n_on = mask.sum().item()
            if n_on == 0:
                step_lifts.append(0)
                continue
            rate_on = (fi_active & mask).sum().item() / n_on
            lift = rate_on / base_rate
            step_lifts.append(lift)
            if lift > best_lift:
                best_lift = lift
                best_step = step_idx

        # Also compute anti-correlation (which step does it avoid most?)
        worst_lift = min(step_lifts) if step_lifts else 1.0

        results.append({
            "feature": fi,
            "freq": freqs[fi].item(),
            "base_rate": base_rate,
            "best_step": best_step,
            "best_lift": best_lift,
            "worst_lift": worst_lift,
            "step_lifts": step_lifts,
        })

    results.sort(key=lambda x: x["best_lift"], reverse=True)
    return results


def enrich_with_position(results, z, meta):
    """Add positional/value/tableau preferences to results."""
    pos = meta["positions"]
    vals = meta["values"]
    tabs = pos[:, :, 2]

    for r in results:
        fi = r["feature"]
        zi = z[:, :, fi]

        # Tableau preference
        p_act = zi[tabs == 0].mean().item() if (tabs == 0).any() else 0
        q_act = zi[tabs == 1].mean().item() if (tabs == 1).any() else 0
        r["p_pref"] = p_act
        r["q_pref"] = q_act

        # Row preference
        rows = pos[:, :, 0]
        row_acts = {}
        for rv in rows.unique().tolist():
            m = rows == rv
            if m.any():
                row_acts[rv] = zi[m].mean().item()
        r["by_row"] = row_acts

        # Value preference
        val_acts = {}
        for vv in vals.unique().tolist():
            if vv == 0:
                continue
            m = vals == vv
            if m.any():
                val_acts[vv] = zi[m].mean().item()
        r["by_value"] = val_acts


def print_results(results, layer, n=25):
    """Print top features by per-step bumping path lift."""
    print(f"\n{'='*80}")
    print(f"Top {n} per-step bumping path features at {layer}")
    print(f"{'='*80}")
    print(f"{'Feat':>6} {'BestLift':>9} {'Step':>5} {'Freq':>6} {'Base':>6} "
          f"{'P/Q':>10}  StepProfile")
    print(f"{'─'*80}")

    for r in results[:n]:
        p_ratio = f"{r['p_pref']:.2f}/{r['q_pref']:.2f}"
        # Compact step profile: show lift for each step
        profile = " ".join(f"{l:.1f}" for l in r["step_lifts"])
        print(f"  {r['feature']:>4} {r['best_lift']:>9.2f} "
              f"  k={r['best_step']:>2} {r['freq']:>6.1%} {r['base_rate']:>6.3f} "
              f"{p_ratio:>10}  [{profile}]")


def print_feature_detail(r, z, meta, annotations):
    """Print detailed analysis of one feature."""
    fi = r["feature"]
    print(f"\n{'─'*60}")
    print(f"Feature {fi}")
    print(f"  Frequency: {r['freq']:.1%}")
    print(f"  Tableau: P={r['p_pref']:.3f}  Q={r['q_pref']:.3f}")
    print(f"  Best step: k={r['best_step']} (lift {r['best_lift']:.2f})")

    # Row heatmap
    row_str = "  By row: " + ", ".join(f"r{k}:{v:.3f}" for k, v in sorted(r["by_row"].items()))
    print(row_str)

    # Value heatmap
    top_vals = sorted(r["by_value"].items(), key=lambda x: x[1], reverse=True)[:5]
    val_str = "  Top values: " + ", ".join(f"v={k}:{v:.3f}" for k, v in top_vals)
    print(val_str)

    # Step profile with step descriptions
    print(f"  Step profile (lift per step):")
    for step_idx, lift in enumerate(r["step_lifts"]):
        marker = " <<<" if lift == r["best_lift"] and lift > 1.5 else ""
        # Show what this step does
        ann = annotations[0][step_idx]  # example from first sample
        print(f"    Step {step_idx} (removes Q={ann['q_entry']}, "
              f"path len={len(ann['bumping_path'])}): lift={lift:.2f}{marker}")

    # Top activating tokens
    zi = z[:, :, fi].reshape(-1)
    k = min(10, zi.shape[0])
    vals_top, idxs = zi.topk(k)
    T = z.shape[1]
    print(f"  Top-10 activating tokens:")
    for a, idx in zip(vals_top, idxs):
        si, ti = divmod(idx.item(), T)
        v = meta["values"][si, ti].item()
        rr, c, t = meta["positions"][si, ti].tolist()
        tab = "P" if t == 0 else "Q"
        print(f"    act={a.item():.2f}  val={v} r={rr} c={c} {tab}  sample={si}")


if __name__ == "__main__":
    layers = ["layer_0", "layer_2", "layer_5"]

    # Compute annotations once (shared across layers)
    print("Loading metadata...")
    metadata = torch.load(ACTS_DIR / "metadata.pt", map_location="cpu", weights_only=True)
    meta = {k: v[:N_SAMPLES] for k, v in metadata.items()}
    annotations = compute_annotations(meta)
    step_masks, is_p = build_step_masks(annotations, meta)
    print(f"  {len(step_masks)} steps, path sizes: "
          + ", ".join(f"k{i}={step_masks[i].sum().item()}" for i in range(len(step_masks))))

    for layer in layers:
        print(f"\n{'#'*80}")
        print(f"# {layer}")
        print(f"{'#'*80}")
        sae, acts, _ = load_sae_and_data(layer)
        z = compute_feature_activations(sae, acts)
        results = per_step_feature_scan(z, step_masks, is_p)
        enrich_with_position(results, z, meta)
        print_results(results, layer)

    # Detailed analysis of top 3 features at the critical layer (layer_2)
    print(f"\n\n{'#'*80}")
    print(f"# Detailed analysis: layer_2 top features")
    print(f"{'#'*80}")
    sae, acts, _ = load_sae_and_data("layer_2")
    z = compute_feature_activations(sae, acts)
    results = per_step_feature_scan(z, step_masks, is_p)
    enrich_with_position(results, z, meta)
    for r in results[:5]:
        print_feature_detail(r, z, meta, annotations)
