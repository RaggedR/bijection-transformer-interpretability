"""
SAE analysis for cylindric plane partition model.

The cylindric bijection has NO alternative algorithm — it MUST use the Burge
local rule at each face. If SAE features correlate with specific local rule
applications (partition triples), that's direct evidence of learned local rules.

Pipeline: collect activations → train SAEs → annotate growth diagram faces →
scan for face-correlated features.

Usage:
    python3 sae/cylindric_sae.py                    # full pipeline
    python3 sae/cylindric_sae.py --analyze-only      # skip collection/training
"""

from __future__ import annotations

import sys
from pathlib import Path

rsk_dir = str(Path(__file__).resolve().parent.parent / "rsk")
sae_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, rsk_dir)
sys.path.insert(0, sae_dir)

import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from model import RSKEncoder
from data import CylindricSamplingDataset
from rsk import growth_diagram_inverse, _non_wrapping_inversions, _swap_profile
from sae import (
    SAEConfig, SparseAutoencoder, ActivationCollector, train_sae,
)

PROFILE = (1, 0, 1, 0, 1, 0, 1, 0)
T = len(PROFILE)
MAX_LABEL = 3
MAX_GAMMA_PARTS = 3
MAX_GAMMA_SIZE = 4
N_COLLECT = 5000
N_ANALYZE = 2000

CKPT_DIR = Path(__file__).resolve().parent.parent / "rsk" / "checkpoints"
MODEL_DIR = CKPT_DIR / "encoder_cyl_10101010_m3"
SAE_DIR = CKPT_DIR / "sae_cyl_10101010"


def load_model():
    """Load trained cylindric model."""
    ckpt = torch.load(MODEL_DIR / "best.pt", map_location="cpu", weights_only=False)
    config = ckpt["model_config"]
    model = RSKEncoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded cylindric model: profile={PROFILE}, {model.count_parameters():,} params")
    print(f"  seq_len={config.seq_len}, vocab_size={config.vocab_size}, "
          f"num_tokens={config.num_tokens}")
    return model, config


def collect_activations(model, config):
    """Collect residual stream activations from cylindric model."""
    print(f"\nCollecting activations from {N_COLLECT} samples...")
    dataset = CylindricSamplingDataset(
        profile=PROFILE,
        max_label=MAX_LABEL,
        max_gamma_parts=MAX_GAMMA_PARTS,
        max_gamma_size=MAX_GAMMA_SIZE,
        size=N_COLLECT,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    collector = ActivationCollector(model, torch.device("cpu"))
    activations, metadata = collector.collect(loader, max_samples=N_COLLECT)

    SAE_DIR.mkdir(parents=True, exist_ok=True)
    acts_dir = SAE_DIR / "activations"
    acts_dir.mkdir(exist_ok=True)
    for name, tensor in activations.items():
        torch.save(tensor, acts_dir / f"{name}.pt")
        print(f"  {name}: {tensor.shape}")
    torch.save(metadata, acts_dir / "metadata.pt")
    print(f"  metadata: values={metadata['values'].shape}, "
          f"positions={metadata['positions'].shape}")
    return activations, metadata


def train_saes(activations):
    """Train SAEs on layers 0, 2, 5."""
    saes_dir = SAE_DIR / "saes"
    saes_dir.mkdir(exist_ok=True)

    for layer in ["layer_0", "layer_2", "layer_5"]:
        print(f"\nTraining SAE for {layer}...")
        acts = activations[layer]
        sae_config = SAEConfig(d_input=128, d_dict=1024, top_k=32, epochs=10)
        sae, metrics = train_sae(acts, sae_config, torch.device("cpu"))
        torch.save({
            "sae_state_dict": sae.state_dict(),
            "sae_config": sae_config,
            "metrics": metrics,
        }, saes_dir / f"{layer}_d1024.pt")
        print(f"  {layer}: loss={metrics['final_loss']:.4f}, "
              f"active={metrics['active_features']}/1024")


# ─── Growth Diagram Annotation ─────────────────────────────────────────────────

def annotate_growth_diagram(
    profile: tuple[int, ...],
    cpp: list[list[int]],
) -> list[dict]:
    """
    Trace the growth diagram inverse, recording which partition triple
    is involved at each face (local rule application).

    Returns one dict per face/step:
        step:       step index (0, 1, ..., num_labels-1)
        position:   index i in profile where inversion is processed
        left:       (i-1) % T — left neighbor partition index
        right:      (i+1) % T — right neighbor partition index
        center:     i — center partition index
        label:      ALCD label value at this face
        profile:    profile state at this step
    """
    annotations = []
    current_profile = list(profile)
    current_cpp = [list(p) for p in cpp]
    step = 0

    while True:
        tp = tuple(current_profile)
        # Check if pi_min
        ones = [i for i, b in enumerate(current_profile) if b == 1]
        zeros = [i for i, b in enumerate(current_profile) if b == 0]
        if all(z > o for o in ones for z in zeros if True) or not _non_wrapping_inversions(tp):
            break

        inv_pos = _non_wrapping_inversions(tp)
        i = inv_pos[0]

        left = (i - 1) % len(current_profile)
        right = (i + 1) % len(current_profile)

        # Get the ALCD label by applying inverse rule
        alpha = current_cpp[left]
        beta = current_cpp[right]
        lam = current_cpp[i]

        from rsk import burge_inverse_rule
        m, nu = burge_inverse_rule(alpha, beta, lam)

        annotations.append({
            "step": step,
            "position": i,
            "left": left,
            "right": right,
            "center": i,
            "label": m,
            "profile": tuple(current_profile),
        })

        # Reduce
        current_cpp[i] = nu
        current_profile[i], current_profile[i + 1] = current_profile[i + 1], current_profile[i]
        step += 1

    return annotations


def build_face_masks(annotations_list, meta, config):
    """
    Build per-face masks: which tokens belong to the partition triple
    involved in each local rule application?

    For face at position i with neighbors (i-1, i+1):
    mask includes all tokens from partitions i-1, i, i+1.
    """
    N = len(annotations_list)
    T_tokens = meta["positions"].shape[1]
    pos = meta["positions"]

    # How many faces (should be same for all samples with same profile)
    num_faces = len(annotations_list[0])

    # Build partition_idx → token indices mapping
    # positions[:, :, 0] is partition index, positions[:, :, 1] is part index
    face_masks = {}
    for face_idx in range(num_faces):
        mask = torch.zeros(N, T_tokens, dtype=torch.bool)
        for sample_i, ann_list in enumerate(annotations_list):
            ann = ann_list[face_idx]
            involved = {ann["left"], ann["center"], ann["right"]}
            for tok_j in range(T_tokens):
                part_idx = pos[sample_i, tok_j, 0].item()  # partition index
                val = meta["values"][sample_i, tok_j].item()
                if part_idx in involved and val > 0:  # skip padding
                    mask[sample_i, tok_j] = True
        face_masks[face_idx] = mask

    return face_masks


def per_face_feature_scan(z, face_masks, meta):
    """For each feature, find which growth diagram face it correlates with most."""
    N, T, d_dict = z.shape
    active = z > 0
    freqs = active.float().reshape(-1, d_dict).mean(dim=0)
    alive = (freqs > 0.001).nonzero(as_tuple=True)[0].tolist()
    print(f"  Scanning {len(alive)} features with freq > 0.1%...")

    # Non-padding tokens
    is_nonpad = meta["values"] > 0

    results = []
    for fi in alive:
        fi_active = active[:, :, fi]
        n_nonpad = is_nonpad.sum().item()
        base_rate = (fi_active & is_nonpad).sum().item() / max(n_nonpad, 1)
        if base_rate < 1e-6:
            continue

        best_face = -1
        best_lift = 0
        face_lifts = []
        for face_idx, mask in face_masks.items():
            n_on = mask.sum().item()
            if n_on == 0:
                face_lifts.append(0)
                continue
            rate_on = (fi_active & mask).sum().item() / n_on
            lift = rate_on / base_rate
            face_lifts.append(lift)
            if lift > best_lift:
                best_lift = lift
                best_face = face_idx

        results.append({
            "feature": fi,
            "freq": freqs[fi].item(),
            "base_rate": base_rate,
            "best_face": best_face,
            "best_lift": best_lift,
            "face_lifts": face_lifts,
        })

    results.sort(key=lambda x: x["best_lift"], reverse=True)
    return results


def print_results(results, layer, annotations_example, n=25):
    """Print top features by per-face lift."""
    print(f"\n{'='*90}")
    print(f"Top {n} local-rule-correlated features at {layer}")
    print(f"{'='*90}")
    print(f"{'Feat':>6} {'BestLift':>9} {'Face':>5} {'Pos':>4} "
          f"{'Triple':>12} {'Freq':>6} {'Base':>6}  FaceProfile")
    print(f"{'─'*90}")

    for r in results[:n]:
        face_idx = r["best_face"]
        if face_idx >= 0 and face_idx < len(annotations_example):
            ann = annotations_example[face_idx]
            triple = f"({ann['left']},{ann['center']},{ann['right']})"
            pos_str = str(ann["position"])
        else:
            triple = "?"
            pos_str = "?"
        profile = " ".join(f"{l:.1f}" for l in r["face_lifts"])
        print(f"  {r['feature']:>4} {r['best_lift']:>9.2f} "
              f"  f={face_idx:>2}  {pos_str:>3} "
              f"{triple:>12} {r['freq']:>6.1%} {r['base_rate']:>6.3f}  [{profile}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    acts_dir = SAE_DIR / "activations"
    saes_dir = SAE_DIR / "saes"

    if not args.analyze_only:
        model, config = load_model()
        activations, metadata = collect_activations(model, config)
        train_saes(activations)
    else:
        print("Loading pre-computed activations and SAEs...")

    # Load for analysis
    meta = torch.load(acts_dir / "metadata.pt", map_location="cpu", weights_only=True)
    meta = {k: v[:N_ANALYZE] for k, v in meta.items()}

    # Reconstruct CPPs from metadata to annotate growth diagram
    print(f"\nAnnotating growth diagram faces for {N_ANALYZE} samples...")
    from data import CylindricSamplingDataset
    dataset = CylindricSamplingDataset(
        profile=PROFILE, max_label=MAX_LABEL,
        max_gamma_parts=MAX_GAMMA_PARTS, max_gamma_size=MAX_GAMMA_SIZE,
        size=N_ANALYZE, seed=42,
    )

    annotations_list = []
    max_parts = MAX_GAMMA_PARTS + 10  # enough room
    for i in range(N_ANALYZE):
        vals, pos, target = dataset[i]
        # Reconstruct CPP from tokens
        cpp = [[] for _ in range(T)]
        for j in range(vals.shape[0]):
            v = vals[j].item()
            pi = pos[j, 0].item()  # partition index
            if v > 0 and pi < T:
                cpp[pi].append(v)
        annotations_list.append(annotate_growth_diagram(PROFILE, cpp))
        if (i + 1) % 500 == 0:
            print(f"  annotated {i+1}/{N_ANALYZE}")

    # Verify all have same number of faces
    face_counts = set(len(a) for a in annotations_list)
    print(f"  Face counts: {face_counts}")

    # Show face structure from first sample
    print(f"\nGrowth diagram face structure (sample 0):")
    for ann in annotations_list[0]:
        print(f"  Face {ann['step']}: pos={ann['position']}, "
              f"triple=({ann['left']},{ann['center']},{ann['right']}), "
              f"label={ann['label']}, profile={ann['profile']}")

    face_masks = build_face_masks(annotations_list, meta,
                                  torch.load(MODEL_DIR / "best.pt",
                                             map_location="cpu",
                                             weights_only=False)["model_config"])

    # Analyze each layer
    for layer in ["layer_0", "layer_2", "layer_5"]:
        print(f"\n{'#'*90}")
        print(f"# {layer}")
        print(f"{'#'*90}")
        acts = torch.load(acts_dir / f"{layer}.pt", map_location="cpu", weights_only=True)[:N_ANALYZE]
        sae_data = torch.load(saes_dir / f"{layer}_d1024.pt", map_location="cpu", weights_only=False)
        sae = SparseAutoencoder(sae_data["sae_config"])
        sae.load_state_dict(sae_data["sae_state_dict"])
        sae.eval()

        # Compute feature activations
        flat = acts.reshape(-1, 128)
        parts = []
        with torch.no_grad():
            for s in range(0, flat.shape[0], 8192):
                _, z_part, _ = sae(flat[s:s + 8192])
                parts.append(z_part.cpu())
        z = torch.cat(parts).reshape(N_ANALYZE, acts.shape[1], -1)

        results = per_face_feature_scan(z, face_masks, meta)
        print_results(results, layer, annotations_list[0])
