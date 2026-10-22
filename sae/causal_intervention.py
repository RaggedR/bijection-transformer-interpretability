"""
Causal feature intervention experiment for the RSK transformer.

For each SAE feature at a given layer, zero its activation in the SAE's
latent space, reconstruct the patched activation, and measure accuracy drop.
This identifies which features are causally necessary vs merely correlated.

Connects to Gauderis et al.'s compositional interpretability framework
(arXiv:2605.08934): features that are both causally important AND correlate
with specific algorithmic steps are compositionally faithful — the structural
label (I_S) and behavioural observation (I_C) commute.

Usage:
    # Permutation RSK (encoder_n10)
    python3 sae/causal_intervention.py --task permutation

    # Cylindric (encoder_cyl_10101010_m3)
    python3 sae/causal_intervention.py --task cylindric

    # Single layer, fewer samples (quick test)
    python3 sae/causal_intervention.py --task permutation --layers layer_2 --num-samples 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

rsk_dir = str(Path(__file__).resolve().parent.parent / "rsk")
sae_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, rsk_dir)
sys.path.insert(0, sae_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from model import RSKEncoder
from data import RSKSamplingDataset, CylindricSamplingDataset
from train import compute_metrics
from sae import SparseAutoencoder, SAEConfig, _compat_model_config


# ─── Configuration ────────────────────────────────────────────────────────────

PERM_CKPT = Path(__file__).resolve().parent.parent / "rsk" / "checkpoints"
PERM_MODEL = PERM_CKPT / "encoder_n10"
PERM_SAE = PERM_CKPT / "sae_encoder_n10"

CYL_PROFILE = (1, 0, 1, 0, 1, 0, 1, 0)
CYL_MAX_LABEL = 3
CYL_MAX_GAMMA_PARTS = 3
CYL_MAX_GAMMA_SIZE = 4
CYL_MODEL = PERM_CKPT / "encoder_cyl_10101010_m3"
CYL_SAE = PERM_CKPT / "sae_cyl_10101010"


# ─── Model and SAE Loading ────────────────────────────────────────────────────


def load_model(model_dir: Path, device: torch.device) -> tuple[RSKEncoder, ModelConfig]:
    ckpt = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    mc = _compat_model_config(ckpt["model_config"])
    model = RSKEncoder(mc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, mc


def load_sae(sae_dir: Path, layer: str, device: torch.device, d_dict: int = 1024) -> SparseAutoencoder:
    sp = sae_dir / "saes" / f"{layer}_d{d_dict}.pt"
    sc = torch.load(sp, map_location=device, weights_only=False)
    sae = SparseAutoencoder(sc["sae_config"]).to(device)
    sae.load_state_dict(sc["sae_state_dict"])
    sae.eval()
    return sae


def get_test_data(
    mc: ModelConfig, num_samples: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load test data, return (values, positions, targets) on CPU."""
    if mc.task == "cylindric":
        ds = CylindricSamplingDataset(
            profile=mc.profile, max_label=mc.max_label,
            max_gamma_parts=mc.max_gamma_parts, max_gamma_size=mc.max_gamma_size,
            size=num_samples, seed=99999,
        )
    else:
        ds = RSKSamplingDataset(mc.n, num_samples, seed=99999)

    loader = DataLoader(ds, batch_size=256, shuffle=False)
    all_v, all_p, all_t = [], [], []
    for v, p, t in loader:
        all_v.append(v)
        all_p.append(p)
        all_t.append(t)
    return torch.cat(all_v), torch.cat(all_p), torch.cat(all_t)


# ─── Evaluation Helpers ───────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: RSKEncoder, mc: ModelConfig,
    all_v: torch.Tensor, all_p: torch.Tensor, all_t: torch.Tensor,
    device: torch.device, batch_size: int = 256,
) -> dict:
    """Run model on test data, return metrics dict."""
    logits_list = []
    for i in range(0, len(all_v), batch_size):
        v = all_v[i:i + batch_size].to(device)
        p = all_p[i:i + batch_size].to(device)
        logits_list.append(model(v, p).cpu())
    return compute_metrics(torch.cat(logits_list), all_t, task=mc.task)


def get_target_module(model: RSKEncoder, layer: str) -> nn.Module:
    if layer == "embed":
        return model.embedding
    return model.encoder.layers[int(layer.split("_")[1])]


# ─── Feature Ablation ─────────────────────────────────────────────────────────


def make_sae_hook(sae: SparseAutoencoder, zero_features: list[int] | None = None):
    """Create a forward hook that patches activations through the SAE.

    If zero_features is None, reconstruct faithfully (SAE baseline).
    If zero_features is a list of feature indices, zero those before decoding.
    """
    def hook(module, inp, out):
        b, t, d = out.shape
        flat = out.reshape(-1, d)

        # Encode
        h_norm = (flat - sae.act_mean) / sae.act_std
        z = torch.relu(sae.W_enc(h_norm - sae.W_dec.bias))

        # TopK sparsity
        if sae.config.top_k is not None:
            topk_vals, topk_idx = z.topk(sae.config.top_k, dim=-1)
            z = torch.zeros_like(z).scatter(-1, topk_idx, topk_vals)

        # Ablation: zero specified features
        if zero_features is not None:
            for fi in zero_features:
                z[:, fi] = 0

        # Decode
        h_hat_norm = sae.W_dec(z)
        h_hat = h_hat_norm * sae.act_std + sae.act_mean
        return h_hat.reshape(b, t, d)

    return hook


@torch.no_grad()
def evaluate_with_hook(
    model: RSKEncoder, mc: ModelConfig, layer: str,
    hook_fn,
    all_v: torch.Tensor, all_p: torch.Tensor, all_t: torch.Tensor,
    device: torch.device, batch_size: int = 256,
) -> dict:
    """Evaluate model with a forward hook on the specified layer."""
    target = get_target_module(model, layer)
    handle = target.register_forward_hook(hook_fn)
    try:
        return evaluate(model, mc, all_v, all_p, all_t, device, batch_size)
    finally:
        handle.remove()


def find_alive_features(
    sae: SparseAutoencoder, acts_dir: Path, layer: str,
    num_samples: int = 2000, min_freq: float = 0.001,
) -> list[int]:
    """Find features with activation frequency > min_freq."""
    acts = torch.load(acts_dir / f"{layer}.pt", map_location="cpu", weights_only=True)[:num_samples]
    N, T, d = acts.shape
    flat = acts.reshape(-1, d)

    parts = []
    with torch.no_grad():
        for s in range(0, flat.shape[0], 8192):
            _, z, _ = sae(flat[s:s + 8192])
            parts.append(z.cpu())
    z = torch.cat(parts)
    freqs = (z > 0).float().mean(dim=0)
    alive = (freqs > min_freq).nonzero(as_tuple=True)[0].tolist()
    return alive


# ─── Per-Step / Per-Face Annotation (reuse existing code) ─────────────────────


def get_permutation_annotations(sae_acts_dir: Path, num_samples: int = 2000):
    """Get per-step bumping path correlational data from find_features.py."""
    from find_features import (
        load_sae_and_data, compute_feature_activations,
        compute_annotations, build_step_masks, per_step_feature_scan,
        enrich_with_position,
    )
    # Override the module-level constant
    import find_features
    old_n = find_features.N_SAMPLES
    find_features.N_SAMPLES = num_samples

    metadata = torch.load(sae_acts_dir / "metadata.pt", map_location="cpu", weights_only=True)
    meta = {k: v[:num_samples] for k, v in metadata.items()}
    annotations = compute_annotations(meta)
    step_masks, is_p = build_step_masks(annotations, meta)

    find_features.N_SAMPLES = old_n
    return meta, annotations, step_masks, is_p


def get_cylindric_annotations(num_samples: int = 2000):
    """Get per-face growth diagram correlational data from cylindric_sae.py."""
    from cylindric_sae import (
        annotate_growth_diagram, build_face_masks,
        N_ANALYZE, SAE_DIR,
    )
    acts_dir = SAE_DIR / "activations"
    meta = torch.load(acts_dir / "metadata.pt", map_location="cpu", weights_only=True)
    meta = {k: v[:num_samples] for k, v in meta.items()}

    ds = CylindricSamplingDataset(
        profile=CYL_PROFILE, max_label=CYL_MAX_LABEL,
        max_gamma_parts=CYL_MAX_GAMMA_PARTS, max_gamma_size=CYL_MAX_GAMMA_SIZE,
        size=num_samples, seed=42,
    )

    T = len(CYL_PROFILE)
    annotations_list = []
    for i in range(num_samples):
        vals, pos, target = ds[i]
        cpp = [[] for _ in range(T)]
        for j in range(vals.shape[0]):
            v = vals[j].item()
            pi = pos[j, 0].item()
            if v > 0 and pi < T:
                cpp[pi].append(v)
        annotations_list.append(annotate_growth_diagram(CYL_PROFILE, cpp))

    mc_ckpt = torch.load(CYL_MODEL / "best.pt", map_location="cpu", weights_only=False)
    face_masks = build_face_masks(annotations_list, meta, mc_ckpt["model_config"])
    return meta, annotations_list, face_masks


def correlational_scan_permutation(sae, acts_dir, layer, step_masks, is_p, num_samples=2000):
    """Run per-step correlational scan, return {feature_idx: {best_step, best_lift, step_lifts}}."""
    from find_features import compute_feature_activations, per_step_feature_scan
    acts = torch.load(acts_dir / f"{layer}.pt", map_location="cpu", weights_only=True)[:num_samples]
    z = compute_feature_activations(sae, acts)
    results = per_step_feature_scan(z, step_masks, is_p)
    return {r["feature"]: r for r in results}


def correlational_scan_cylindric(sae, acts_dir, layer, face_masks, meta, num_samples=2000):
    """Run per-face correlational scan, return {feature_idx: {best_face, best_lift, face_lifts}}."""
    from cylindric_sae import per_face_feature_scan
    acts = torch.load(acts_dir / f"{layer}.pt", map_location="cpu", weights_only=True)[:num_samples]
    flat = acts.reshape(-1, 128)
    parts = []
    with torch.no_grad():
        for s in range(0, flat.shape[0], 8192):
            _, z_part, _ = sae(flat[s:s + 8192])
            parts.append(z_part.cpu())
    z = torch.cat(parts).reshape(num_samples, acts.shape[1], -1)
    results = per_face_feature_scan(z, face_masks, meta)
    return {r["feature"]: r for r in results}


# ─── Main Experiment ──────────────────────────────────────────────────────────


def run_experiment(
    task: str,
    layers: list[str],
    num_samples: int,
    device: torch.device,
):
    print(f"\n{'='*80}")
    print(f"Causal Feature Intervention Experiment")
    print(f"  Task: {task}")
    print(f"  Layers: {layers}")
    print(f"  Test samples: {num_samples}")
    print(f"  Device: {device}")
    print(f"{'='*80}")

    # Load model
    if task == "permutation":
        model, mc = load_model(PERM_MODEL, device)
        sae_base = PERM_SAE
        acts_dir = PERM_SAE / "activations"
    else:
        model, mc = load_model(CYL_MODEL, device)
        sae_base = CYL_SAE
        acts_dir = CYL_SAE / "activations"

    print(f"\nLoaded model: {mc.task}, params={sum(p.numel() for p in model.parameters()):,}")

    # Load test data
    print("Generating test data...")
    all_v, all_p, all_t = get_test_data(mc, num_samples, device)

    # Baseline (no intervention)
    print("Computing baseline accuracy...")
    baseline = evaluate(model, mc, all_v, all_p, all_t, device)
    print(f"  Baseline: {baseline}")

    # Load correlational annotations
    print("Loading correlational annotations...")
    if task == "permutation":
        meta, annotations, step_masks, is_p = get_permutation_annotations(acts_dir, num_samples=2000)
    else:
        meta, annotations_list, face_masks = get_cylindric_annotations(num_samples=2000)

    # Per-layer experiment
    all_results = {}
    for layer in layers:
        print(f"\n{'#'*80}")
        print(f"# Layer: {layer}")
        print(f"{'#'*80}")

        sae = load_sae(sae_base, layer, device)
        print(f"  SAE loaded: d_dict={sae.config.d_dict}, top_k={sae.config.top_k}")

        # SAE reconstruction baseline (all features active)
        sae_hook = make_sae_hook(sae)
        sae_baseline = evaluate_with_hook(model, mc, layer, sae_hook, all_v, all_p, all_t, device)
        print(f"  SAE baseline: {sae_baseline}")

        # Zero ALL features (sanity check)
        all_features = list(range(sae.config.d_dict))
        zero_all_hook = make_sae_hook(sae, zero_features=all_features)
        zero_all = evaluate_with_hook(model, mc, layer, zero_all_hook, all_v, all_p, all_t, device)
        print(f"  All zeroed:   {zero_all}")

        # Find alive features
        alive = find_alive_features(sae, acts_dir, layer)
        print(f"  Alive features: {len(alive)}")

        # Correlational scan
        print("  Running correlational scan...")
        if task == "permutation":
            corr = correlational_scan_permutation(sae, acts_dir, layer, step_masks, is_p)
        else:
            corr = correlational_scan_cylindric(sae, acts_dir, layer, face_masks, meta)

        # Single-feature ablation scan
        print(f"  Ablating {len(alive)} features individually...")
        metric_key = "greedy_exact_match" if task == "permutation" else "exact_match"
        sae_base_val = sae_baseline[metric_key]

        feature_results = []
        t0 = time.time()
        for idx, fi in enumerate(alive):
            hook_fn = make_sae_hook(sae, zero_features=[fi])
            metrics = evaluate_with_hook(model, mc, layer, hook_fn, all_v, all_p, all_t, device)
            delta = sae_base_val - metrics[metric_key]
            delta_pp = sae_baseline["per_position"] - metrics["per_position"]

            result = {
                "feature": fi,
                "delta_exact": delta,
                "delta_per_pos": delta_pp,
                "ablated_exact": metrics[metric_key],
                "ablated_per_pos": metrics["per_position"],
            }

            # Add correlational data
            if fi in corr:
                c = corr[fi]
                if task == "permutation":
                    result["best_step"] = c["best_step"]
                    result["best_lift"] = c["best_lift"]
                    result["step_lifts"] = c.get("step_lifts", [])
                else:
                    result["best_face"] = c["best_face"]
                    result["best_lift"] = c["best_lift"]
                    result["face_lifts"] = c.get("face_lifts", [])
            else:
                result["best_lift"] = 0.0

            feature_results.append(result)

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = (len(alive) - idx - 1) / rate
                print(f"    {idx+1}/{len(alive)} features "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        # Sort by causal importance
        feature_results.sort(key=lambda x: x["delta_exact"], reverse=True)
        all_results[layer] = feature_results

        # Print results
        print_causal_results(feature_results, layer, task, sae_baseline, baseline)

    # Cross-layer summary
    print_summary(all_results, task, baseline)

    return all_results


# ─── Output ───────────────────────────────────────────────────────────────────


def print_causal_results(results, layer, task, sae_baseline, baseline):
    metric_key = "greedy_exact_match" if task == "permutation" else "exact_match"
    step_or_face = "Step" if task == "permutation" else "Face"

    print(f"\n{'='*90}")
    print(f"Causal importance ranking: {layer}")
    print(f"  Baseline: {baseline[metric_key]:.4f}  "
          f"SAE baseline: {sae_baseline[metric_key]:.4f}  "
          f"SAE per-pos: {sae_baseline['per_position']:.4f}")
    print(f"{'='*90}")
    print(f"{'Feat':>6} {'Δ Exact':>9} {'Δ PerPos':>9} {'Lift':>7} "
          f"{step_or_face:>5}  {'Ablated':>8}  Assessment")
    print(f"{'─'*90}")

    causal_count = 0
    for r in results[:30]:
        fi = r["feature"]
        de = r["delta_exact"]
        dp = r["delta_per_pos"]
        lift = r.get("best_lift", 0)
        if task == "permutation":
            sf = r.get("best_step", -1)
            sf_str = f"k={sf:>2}" if sf >= 0 else "  --"
        else:
            sf = r.get("best_face", -1)
            sf_str = f"f={sf:>2}" if sf >= 0 else "  --"

        # Classify
        if de > 0.001 and lift > 1.5:
            assessment = "CAUSAL + CORRELATED ★"
            causal_count += 1
        elif de > 0.001:
            assessment = "causal only"
        elif lift > 1.5:
            assessment = "correlated only"
        else:
            assessment = ""

        print(f"  {fi:>4} {de:>+9.4f} {dp:>+9.4f} {lift:>7.2f} "
              f"{sf_str:>5}  {r['ablated_exact']:>8.4f}  {assessment}")

    # Count features with nonzero causal impact
    causal = [r for r in results if r["delta_exact"] > 0.001]
    correlated = [r for r in results if r.get("best_lift", 0) > 1.5]
    both = [r for r in results if r["delta_exact"] > 0.001 and r.get("best_lift", 0) > 1.5]

    print(f"\n  Summary:")
    print(f"    Causally important (Δ > 0.1%): {len(causal)}")
    print(f"    Correlationally strong (lift > 1.5): {len(correlated)}")
    print(f"    Both (compositionally faithful): {len(both)}")

    if causal and correlated:
        # Overlap analysis: top-20 of each
        top_causal = set(r["feature"] for r in sorted(results, key=lambda x: x["delta_exact"], reverse=True)[:20])
        top_corr = set(r["feature"] for r in sorted(results, key=lambda x: x.get("best_lift", 0), reverse=True)[:20])
        overlap = top_causal & top_corr
        print(f"    Top-20 overlap (causal ∩ correlated): {len(overlap)} features")
        if overlap:
            print(f"      Features: {sorted(overlap)}")


def print_summary(all_results, task, baseline):
    print(f"\n\n{'#'*80}")
    print(f"# EXPERIMENT SUMMARY")
    print(f"{'#'*80}")

    metric_key = "greedy_exact_match" if task == "permutation" else "exact_match"
    print(f"\nBaseline accuracy: {baseline[metric_key]:.4f}")

    for layer, results in all_results.items():
        causal = [r for r in results if r["delta_exact"] > 0.001]
        both = [r for r in results if r["delta_exact"] > 0.001 and r.get("best_lift", 0) > 1.5]
        if causal:
            top = causal[0]
            print(f"\n  {layer}:")
            print(f"    Causal features: {len(causal)}")
            print(f"    Compositionally faithful: {len(both)}")
            print(f"    Most impactful: feature {top['feature']} "
                  f"(Δ={top['delta_exact']:+.4f}, lift={top.get('best_lift', 0):.2f})")
        else:
            print(f"\n  {layer}: no causally important features found")

    print(f"\n  Gauderis assessment:")
    total_both = sum(
        len([r for r in res if r["delta_exact"] > 0.001 and r.get("best_lift", 0) > 1.5])
        for res in all_results.values()
    )
    total_causal = sum(
        len([r for r in res if r["delta_exact"] > 0.001])
        for res in all_results.values()
    )
    if total_causal > 0:
        ratio = total_both / total_causal
        print(f"    {total_both}/{total_causal} causal features are compositionally faithful "
              f"({ratio:.0%})")
        if ratio > 0.5:
            print(f"    → Strong evidence for explanatory optimism (Conjecture 4.3)")
        elif ratio > 0.2:
            print(f"    → Moderate evidence for explanatory optimism")
        else:
            print(f"    → Weak evidence — many causal features lack interpretable correlates")
    else:
        print(f"    No causal features found at these layers")


# ─── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Causal feature intervention experiment for RSK transformer"
    )
    parser.add_argument("--task", choices=["permutation", "cylindric"], default="permutation")
    parser.add_argument("--layers", nargs="+", default=None,
                        help="Layers to scan (default: layer_0 layer_2 layer_5)")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    layers = args.layers or ["layer_0", "layer_2", "layer_5"]
    device = torch.device(args.device)

    run_experiment(args.task, layers, args.num_samples, device)
