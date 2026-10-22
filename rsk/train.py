"""
Training loop and evaluation for RSK neural network experiments.

Features:
- AdamW optimizer with cosine decay and linear warmup
- Masked greedy decoding for valid permutation inference
- Exact-match and per-position accuracy metrics
- Early stopping on validation exact-match accuracy
- Checkpoint saving
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import ModelConfig, TrainConfig
from model import RSKEncoder, RSKLevelDecoder, BaselineMLP
from data import make_dataloaders, verify_hf_against_rsk, load_hf_dataset


def masked_greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """
    Masked greedy decoding to produce valid permutations.

    Process: pick the highest-confidence (head, value) pair across all heads,
    assign it, mask that value from all other heads, repeat.

    Args:
        logits: (batch, n, n) — raw logits from model

    Returns:
        preds: (batch, n) — predicted permutation (0-indexed)
    """
    batch, n, _ = logits.shape
    preds = torch.full((batch, n), -1, dtype=torch.long, device=logits.device)

    # Work with a copy we can mask
    scores = logits.clone()

    for _ in range(n):
        # Find the global argmax across all (head, value) pairs per batch
        # Reshape to (batch, n*n) to find global max
        flat = scores.view(batch, -1)
        max_idx = flat.argmax(dim=1)  # (batch,)

        head_idx = max_idx // n  # which position
        val_idx = max_idx % n  # which value

        # Assign
        preds[torch.arange(batch, device=logits.device), head_idx] = val_idx

        # Mask: set this head and this value to -inf everywhere
        for b in range(batch):
            h, v = head_idx[b].item(), val_idx[b].item()
            scores[b, h, :] = float("-inf")  # this position is decided
            scores[b, :, v] = float("-inf")  # this value is used

    return preds


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor,
    task: str = "permutation", model_name: str = "encoder",
) -> dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        logits: (batch, seq_len, vocab_size) — model output
        targets: (batch, seq_len) — ground truth (0-indexed)
        task: "permutation" or "word"
        model_name: "encoder", "mlp", or "leveldecoder"

    Returns:
        dict with 'exact_match', 'per_position', and 'greedy_exact_match' (permutations only)
    """
    if model_name == "leveldecoder":
        # Binary logits → ordinal reconstruction: sum of (logit > 0) - 1
        # logits: (batch, m, k) binary logits
        preds = (logits > 0).long().sum(dim=-1) - 1  # (batch, m) 0-indexed
        # Clamp to valid range [0, k-1]
        preds = preds.clamp(0, logits.shape[-1] - 1)
    else:
        # Classification logits → argmax
        preds = logits.argmax(dim=-1)  # (batch, seq_len)

    per_pos = (preds == targets).float().mean().item()
    exact = (preds == targets).all(dim=1).float().mean().item()

    metrics = {
        "exact_match": exact,
        "per_position": per_pos,
    }

    # Masked greedy decoding only for permutations (no repeated values in output)
    if task == "permutation" and model_name != "leveldecoder":
        greedy_preds = masked_greedy_decode(logits)
        greedy_exact = (greedy_preds == targets).all(dim=1).float().mean().item()
        metrics["greedy_exact_match"] = greedy_exact

    return metrics


def _make_ordinal_targets(targets: torch.Tensor, k: int) -> torch.Tensor:
    """
    Convert 0-indexed integer targets to ordinal binary targets.

    Args:
        targets: (batch, m) — 0-indexed word values
        k: number of levels (vocab_size)

    Returns:
        ordinal: (batch, m, k) — float tensor where ordinal[b,i,l] = 1 if targets[b,i] >= l
    """
    levels = torch.arange(k, device=targets.device)  # (k,)
    # targets[:, :, None] >= levels[None, None, :] → (batch, m, k)
    return (targets.unsqueeze(-1) >= levels.unsqueeze(0).unsqueeze(0)).float()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    log_every: int = 50,
    model_name: str = "encoder",
) -> dict[str, float]:
    """Train for one epoch. Returns average metrics."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_positions = 0
    num_batches = 0

    for batch_idx, (values, positions, targets) in enumerate(loader):
        values = values.to(device)
        positions = positions.to(device)
        targets = targets.to(device)

        logits = model(values, positions)  # (batch, seq_len, vocab_size)
        batch_size, seq_len, vocab_size = logits.shape

        if model_name == "leveldecoder":
            # Ordinal BCE loss: convert targets to binary ordinal encoding
            ordinal = _make_ordinal_targets(targets, vocab_size)
            loss = F.binary_cross_entropy_with_logits(logits, ordinal)
            # Per-position accuracy via ordinal reconstruction
            preds = (logits > 0).long().sum(dim=-1) - 1  # (batch, m)
            preds = preds.clamp(0, vocab_size - 1)
        else:
            # Cross-entropy across all heads
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1)
            )
            preds = logits.argmax(dim=-1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_correct += (preds == targets).sum().item()
        total_positions += batch_size * seq_len
        num_batches += 1

        if log_every and (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_correct / total_positions
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  batch {batch_idx + 1:4d}/{len(loader)}: "
                f"loss={avg_loss:.4f} pos_acc={avg_acc:.4f} lr={lr:.2e}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "per_position": total_correct / max(total_positions, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    task: str = "permutation",
    model_name: str = "encoder",
) -> dict[str, float]:
    """Evaluate on a dataset. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    num_batches = 0

    for values, positions, targets in loader:
        values = values.to(device)
        positions = positions.to(device)
        targets = targets.to(device)

        logits = model(values, positions)
        batch_size, seq_len, vocab_size = logits.shape

        if model_name == "leveldecoder":
            ordinal = _make_ordinal_targets(targets, vocab_size)
            loss = F.binary_cross_entropy_with_logits(logits, ordinal)
        else:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        total_loss += loss.item()
        num_batches += 1

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_logits, all_targets, task=task, model_name=model_name)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def train(
    model_name: str = "encoder",
    n: int = 8,
    source: str = "hf",
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    resume: bool = False,
    task: str = "permutation",
    a_dim: int | None = None,
    b_dim: int | None = None,
    shape: tuple[int, ...] | None = None,
    max_entry: int | None = None,
    profile: tuple[int, ...] | None = None,
    max_label: int | None = None,
    max_gamma_parts: int = 3,
    max_gamma_size: int = 4,
):
    """
    Full training pipeline.

    Args:
        model_name: "encoder" for RSKEncoder, "mlp" for BaselineMLP
        n: permutation size, word length, or total_n for matrices (seq_len)
        source: "hf" for HuggingFace, "generate" for our own data, "sample" for random
        model_config: override model config
        train_config: override train config
        resume: if True, load existing checkpoint and continue training
        task: "permutation", "word", or "matrix"
        a_dim: matrix rows (for task="matrix")
        b_dim: matrix cols (for task="matrix")
    """
    if model_config is None:
        model_config = ModelConfig(n=n)
    if train_config is None:
        train_config = TrainConfig()

    device = torch.device(train_config.device)
    print(f"Device: {device}")
    if task == "word":
        print(f"Task: word (m={model_config.seq_len}, k={model_config.vocab_size})")
    elif task == "matrix":
        print(f"Task: matrix (a={a_dim}, b={b_dim}, N={model_config.seq_len})")
    elif task == "rpp":
        print(f"Task: rpp (shape={shape}, max_entry={max_entry}, |λ|={model_config.seq_len})")
    elif task == "cylindric":
        print(f"Task: cylindric (profile={profile}, max_label={max_label}, "
              f"num_labels={model_config.seq_len}, γ_max=({max_gamma_parts},{max_gamma_size}))")

    # Data
    print(f"\nLoading data for n={n} (source={source})...")
    train_loader, val_loader, test_loader = make_dataloaders(
        n, train_config, source,
        train_size=train_size, val_size=val_size, test_size=test_size,
        task=task, vocab_size=model_config.vocab_size if task == "word" else None,
        a_dim=a_dim, b_dim=b_dim,
        shape=shape, max_entry=max_entry,
        profile=profile, max_label=max_label,
        max_gamma_parts=max_gamma_parts, max_gamma_size=max_gamma_size,
    )
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")

    # Verify HuggingFace data if using it
    if source == "hf":
        print("\nCross-checking HuggingFace data against our RSK...")
        train_data, _ = load_hf_dataset(n)
        checked, matched = verify_hf_against_rsk(train_data, n, max_check=200)
        print(f"  {matched}/{checked} match our RSK implementation")
        if matched < checked:
            print("  WARNING: Some mismatches detected!")
        del train_data

    # Model
    if model_name == "encoder":
        model = RSKEncoder(model_config)
    elif model_name == "leveldecoder":
        model = RSKLevelDecoder(model_config)
    elif model_name == "mlp":
        model = BaselineMLP(model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    print(f"\n{model_name}: {model.count_parameters():,} parameters")

    # Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    total_steps = len(train_loader) * train_config.epochs
    warmup_steps = int(train_config.warmup_fraction * total_steps)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    # Checkpoint directory
    if task == "rpp":
        shape_str = "x".join(str(s) for s in shape)
        ckpt_dir = Path(train_config.checkpoint_dir) / f"{model_name}_rpp_{shape_str}_m{max_entry}"
    elif task == "cylindric":
        prof_str = "".join(str(b) for b in profile)
        ckpt_dir = Path(train_config.checkpoint_dir) / f"{model_name}_cyl_{prof_str}_m{max_label}"
    elif task == "matrix":
        ckpt_dir = Path(train_config.checkpoint_dir) / f"{model_name}_matrix_a{a_dim}_b{b_dim}_N{model_config.seq_len}"
    elif task == "word":
        ckpt_dir = Path(train_config.checkpoint_dir) / f"{model_name}_word_m{model_config.seq_len}_k{model_config.vocab_size}"
    else:
        ckpt_dir = Path(train_config.checkpoint_dir) / f"{model_name}_n{n}"
    if model_config.ablation:
        ckpt_dir = ckpt_dir.parent / f"{ckpt_dir.name}_ablate-{model_config.ablation}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_exact = 0.0
    patience_counter = 0

    # Which metric to track for early stopping
    # Greedy decode only for permutations (no repeated values); word/matrix use argmax exact
    tracking_metric = "greedy_exact_match" if task == "permutation" else "exact_match"

    if resume:
        ckpt_path = ckpt_dir / "best.pt"
        if ckpt_path.exists():
            print(f"\nResuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_exact = ckpt["val_metrics"].get(tracking_metric, 0.0)
            # Advance scheduler to the right step
            steps_done = len(train_loader) * ckpt["epoch"]
            for _ in range(steps_done):
                scheduler.step()
            print(f"  Resumed at epoch {start_epoch}, best {tracking_metric}={best_val_exact:.4f}")
        else:
            print(f"\nNo checkpoint found at {ckpt_path}, starting fresh.")

    print(f"\nTraining for up to {train_config.epochs} epochs...")
    print(f"{'='*80}")

    for epoch in range(start_epoch, train_config.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            log_every=train_config.log_every,
            model_name=model_name,
        )

        val_metrics = evaluate(model, val_loader, device, task=task, model_name=model_name)
        elapsed = time.time() - t0

        log_line = (
            f"Epoch {epoch:3d}/{train_config.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} train_pos={train_metrics['per_position']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_pos={val_metrics['per_position']:.4f} "
            f"val_exact={val_metrics['exact_match']:.4f}"
        )
        if task == "permutation":
            log_line += f" val_greedy={val_metrics['greedy_exact_match']:.4f}"
        print(log_line)

        # Save checkpoint every epoch (always have something to evaluate)
        ckpt_path = ckpt_dir / "best.pt"

        # Early stopping on tracked metric
        val_exact = val_metrics[tracking_metric]
        if val_exact >= 1.0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "model_config": model_config,
            }, ckpt_dir / "best.pt")
            print(f"  -> Perfect {tracking_metric}! Saved and stopping.")
            break
        if val_exact > best_val_exact or epoch == 1:
            best_val_exact = val_exact
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "model_config": model_config,
            }, ckpt_path)
            if val_exact > 0:
                print(f"  -> New best! Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={train_config.patience})")
                break

    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("Loading best model for test evaluation...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, task=task, model_name=model_name)
    if task == "rpp":
        shape_str = "x".join(str(s) for s in shape)
        print(f"\nTest Results (rpp shape={shape_str}, max_entry={max_entry}, {model_name}):")
    elif task == "matrix":
        print(f"\nTest Results (matrix a={a_dim}, b={b_dim}, N={model_config.seq_len}, {model_name}):")
    elif task == "word":
        print(f"\nTest Results (word m={model_config.seq_len}, k={model_config.vocab_size}, {model_name}):")
    else:
        print(f"\nTest Results (n={n}, {model_name}):")
    print(f"  Loss:              {test_metrics['loss']:.4f}")
    print(f"  Per-position acc:  {test_metrics['per_position']:.4f}")
    print(f"  Exact match:       {test_metrics['exact_match']:.4f}")
    if task == "permutation":
        print(f"  Greedy exact match:{test_metrics['greedy_exact_match']:.4f}")
    print(f"  Best val {tracking_metric}: {best_val_exact:.4f} (epoch {ckpt['epoch']})")

    return test_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RSK neural network")
    parser.add_argument("--model", choices=["encoder", "leveldecoder", "mlp"], default="encoder")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--task", choices=["permutation", "word", "matrix", "rpp", "cylindric"], default="permutation")
    parser.add_argument("--seq-len", type=int, default=None, help="Word length m (defaults to n)")
    parser.add_argument("--vocab-size", type=int, default=None, help="Alphabet size k (defaults to n)")
    parser.add_argument("--a-dim", type=int, default=None, help="Matrix rows a (for --task matrix)")
    parser.add_argument("--b-dim", type=int, default=None, help="Matrix cols b (for --task matrix)")
    parser.add_argument("--total-n", type=int, default=None, help="Total matrix entries |λ| (for --task matrix)")
    parser.add_argument("--source", choices=["hf", "generate", "sample"], default="hf")
    parser.add_argument("--train-size", type=int, default=None, help="Training set size (for --source sample)")
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--shape", type=str, default=None, help="Partition shape for RPP (e.g. 3,2,1)")
    parser.add_argument("--max-entry", type=int, default=None, help="Max filling value for RPP")
    parser.add_argument("--profile", type=str, default=None, help="Binary profile for cylindric (e.g. 010)")
    parser.add_argument("--max-label", type=int, default=None, help="Max ALCD face label for cylindric")
    parser.add_argument("--max-gamma-parts", type=int, default=3, help="Max parts in base partition γ")
    parser.add_argument("--max-gamma-size", type=int, default=4, help="Max part size in base partition γ")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--ablate", type=str, default=None,
        choices=["drop-row", "drop-col", "drop-tab", "drop-row-col", "1d-pos", "concat"],
        help="Embedding ablation mode")
    args = parser.parse_args()

    # Resolve seq_len and vocab_size based on task
    if args.task == "rpp":
        if args.shape is None or args.max_entry is None:
            parser.error("--task rpp requires --shape and --max-entry")
        shape = tuple(int(x) for x in args.shape.split(","))
        seq_len = sum(shape)
        vocab_size = args.max_entry + 1

        model_config = ModelConfig(
            n=seq_len,
            task="rpp",
            shape=shape,
            max_entry=args.max_entry,
            d_model=args.d_model,
            num_layers=args.num_layers,
            nhead=args.nhead,
            ablation=args.ablate,
        )
        train_config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

        source = "sample"
        shape_str = "x".join(str(s) for s in shape)
        ckpt_subdir = f"{args.model}_rpp_{shape_str}_m{args.max_entry}"

        print(f"Task: rpp (shape={shape}, max_entry={args.max_entry}, |λ|={seq_len})")

        train(
            model_name=args.model,
            n=seq_len,
            source=source,
            model_config=model_config,
            train_config=train_config,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            resume=args.resume,
            task="rpp",
            shape=shape,
            max_entry=args.max_entry,
        )
        import sys
        sys.exit(0)

    elif args.task == "cylindric":
        if args.profile is None or args.max_label is None:
            parser.error("--task cylindric requires --profile and --max-label")
        profile = tuple(int(b) for b in args.profile)
        from rsk import _num_alcd_labels
        num_labels = _num_alcd_labels(profile)

        model_config = ModelConfig(
            n=num_labels,
            task="cylindric",
            profile=profile,
            max_label=args.max_label,
            max_gamma_parts=args.max_gamma_parts,
            max_gamma_size=args.max_gamma_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            nhead=args.nhead,
            ablation=args.ablate,
        )
        train_config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

        prof_str = "".join(str(b) for b in profile)
        print(f"Task: cylindric (profile={prof_str}, T={len(profile)}, "
              f"max_label={args.max_label}, num_labels={num_labels})")

        train(
            model_name=args.model,
            n=num_labels,
            source="sample",
            model_config=model_config,
            train_config=train_config,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            resume=args.resume,
            task="cylindric",
            profile=profile,
            max_label=args.max_label,
            max_gamma_parts=args.max_gamma_parts,
            max_gamma_size=args.max_gamma_size,
        )
        import sys
        sys.exit(0)

    elif args.task == "matrix":
        if args.a_dim is None or args.b_dim is None or args.total_n is None:
            parser.error("--task matrix requires --a-dim, --b-dim, and --total-n")
        seq_len = args.total_n
        vocab_size = args.b_dim
    else:
        seq_len = args.seq_len if args.seq_len is not None else args.n
        vocab_size = args.vocab_size

    model_config = ModelConfig(
        n=seq_len,
        task=args.task,
        seq_len=seq_len if args.task == "matrix" else args.seq_len,
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        ablation=args.ablate,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # For word/matrix task, force sample source
    source = args.source
    if args.task in ("word", "matrix", "rpp", "cylindric") and source != "sample":
        print(f"Note: {args.task} task requires --source sample, overriding --source {source}")
        source = "sample"

    train(
        model_name=args.model,
        n=seq_len,
        source=source,
        model_config=model_config,
        train_config=train_config,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        resume=args.resume,
        task=args.task,
        a_dim=args.a_dim,
        b_dim=args.b_dim,
    )
