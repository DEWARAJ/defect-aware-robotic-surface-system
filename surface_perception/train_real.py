from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from .metrics import metrics_from_counts
from .real_data import load_manifest
from .torch_data import SegmentationManifestDataset, load_record_arrays
from .torch_model import build_tiny_unet


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _class_balance(manifest_path: Path, image_size: int) -> dict[str, float | int]:
    payload = load_manifest(manifest_path)
    records = [record for record in payload["records"] if record["split"] == "train"]
    positive = 0
    total = 0
    anomalous_samples = 0
    for record in records:
        _, mask = load_record_arrays(record, image_size=image_size)
        positive += int(mask.sum())
        total += int(mask.size)
        anomalous_samples += int(bool(record["anomaly"]))
    if positive == 0:
        raise ValueError(
            "the training split contains no positive defect pixels; use the documented "
            "supervised-development protocol or add annotated custom images"
        )
    negative = total - positive
    return {
        "positive_pixels": positive,
        "negative_pixels": negative,
        "positive_fraction": positive / total,
        "anomalous_samples": anomalous_samples,
        "pos_weight": float(np.clip(negative / positive, 1.0, 30.0)),
    }


def _evaluate_thresholds(model, loader, device, thresholds, torch) -> tuple[float, dict]:
    counts = {
        float(threshold): {key: 0 for key in ("true_positive", "true_negative", "false_positive", "false_negative")}
        for threshold in thresholds
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            targets = batch["mask"].to(device) > 0.5
            probabilities = torch.sigmoid(model(batch["image"].to(device)))
            for threshold in thresholds:
                prediction = probabilities >= float(threshold)
                bucket = counts[float(threshold)]
                bucket["true_positive"] += int(torch.logical_and(prediction, targets).sum().item())
                bucket["true_negative"] += int(torch.logical_and(~prediction, ~targets).sum().item())
                bucket["false_positive"] += int(torch.logical_and(prediction, ~targets).sum().item())
                bucket["false_negative"] += int(torch.logical_and(~prediction, targets).sum().item())
    metrics = {threshold: metrics_from_counts(value) for threshold, value in counts.items()}
    best_threshold = max(metrics, key=lambda threshold: metrics[threshold]["iou"])
    return float(best_threshold), metrics[best_threshold]


def train_real_model(
    manifest_path: Path,
    output_dir: Path,
    *,
    image_size: int = 256,
    base_channels: int = 16,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    patience: int = 7,
    seed: int = 42,
    workers: int = 0,
) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for real-data training. Install the optional ML dependencies."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_dataset = SegmentationManifestDataset(
        manifest_path, "train", image_size=image_size, augment=True, seed=seed
    )
    validation_dataset = SegmentationManifestDataset(
        manifest_path, "validation", image_size=image_size, augment=False, seed=seed
    )
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    balance = _class_balance(manifest_path, image_size)
    model = build_tiny_unet(base_channels=base_channels).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    positive_weight = torch.tensor([balance["pos_weight"]], dtype=torch.float32, device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def loss_function(logits, targets):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return bce(logits, targets) + dice_loss

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    thresholds = np.linspace(0.20, 0.80, 25)
    best_iou = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []
    best_threshold = 0.5

    for epoch in range(1, epochs + 1):
        training_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        samples_seen = 0
        for batch in training_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_function(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.shape[0]
            samples_seen += int(images.shape[0])

        threshold, validation_metrics = _evaluate_thresholds(
            model, validation_loader, device, thresholds, torch
        )
        validation_iou = float(validation_metrics["iou"])
        scheduler.step(validation_iou)
        epoch_report = {
            "epoch": epoch,
            "training_loss": running_loss / max(1, samples_seen),
            "validation": validation_metrics,
            "threshold": threshold,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True))

        if validation_iou > best_iou + 1e-6:
            best_iou = validation_iou
            best_threshold = threshold
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "model_config": {"architecture": "tiny_unet", "base_channels": base_channels},
                    "training_config": {
                        "image_size": image_size,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "seed": seed,
                    },
                    "manifest_sha256": _manifest_hash(manifest_path),
                    "best_validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    report = {
        "model": "tiny_unet",
        "device": str(device),
        "parameter_count": parameter_count,
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _manifest_hash(manifest_path),
        "class_balance": balance,
        "completed_epochs": len(history),
        "requested_epochs": epochs,
        "best_validation_iou": best_iou,
        "selected_threshold": best_threshold,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny U-Net on a real-data manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    train_real_model(
        args.manifest,
        args.output,
        image_size=args.image_size,
        base_channels=args.base_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
