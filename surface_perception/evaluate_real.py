from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import add_counts, confusion_counts, metrics_from_counts
from .torch_data import SegmentationManifestDataset
from .torch_model import load_tiny_unet_checkpoint


def _synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _preview(image_chw, target, probability, threshold: float, output_path: Path) -> None:
    image = np.clip(image_chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    truth = target.astype(bool)
    prediction = probability >= threshold
    overlay = image.astype(np.float32)
    overlay[prediction, 0] = 255
    overlay[prediction, 1:] *= 0.42
    overlay[truth & ~prediction, 2] = 255
    overlay[truth & ~prediction, :2] *= 0.45
    target_rgb = np.repeat((truth[..., None] * 255).astype(np.uint8), 3, axis=2)
    probability_rgb = np.zeros_like(image)
    probability_rgb[..., 0] = np.clip(probability * 255, 0, 255).astype(np.uint8)
    probability_rgb[..., 1] = np.clip((1.0 - np.abs(probability - 0.5) * 2.0) * 180, 0, 255).astype(np.uint8)
    combined = np.concatenate(
        [image, target_rgb, probability_rgb, np.clip(overlay, 0, 255).astype(np.uint8)], axis=1
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(output_path)


def evaluate_real_model(
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    split: str = "test",
    image_size: int | None = None,
    preview_count: int = 12,
) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for real-data evaluation.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_tiny_unet_checkpoint(checkpoint_path, device=device)
    configured_size = int(checkpoint.get("training_config", {}).get("image_size", 256))
    image_size = int(image_size or configured_size)
    threshold = float(checkpoint.get("threshold", 0.5))
    dataset = SegmentationManifestDataset(
        manifest_path, split, image_size=image_size, augment=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    total_counts: dict[str, int] = {}
    grouped_counts: dict[str, dict[str, int]] = defaultdict(dict)
    per_image: list[dict] = []
    latencies: list[float] = []
    preview_candidates: list[tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            image_tensor = batch["image"].to(device)
            _synchronize(torch, device)
            start = time.perf_counter()
            logits = model(image_tensor)
            _synchronize(torch, device)
            latencies.append((time.perf_counter() - start) * 1000.0)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
            target = batch["mask"][0, 0].numpy() > 0.5
            prediction = probability >= threshold
            counts = confusion_counts(prediction, target)
            total_counts = add_counts(total_counts, counts)
            category = str(batch["category"][0])
            defect_type = str(batch["defect_type"][0])
            group_key = f"{category}:{defect_type}"
            grouped_counts[group_key] = add_counts(grouped_counts[group_key], counts)
            metrics = metrics_from_counts(counts)
            anomaly = bool(batch["anomaly"][0].item())
            failure_score = (1.0 - metrics["iou"]) if anomaly else counts["false_positive"] / target.size
            record = {
                "sample_id": str(batch["sample_id"][0]),
                "category": category,
                "defect_type": defect_type,
                "anomaly": anomaly,
                **metrics,
            }
            per_image.append(record)
            preview_candidates.append(
                (
                    failure_score,
                    record,
                    batch["image"][0].numpy(),
                    target,
                    probability,
                )
            )

    output_dir = Path(output_dir)
    previews_dir = output_dir / "failure_gallery"
    for rank, candidate in enumerate(
        sorted(preview_candidates, key=lambda item: item[0], reverse=True)[:preview_count], start=1
    ):
        _, record, image, target, probability = candidate
        _preview(
            image,
            target,
            probability,
            threshold,
            previews_dir / f"{rank:02d}_{record['sample_id']}.png",
        )

    report = {
        "split": split,
        "samples": len(dataset),
        "device": str(device),
        "image_size": image_size,
        "threshold": threshold,
        "aggregate": metrics_from_counts(total_counts),
        "by_category_and_defect": {
            key: metrics_from_counts(counts) for key, counts in sorted(grouped_counts.items())
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "per_image": per_image,
        "gallery_note": "panels are RGB input, ground truth, probability heatmap, and overlay",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a real-data Tiny U-Net checkpoint")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()
    report = evaluate_real_model(
        args.manifest,
        args.checkpoint,
        args.output,
        split=args.split,
        image_size=args.image_size,
        preview_count=args.preview_count,
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
