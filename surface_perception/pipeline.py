from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .linear_model import PixelLogisticSegmenter
from .metrics import add_counts, confusion_counts, metrics_from_counts


def load_split(dataset_dir: Path, split: str) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    image_dir = Path(dataset_dir) / split / "images"
    mask_dir = Path(dataset_dir) / split / "masks"
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"no images found for split '{split}' in {image_dir}")
    images, masks, sample_ids = [], [], []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"missing mask for {image_path.name}")
        images.append(np.asarray(Image.open(image_path).convert("RGB")))
        masks.append(np.asarray(Image.open(mask_path).convert("L")))
        sample_ids.append(image_path.stem)
    return images, masks, sample_ids


def _best_threshold(model: PixelLogisticSegmenter, images: list[np.ndarray], masks: list[np.ndarray]) -> float:
    best_threshold, best_iou = 0.5, -1.0
    probabilities = [model.predict_proba(image) for image in images]
    for threshold in np.linspace(0.2, 0.8, 25):
        total: dict[str, int] = {}
        for probability, mask in zip(probabilities, masks):
            total = add_counts(total, confusion_counts(probability >= threshold, mask > 0))
        iou = metrics_from_counts(total)["iou"]
        if iou > best_iou:
            best_iou, best_threshold = iou, float(threshold)
    return best_threshold


def train_baseline(
    dataset_dir: Path,
    model_path: Path,
    report_path: Path,
    training_config: dict,
    seed: int,
) -> dict:
    train_images, train_masks, _ = load_split(dataset_dir, "train")
    validation_images, validation_masks, _ = load_split(dataset_dir, "validation")
    model = PixelLogisticSegmenter()
    history = model.fit(train_images, train_masks, seed=seed, **training_config)
    model.threshold = _best_threshold(model, validation_images, validation_masks)
    model.save(model_path)
    report = {
        "model": "pixel_logistic_segmenter",
        "seed": seed,
        "training_samples": len(train_images),
        "validation_samples": len(validation_images),
        "sampled_pixels": history.sampled_pixels,
        "epochs": len(history.losses),
        "initial_loss": history.losses[0],
        "final_loss": history.losses[-1],
        "selected_threshold": model.threshold,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _save_preview(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    overlay = image.astype(np.float32).copy()
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    overlay[pred, 0] = 255
    overlay[pred, 1:] *= 0.45
    overlay[truth & ~pred, 2] = 255
    overlay[truth & ~pred, :2] *= 0.55
    target_rgb = np.repeat((truth[..., None] * 255).astype(np.uint8), 3, axis=-1)
    combined = np.concatenate([image, target_rgb, np.clip(overlay, 0, 255).astype(np.uint8)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(output_path)


def evaluate_model(
    dataset_dir: Path,
    model_path: Path,
    output_dir: Path,
    split: str = "test",
    preview_count: int = 8,
) -> dict:
    images, masks, sample_ids = load_split(dataset_dir, split)
    model = PixelLogisticSegmenter.load(model_path)
    total_counts: dict[str, int] = {}
    per_image: list[dict] = []
    latencies_ms: list[float] = []
    for index, (image, mask, sample_id) in enumerate(zip(images, masks, sample_ids)):
        start = time.perf_counter()
        probability = model.predict_proba(image)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        prediction = probability >= model.threshold
        counts = confusion_counts(prediction, mask > 0)
        total_counts = add_counts(total_counts, counts)
        per_image.append({"sample_id": sample_id, **metrics_from_counts(counts)})
        if index < preview_count:
            _save_preview(image, mask > 0, prediction, Path(output_dir) / "previews" / f"{sample_id}.png")

    report = {
        "split": split,
        "samples": len(images),
        "threshold": model.threshold,
        "aggregate": metrics_from_counts(total_counts),
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
        },
        "per_image": per_image,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
