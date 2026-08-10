from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    split: str
    defect_type: str
    defect_fraction: float
    seed: int


def _line_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    center = np.array([rng.uniform(0, width), rng.uniform(0, height)])
    angle = rng.uniform(0, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    length = rng.uniform(width * 0.35, width * 0.95)
    thickness = rng.uniform(0.7, 2.3)
    rel = np.stack([xx - center[0], yy - center[1]], axis=-1)
    along = rel @ direction
    normal = rel[..., 0] * (-direction[1]) + rel[..., 1] * direction[0]
    return (np.abs(normal) <= thickness) & (np.abs(along) <= length / 2)


def _ellipse_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cx, cy = rng.uniform(0.15 * width, 0.85 * width), rng.uniform(0.15 * height, 0.85 * height)
    rx, ry = rng.uniform(0.05 * width, 0.2 * width), rng.uniform(0.04 * height, 0.16 * height)
    angle = rng.uniform(0, np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x0, y0 = xx - cx, yy - cy
    xr = x0 * cos_a + y0 * sin_a
    yr = -x0 * sin_a + y0 * cos_a
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def _pit_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    mask = np.zeros((height, width), dtype=bool)
    for _ in range(int(rng.integers(4, 13))):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        radius = rng.uniform(1.0, max(2.0, width * 0.035))
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    return mask


def make_surface_sample(size: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Create a deterministic RGB surface image and binary defect mask."""
    if size < 32:
        raise ValueError("image size must be at least 32 pixels")

    rng = np.random.default_rng(seed)
    height = width = size
    yy, xx = np.mgrid[:height, :width]

    base_level = rng.uniform(0.48, 0.72)
    brushed = 0.035 * np.sin(yy * rng.uniform(0.45, 0.9) + rng.uniform(0, np.pi))
    fine_noise = rng.normal(0, 0.025, size=(height, width))
    lighting = rng.uniform(-0.11, 0.11) * (xx / max(1, width - 1) - 0.5)
    gray = np.clip(base_level + brushed + fine_noise + lighting, 0, 1)
    image = np.stack(
        [gray * rng.uniform(0.96, 1.04), gray, gray * rng.uniform(0.9, 1.03)], axis=-1
    )

    defect_type = str(rng.choice(["scratch", "corrosion", "pitting"]))
    if defect_type == "scratch":
        mask = _line_mask(height, width, rng)
        image[mask] *= rng.uniform(0.18, 0.48)
        halo = np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=1)
        image[halo & ~mask] *= 0.82
    elif defect_type == "corrosion":
        mask = np.zeros((height, width), dtype=bool)
        for _ in range(int(rng.integers(1, 4))):
            mask |= _ellipse_mask(height, width, rng)
        mottling = rng.uniform(0.55, 1.0, size=(height, width, 1))
        corrosion_color = np.array([0.46, 0.22, 0.08])[None, None, :]
        image[mask] = 0.55 * image[mask] + 0.45 * (corrosion_color * mottling)[mask]
    else:
        mask = _pit_mask(height, width, rng)
        image[mask] *= rng.uniform(0.12, 0.38)

    sensor_noise = rng.normal(0, 0.01, size=image.shape)
    image = np.clip(image + sensor_noise, 0, 1)
    return (image * 255).astype(np.uint8), (mask.astype(np.uint8) * 255), defect_type


def _split_for(index: int, total: int, train_fraction: float, validation_fraction: float) -> str:
    position = index / max(1, total)
    if position < train_fraction:
        return "train"
    if position < train_fraction + validation_fraction:
        return "validation"
    return "test"


def generate_dataset(
    output_dir: Path,
    samples: int = 120,
    image_size: int = 96,
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> list[SampleMetadata]:
    if samples < 12:
        raise ValueError("at least 12 samples are required for train/validation/test splits")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("training and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("training plus validation fractions must be less than one")

    output_dir = Path(output_dir)
    records: list[SampleMetadata] = []
    for split in ("train", "validation", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    order_rng = np.random.default_rng(seed)
    seeds = order_rng.permutation(np.arange(seed, seed + samples * 13, 13, dtype=np.int64))
    for index, sample_seed in enumerate(seeds):
        split = _split_for(index, samples, train_fraction, validation_fraction)
        image, mask, defect_type = make_surface_sample(image_size, int(sample_seed))
        sample_id = f"surface_{index:04d}"
        Image.fromarray(image).save(output_dir / split / "images" / f"{sample_id}.png")
        Image.fromarray(mask).save(output_dir / split / "masks" / f"{sample_id}.png")
        records.append(
            SampleMetadata(
                sample_id=sample_id,
                split=split,
                defect_type=defect_type,
                defect_fraction=float((mask > 0).mean()),
                seed=int(sample_seed),
            )
        )

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records
