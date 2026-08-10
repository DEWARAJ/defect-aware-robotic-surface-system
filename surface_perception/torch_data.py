from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from .real_data import load_manifest


def _sample_seed(seed: int, epoch: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def augment_pair(
    image: np.ndarray, mask: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply deterministic paired geometry and photometric augmentation."""
    rng = np.random.default_rng(seed)
    image_array = np.asarray(image, dtype=np.uint8)
    mask_array = np.asarray(mask, dtype=np.uint8)
    if rng.random() < 0.5:
        image_array = np.flip(image_array, axis=1)
        mask_array = np.flip(mask_array, axis=1)
    if rng.random() < 0.25:
        image_array = np.flip(image_array, axis=0)
        mask_array = np.flip(mask_array, axis=0)
    rotations = int(rng.integers(0, 4))
    if rotations:
        image_array = np.rot90(image_array, rotations)
        mask_array = np.rot90(mask_array, rotations)

    pil_image = Image.fromarray(np.ascontiguousarray(image_array))
    pil_image = ImageEnhance.Brightness(pil_image).enhance(float(rng.uniform(0.78, 1.22)))
    pil_image = ImageEnhance.Contrast(pil_image).enhance(float(rng.uniform(0.82, 1.20)))
    image_array = np.asarray(pil_image, dtype=np.float32)
    noise_sigma = float(rng.uniform(0.0, 7.0))
    image_array = np.clip(
        image_array + rng.normal(0.0, noise_sigma, size=image_array.shape), 0, 255
    ).astype(np.uint8)
    return np.ascontiguousarray(image_array), np.ascontiguousarray(mask_array)


def load_record_arrays(
    record: dict,
    *,
    image_size: int,
    augment: bool = False,
    seed: int = 42,
    epoch: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(Path(record["image_path"])).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if record.get("mask_path"):
        mask = Image.open(Path(record["mask_path"])).convert("L")
        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        mask_array = (np.asarray(mask) > 0).astype(np.uint8)
    else:
        mask_array = np.zeros((image_size, image_size), dtype=np.uint8)
    image_array = np.asarray(image, dtype=np.uint8)
    if augment:
        image_array, mask_array = augment_pair(
            image_array,
            mask_array,
            seed=_sample_seed(seed, epoch, str(record["sample_id"])),
        )
    image_chw = np.ascontiguousarray(image_array.transpose(2, 0, 1), dtype=np.float32) / 255.0
    mask_chw = np.ascontiguousarray(mask_array[None, ...], dtype=np.float32)
    return image_chw, mask_chw


class SegmentationManifestDataset:
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        *,
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        payload = load_manifest(manifest_path)
        self.records = [record for record in payload["records"] if record["split"] == split]
        if not self.records:
            raise ValueError(f"manifest contains no records for split '{split}'")
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the optional ML dependencies to load torch tensors.") from exc
        record = self.records[index]
        image, mask = load_record_arrays(
            record,
            image_size=self.image_size,
            augment=self.augment,
            seed=self.seed,
            epoch=self.epoch,
        )
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "sample_id": record["sample_id"],
            "category": record["category"],
            "defect_type": record["defect_type"],
            "anomaly": bool(record["anomaly"]),
            "image_path": record["image_path"],
        }
