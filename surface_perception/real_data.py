from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SegmentationRecord:
    sample_id: str
    image_path: str
    mask_path: str | None
    split: str
    source: str
    category: str
    defect_type: str
    anomaly: bool


def _image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _stable_order(paths: Iterable[Path], seed: int, namespace: str) -> list[Path]:
    def key(path: Path) -> str:
        value = f"{seed}:{namespace}:{path.as_posix()}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    return sorted(paths, key=key)


def _three_way_split(paths: list[Path], seed: int, namespace: str) -> dict[Path, str]:
    ordered = _stable_order(paths, seed, namespace)
    count = len(ordered)
    if count == 0:
        return {}
    if count == 1:
        return {ordered[0]: "test"}
    if count == 2:
        return {ordered[0]: "train", ordered[1]: "test"}

    train_count = max(1, int(round(count * 0.60)))
    validation_count = max(1, int(round(count * 0.20)))
    if train_count + validation_count >= count:
        train_count = max(1, count - 2)
        validation_count = 1
    result: dict[Path, str] = {}
    for index, path in enumerate(ordered):
        if index < train_count:
            result[path] = "train"
        elif index < train_count + validation_count:
            result[path] = "validation"
        else:
            result[path] = "test"
    return result


def _train_validation_split(paths: list[Path], seed: int, namespace: str) -> dict[Path, str]:
    ordered = _stable_order(paths, seed, namespace)
    if len(ordered) < 2:
        return {path: "train" for path in ordered}
    validation_count = max(1, int(round(len(ordered) * 0.15)))
    validation = set(ordered[:validation_count])
    return {path: "validation" if path in validation else "train" for path in ordered}


def _record(
    image_path: Path,
    mask_path: Path | None,
    split: str,
    source: str,
    category: str,
    defect_type: str,
    anomaly: bool,
) -> SegmentationRecord:
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return SegmentationRecord(
        sample_id=f"{source}_{category}_{defect_type}_{image_path.stem}_{digest}",
        image_path=str(image_path.resolve()),
        mask_path=str(mask_path.resolve()) if mask_path is not None else None,
        split=split,
        source=source,
        category=category,
        defect_type=defect_type,
        anomaly=anomaly,
    )


def _validate_pair(image_path: Path, mask_path: Path) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(
                f"image/mask size mismatch for {image_path.name}: {image.size} != {mask.size}"
            )
        mask_array = np.asarray(mask.convert("L"))
        if not np.any(mask_array > 0):
            raise ValueError(f"anomaly mask is empty: {mask_path}")


def _mvtec_records(root: Path, categories: list[str] | None, seed: int) -> list[SegmentationRecord]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MVTec root does not exist: {root}")
    discovered = sorted(
        path.name for path in root.iterdir() if path.is_dir() and (path / "train" / "good").exists()
    )
    selected = categories or discovered
    if not selected:
        raise ValueError(f"no MVTec categories were found under {root}")

    records: list[SegmentationRecord] = []
    for category in selected:
        category_root = root / category
        if not (category_root / "train" / "good").exists():
            raise FileNotFoundError(f"invalid MVTec category structure: {category_root}")

        good_training = _image_files(category_root / "train" / "good")
        good_splits = _train_validation_split(good_training, seed, f"mvtec:{category}:good")
        for image_path, split in good_splits.items():
            records.append(_record(image_path, None, split, "mvtec_ad", category, "good", False))

        for image_path in _image_files(category_root / "test" / "good"):
            records.append(_record(image_path, None, "test", "mvtec_ad", category, "good", False))

        test_root = category_root / "test"
        defect_directories = sorted(
            path for path in test_root.iterdir() if path.is_dir() and path.name != "good"
        )
        for defect_directory in defect_directories:
            defect_type = defect_directory.name
            images = _image_files(defect_directory)
            splits = _three_way_split(images, seed, f"mvtec:{category}:{defect_type}")
            for image_path, split in splits.items():
                mask_path = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"missing MVTec mask for {image_path}: {mask_path}")
                _validate_pair(image_path, mask_path)
                records.append(
                    _record(
                        image_path,
                        mask_path,
                        split,
                        "mvtec_ad",
                        category,
                        defect_type,
                        True,
                    )
                )
    return records


def _custom_records(root: Path, seed: int) -> list[SegmentationRecord]:
    root = Path(root)
    image_root = root / "images"
    mask_root = root / "masks"
    images = _image_files(image_root)
    if not images:
        raise ValueError(f"no custom images found under {image_root}")
    splits = _three_way_split(images, seed, "custom")
    records: list[SegmentationRecord] = []
    for image_path, split in splits.items():
        candidates = [
            mask_root / f"{image_path.stem}{suffix}" for suffix in sorted(IMAGE_EXTENSIONS)
        ]
        mask_path = next((path for path in candidates if path.exists()), None)
        if mask_path is None:
            raise FileNotFoundError(f"missing custom mask for {image_path.name} in {mask_root}")
        _validate_pair(image_path, mask_path)
        records.append(
            _record(image_path, mask_path, split, "custom", root.name, "annotated_defect", True)
        )
    return records


def _dataset_report(records: list[SegmentationRecord], protocol: str) -> dict:
    split_counts = Counter(record.split for record in records)
    anomaly_counts = Counter(f"{record.split}:{'anomaly' if record.anomaly else 'good'}" for record in records)
    category_counts = Counter(f"{record.source}:{record.category}" for record in records)
    return {
        "protocol": protocol,
        "records": len(records),
        "splits": dict(sorted(split_counts.items())),
        "split_labels": dict(sorted(anomaly_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "license_notice": {
            "mvtec_ad": "CC BY-NC-SA 4.0; research/non-commercial use only",
            "custom": "user-provided; document provenance and permissions separately",
        },
        "benchmark_warning": (
            "supervised-development reallocates MVTec AD anomaly test images across development "
            "splits and is not the official unsupervised MVTec AD benchmark protocol"
        ),
    }


def build_segmentation_manifest(
    output_path: Path,
    *,
    mvtec_root: Path | None = None,
    custom_root: Path | None = None,
    categories: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Build an explicit supervised-development manifest from MVTec AD and/or custom pairs."""
    records: list[SegmentationRecord] = []
    if mvtec_root is not None:
        records.extend(_mvtec_records(Path(mvtec_root), categories, seed))
    if custom_root is not None:
        records.extend(_custom_records(Path(custom_root), seed))
    if not records:
        raise ValueError("provide at least one MVTec AD root or custom dataset root")

    records.sort(key=lambda record: (record.split, record.source, record.category, record.sample_id))
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest contains duplicate sample identifiers")
    image_splits: dict[str, str] = {}
    for record in records:
        previous = image_splits.setdefault(record.image_path, record.split)
        if previous != record.split:
            raise ValueError(f"data leakage: {record.image_path} appears in multiple splits")

    report = _dataset_report(records, "supervised-development")
    payload = {
        "schema_version": "1.0",
        "seed": seed,
        "protocol": "supervised-development",
        "report": report,
        "records": [asdict(record) for record in records],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_manifest(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or not isinstance(payload.get("records"), list):
        raise ValueError(f"unsupported segmentation manifest: {path}")
    return payload
