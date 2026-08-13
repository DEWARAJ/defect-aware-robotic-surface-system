from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SIM_SCHEMA_VERSION = "1.0"
OUTPUT_SCHEMA_VERSION = "isaac-surface-1.0"

MODALITY_ALIASES = {
    "rgb": ("rgb",),
    "semantic_segmentation": ("semantic_segmentation",),
    "distance_to_image_plane": ("distance_to_image_plane", "depth"),
    "normals": ("normals",),
    "camera_params": ("camera_params",),
}

MODALITY_EXTENSIONS = {
    "rgb": {".png", ".jpg", ".jpeg"},
    "semantic_segmentation": {".png", ".npy"},
    "distance_to_image_plane": {".npy", ".npz", ".exr"},
    "normals": {".png", ".npy", ".exr"},
    "camera_params": {".json"},
}


def _number_range(value: Any, name: str, *, minimum: float | None = None) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value list")
    result = [float(value[0]), float(value[1])]
    if result[0] > result[1]:
        raise ValueError(f"{name} minimum exceeds maximum")
    if minimum is not None and result[0] < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive_triplet(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    result = [float(item) for item in value]
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _color_triplet(value: Any, name: str) -> list[float]:
    result = _positive_triplet(value, name)
    if any(item > 1.0 for item in result):
        raise ValueError(f"{name} values cannot exceed 1.0")
    return result


def validate_sim_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the Isaac Sim surface-generation configuration."""
    normalized = copy.deepcopy(config)
    if normalized.get("schema_version") != SIM_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SIM_SCHEMA_VERSION}")
    if normalized.get("renderer") not in {"RayTracedLighting", "PathTracing"}:
        raise ValueError("renderer must be RayTracedLighting or PathTracing")

    seed = normalized.get("seed")
    frames = normalized.get("frames")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(frames, int) or frames < 3:
        raise ValueError("frames must be an integer of at least 3")

    resolution = normalized.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(not isinstance(value, int) or value < 64 for value in resolution)
    ):
        raise ValueError("resolution must contain two integer values of at least 64")

    panel = normalized.get("panel", {})
    panel["size_m"] = _positive_triplet(panel.get("size_m"), "panel.size_m")
    panel["roughness"] = _number_range(
        panel.get("roughness"), "panel.roughness", minimum=0.0
    )
    panel["metallic"] = _number_range(
        panel.get("metallic"), "panel.metallic", minimum=0.0
    )
    for name in ("roughness", "metallic"):
        if panel[name][1] > 1.0:
            raise ValueError(f"panel.{name} values cannot exceed 1.0")
    panel["diffuse_min"] = _color_triplet(panel.get("diffuse_min"), "panel.diffuse_min")
    panel["diffuse_max"] = _color_triplet(panel.get("diffuse_max"), "panel.diffuse_max")
    if any(low > high for low, high in zip(panel["diffuse_min"], panel["diffuse_max"])):
        raise ValueError("panel.diffuse_min cannot exceed panel.diffuse_max")
    normalized["panel"] = panel

    camera = normalized.get("camera", {})
    for axis in ("x_m", "y_m", "z_m"):
        camera[axis] = _number_range(camera.get(axis), f"camera.{axis}")
    camera["focal_length_mm"] = _number_range(
        camera.get("focal_length_mm"), "camera.focal_length_mm", minimum=1.0
    )
    if camera["z_m"][0] <= panel["size_m"][2] / 2:
        raise ValueError("camera.z_m must remain above the panel surface")
    normalized["camera"] = camera

    lighting = normalized.get("lighting", {})
    lighting["count"] = int(lighting.get("count", 0))
    if lighting["count"] < 1:
        raise ValueError("lighting.count must be at least 1")
    lighting["intensity"] = _number_range(
        lighting.get("intensity"), "lighting.intensity", minimum=0.0
    )
    lighting["color_temperature_k"] = _number_range(
        lighting.get("color_temperature_k"),
        "lighting.color_temperature_k",
        minimum=1000.0,
    )
    normalized["lighting"] = lighting

    defects = normalized.get("defects", {})
    required_defects = {"scratch", "corrosion", "pit"}
    if set(defects) != required_defects:
        raise ValueError(f"defects must contain exactly {sorted(required_defects)}")
    for defect_name, settings in defects.items():
        count = settings.get("count")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"defects.{defect_name}.count must be a positive integer")
        settings["length_m"] = _number_range(
            settings.get("length_m"), f"defects.{defect_name}.length_m", minimum=0.001
        )
        settings["width_m"] = _number_range(
            settings.get("width_m"), f"defects.{defect_name}.width_m", minimum=0.001
        )
    normalized["defects"] = defects

    protected = normalized.get("protected_objects", {})
    if not isinstance(protected.get("fastener_count"), int) or protected["fastener_count"] < 1:
        raise ValueError("protected_objects.fastener_count must be a positive integer")
    if not isinstance(protected.get("seam_count"), int) or protected["seam_count"] < 1:
        raise ValueError("protected_objects.seam_count must be a positive integer")

    outputs = normalized.get("outputs", {})
    unknown_outputs = set(outputs) - set(MODALITY_ALIASES)
    if unknown_outputs:
        raise ValueError(f"unknown output modalities: {sorted(unknown_outputs)}")
    for modality in MODALITY_ALIASES:
        if not isinstance(outputs.get(modality), bool):
            raise ValueError(f"outputs.{modality} must be true or false")
    if not outputs["rgb"] or not outputs["semantic_segmentation"]:
        raise ValueError("RGB and semantic segmentation outputs are required")

    splits = normalized.get("splits", {})
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("splits must contain train, validation, and test")
    split_total = 0.0
    for name, fraction in splits.items():
        if not isinstance(fraction, (int, float)) or float(fraction) <= 0:
            raise ValueError(f"splits.{name} must be positive")
        splits[name] = float(fraction)
        split_total += splits[name]
    if abs(split_total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    classes = normalized.get("semantic_classes", {})
    required_classes = {
        "panel",
        "defect_scratch",
        "defect_corrosion",
        "defect_pit",
        "protected_fastener",
        "protected_seam",
    }
    if set(classes) != required_classes:
        raise ValueError(f"semantic_classes must contain exactly {sorted(required_classes)}")
    class_ids = list(classes.values())
    if any(not isinstance(value, int) or value < 0 for value in class_ids):
        raise ValueError("semantic class IDs must be non-negative integers")
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("semantic class IDs must be unique")
    return normalized


def load_sim_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_sim_config(config)


def _ordered_frame_indices(frame_count: int, seed: int) -> list[int]:
    def key(index: int) -> str:
        return hashlib.sha256(f"{seed}:isaac-surface:{index}".encode("utf-8")).hexdigest()

    return sorted(range(frame_count), key=key)


def planned_frame_splits(config: dict[str, Any]) -> dict[int, str]:
    normalized = validate_sim_config(config)
    frame_count = int(normalized["frames"])
    ordered = _ordered_frame_indices(frame_count, int(normalized["seed"]))
    train_count = max(1, int(round(frame_count * normalized["splits"]["train"])))
    validation_count = max(
        1, int(round(frame_count * normalized["splits"]["validation"]))
    )
    if train_count + validation_count >= frame_count:
        train_count = frame_count - 2
        validation_count = 1
    split_by_frame: dict[int, str] = {}
    for order, index in enumerate(ordered):
        if order < train_count:
            split_by_frame[index] = "train"
        elif order < train_count + validation_count:
            split_by_frame[index] = "validation"
        else:
            split_by_frame[index] = "test"
    return split_by_frame


def build_sim_run_plan(config: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_sim_config(config)
    splits = planned_frame_splits(normalized)
    modalities = [name for name, enabled in normalized["outputs"].items() if enabled]
    frames = [
        {
            "frame_index": index,
            "sample_id": f"isaac_surface_{index:06d}",
            "split": splits[index],
        }
        for index in range(int(normalized["frames"]))
    ]
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "generator": "isaac_sim_replicator",
        "status": "planned",
        "seed": normalized["seed"],
        "resolution": normalized["resolution"],
        "modalities": modalities,
        "semantic_classes": normalized["semantic_classes"],
        "defect_classes": [
            "defect_scratch",
            "defect_corrosion",
            "defect_pit",
        ],
        "protected_classes": ["protected_fastener", "protected_seam"],
        "split_counts": dict(sorted(Counter(splits.values()).items())),
        "frames": frames,
    }


def save_sim_run_plan(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    plan = build_sim_run_plan(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan


def _frame_index(path: Path, aliases: tuple[str, ...]) -> int | None:
    stem = path.stem.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        match = re.search(rf"(?:^|_){re.escape(alias)}_(\d+)(?:_|$)", stem)
        if match:
            return int(match.group(1))
    return None


def discover_replicator_outputs(root: Path) -> dict[str, dict[int, Path]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Replicator output directory does not exist: {root}")
    discovered: dict[str, dict[int, Path]] = {name: {} for name in MODALITY_ALIASES}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        for modality, aliases in MODALITY_ALIASES.items():
            if path.suffix.lower() not in MODALITY_EXTENSIONS[modality]:
                continue
            index = _frame_index(path, aliases)
            if index is None:
                continue
            existing = discovered[modality].get(index)
            if existing is not None:
                raise ValueError(
                    f"duplicate {modality} output for frame {index}: {existing} and {path}"
                )
            discovered[modality][index] = path
            break
    return discovered


def _validate_raster_size(path: Path, resolution: list[int], modality: str) -> None:
    suffix = path.suffix.lower()
    actual_size: list[int] | None = None
    if suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            actual_size = list(image.size)
    elif suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        if array.ndim < 2:
            raise ValueError(f"{modality} array must have at least two dimensions: {path.name}")
        actual_size = [int(array.shape[1]), int(array.shape[0])]
    elif suffix == ".npz":
        with np.load(path) as archive:
            if not archive.files:
                raise ValueError(f"{modality} archive is empty: {path.name}")
            array = archive[archive.files[0]]
            if array.ndim < 2:
                raise ValueError(
                    f"{modality} array must have at least two dimensions: {path.name}"
                )
            actual_size = [int(array.shape[1]), int(array.shape[0])]
    if actual_size is not None and actual_size != resolution:
        raise ValueError(
            f"{modality} size mismatch for {path.name}: "
            f"{tuple(actual_size)} != {tuple(resolution)}"
        )


def validate_replicator_dataset(
    output_root: Path,
    config: dict[str, Any],
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    normalized = validate_sim_config(config)
    root = Path(output_root)
    discovered = discover_replicator_outputs(root)
    expected = set(range(int(normalized["frames"])))
    required = [name for name, enabled in normalized["outputs"].items() if enabled]
    missing: dict[str, list[int]] = {}
    extras: dict[str, list[int]] = {}
    for modality in required:
        actual = set(discovered[modality])
        if expected - actual:
            missing[modality] = sorted(expected - actual)
        if actual - expected:
            extras[modality] = sorted(actual - expected)
    if missing or extras:
        raise ValueError(
            "Replicator output is incomplete: "
            f"missing={{{', '.join(f'{key}: {len(value)}' for key, value in missing.items())}}}, "
            f"extras={{{', '.join(f'{key}: {len(value)}' for key, value in extras.items())}}}"
        )

    splits = planned_frame_splits(normalized)
    records = []
    for index in range(int(normalized["frames"])):
        paths: dict[str, str] = {}
        for modality in required:
            path = discovered[modality][index]
            _validate_raster_size(path, normalized["resolution"], modality)
            paths[modality] = str(path.resolve())
        records.append(
            {
                "sample_id": f"isaac_surface_{index:06d}",
                "frame_index": index,
                "split": splits[index],
                "paths": paths,
            }
        )

    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "generator": "isaac_sim_replicator",
        "status": "validated",
        "seed": normalized["seed"],
        "resolution": normalized["resolution"],
        "records": records,
        "report": {
            "frames": len(records),
            "modalities": required,
            "split_counts": dict(sorted(Counter(record["split"] for record in records).items())),
            "missing_outputs": 0,
            "extra_outputs": 0,
        },
        "semantic_classes": normalized["semantic_classes"],
        "defect_classes": ["defect_scratch", "defect_corrosion", "defect_pit"],
        "protected_classes": ["protected_fastener", "protected_seam"],
    }
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def render_sim_plan_preview(config: dict[str, Any], output_path: Path) -> Path:
    """Render a deterministic planning schematic; this is not an Isaac Sim sensor image."""
    normalized = validate_sim_config(config)
    width, height = 1400, 800
    image = Image.new("RGB", (width, height), (7, 13, 24))
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(int(normalized["seed"]))

    draw.text((55, 38), "ISAAC SIM SURFACE INTELLIGENCE / v0.5", fill=(111, 225, 255))
    draw.text(
        (55, 70),
        "PRE-CAPTURE CONFIGURATION PREVIEW - NOT AN ISAAC SIM SENSOR FRAME",
        fill=(145, 157, 180),
    )

    panel_box = (70, 145, 930, 705)
    draw.rounded_rectangle(
        panel_box,
        radius=50,
        fill=(92, 104, 116),
        outline=(176, 197, 212),
        width=4,
    )
    for x in range(105, 900, 26):
        draw.line((x, 165, x - 70, 682), fill=(108, 121, 132), width=2)

    for _ in range(normalized["defects"]["scratch"]["count"] * 3):
        x = int(rng.integers(160, 820))
        y = int(rng.integers(225, 620))
        length = int(rng.integers(35, 110))
        draw.line((x, y, x + length, y + int(rng.integers(-30, 31))), fill=(255, 92, 115), width=5)
    for _ in range(normalized["defects"]["corrosion"]["count"] * 3):
        x = int(rng.integers(170, 850))
        y = int(rng.integers(220, 620))
        radius = int(rng.integers(12, 35))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(236, 154, 75))
    for _ in range(normalized["defects"]["pit"]["count"] * 2):
        x = int(rng.integers(160, 850))
        y = int(rng.integers(215, 630))
        radius = int(rng.integers(5, 12))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(34, 42, 50))

    for index in range(normalized["protected_objects"]["fastener_count"]):
        angle = 2 * np.pi * index / normalized["protected_objects"]["fastener_count"]
        x = int(500 + 360 * np.cos(angle))
        y = int(425 + 225 * np.sin(angle))
        draw.ellipse(
            (x - 12, y - 12, x + 12, y + 12),
            fill=(89, 224, 194),
            outline=(214, 255, 246),
            width=2,
        )
    draw.line((470, 150, 470, 700), fill=(89, 224, 194), width=8)

    camera = [(490, 105), (400, 145), (580, 145)]
    draw.polygon(camera, outline=(255, 207, 92), fill=(64, 52, 31))
    draw.line((490, 145, 275, 395), fill=(255, 207, 92), width=2)
    draw.line((490, 145, 705, 395), fill=(255, 207, 92), width=2)

    card_x = 990
    draw.rounded_rectangle(
        (card_x, 145, 1345, 705),
        radius=30,
        fill=(13, 25, 43),
        outline=(42, 68, 91),
        width=3,
    )
    draw.text((card_x + 30, 175), "CAPTURE CONTRACT", fill=(111, 225, 255))
    lines = [
        f"Frames           {normalized['frames']}",
        f"Resolution       {normalized['resolution'][0]} x {normalized['resolution'][1]}",
        f"Seed             {normalized['seed']}",
        "",
        "OUTPUTS",
        "RGB",
        "Semantic labels",
        "Metric depth",
        "Surface normals",
        "Camera parameters",
        "",
        "RANDOMIZATION",
        "Camera orbit / focal length",
        "Light pose / intensity / CCT",
        "Metal roughness / reflectance",
        "Defect scale / pose / density",
    ]
    y = 225
    for line in lines:
        color = (214, 223, 235) if line not in {"OUTPUTS", "RANDOMIZATION"} else (255, 207, 92)
        draw.text((card_x + 30, y), line, fill=color)
        y += 29

    draw.text(
        (72, 735),
        "red: scratch  /  amber: corrosion  /  dark: pit  /  cyan: protected",
        fill=(174, 190, 210),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
