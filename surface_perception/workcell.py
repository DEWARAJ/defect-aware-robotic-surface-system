from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synthetic import make_surface_sample


@dataclass(frozen=True)
class WorkcellScene:
    image: np.ndarray
    workpiece_mask: np.ndarray
    sanding_mask: np.ndarray
    protected_mask: np.ndarray
    defect_mask: np.ndarray
    seed: int


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]


def _shift(mask: np.ndarray, dy: int, dx: int, fill: bool = False) -> np.ndarray:
    height, width = mask.shape
    shifted = np.full(mask.shape, fill, dtype=bool)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    target_y0, target_y1 = max(0, dy), min(height, height + dy)
    target_x0, target_x1 = max(0, dx), min(width, width + dx)
    if source_y0 < source_y1 and source_x0 < source_x1:
        shifted[target_y0:target_y1, target_x0:target_x1] = mask[
            source_y0:source_y1, source_x0:source_x1
        ]
    return shifted


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    result = np.zeros(source.shape, dtype=bool)
    for dy, dx in _disk_offsets(radius):
        result |= _shift(source, dy, dx)
    return result


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    result = np.ones(source.shape, dtype=bool)
    for dy, dx in _disk_offsets(radius):
        result &= _shift(source, dy, dx)
    return result


def _local_coordinates(size: int, angle: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    centered_x = xx - (size - 1) / 2
    centered_y = yy - (size - 1) / 2
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    local_x = centered_x * cos_a + centered_y * sin_a
    local_y = -centered_x * sin_a + centered_y * cos_a
    return local_x, local_y


def make_workcell_scene(size: int = 384, seed: int = 42) -> WorkcellScene:
    """Create a deterministic aerospace-panel-like workcell scene and semantic masks."""
    if size < 96:
        raise ValueError("workcell scene size must be at least 96 pixels")

    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-0.09, 0.09))
    local_x, local_y = _local_coordinates(size, angle)

    half_width = size * 0.43
    half_height = size * 0.34
    workpiece = (np.abs(local_x) / half_width) ** 8 + (
        np.abs(local_y) / half_height
    ) ** 8 <= 1.0

    port = ((local_x + size * 0.22) / (size * 0.075)) ** 2 + (
        (local_y + size * 0.035) / (size * 0.055)
    ) ** 2 <= 1.0
    seam = (
        (np.abs(local_y - size * 0.145) <= max(2.0, size * 0.009))
        & (np.abs(local_x) < size * 0.34)
    )
    protected = port | seam
    rivet_radius = max(3.0, size * 0.014)
    for center_x, center_y in (
        (-0.31, -0.22),
        (0.31, -0.22),
        (-0.31, 0.23),
        (0.31, 0.23),
        (0.02, -0.26),
    ):
        protected |= (local_x - center_x * size) ** 2 + (
            local_y - center_y * size
        ) ** 2 <= rivet_radius**2
    protected &= workpiece

    surface = None
    defect = None
    for attempt in range(16):
        candidate, candidate_mask, _ = make_surface_sample(size, seed + attempt * 997)
        candidate_defect = (candidate_mask > 0) & workpiece & ~protected
        if int(candidate_defect.sum()) >= max(24, int(size * size * 0.0008)):
            surface = candidate
            defect = candidate_defect
            break
    if surface is None or defect is None:
        raise RuntimeError("unable to place a visible defect inside the generated workpiece")

    background_level = rng.integers(15, 25, size=(size, size, 1), dtype=np.uint8)
    background = np.repeat(background_level, 3, axis=2)
    image = background
    image[workpiece] = surface[workpiece]
    image[protected] = np.array([32, 48, 64], dtype=np.uint8)

    panel_edge = workpiece & ~binary_erode(workpiece, max(1, size // 160))
    protected_edge = binary_dilate(protected, max(1, size // 192)) & ~protected
    image[panel_edge] = np.array([188, 208, 216], dtype=np.uint8)
    image[protected_edge & workpiece] = np.array([235, 181, 65], dtype=np.uint8)

    sanding = workpiece & ~protected
    return WorkcellScene(
        image=image,
        workpiece_mask=workpiece,
        sanding_mask=sanding,
        protected_mask=protected,
        defect_mask=defect,
        seed=seed,
    )
