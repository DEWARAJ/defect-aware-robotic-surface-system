from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "red",
    "green",
    "blue",
    "luminance",
    "gradient_x",
    "gradient_y",
    "local_contrast",
    "local_range",
)


def _local_statistics(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(gray, 1, mode="reflect")
    neighbors = [
        padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
        for dy in range(3)
        for dx in range(3)
    ]
    stack = np.stack(neighbors, axis=0)
    return stack.mean(axis=0), stack.max(axis=0) - stack.min(axis=0)


def extract_features(image: np.ndarray) -> np.ndarray:
    """Return per-pixel features with shape [height, width, feature_count]."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("expected an RGB image with shape [height, width, 3]")
    rgb = image.astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    grad_y[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    local_mean, local_range = _local_statistics(gray)
    return np.stack(
        [
            rgb[..., 0],
            rgb[..., 1],
            rgb[..., 2],
            gray,
            np.abs(grad_x),
            np.abs(grad_y),
            np.abs(gray - local_mean),
            local_range,
        ],
        axis=-1,
    )

