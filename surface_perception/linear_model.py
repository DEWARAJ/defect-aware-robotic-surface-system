from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import FEATURE_NAMES, extract_features


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class FitHistory:
    losses: list[float]
    sampled_pixels: int


class PixelLogisticSegmenter:
    """A deterministic learned baseline for validating the full segmentation pipeline."""

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.threshold: float = 0.5

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None and self.mean is not None and self.scale is not None

    def _balanced_sample(
        self,
        images: Iterable[np.ndarray],
        masks: Iterable[np.ndarray],
        pixels_per_image: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        sampled_x: list[np.ndarray] = []
        sampled_y: list[np.ndarray] = []
        for image, mask in zip(images, masks):
            features = extract_features(image).reshape(-1, len(FEATURE_NAMES))
            labels = (mask.reshape(-1) > 0).astype(np.float32)
            positive = np.flatnonzero(labels == 1)
            negative = np.flatnonzero(labels == 0)
            if not len(positive) or not len(negative):
                continue
            positive_count = min(len(positive), max(1, pixels_per_image // 2))
            negative_count = min(len(negative), pixels_per_image - positive_count)
            selected = np.concatenate(
                [
                    rng.choice(positive, positive_count, replace=False),
                    rng.choice(negative, negative_count, replace=False),
                ]
            )
            rng.shuffle(selected)
            sampled_x.append(features[selected])
            sampled_y.append(labels[selected])
        if not sampled_x:
            raise ValueError("training data did not contain both defect and background pixels")
        return np.concatenate(sampled_x), np.concatenate(sampled_y)

    def fit(
        self,
        images: Iterable[np.ndarray],
        masks: Iterable[np.ndarray],
        *,
        epochs: int = 45,
        learning_rate: float = 0.08,
        batch_size: int = 4096,
        pixels_per_image: int = 1500,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> FitHistory:
        rng = np.random.default_rng(seed)
        x, y = self._balanced_sample(images, masks, pixels_per_image, rng)
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-5)
        x = (x - self.mean) / self.scale
        x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        self.weights = np.zeros(x.shape[1], dtype=np.float64)
        losses: list[float] = []

        for _ in range(epochs):
            order = rng.permutation(x.shape[0])
            epoch_loss = 0.0
            seen = 0
            for start in range(0, x.shape[0], batch_size):
                indices = order[start : start + batch_size]
                xb, yb = x[indices], y[indices]
                probabilities = _sigmoid(xb @ self.weights)
                error = probabilities - yb
                gradient = xb.T @ error / max(1, len(indices))
                gradient[:-1] += l2 * self.weights[:-1]
                self.weights -= learning_rate * gradient
                batch_loss = -np.mean(
                    yb * np.log(probabilities + 1e-8)
                    + (1 - yb) * np.log(1 - probabilities + 1e-8)
                )
                epoch_loss += float(batch_loss) * len(indices)
                seen += len(indices)
            losses.append(epoch_loss / max(1, seen))
        return FitHistory(losses=losses, sampled_pixels=int(x.shape[0]))

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("model must be fitted before prediction")
        assert self.mean is not None and self.scale is not None and self.weights is not None
        features = extract_features(image)
        flat = (features.reshape(-1, len(FEATURE_NAMES)) - self.mean) / self.scale
        flat = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)], axis=1)
        return _sigmoid(flat @ self.weights).reshape(image.shape[:2])

    def predict(self, image: np.ndarray) -> np.ndarray:
        return self.predict_proba(image) >= self.threshold

    def save(self, path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted model")
        assert self.mean is not None and self.scale is not None and self.weights is not None
        payload = {
            "model_type": "pixel_logistic_segmenter",
            "version": 1,
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "threshold": self.threshold,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PixelLogisticSegmenter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "pixel_logistic_segmenter":
            raise ValueError("unsupported model format")
        if tuple(payload.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("model features do not match this package version")
        model = cls()
        model.weights = np.asarray(payload["weights"], dtype=np.float64)
        model.mean = np.asarray(payload["mean"], dtype=np.float64)
        model.scale = np.asarray(payload["scale"], dtype=np.float64)
        model.threshold = float(payload["threshold"])
        return model

