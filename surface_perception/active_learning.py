from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .metrics import confusion_counts, metrics_from_counts
from .real_data import load_manifest
from .torch_data import load_record_arrays
from .torch_model import load_segmentation_checkpoint


EPSILON = 1e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def probability_to_logit(probability: float) -> float:
    value = float(np.clip(probability, EPSILON, 1.0 - EPSILON))
    return float(math.log(value / (1.0 - value)))


def calibrated_threshold(original_threshold: float, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return float(sigmoid(np.asarray(probability_to_logit(original_threshold) / temperature)))


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    entropy = -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))
    return entropy / math.log(2.0)


def _binary_nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    truth = np.asarray(targets, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, scaled) - truth * scaled))


def _sample_pixels(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat_logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    flat_targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if maximum <= 0:
        raise ValueError("maximum calibration pixels must be positive")
    if flat_logits.size <= maximum:
        return flat_logits, flat_targets
    indexes = np.random.default_rng(seed).choice(flat_logits.size, size=maximum, replace=False)
    return flat_logits[indexes], flat_targets[indexes]


def fit_temperature(
    validation_logits: np.ndarray,
    validation_targets: np.ndarray,
    *,
    maximum_pixels: int = 250_000,
    seed: int = 42,
) -> dict:
    """Fit one positive temperature on validation NLL using deterministic golden search."""
    logits, targets = _sample_pixels(
        validation_logits,
        validation_targets,
        maximum=maximum_pixels,
        seed=seed,
    )
    lower = math.log(0.05)
    upper = math.log(20.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_loss = _binary_nll(logits, targets, math.exp(left))
    right_loss = _binary_nll(logits, targets, math.exp(right))
    for _ in range(64):
        if left_loss <= right_loss:
            upper = right
            right = left
            right_loss = left_loss
            left = upper - ratio * (upper - lower)
            left_loss = _binary_nll(logits, targets, math.exp(left))
        else:
            lower = left
            left = right
            left_loss = right_loss
            right = lower + ratio * (upper - lower)
            right_loss = _binary_nll(logits, targets, math.exp(right))
    temperature = math.exp((lower + upper) / 2.0)
    return {
        "temperature": float(temperature),
        "objective": "unweighted binary pixel NLL",
        "fit_split": "validation",
        "sampled_pixels": int(logits.size),
        "positive_fraction": float(targets.mean()),
        "nll_before": _binary_nll(logits, targets, 1.0),
        "nll_after": _binary_nll(logits, targets, temperature),
        "search_bounds": [0.05, 20.0],
        "bound_hit": bool(temperature <= 0.051 or temperature >= 19.9),
    }


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
) -> dict:
    if bins < 2:
        raise ValueError("ECE bins must be at least 2")
    values = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0, 1.0)
    truth = np.asarray(targets, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indexes = np.minimum(np.searchsorted(edges, values, side="right") - 1, bins - 1)
    indexes = np.maximum(indexes, 0)
    ece = 0.0
    maximum_gap = 0.0
    records = []
    for index in range(bins):
        selected = indexes == index
        count = int(selected.sum())
        if count:
            confidence = float(values[selected].mean())
            accuracy = float(truth[selected].mean())
            gap = abs(confidence - accuracy)
            ece += gap * count / max(1, values.size)
            maximum_gap = max(maximum_gap, gap)
        else:
            confidence = 0.0
            accuracy = 0.0
            gap = 0.0
        records.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": confidence,
                "positive_fraction": accuracy,
                "absolute_gap": float(gap),
            }
        )
    return {
        "ece": float(ece),
        "maximum_calibration_gap": float(maximum_gap),
        "bins": records,
    }


def calibration_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    temperature: float,
    maximum_pixels: int,
    bins: int,
    seed: int,
) -> dict:
    sampled_logits, sampled_targets = _sample_pixels(
        logits,
        targets,
        maximum=maximum_pixels,
        seed=seed,
    )
    probabilities = sigmoid(sampled_logits / temperature)
    ece = expected_calibration_error(probabilities, sampled_targets, bins=bins)
    positives = sampled_targets > 0.5
    negatives = ~positives

    def group_average(values: np.ndarray, selected: np.ndarray) -> float:
        return float(values[selected].mean()) if np.any(selected) else 0.0

    per_pixel_nll = (
        np.logaddexp(0.0, sampled_logits / temperature)
        - sampled_targets * sampled_logits / temperature
    )
    squared_error = (probabilities - sampled_targets) ** 2
    return {
        "temperature": float(temperature),
        "pixels": int(sampled_logits.size),
        "positive_fraction": float(sampled_targets.mean()),
        "nll": float(per_pixel_nll.mean()),
        "brier": float(squared_error.mean()),
        "class_balanced_nll": float(
            0.5
            * (
                group_average(per_pixel_nll, positives)
                + group_average(per_pixel_nll, negatives)
            )
        ),
        "class_balanced_brier": float(
            0.5
            * (
                group_average(squared_error, positives)
                + group_average(squared_error, negatives)
            )
        ),
        **ece,
    }


def image_feature_vector(image_chw: np.ndarray) -> np.ndarray:
    image = np.asarray(image_chw, dtype=np.float64).transpose(1, 2, 0)
    means = image.mean(axis=(0, 1))
    standard_deviations = image.std(axis=(0, 1))
    quantiles = np.quantile(image.reshape(-1, 3), [0.10, 0.50, 0.90], axis=0).reshape(-1)
    gray = image.mean(axis=2)
    histogram, _ = np.histogram(gray, bins=8, range=(0.0, 1.0), density=False)
    histogram = histogram.astype(np.float64) / max(1, gray.size)
    gradient_y, gradient_x = np.gradient(gray)
    gradient = np.hypot(gradient_x, gradient_y)
    gradient_features = np.asarray(
        [gradient.mean(), gradient.std(), np.quantile(gradient, 0.90)], dtype=np.float64
    )
    return np.concatenate(
        [means, standard_deviations, quantiles, histogram, gradient_features]
    )


def sample_observable_features(
    image_chw: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Summarize label-free image and model-output signals for sample-level risk."""
    values = np.asarray(probability, dtype=np.float64)
    entropy = binary_entropy(values)
    prediction_features = np.asarray(
        [
            values.mean(),
            values.std(),
            values.max(),
            *np.quantile(values, [0.50, 0.90, 0.95, 0.99]),
            np.mean(values >= threshold),
            entropy.mean(),
            np.quantile(entropy, 0.95),
        ],
        dtype=np.float64,
    )
    return np.concatenate([image_feature_vector(image_chw), prediction_features])


def predict_knn_failure_risk(
    validation_features: np.ndarray,
    validation_failures: np.ndarray,
    query_features: np.ndarray,
    *,
    neighbors: int,
    leave_one_out: bool = False,
) -> np.ndarray:
    """Predict failure severity using only labeled validation samples as neighbors."""
    train = np.asarray(validation_features, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    failures = np.asarray(validation_failures, dtype=np.float64)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("validation and query features must be compatible matrices")
    if failures.shape != (train.shape[0],):
        raise ValueError("validation failure scores must match validation features")
    maximum_neighbors = train.shape[0] - (1 if leave_one_out else 0)
    if not 1 <= neighbors <= maximum_neighbors:
        raise ValueError("neighbors exceeds available validation samples")
    if leave_one_out and query.shape != train.shape:
        raise ValueError("leave-one-out prediction requires the validation matrix as query")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized_train = (train - mean) / scale
    standardized_query = (query - mean) / scale
    distances = np.linalg.norm(
        standardized_query[:, None, :] - standardized_train[None, :, :], axis=2
    )
    if leave_one_out:
        np.fill_diagonal(distances, np.inf)
    indexes = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    neighbor_distances = np.take_along_axis(distances, indexes, axis=1)
    weights = 1.0 / (neighbor_distances + 1e-6)
    neighbor_failures = failures[indexes]
    return (neighbor_failures * weights).sum(axis=1) / weights.sum(axis=1)


def fit_shift_model(features: np.ndarray, *, percentile: float = 95.0) -> dict:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("at least three validation feature vectors are required")
    if not 50.0 <= percentile < 100.0:
        raise ValueError("shift percentile must be in [50, 100)")
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)
    ridge = max(float(np.trace(covariance) / covariance.shape[0]) * 0.01, 1e-6)
    inverse = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * ridge)
    scores = shift_scores(values, mean, inverse)
    return {
        "mean": mean,
        "inverse_covariance": inverse,
        "ridge": ridge,
        "threshold_percentile": float(percentile),
        "threshold": float(np.percentile(scores, percentile)),
        "validation_scores": scores,
    }


def shift_scores(
    features: np.ndarray, mean: np.ndarray, inverse_covariance: np.ndarray
) -> np.ndarray:
    difference = np.asarray(features, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    squared = np.einsum("ij,jk,ik->i", difference, inverse_covariance, difference)
    return np.sqrt(np.maximum(squared, 0.0))


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    numbers = np.asarray(values, dtype=np.float64)
    order = np.argsort(numbers, kind="stable")
    ranks = np.empty(numbers.size, dtype=np.float64)
    ranks[order] = np.arange(numbers.size, dtype=np.float64)
    return ranks / max(1, numbers.size - 1)


def select_review_queue(
    review_ids: list[str],
    uncertainty: np.ndarray,
    shift: np.ndarray,
    features: np.ndarray,
    *,
    budget: int,
    predicted_failure_risk: np.ndarray | None = None,
    failure_risk_weight: float = 0.0,
    uncertainty_weight: float = 0.70,
    shift_weight: float = 0.30,
    diversity_weight: float = 0.25,
) -> list[dict]:
    """Create a label-free queue using uncertainty, input shift, and farthest-first diversity."""
    count = len(review_ids)
    if count == 0 or len(set(review_ids)) != count:
        raise ValueError("review identifiers must be non-empty and unique")
    if not 1 <= budget <= count:
        raise ValueError("review budget must be between one and pool size")
    if failure_risk_weight < 0 or uncertainty_weight < 0 or shift_weight < 0:
        raise ValueError("acquisition weights must be non-negative")
    if failure_risk_weight + uncertainty_weight + shift_weight <= 0:
        raise ValueError("acquisition weights must have a positive sum")
    if not 0.0 <= diversity_weight < 1.0:
        raise ValueError("diversity_weight must be in [0, 1)")
    values = np.asarray(features, dtype=np.float64)
    if values.shape[0] != count:
        raise ValueError("feature rows must match review identifiers")
    scale = values.std(axis=0)
    standardized = (values - values.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)
    uncertainty_rank = _rank_fraction(uncertainty)
    shift_rank = _rank_fraction(shift)
    if predicted_failure_risk is None:
        risk_values = np.zeros(count, dtype=np.float64)
    else:
        risk_values = np.asarray(predicted_failure_risk, dtype=np.float64)
        if risk_values.shape != (count,):
            raise ValueError("predicted failure risk must match review identifiers")
    risk_rank = _rank_fraction(risk_values)
    total_weight = failure_risk_weight + uncertainty_weight + shift_weight
    base = (
        failure_risk_weight * risk_rank
        + uncertainty_weight * uncertainty_rank
        + shift_weight * shift_rank
    ) / total_weight
    selected: list[int] = []
    selected_acquisition: list[float] = []
    remaining = set(range(count))
    for rank in range(1, budget + 1):
        if not selected:
            diversity = np.zeros(count, dtype=np.float64)
        else:
            difference = standardized[:, None, :] - standardized[np.asarray(selected)][None, :, :]
            diversity = np.linalg.norm(difference, axis=2).min(axis=1)
        diversity_rank = _rank_fraction(diversity)
        acquisition = (1.0 - diversity_weight) * base + diversity_weight * diversity_rank
        chosen = max(
            remaining,
            key=lambda index: (float(acquisition[index]), review_ids[index]),
        )
        selected.append(chosen)
        selected_acquisition.append(float(acquisition[chosen]))
        remaining.remove(chosen)
    queue = []
    for rank, (index, acquisition_score) in enumerate(
        zip(selected, selected_acquisition, strict=True), start=1
    ):
        reasons = []
        if failure_risk_weight > 0 and risk_rank[index] >= 0.75:
            reasons.append("high validation-learned failure risk")
        if uncertainty_rank[index] >= 0.75:
            reasons.append("high predictive entropy")
        if shift_rank[index] >= 0.75:
            reasons.append("input distribution shift")
        if rank > 1:
            reasons.append("pool diversity")
        queue.append(
            {
                "rank": rank,
                "review_id": review_ids[index],
                "acquisition_score": acquisition_score,
                "predicted_failure_risk": float(risk_values[index]),
                "predicted_failure_risk_percentile": float(risk_rank[index]),
                "uncertainty_score": float(uncertainty[index]),
                "uncertainty_percentile": float(uncertainty_rank[index]),
                "shift_score": float(shift[index]),
                "shift_percentile": float(shift_rank[index]),
                "reasons": reasons or ["combined acquisition score"],
            }
        )
    return queue


def select_acquisition_policy(
    review_ids: list[str],
    validation_features: np.ndarray,
    validation_failures: np.ndarray,
    uncertainty: np.ndarray,
    shift: np.ndarray,
    *,
    budget: int,
    neighbor_candidates: list[int],
    weight_candidates: list[dict],
    worst_fraction: float = 0.25,
) -> dict:
    """Choose kNN and acquisition weights through leave-one-out validation only."""
    if not neighbor_candidates or not weight_candidates:
        raise ValueError("acquisition policy search requires neighbor and weight candidates")
    labeled_records = [
        {
            "review_id": review_id,
            "failure_score": float(validation_failures[index]),
            "uncertainty_score": float(uncertainty[index]),
            "shift_score": float(shift[index]),
        }
        for index, review_id in enumerate(review_ids)
    ]
    candidates = []
    for neighbors in neighbor_candidates:
        risk = predict_knn_failure_risk(
            validation_features,
            validation_failures,
            validation_features,
            neighbors=int(neighbors),
            leave_one_out=True,
        )
        for weights in weight_candidates:
            queue = select_review_queue(
                review_ids,
                uncertainty,
                shift,
                validation_features,
                budget=budget,
                predicted_failure_risk=risk,
                failure_risk_weight=float(weights["failure_risk"]),
                uncertainty_weight=float(weights["uncertainty"]),
                shift_weight=float(weights["shift"]),
                diversity_weight=float(weights["diversity"]),
            )
            result = evaluate_queue_retrospectively(
                queue, labeled_records, worst_fraction=worst_fraction
            )["active_queue"]
            candidates.append(
                {
                    "neighbors": int(neighbors),
                    "weights": {
                        key: float(weights[key])
                        for key in ("failure_risk", "uncertainty", "shift", "diversity")
                    },
                    "validation_worst_quartile_hits": result["worst_quartile_hits"],
                    "validation_worst_quartile_recall": result["worst_quartile_recall"],
                    "validation_mean_failure_score": result["mean_failure_score"],
                }
            )
    selected = max(
        candidates,
        key=lambda item: (
            item["validation_worst_quartile_hits"],
            item["validation_mean_failure_score"],
            -item["neighbors"],
            -item["weights"]["diversity"],
        ),
    )
    return {
        "selection_split": "validation",
        "method": "leave-one-out inverse-distance kNN failure-risk policy search",
        "validation_budget": budget,
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    positive = int(truth.sum())
    negative = int((~truth).sum())
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires positive and negative examples")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[truth].sum())
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def apply_corruption(
    image_chw: np.ndarray,
    name: str,
    severity: float,
    *,
    seed: int,
) -> np.ndarray:
    image = np.asarray(image_chw, dtype=np.float32)
    if severity <= 0:
        raise ValueError("corruption severity must be positive")
    if name == "gaussian_noise":
        noise = np.random.default_rng(seed).normal(0.0, severity, size=image.shape)
        return np.clip(image + noise, 0.0, 1.0).astype(np.float32)
    if name == "brightness":
        return np.clip(image * severity, 0.0, 1.0).astype(np.float32)
    if name == "blur":
        uint8 = np.clip(image.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        blurred = Image.fromarray(uint8).filter(ImageFilter.GaussianBlur(radius=severity))
        return np.asarray(blurred, dtype=np.float32).transpose(2, 0, 1) / 255.0
    raise ValueError(f"unsupported controlled corruption '{name}'")


def _review_id(sample_id: str) -> str:
    return "review_" + hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]


def _uncertainty_score(probability: np.ndarray, top_fraction: float) -> float:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("uncertainty_top_fraction must be in (0, 1]")
    entropy = binary_entropy(probability).reshape(-1)
    count = max(1, int(math.ceil(entropy.size * top_fraction)))
    return float(np.partition(entropy, entropy.size - count)[-count:].mean())


def _collect_outputs(
    model,
    records: list[dict],
    *,
    image_size: int,
    batch_size: int,
    device,
    torch,
) -> list[dict]:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            arrays = [load_record_arrays(record, image_size=image_size) for record in batch_records]
            images = np.stack([item[0] for item in arrays]).astype(np.float32)
            logits = model(torch.from_numpy(images).to(device)).detach().cpu().numpy()[:, 0]
            for record, image, (_, target), logit in zip(
                batch_records, images, arrays, logits, strict=True
            ):
                outputs.append(
                    {
                        "record": record,
                        "image": image,
                        "target": target[0].astype(bool),
                        "logits": logit.astype(np.float32),
                    }
                )
    return outputs


def _failure_score(prediction: np.ndarray, target: np.ndarray, anomaly: bool) -> tuple[float, dict]:
    counts = confusion_counts(prediction, target)
    metrics = metrics_from_counts(counts)
    if anomaly:
        score = 1.0 - metrics["iou"]
    else:
        score = counts["false_positive"] / max(1, target.size)
    return float(score), metrics


def evaluate_queue_retrospectively(
    queue: list[dict],
    labeled_records: list[dict],
    *,
    worst_fraction: float = 0.25,
) -> dict:
    if not 0.0 < worst_fraction < 1.0:
        raise ValueError("worst_fraction must be in (0, 1)")
    by_id = {record["review_id"]: record for record in labeled_records}
    selected_ids = [item["review_id"] for item in queue]
    if any(review_id not in by_id for review_id in selected_ids):
        raise ValueError("queue contains an unknown review identifier")
    ordered = sorted(
        labeled_records,
        key=lambda record: (-record["failure_score"], record["review_id"]),
    )
    worst_count = max(1, int(math.ceil(len(ordered) * worst_fraction)))
    worst_ids = {record["review_id"] for record in ordered[:worst_count]}

    def strategy(ids: list[str]) -> dict:
        chosen = [by_id[review_id] for review_id in ids]
        hits = len(set(ids) & worst_ids)
        expected_random = len(ids) * worst_count / len(ordered)
        return {
            "budget": len(ids),
            "worst_quartile_hits": hits,
            "worst_quartile_precision": hits / max(1, len(ids)),
            "worst_quartile_recall": hits / worst_count,
            "hit_lift_vs_random_expectation": hits / expected_random if expected_random else 0.0,
            "mean_failure_score": float(
                np.mean([record["failure_score"] for record in chosen])
            ),
        }

    budget = len(selected_ids)
    uncertainty_ids = [
        record["review_id"]
        for record in sorted(
            labeled_records,
            key=lambda record: (-record["uncertainty_score"], record["review_id"]),
        )[:budget]
    ]
    shift_ids = [
        record["review_id"]
        for record in sorted(
            labeled_records,
            key=lambda record: (-record["shift_score"], record["review_id"]),
        )[:budget]
    ]
    random_expected_hits = budget * worst_count / len(ordered)
    selected = strategy(selected_ids)
    report = {
        "label_use": "labels revealed only after the queue was frozen within this run",
        "pool_size": len(ordered),
        "worst_fraction": float(worst_fraction),
        "worst_sample_count": worst_count,
        "pool_mean_failure_score": float(
            np.mean([record["failure_score"] for record in ordered])
        ),
        "active_queue": selected,
        "uncertainty_only": strategy(uncertainty_ids),
        "shift_only": strategy(shift_ids),
        "random_expectation": {
            "expected_worst_quartile_hits": float(random_expected_hits),
            "expected_worst_quartile_precision": worst_count / len(ordered),
        },
        "oracle_upper_bound": {
            "worst_quartile_hits": min(budget, worst_count),
            "worst_quartile_recall": min(budget, worst_count) / worst_count,
        },
        "selected_mean_failure_lift_vs_pool": (
            selected["mean_failure_score"]
            / max(EPSILON, float(np.mean([record["failure_score"] for record in ordered])))
        ),
        "selected_labeled_audit": [by_id[review_id] for review_id in selected_ids],
    }
    if all("predicted_failure_risk" in record for record in labeled_records):
        risk_ids = [
            record["review_id"]
            for record in sorted(
                labeled_records,
                key=lambda record: (-record["predicted_failure_risk"], record["review_id"]),
            )[:budget]
        ]
        report["failure_risk_only"] = strategy(risk_ids)
    return report


def _heatmap(values: np.ndarray, *, entropy: bool = False) -> np.ndarray:
    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    red = np.clip(normalized * 2.0, 0.0, 1.0)
    blue = np.clip((1.0 - normalized) * 2.0, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(normalized - 0.5) * 2.0, 0.0, 1.0)
    if entropy:
        blue *= 0.45
    return np.clip(np.stack([red, green, blue], axis=-1) * 255.0, 0, 255).astype(np.uint8)


def _save_review_preview(
    output: dict,
    probability: np.ndarray,
    threshold: float,
    destination: Path,
) -> None:
    image = np.clip(output["image"].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    entropy = binary_entropy(probability)
    overlay = image.astype(np.float32)
    prediction = probability >= threshold
    overlay[prediction, 0] = 255
    overlay[prediction, 1:] *= 0.40
    combined = np.concatenate(
        [image, _heatmap(probability), _heatmap(entropy, entropy=True), overlay.astype(np.uint8)],
        axis=1,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(destination)


def portable_evidence(report: dict) -> dict:
    """Remove sample-level audit details and verbose bins from committed evidence."""
    evidence = json.loads(json.dumps(report))
    evidence.pop("per_sample_retrospective", None)
    for split in ("validation", "held_out_test"):
        for state in ("before", "after"):
            evidence["calibration"][split][state].pop("bins", None)
    selection = evidence["review_queue"]["policy_selection"]
    selection.pop("candidates", None)
    evidence["evidence_scope"] = (
        "portable aggregate development evidence; raw data, weights, sample IDs, and labels omitted"
    )
    return evidence


def _validate_config(config: dict) -> dict:
    defaults = {
        "schema_version": "1.0",
        "image_size": 128,
        "batch_size": 16,
        "seed": 42,
        "calibration_pixels": 250_000,
        "ece_bins": 15,
        "queue_budget": 20,
        "uncertainty_top_fraction": 0.05,
        "neighbor_candidates": [3, 5, 8, 12, 20, 30],
        "acquisition_weight_candidates": [
            {"failure_risk": 1.0, "uncertainty": 0.0, "shift": 0.0, "diversity": 0.0},
            {"failure_risk": 1.0, "uncertainty": 0.0, "shift": 0.0, "diversity": 0.1},
            {"failure_risk": 0.9, "uncertainty": 0.1, "shift": 0.0, "diversity": 0.1},
            {"failure_risk": 0.8, "uncertainty": 0.2, "shift": 0.0, "diversity": 0.15},
            {"failure_risk": 0.7, "uncertainty": 0.3, "shift": 0.0, "diversity": 0.25},
            {"failure_risk": 0.7, "uncertainty": 0.2, "shift": 0.1, "diversity": 0.15},
            {"failure_risk": 0.6, "uncertainty": 0.3, "shift": 0.1, "diversity": 0.15},
            {"failure_risk": 0.5, "uncertainty": 0.4, "shift": 0.1, "diversity": 0.25},
        ],
        "shift_percentile": 95.0,
        "worst_fraction": 0.25,
        "gallery_count": 12,
        "corruptions": [
            {"name": "gaussian_noise", "severity": 0.18},
            {"name": "brightness", "severity": 0.45},
            {"name": "blur", "severity": 2.0},
        ],
    }
    merged = {**defaults, **config}
    if str(merged["schema_version"]) != "1.0":
        raise ValueError("unsupported active-learning config schema_version")
    for key in ("image_size", "batch_size", "calibration_pixels", "ece_bins", "queue_budget"):
        if int(merged[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(merged["ece_bins"]) < 2:
        raise ValueError("ece_bins must be at least 2")
    if not 0.0 < float(merged["uncertainty_top_fraction"]) <= 1.0:
        raise ValueError("uncertainty_top_fraction must be in (0, 1]")
    if not 50.0 <= float(merged["shift_percentile"]) < 100.0:
        raise ValueError("shift_percentile must be in [50, 100)")
    if not 0.0 < float(merged["worst_fraction"]) < 1.0:
        raise ValueError("worst_fraction must be in (0, 1)")
    if int(merged["gallery_count"]) < 0:
        raise ValueError("gallery_count must be non-negative")
    if not isinstance(merged["neighbor_candidates"], list) or not merged["neighbor_candidates"]:
        raise ValueError("neighbor_candidates must be a non-empty list")
    if any(int(value) < 1 for value in merged["neighbor_candidates"]):
        raise ValueError("neighbor_candidates must be positive")
    weight_candidates = merged["acquisition_weight_candidates"]
    if not isinstance(weight_candidates, list) or not weight_candidates:
        raise ValueError("acquisition_weight_candidates must be a non-empty list")
    required_weights = {"failure_risk", "uncertainty", "shift", "diversity"}
    for weights in weight_candidates:
        if set(weights) != required_weights:
            raise ValueError("each acquisition weight candidate must define all four weights")
        if any(float(weights[key]) < 0 for key in required_weights):
            raise ValueError("acquisition weights must be non-negative")
        if sum(float(weights[key]) for key in ("failure_risk", "uncertainty", "shift")) <= 0:
            raise ValueError("acquisition signal weights must have a positive sum")
        if not 0.0 <= float(weights["diversity"]) < 1.0:
            raise ValueError("diversity weight must be in [0, 1)")
    if not isinstance(merged["corruptions"], list) or not merged["corruptions"]:
        raise ValueError("at least one controlled corruption is required")
    for corruption in merged["corruptions"]:
        if corruption.get("name") not in {"gaussian_noise", "brightness", "blur"}:
            raise ValueError("unsupported controlled corruption in config")
        if float(corruption.get("severity", 0.0)) <= 0:
            raise ValueError("corruption severity must be positive")
    return merged


def load_active_learning_config(path: Path) -> dict:
    config_path = Path(path)
    payload = _validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    for key in ("manifest", "checkpoint", "output", "portable_evidence"):
        if key not in payload:
            continue
        value = Path(payload[key])
        payload[key] = value if value.is_absolute() else (config_path.parent / value).resolve()
    return payload


def run_active_learning_study(config: dict) -> dict:
    config = _validate_config(config)
    manifest_path = Path(config["manifest"])
    checkpoint_path = Path(config["checkpoint"])
    output_dir = Path(config["output"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the active-learning study") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_segmentation_checkpoint(checkpoint_path, device=device)
    image_size = int(
        config.get("image_size")
        or checkpoint.get("training_config", {}).get("image_size", 128)
    )
    original_threshold = float(checkpoint.get("threshold", 0.5))
    manifest = load_manifest(manifest_path)
    validation_records = [
        record for record in manifest["records"] if record["split"] == "validation"
    ]
    test_records = [record for record in manifest["records"] if record["split"] == "test"]
    if len(validation_records) < 4:
        raise ValueError("manifest must contain at least four validation records")
    if not test_records:
        raise ValueError("manifest must contain at least one held-out test record")
    validation_outputs = _collect_outputs(
        model,
        validation_records,
        image_size=image_size,
        batch_size=int(config["batch_size"]),
        device=device,
        torch=torch,
    )
    test_outputs = _collect_outputs(
        model,
        test_records,
        image_size=image_size,
        batch_size=int(config["batch_size"]),
        device=device,
        torch=torch,
    )
    validation_logits = np.stack([output["logits"] for output in validation_outputs])
    validation_targets = np.stack([output["target"] for output in validation_outputs])
    test_logits = np.stack([output["logits"] for output in test_outputs])
    test_targets = np.stack([output["target"] for output in test_outputs])
    calibration = fit_temperature(
        validation_logits,
        validation_targets,
        maximum_pixels=int(config["calibration_pixels"]),
        seed=int(config["seed"]),
    )
    temperature = float(calibration["temperature"])
    transformed_threshold = calibrated_threshold(original_threshold, temperature)
    calibration_report = {
        "fit": calibration,
        "validation": {
            "before": calibration_metrics(
                validation_logits,
                validation_targets,
                temperature=1.0,
                maximum_pixels=int(config["calibration_pixels"]),
                bins=int(config["ece_bins"]),
                seed=int(config["seed"]),
            ),
            "after": calibration_metrics(
                validation_logits,
                validation_targets,
                temperature=temperature,
                maximum_pixels=int(config["calibration_pixels"]),
                bins=int(config["ece_bins"]),
                seed=int(config["seed"]),
            ),
        },
        "held_out_test": {
            "before": calibration_metrics(
                test_logits,
                test_targets,
                temperature=1.0,
                maximum_pixels=int(config["calibration_pixels"]),
                bins=int(config["ece_bins"]),
                seed=int(config["seed"] + 1),
            ),
            "after": calibration_metrics(
                test_logits,
                test_targets,
                temperature=temperature,
                maximum_pixels=int(config["calibration_pixels"]),
                bins=int(config["ece_bins"]),
                seed=int(config["seed"] + 1),
            ),
        },
        "decision_boundary": {
            "original_probability_threshold": original_threshold,
            "original_logit_threshold": probability_to_logit(original_threshold),
            "calibrated_probability_threshold": transformed_threshold,
            "mask_semantics": "same original logit boundary after temperature scaling",
        },
    }
    original_masks = sigmoid(test_logits) >= original_threshold
    calibrated_probabilities = sigmoid(test_logits / temperature)
    calibrated_masks = calibrated_probabilities >= transformed_threshold
    mask_agreement = float(np.mean(original_masks == calibrated_masks))
    if mask_agreement < 1.0:
        raise RuntimeError("calibration changed deployed segmentation decisions")
    calibration_report["decision_boundary"]["test_mask_agreement"] = mask_agreement

    validation_probabilities = sigmoid(validation_logits / temperature)
    validation_features = np.stack(
        [
            sample_observable_features(
                output["image"], probability, threshold=transformed_threshold
            )
            for output, probability in zip(
                validation_outputs, validation_probabilities, strict=True
            )
        ]
    )
    test_features = np.stack(
        [
            sample_observable_features(
                output["image"], probability, threshold=transformed_threshold
            )
            for output, probability in zip(test_outputs, calibrated_probabilities, strict=True)
        ]
    )
    validation_image_features = np.stack(
        [image_feature_vector(output["image"]) for output in validation_outputs]
    )
    test_image_features = np.stack(
        [image_feature_vector(output["image"]) for output in test_outputs]
    )
    input_shift_model = fit_shift_model(
        validation_image_features, percentile=float(config["shift_percentile"])
    )
    clean_input_shift = shift_scores(
        test_image_features,
        input_shift_model["mean"],
        input_shift_model["inverse_covariance"],
    )
    shift_model = fit_shift_model(
        validation_features, percentile=float(config["shift_percentile"])
    )
    test_shift = shift_scores(
        test_features,
        shift_model["mean"],
        shift_model["inverse_covariance"],
    )
    validation_uncertainty = np.asarray(
        [
            _uncertainty_score(probability, float(config["uncertainty_top_fraction"]))
            for probability in validation_probabilities
        ]
    )
    uncertainty = np.asarray(
        [
            _uncertainty_score(probability, float(config["uncertainty_top_fraction"]))
            for probability in calibrated_probabilities
        ]
    )
    validation_failures = np.asarray(
        [
            _failure_score(
                probability >= transformed_threshold,
                output["target"],
                bool(output["record"]["anomaly"]),
            )[0]
            for output, probability in zip(
                validation_outputs, validation_probabilities, strict=True
            )
        ]
    )
    validation_shift = shift_scores(
        validation_features,
        shift_model["mean"],
        shift_model["inverse_covariance"],
    )
    validation_budget = min(
        max(1, round(len(validation_outputs) * int(config["queue_budget"]) / len(test_outputs))),
        len(validation_outputs) - 1,
    )
    maximum_neighbors = len(validation_outputs) - 1
    neighbor_candidates = sorted(
        {
            min(int(value), maximum_neighbors)
            for value in config["neighbor_candidates"]
            if int(value) >= 1
        }
    )
    validation_review_ids = [
        _review_id(output["record"]["sample_id"]) for output in validation_outputs
    ]
    policy = select_acquisition_policy(
        validation_review_ids,
        validation_features,
        validation_failures,
        validation_uncertainty,
        validation_shift,
        budget=validation_budget,
        neighbor_candidates=neighbor_candidates,
        weight_candidates=config["acquisition_weight_candidates"],
        worst_fraction=float(config["worst_fraction"]),
    )
    selected_policy = policy["selected"]
    selected_weights = selected_policy["weights"]
    predicted_failure_risk = predict_knn_failure_risk(
        validation_features,
        validation_failures,
        test_features,
        neighbors=int(selected_policy["neighbors"]),
    )
    review_ids = [_review_id(output["record"]["sample_id"]) for output in test_outputs]
    queue = select_review_queue(
        review_ids,
        uncertainty,
        test_shift,
        test_features,
        budget=min(int(config["queue_budget"]), len(test_outputs)),
        predicted_failure_risk=predicted_failure_risk,
        failure_risk_weight=float(selected_weights["failure_risk"]),
        uncertainty_weight=float(selected_weights["uncertainty"]),
        shift_weight=float(selected_weights["shift"]),
        diversity_weight=float(selected_weights["diversity"]),
    )
    queue_by_id = {item["review_id"]: item for item in queue}
    label_free_queue = {
        "schema_version": "1.0",
        "label_use": "no target mask, anomaly flag, or defect type used for acquisition",
        "pool_split": "test split treated as an unlabeled retrospective development pool",
        "budget": len(queue),
        "pool_size": len(test_outputs),
        "items": queue,
    }
    (output_dir / "review_queue.json").write_text(
        json.dumps(label_free_queue, indent=2, sort_keys=True), encoding="utf-8"
    )
    labeled_audit = []
    per_sample = []
    for index, output in enumerate(test_outputs):
        review_id = review_ids[index]
        failure, metrics = _failure_score(
            calibrated_masks[index],
            output["target"],
            bool(output["record"]["anomaly"]),
        )
        record = {
            "review_id": review_id,
            "category": output["record"]["category"],
            "defect_type": output["record"]["defect_type"],
            "anomaly": bool(output["record"]["anomaly"]),
            "failure_score": failure,
            "predicted_failure_risk": float(predicted_failure_risk[index]),
            "uncertainty_score": float(uncertainty[index]),
            "shift_score": float(test_shift[index]),
            "selected": review_id in queue_by_id,
            **metrics,
        }
        labeled_audit.append(record)
        per_sample.append(record)
    retrospective = evaluate_queue_retrospectively(
        queue,
        labeled_audit,
        worst_fraction=float(config["worst_fraction"]),
    )
    (output_dir / "retrospective_evaluation.json").write_text(
        json.dumps(retrospective, indent=2, sort_keys=True), encoding="utf-8"
    )

    controlled_shift = {}
    clean_detection_rate = float(
        np.mean(clean_input_shift > input_shift_model["threshold"])
    )
    for corruption_index, corruption in enumerate(config["corruptions"]):
        name = str(corruption["name"])
        severity = float(corruption["severity"])
        corrupted_images = [
            apply_corruption(
                output["image"],
                name,
                severity,
                seed=int(config["seed"]) + corruption_index * 10_000 + index,
            )
            for index, output in enumerate(test_outputs)
        ]
        corrupted_image_features = np.stack(
            [image_feature_vector(image) for image in corrupted_images]
        )
        corrupted_input_shift = shift_scores(
            corrupted_image_features,
            input_shift_model["mean"],
            input_shift_model["inverse_covariance"],
        )
        labels = np.concatenate(
            [
                np.zeros(clean_input_shift.size, dtype=bool),
                np.ones(corrupted_input_shift.size, dtype=bool),
            ]
        )
        scores = np.concatenate([clean_input_shift, corrupted_input_shift])
        with torch.no_grad():
            corrupted_logits = []
            for start in range(0, len(corrupted_images), int(config["batch_size"])):
                batch = np.stack(
                    corrupted_images[start : start + int(config["batch_size"])]
                ).astype(np.float32)
                values = model(torch.from_numpy(batch).to(device)).detach().cpu().numpy()[:, 0]
                corrupted_logits.extend(values)
        corrupted_probabilities = sigmoid(np.stack(corrupted_logits) / temperature)
        corrupted_uncertainty = np.asarray(
            [
                _uncertainty_score(probability, float(config["uncertainty_top_fraction"]))
                for probability in corrupted_probabilities
            ]
        )
        controlled_shift[name] = {
            "severity": severity,
            "samples": len(corrupted_images),
            "shift_score_auroc": binary_auroc(labels, scores),
            "detection_rate_at_validation_threshold": float(
                np.mean(corrupted_input_shift > input_shift_model["threshold"])
            ),
            "clean_test_detection_rate_at_validation_threshold": clean_detection_rate,
            "mean_shift_score_clean": float(clean_input_shift.mean()),
            "mean_shift_score_corrupted": float(corrupted_input_shift.mean()),
            "mean_uncertainty_clean": float(uncertainty.mean()),
            "mean_uncertainty_corrupted": float(corrupted_uncertainty.mean()),
        }

    selected_indexes = {review_id: index for index, review_id in enumerate(review_ids)}
    gallery_dir = output_dir / "review_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for stale_preview in gallery_dir.glob("*.png"):
        stale_preview.unlink()
    for item in queue[: min(int(config["gallery_count"]), len(queue))]:
        index = selected_indexes[item["review_id"]]
        _save_review_preview(
            test_outputs[index],
            calibrated_probabilities[index],
            transformed_threshold,
            gallery_dir / f"{item['rank']:02d}_{item['review_id']}.png",
        )
    report = {
        "schema_version": "1.0",
        "project_version": "0.11.0",
        "study": "validation-calibrated risk-aware human review acquisition",
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_config": checkpoint.get("model_config", {}),
        "device": str(device),
        "image_size": image_size,
        "splits": {
            "calibration": {"name": "validation", "samples": len(validation_outputs)},
            "simulated_unlabeled_pool": {
                "name": "test split used as retrospective development pool",
                "samples": len(test_outputs),
            },
        },
        "calibration": calibration_report,
        "uncertainty": {
            "aggregation": (
                f"mean of top {float(config['uncertainty_top_fraction']) * 100.0:.1f}% "
                "normalized binary pixel entropy"
            ),
            "pool_mean": float(uncertainty.mean()),
            "pool_p95": float(np.percentile(uncertainty, 95)),
        },
        "input_shift": {
            "feature_contract": (
                "label-free RGB mean/std/quantiles, grayscale histogram, and gradient statistics"
            ),
            "score": "ridge-regularized Mahalanobis distance from validation features",
            "validation_threshold_percentile": float(config["shift_percentile"]),
            "validation_threshold": float(input_shift_model["threshold"]),
            "clean_test_detection_rate": clean_detection_rate,
            "controlled_corruptions": controlled_shift,
            "boundary": (
                "controlled corruptions test sensitivity to known shifts; they are not proof of "
                "open-world OOD performance"
            ),
        },
        "review_queue": {
            "budget": len(queue),
            "pool_size": len(test_outputs),
            "budget_fraction": len(queue) / len(test_outputs),
            "policy_selection": policy,
            "selected_weights": selected_weights,
            "selected_neighbors": int(selected_policy["neighbors"]),
            "label_free_artifact": "review_queue.json",
        },
        "retrospective_queue_evaluation": {
            key: value
            for key, value in retrospective.items()
            if key != "selected_labeled_audit"
        },
        "per_sample_retrospective": per_sample,
        "gallery_note": "panels are RGB, calibrated probability, entropy, and prediction overlay",
        "limitations": [
            (
                "The test split is label-free during each acquisition run, but pool-level results "
                "were inspected while developing v0.11; this is not a pristine final holdout."
            ),
            "This is a retrospective acquisition study and does not retrain after annotation.",
            "An untouched external dataset is required to confirm review-queue lift.",
            "Temperature calibration is fitted only on validation pixels.",
            "Controlled corruptions are not a substitute for an external OOD dataset.",
            "The MVTec protocol remains supervised-development, not the official benchmark.",
            "No aircraft-surface, production, or safety-critical performance claim is made.",
        ],
    }
    (output_dir / "active_learning_study.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "portable_evidence.json").write_text(
        json.dumps(portable_evidence(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    if config.get("portable_evidence"):
        evidence_path = Path(config["portable_evidence"])
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(portable_evidence(report), indent=2, sort_keys=True), encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate uncertainty and build a diversity-aware human review queue"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--portable-evidence", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--queue-budget", type=int)
    parser.add_argument("--calibration-pixels", type=int)
    args = parser.parse_args()
    config = load_active_learning_config(args.config)
    for key in (
        "manifest",
        "checkpoint",
        "output",
        "portable_evidence",
        "image_size",
        "batch_size",
        "queue_budget",
        "calibration_pixels",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    report = run_active_learning_study(config)
    summary = {
        "temperature": report["calibration"]["fit"]["temperature"],
        "test_ece_before": report["calibration"]["held_out_test"]["before"]["ece"],
        "test_ece_after": report["calibration"]["held_out_test"]["after"]["ece"],
        "review_budget": report["review_queue"]["budget"],
        "worst_quartile_recall": report["retrospective_queue_evaluation"]["active_queue"][
            "worst_quartile_recall"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
