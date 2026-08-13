import json
import unittest
from pathlib import Path

import numpy as np

from surface_perception.active_learning import (
    _validate_config,
    apply_corruption,
    binary_auroc,
    calibrated_threshold,
    expected_calibration_error,
    evaluate_queue_retrospectively,
    fit_shift_model,
    fit_temperature,
    predict_knn_failure_risk,
    sample_observable_features,
    select_review_queue,
    shift_scores,
    sigmoid,
)


class TemperatureCalibrationTest(unittest.TestCase):
    def test_transformed_threshold_preserves_binary_decisions(self) -> None:
        logits = np.linspace(-8.0, 8.0, 1001)
        original_threshold = 0.70
        temperature = 2.75
        transformed = calibrated_threshold(original_threshold, temperature)
        np.testing.assert_array_equal(
            sigmoid(logits) >= original_threshold,
            sigmoid(logits / temperature) >= transformed,
        )

    def test_temperature_fit_reduces_validation_nll(self) -> None:
        calibrated_logits = np.concatenate(
            [np.full(100, np.log(0.2 / 0.8)), np.full(100, np.log(0.8 / 0.2))]
        )
        targets = np.concatenate(
            [np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(80), np.zeros(20)]]
        )
        report = fit_temperature(calibrated_logits * 2.5, targets, maximum_pixels=1000)
        self.assertLess(report["nll_after"], report["nll_before"])
        self.assertAlmostEqual(report["temperature"], 2.5, delta=0.1)
        self.assertFalse(report["bound_hit"])

    def test_ece_uses_probability_bins(self) -> None:
        probabilities = np.r_[np.full(10, 0.1), np.full(10, 0.8)]
        targets = np.r_[np.r_[1, np.zeros(9)], np.r_[np.ones(8), np.zeros(2)]]
        report = expected_calibration_error(probabilities, targets, bins=10)
        self.assertAlmostEqual(report["ece"], 0.0)


class ShiftAndAcquisitionTest(unittest.TestCase):
    def test_mahalanobis_shift_ranks_displaced_features_higher(self) -> None:
        rng = np.random.default_rng(7)
        validation = rng.normal(0.0, 0.2, size=(40, 5))
        model = fit_shift_model(validation)
        clean = rng.normal(0.0, 0.2, size=(20, 5))
        shifted = rng.normal(4.0, 0.2, size=(20, 5))
        clean_scores = shift_scores(clean, model["mean"], model["inverse_covariance"])
        shifted_scores = shift_scores(shifted, model["mean"], model["inverse_covariance"])
        self.assertGreater(float(shifted_scores.mean()), float(clean_scores.mean()) * 5.0)

    def test_review_queue_is_label_free_unique_and_budgeted(self) -> None:
        queue = select_review_queue(
            [f"review_{index}" for index in range(6)],
            np.asarray([0.1, 0.9, 0.8, 0.2, 0.3, 0.7]),
            np.asarray([0.1, 0.2, 0.3, 0.9, 0.8, 0.4]),
            np.asarray([[index, index % 2] for index in range(6)], dtype=float),
            budget=3,
        )
        self.assertEqual(len(queue), 3)
        self.assertEqual(len({item["review_id"] for item in queue}), 3)
        forbidden = {"target", "mask", "anomaly", "defect_type", "failure_score"}
        self.assertTrue(all(forbidden.isdisjoint(item) for item in queue))
        self.assertEqual([item["rank"] for item in queue], [1, 2, 3])

    def test_knn_failure_risk_uses_validation_neighbors(self) -> None:
        validation = np.asarray([[0.0], [1.0], [5.0], [6.0]])
        failures = np.asarray([0.0, 0.1, 0.9, 1.0])
        queries = np.asarray([[0.2], [5.8]])
        risk = predict_knn_failure_risk(
            validation, failures, queries, neighbors=2
        )
        self.assertLess(risk[0], 0.2)
        self.assertGreater(risk[1], 0.8)

    def test_observable_features_do_not_require_a_target(self) -> None:
        image = np.full((3, 8, 8), 0.4, dtype=np.float32)
        probability = np.full((8, 8), 0.7, dtype=np.float32)
        features = sample_observable_features(image, probability, threshold=0.6)
        self.assertEqual(features.ndim, 1)
        self.assertTrue(np.isfinite(features).all())

    def test_auroc_handles_ties_and_perfect_separation(self) -> None:
        self.assertAlmostEqual(
            binary_auroc(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9])),
            1.0,
        )
        self.assertAlmostEqual(
            binary_auroc(np.asarray([0, 1]), np.asarray([0.5, 0.5])),
            0.5,
        )

    def test_retrospective_labels_are_revealed_after_queue_selection(self) -> None:
        labeled = [
            {
                "review_id": f"review_{index}",
                "failure_score": float(index),
                "uncertainty_score": float(index),
                "shift_score": float(5 - index),
            }
            for index in range(6)
        ]
        queue = [{"review_id": "review_5"}, {"review_id": "review_4"}]
        report = evaluate_queue_retrospectively(queue, labeled, worst_fraction=1 / 3)
        self.assertEqual(report["active_queue"]["worst_quartile_hits"], 2)
        self.assertEqual(report["active_queue"]["worst_quartile_recall"], 1.0)
        self.assertIn("after the queue was frozen", report["label_use"])


class ActiveLearningUtilityTest(unittest.TestCase):
    def test_corruption_is_deterministic_and_bounded(self) -> None:
        image = np.full((3, 12, 12), 0.5, dtype=np.float32)
        first = apply_corruption(image, "gaussian_noise", 0.2, seed=9)
        second = apply_corruption(image, "gaussian_noise", 0.2, seed=9)
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_config_rejects_invalid_acquisition_weights(self) -> None:
        base = {"manifest": "m.json", "checkpoint": "c.pt", "output": "out"}
        config = _validate_config(base)
        self.assertEqual(config["queue_budget"], 20)
        with self.assertRaisesRegex(ValueError, "positive sum"):
            _validate_config(
                {
                    **base,
                    "acquisition_weight_candidates": [
                        {
                            "failure_risk": 0,
                            "uncertainty": 0,
                            "shift": 0,
                            "diversity": 0.2,
                        }
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "diversity weight"):
            _validate_config(
                {
                    **base,
                    "acquisition_weight_candidates": [
                        {
                            "failure_risk": 1,
                            "uncertainty": 0,
                            "shift": 0,
                            "diversity": 1.0,
                        }
                    ],
                }
            )

    def test_reference_evidence_is_self_consistent(self) -> None:
        evidence_path = (
            Path(__file__).parents[1]
            / "artifacts"
            / "reference"
            / "active_learning_v11.json"
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        calibration = evidence["calibration"]
        self.assertEqual(evidence["project_version"], "0.11.0")
        self.assertEqual(
            calibration["decision_boundary"]["test_mask_agreement"], 1.0
        )
        self.assertLessEqual(
            calibration["validation"]["after"]["nll"],
            calibration["validation"]["before"]["nll"] + 1e-12,
        )
        queue = evidence["review_queue"]
        self.assertLess(queue["budget"], queue["pool_size"])
        self.assertIn("label_free_artifact", queue)
        self.assertIn("controlled_corruptions", evidence["input_shift"])


if __name__ == "__main__":
    unittest.main()
