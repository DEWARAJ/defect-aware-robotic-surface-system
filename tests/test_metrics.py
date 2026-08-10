import unittest

import numpy as np

from surface_perception.metrics import segmentation_metrics


class SegmentationMetricsTest(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        metrics = segmentation_metrics(target, target)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["pixel_accuracy"], 1.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            segmentation_metrics(np.zeros((2, 2)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()

