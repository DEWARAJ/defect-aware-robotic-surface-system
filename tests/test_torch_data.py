import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from surface_perception.torch_data import augment_pair, load_record_arrays


class TorchDataPreprocessingTest(unittest.TestCase):
    def test_augmentation_is_deterministic_and_keeps_mask_binary(self) -> None:
        image = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[10:25, 13:32] = 1
        first_image, first_mask = augment_pair(image, mask, seed=99)
        second_image, second_mask = augment_pair(image, mask, seed=99)
        self.assertTrue(np.array_equal(first_image, second_image))
        self.assertTrue(np.array_equal(first_mask, second_mask))
        self.assertEqual(set(np.unique(first_mask)), {0, 1})
        self.assertEqual(int(first_mask.sum()), int(mask.sum()))

    def test_good_record_creates_zero_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "good.png"
            Image.fromarray(np.full((30, 50, 3), 127, dtype=np.uint8)).save(image_path)
            record = {
                "sample_id": "good_001",
                "image_path": str(image_path),
                "mask_path": None,
            }
            image, mask = load_record_arrays(record, image_size=32)
            self.assertEqual(image.shape, (3, 32, 32))
            self.assertEqual(mask.shape, (1, 32, 32))
            self.assertEqual(float(mask.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
