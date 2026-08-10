import tempfile
import unittest
from pathlib import Path

import numpy as np

from surface_perception.synthetic import generate_dataset, make_surface_sample


class SyntheticDatasetTest(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = make_surface_sample(48, 11)
        second = make_surface_sample(48, 11)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])
        self.assertGreater((first[1] > 0).sum(), 0)

    def test_dataset_has_all_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = generate_dataset(output, samples=12, image_size=32, seed=7)
            self.assertEqual(len(records), 12)
            for split in ("train", "validation", "test"):
                self.assertTrue(any((output / split / "images").glob("*.png")))
                self.assertTrue(any((output / split / "masks").glob("*.png")))


if __name__ == "__main__":
    unittest.main()

