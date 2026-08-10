import tempfile
import unittest
from pathlib import Path

import numpy as np

from surface_perception.linear_model import PixelLogisticSegmenter
from surface_perception.synthetic import make_surface_sample


class PixelLogisticSegmenterTest(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        samples = [make_surface_sample(32, seed) for seed in range(10, 18)]
        images = [item[0] for item in samples]
        masks = [item[1] for item in samples]
        model = PixelLogisticSegmenter()
        model.fit(images, masks, epochs=3, pixels_per_image=300, seed=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = PixelLogisticSegmenter.load(path)
            np.testing.assert_allclose(model.predict_proba(images[0]), loaded.predict_proba(images[0]))


if __name__ == "__main__":
    unittest.main()

