import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from surface_perception.real_data import build_segmentation_manifest, load_manifest


def _save_rgb(path: Path, value: int, size: tuple[int, int] = (40, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8)).save(path)


def _save_mask(path: Path, size: tuple[int, int] = (40, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[8:18, 12:25] = 255
    Image.fromarray(mask).save(path)


class RealDataManifestTest(unittest.TestCase):
    def test_mvtec_manifest_is_deterministic_and_leak_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "mvtec"
            category = root / "metal_nut"
            for index in range(8):
                _save_rgb(category / "train" / "good" / f"{index:03d}.png", 90 + index)
            for index in range(3):
                _save_rgb(category / "test" / "good" / f"{index:03d}.png", 120 + index)
            for index in range(7):
                _save_rgb(category / "test" / "scratch" / f"{index:03d}.png", 150 + index)
                _save_mask(category / "ground_truth" / "scratch" / f"{index:03d}_mask.png")

            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            first = build_segmentation_manifest(
                first_path, mvtec_root=root, categories=["metal_nut"], seed=17
            )
            second = build_segmentation_manifest(
                second_path, mvtec_root=root, categories=["metal_nut"], seed=17
            )
            self.assertEqual(first, second)
            self.assertEqual(first, load_manifest(first_path))
            self.assertIn("train", first["report"]["splits"])
            self.assertIn("validation", first["report"]["splits"])
            self.assertIn("test", first["report"]["splits"])
            paths = [record["image_path"] for record in first["records"]]
            self.assertEqual(len(paths), len(set(paths)))
            anomaly_splits = {
                record["split"] for record in first["records"] if record["anomaly"]
            }
            self.assertEqual(anomaly_splits, {"train", "validation", "test"})
            self.assertIn("not the official", first["report"]["benchmark_warning"])

    def test_custom_pairs_require_matching_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "custom"
            _save_rgb(root / "images" / "sample.png", 120)
            _save_mask(root / "masks" / "sample.png", size=(20, 20))
            with self.assertRaises(ValueError):
                build_segmentation_manifest(Path(directory) / "manifest.json", custom_root=root)


if __name__ == "__main__":
    unittest.main()
