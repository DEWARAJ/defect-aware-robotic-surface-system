import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from surface_perception.sim_data import (
    build_sim_run_plan,
    load_sim_config,
    validate_replicator_dataset,
    validate_sim_config,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "isaac_sim_surface.json"


def _fixture_config(frames: int = 6) -> dict:
    config = load_sim_config(CONFIG_PATH)
    config["frames"] = frames
    config["resolution"] = [64, 64]
    return config


def _write_frame(root: Path, index: int) -> None:
    rgb = np.full((64, 64, 3), 40 + index, dtype=np.uint8)
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[12:28, 18:34] = 10
    normals = np.zeros((64, 64, 3), dtype=np.uint8)
    normals[..., 2] = 255

    Image.fromarray(rgb).save(root / f"rgb_{index:06d}.png")
    Image.fromarray(labels).save(root / f"semantic_segmentation_{index:06d}.png")
    Image.fromarray(normals).save(root / f"normals_{index:06d}.png")
    np.save(root / f"distance_to_image_plane_{index:06d}.npy", np.ones((64, 64)))
    (root / f"camera_params_{index:06d}.json").write_text(
        json.dumps({"cameraViewTransform": np.eye(4).tolist()}),
        encoding="utf-8",
    )


class IsaacSimDataContractTest(unittest.TestCase):
    def test_capture_plan_is_deterministic_and_has_exact_splits(self) -> None:
        config = _fixture_config()
        first = build_sim_run_plan(config)
        second = build_sim_run_plan(config)

        self.assertEqual(first, second)
        self.assertEqual(first["split_counts"], {"test": 1, "train": 4, "validation": 1})
        self.assertEqual(len(first["frames"]), 6)

    def test_invalid_split_fractions_are_rejected(self) -> None:
        config = copy.deepcopy(_fixture_config())
        config["splits"] = {"train": 0.8, "validation": 0.2, "test": 0.2}
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            validate_sim_config(config)

    def test_complete_replicator_output_writes_validated_manifest(self) -> None:
        config = _fixture_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(config["frames"]):
                _write_frame(root, index)

            manifest_path = root / "validated_manifest.json"
            payload = validate_replicator_dataset(root, config, manifest_path)

            self.assertEqual(payload["status"], "validated")
            self.assertEqual(payload["report"]["frames"], 6)
            self.assertEqual(payload["report"]["missing_outputs"], 0)
            self.assertEqual(len(payload["records"]), 6)
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")), payload
            )

    def test_missing_modality_frame_is_rejected(self) -> None:
        config = _fixture_config(frames=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(config["frames"]):
                _write_frame(root, index)
            (root / "normals_000002.png").unlink()

            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_replicator_dataset(root, config)

    def test_numeric_depth_dimensions_are_validated(self) -> None:
        config = _fixture_config(frames=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(config["frames"]):
                _write_frame(root, index)
            np.save(root / "distance_to_image_plane_000001.npy", np.ones((32, 64)))

            with self.assertRaisesRegex(ValueError, "size mismatch"):
                validate_replicator_dataset(root, config)


if __name__ == "__main__":
    unittest.main()
