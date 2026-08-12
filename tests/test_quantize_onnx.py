import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from surface_perception.quantize_onnx import (
    ManifestCalibrationReader,
    run_quantization_experiment,
)


HAS_ML_STACK = all(
    importlib.util.find_spec(module) is not None
    for module in ("onnx", "onnxruntime", "torch")
)


def _write_manifest(root: Path, image_size: int = 32) -> Path:
    records = []
    for split in ("validation", "test"):
        for index in range(3):
            image_path = root / split / "images" / f"sample_{index}.png"
            mask_path = root / split / "masks" / f"sample_{index}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            image[8:24, 8:24] = (80 + index * 25, 120, 160)
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            mask[10:22, 10:22] = 255
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            records.append(
                {
                    "sample_id": f"{split}_{index}",
                    "split": split,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "category": "panel",
                    "defect_type": "scratch",
                    "anomaly": True,
                }
            )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "1.0", "records": records}),
        encoding="utf-8",
    )
    return manifest_path


class CalibrationContractTest(unittest.TestCase):
    def test_test_split_cannot_be_used_for_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_manifest(Path(directory))
            with self.assertRaisesRegex(ValueError, "held-out test split"):
                ManifestCalibrationReader(manifest, "image", 32, split="test")

    @unittest.skipUnless(HAS_ML_STACK, "optional ONNX/PyTorch stack is not installed")
    def test_static_int8_smoke_experiment(self) -> None:
        import torch
        from torch import nn

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root)
            model_path = root / "model.onnx"
            model = nn.Sequential(
                nn.Conv2d(3, 4, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4, 1, 1),
            ).eval()
            sample = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
            torch.onnx.export(
                model,
                sample,
                model_path,
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "logits": {0: "batch", 2: "height", 3: "width"},
                },
                opset_version=18,
                dynamo=False,
            )
            training_report = root / "training_report.json"
            training_report.write_text(
                json.dumps({"selected_threshold": 0.5}), encoding="utf-8"
            )
            output = root / "optimized"
            report = run_quantization_experiment(
                model_path,
                manifest,
                training_report,
                output,
                image_size=32,
                calibration_samples=3,
                iterations=3,
                warmup=1,
                trials=1,
            )

            self.assertEqual(report["quantization"]["calibration_split"], "validation")
            self.assertEqual(report["quantization"]["calibration_samples"], 3)
            self.assertEqual(report["quality"]["samples"], 3)
            self.assertTrue((output / "model_int8_qdq.onnx").exists())
            self.assertTrue((output / "quantization_report.json").exists())


if __name__ == "__main__":
    unittest.main()
