import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from surface_perception.model_compression import _validate_config, select_deployment_candidate


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class CompressionConfigTest(unittest.TestCase):
    def test_defaults_and_quality_budget_are_validated(self) -> None:
        config = _validate_config(
            {
                "manifest": "manifest.json",
                "teacher_checkpoint": "teacher.pt",
                "output": "output",
            }
        )
        self.assertEqual(config["student_base_channels"], 4)
        self.assertAlmostEqual(config["max_absolute_validation_iou_drop"], 0.03)
        self.assertAlmostEqual(config["max_absolute_test_iou_drop"], 0.03)
        with self.assertRaisesRegex(ValueError, "distillation_alpha"):
            _validate_config({**config, "distillation_alpha": 1.0})
        with self.assertRaisesRegex(ValueError, "pruning_fraction"):
            _validate_config({**config, "pruning_fraction": -0.1})
        with self.assertRaisesRegex(ValueError, "max_absolute_test_iou_drop"):
            _validate_config({**config, "max_absolute_test_iou_drop": -0.1})

    def test_selection_uses_validation_then_test_only_as_guardrail(self) -> None:
        def candidate(validation_iou, test_iou, parameters=100, latency=1.0):
            return {
                "validation": {"iou": validation_iou},
                "held_out_test": {"iou": test_iou},
                "profile": {"parameter_count": parameters},
                "onnx_benchmark": {"latency_ms": {"mean": latency}},
            }

        candidates = {
            "teacher": candidate(0.30, 0.30, parameters=300),
            "student_supervised": candidate(0.29, 0.10),
            "student_distilled": candidate(0.27, 0.29),
            "teacher_pruned": candidate(0.20, 0.20, parameters=300),
        }
        selection = select_deployment_candidate(
            candidates,
            maximum_validation_iou_drop=0.05,
            maximum_test_iou_drop=0.03,
        )
        self.assertEqual(
            selection["provisional_candidate_from_validation"], "student_supervised"
        )
        self.assertEqual(selection["selected_candidate"], "teacher")
        self.assertNotEqual(selection["selected_candidate"], "student_distilled")

    def test_reference_evidence_is_self_consistent(self) -> None:
        evidence_path = (
            Path(__file__).parents[1]
            / "artifacts"
            / "reference"
            / "model_compression_v10.json"
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        teacher = evidence["candidates"]["teacher"]
        student = evidence["candidates"]["student_distilled"]
        relative = student["relative_to_teacher"]
        self.assertEqual(evidence["selection"]["selected_candidate"], "student_distilled")
        self.assertTrue(evidence["selection"]["held_out_guardrail_pass"])
        self.assertAlmostEqual(
            relative["test_iou_delta"],
            student["test"]["iou"] - teacher["test"]["iou"],
        )
        self.assertAlmostEqual(
            relative["parameter_reduction_fraction"],
            1.0 - student["parameters"] / teacher["parameters"],
        )
        self.assertAlmostEqual(
            relative["onnx_size_reduction_fraction"],
            1.0 - student["onnx"]["bytes"] / teacher["onnx"]["bytes"],
        )
        self.assertAlmostEqual(
            relative["mean_latency_speedup"],
            teacher["onnx"]["mean_latency_ms"] / student["onnx"]["mean_latency_ms"],
        )
        self.assertGreater(student["test"]["iou"], teacher["test"]["iou"])
        self.assertEqual(student["onnx"]["minimum_binary_mask_agreement"], 1.0)


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class CompressionTorchTest(unittest.TestCase):
    def test_compact_model_reduces_layers_parameters_and_state_size(self) -> None:
        from surface_perception.model_compression import profile_model
        from surface_perception.torch_model import build_compact_unet, build_tiny_unet

        teacher = build_tiny_unet(base_channels=4)
        student = build_compact_unet(base_channels=4)
        teacher_profile = profile_model(teacher)
        student_profile = profile_model(student)
        self.assertEqual(teacher_profile["spatial_convolution_layers"], 10)
        self.assertEqual(student_profile["spatial_convolution_layers"], 6)
        self.assertLess(student_profile["parameter_count"], teacher_profile["parameter_count"])
        self.assertLess(
            student_profile["serialized_state_dict_bytes"],
            teacher_profile["serialized_state_dict_bytes"],
        )

    def test_architecture_aware_checkpoint_roundtrip(self) -> None:
        import torch

        from surface_perception.torch_model import (
            build_compact_unet,
            load_segmentation_checkpoint,
        )

        model = build_compact_unet(base_channels=4).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "student.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "threshold": 0.57,
                    "model_config": {
                        "architecture": "compact_unet",
                        "base_channels": 4,
                    },
                },
                checkpoint,
            )
            restored, metadata = load_segmentation_checkpoint(checkpoint)
            result = restored(torch.zeros(1, 3, 32, 32))
            self.assertEqual(tuple(result.shape), (1, 1, 32, 32))
            self.assertAlmostEqual(metadata["threshold"], 0.57)

    def test_historical_tiny_unet_checkpoint_keys_remain_compatible(self) -> None:
        import torch

        from surface_perception.torch_model import build_tiny_unet, load_segmentation_checkpoint

        model = build_tiny_unet(base_channels=4)
        self.assertIn("encoder_1.layers.0.weight", model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "historical.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "threshold": 0.5,
                    "model_config": {"architecture": "tiny_unet", "base_channels": 4},
                },
                checkpoint,
            )
            restored, _ = load_segmentation_checkpoint(checkpoint)
            self.assertEqual(
                tuple(restored(torch.zeros(1, 3, 32, 32)).shape),
                (1, 1, 32, 32),
            )

    def test_structured_pruning_disables_complete_filters(self) -> None:
        from surface_perception.model_compression import (
            profile_model,
            structured_prune_filters,
        )
        from surface_perception.torch_model import build_tiny_unet

        teacher = build_tiny_unet(base_channels=4)
        pruned, report = structured_prune_filters(teacher, 0.25)
        profile = profile_model(pruned)
        self.assertGreater(report["pruned_filters"], 0)
        self.assertGreater(profile["inactive_filters"], 0)
        self.assertGreater(profile["parameter_zero_fraction"], 0.0)
        self.assertFalse(report["shape_compacted"])

    def test_distillation_loss_uses_soft_teacher_targets(self) -> None:
        import torch

        from surface_perception.model_compression import combined_distillation_loss

        student_logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
        teacher_logits = torch.full((1, 1, 8, 8), 2.0)
        targets = torch.zeros(1, 1, 8, 8)
        loss, parts = combined_distillation_loss(
            student_logits,
            targets,
            torch.tensor([2.0]),
            teacher_logits=teacher_logits,
            alpha=0.4,
            temperature=2.0,
        )
        loss.backward()
        self.assertGreater(parts["distillation"], 0.0)
        self.assertIsNotNone(student_logits.grad)


if __name__ == "__main__":
    unittest.main()
