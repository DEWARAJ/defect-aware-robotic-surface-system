import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class OptionalTorchModelTest(unittest.TestCase):
    def test_tiny_unet_preserves_spatial_shape(self) -> None:
        import torch

        from surface_perception.torch_model import build_tiny_unet

        model = build_tiny_unet(base_channels=4)
        output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 1, 64, 64))

    def test_metadata_checkpoint_roundtrip(self) -> None:
        import torch

        from surface_perception.torch_model import build_tiny_unet, load_tiny_unet_checkpoint

        model = build_tiny_unet(base_channels=4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "threshold": 0.43,
                    "model_config": {"architecture": "tiny_unet", "base_channels": 4},
                },
                checkpoint,
            )
            restored, metadata = load_tiny_unet_checkpoint(checkpoint)
            output = restored(torch.zeros(1, 3, 32, 32))
            self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
            self.assertAlmostEqual(float(metadata["threshold"]), 0.43)


if __name__ == "__main__":
    unittest.main()
