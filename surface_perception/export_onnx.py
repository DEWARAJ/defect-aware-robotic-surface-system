from __future__ import annotations

import argparse
from pathlib import Path

from .torch_model import load_tiny_unet_checkpoint


def export_checkpoint(checkpoint: Path, output: Path, image_size: int = 256) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional ML dependencies before exporting ONNX.") from exc

    model, _ = load_tiny_unet_checkpoint(checkpoint, device="cpu")
    sample = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the Tiny U-Net checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, args.image_size)
    print(f"exported ONNX model: {args.output}")


if __name__ == "__main__":
    main()
