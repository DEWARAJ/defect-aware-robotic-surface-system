from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .real_data import load_manifest
from .torch_data import load_record_arrays
from .torch_model import load_tiny_unet_checkpoint


def verify_onnx_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    manifest_path: Path,
    *,
    split: str = "test",
    samples: int = 10,
    image_size: int | None = None,
) -> dict:
    try:
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch and ONNX Runtime are required for parity verification.") from exc

    model, checkpoint = load_tiny_unet_checkpoint(checkpoint_path, device="cpu")
    configured_size = int(checkpoint.get("training_config", {}).get("image_size", 256))
    image_size = int(image_size or configured_size)
    threshold = float(checkpoint.get("threshold", 0.5))
    available = set(ort.get_available_providers())
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    records = [
        record for record in load_manifest(manifest_path)["records"] if record["split"] == split
    ][:samples]
    if not records:
        raise ValueError(f"manifest contains no records for split '{split}'")

    comparisons = []
    for record in records:
        image, _ = load_record_arrays(record, image_size=image_size)
        batch = image[None, ...].astype(np.float32)
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(batch)).cpu().numpy()
        onnx_logits = session.run(None, {input_name: batch})[0]
        difference = np.abs(torch_logits - onnx_logits)
        torch_probability = 1.0 / (1.0 + np.exp(-np.clip(torch_logits, -30, 30)))
        onnx_probability = 1.0 / (1.0 + np.exp(-np.clip(onnx_logits, -30, 30)))
        mask_agreement = float(
            np.mean((torch_probability >= threshold) == (onnx_probability >= threshold))
        )
        comparisons.append(
            {
                "sample_id": record["sample_id"],
                "maximum_absolute_logit_error": float(difference.max()),
                "mean_absolute_logit_error": float(difference.mean()),
                "binary_mask_agreement": mask_agreement,
            }
        )

    report = {
        "samples": len(comparisons),
        "image_size": image_size,
        "providers": session.get_providers(),
        "maximum_absolute_logit_error": max(
            item["maximum_absolute_logit_error"] for item in comparisons
        ),
        "mean_absolute_logit_error": float(
            np.mean([item["mean_absolute_logit_error"] for item in comparisons])
        ),
        "minimum_binary_mask_agreement": min(
            item["binary_mask_agreement"] for item in comparisons
        ),
        "pass": bool(
            max(item["maximum_absolute_logit_error"] for item in comparisons) <= 1e-4
            and min(item["binary_mask_agreement"] for item in comparisons) >= 0.999
        ),
        "per_sample": comparisons,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PyTorch and ONNX numerical parity")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_onnx_parity(
        args.checkpoint,
        args.onnx,
        args.manifest,
        split=args.split,
        samples=args.samples,
        image_size=args.image_size,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not report["pass"]:
        raise SystemExit("ONNX parity thresholds were not met")


if __name__ == "__main__":
    main()
