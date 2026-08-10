from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _base_image(size: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    gray = 145 + 13 * np.sin(yy * 0.55) + rng.normal(0, 5, (size, size)) + xx * 0.08
    rgb = np.stack([gray * 1.02, gray, gray * 0.98], axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def create_fixture(output_root: Path, *, size: int = 64) -> None:
    category_root = Path(output_root) / "metal_panel"
    for index in range(14):
        path = category_root / "train" / "good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        _base_image(size, 100 + index).save(path)
    for index in range(4):
        path = category_root / "test" / "good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        _base_image(size, 300 + index).save(path)

    for defect_type in ("scratch", "corrosion"):
        for index in range(12):
            image = _base_image(size, 500 + index + (100 if defect_type == "corrosion" else 0))
            mask = Image.new("L", (size, size), 0)
            image_draw = ImageDraw.Draw(image)
            mask_draw = ImageDraw.Draw(mask)
            if defect_type == "scratch":
                y = 10 + (index * 4) % max(1, size - 22)
                image_draw.line((8, y, size - 8, y + 5), fill=(45, 38, 32), width=3)
                mask_draw.line((8, y, size - 8, y + 5), fill=255, width=3)
            else:
                x = 10 + (index * 3) % max(1, size - 30)
                y = 12 + (index * 5) % max(1, size - 28)
                box = (x, y, x + 16, y + 12)
                image_draw.ellipse(box, fill=(118, 57, 25))
                mask_draw.ellipse(box, fill=255)
            image_path = category_root / "test" / defect_type / f"{index:03d}.png"
            mask_path = category_root / "ground_truth" / defect_type / f"{index:03d}_mask.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
            mask.save(mask_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny MVTec-layout fixture for ML CI")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    create_fixture(args.output, size=args.size)
    print(f"smoke fixture: {args.output.resolve()}")


if __name__ == "__main__":
    main()
