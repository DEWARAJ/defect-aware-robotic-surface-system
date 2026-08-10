from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .torch_model import build_tiny_unet


def train(
    dataset_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional ML dependencies before training Tiny U-Net.") from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class SurfaceDataset(Dataset):
        def __init__(self, root: Path, split: str) -> None:
            self.image_paths = sorted((root / split / "images").glob("*.png"))
            self.mask_dir = root / split / "masks"
            if not self.image_paths:
                raise FileNotFoundError(f"no images found for {split} split")

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, index: int):
            image_path = self.image_paths[index]
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
            mask = np.asarray(
                Image.open(self.mask_dir / image_path.name).convert("L"), dtype=np.float32
            )
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)[None, ...])
            return image_tensor, mask_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        SurfaceDataset(dataset_dir, "train"), batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        SurfaceDataset(dataset_dir, "validation"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = build_tiny_unet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    def loss_function(logits, targets):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return bce(logits, targets) + dice_loss

    def validation_iou() -> float:
        model.eval()
        intersection = 0
        union = 0
        with torch.no_grad():
            for images, targets in validation_loader:
                probabilities = torch.sigmoid(model(images.to(device)))
                predictions = probabilities >= 0.5
                truth = targets.to(device) > 0
                intersection += int(torch.logical_and(predictions, truth).sum().item())
                union += int(torch.logical_or(predictions, truth).sum().item())
        return float(intersection / union) if union else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        samples_seen = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.shape[0]
            samples_seen += images.shape[0]
        iou = validation_iou()
        epoch_report = {
            "epoch": epoch,
            "training_loss": running_loss / max(1, samples_seen),
            "validation_iou": iou,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True))
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), output_dir / "best.pt")

    report = {
        "model": "tiny_unet",
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_validation_iou": best_iou,
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny U-Net on the generated surface data")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

