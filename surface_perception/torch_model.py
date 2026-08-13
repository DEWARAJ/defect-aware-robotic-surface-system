from __future__ import annotations


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the optional ML dependencies with "
            "'python -m pip install -e .[ml]'."
        ) from exc
    return torch, nn


def _conv_block(nn, input_channels: int, output_channels: int):
    class ConvBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs):
            return self.layers(inputs)

    return ConvBlock()


def build_tiny_unet(base_channels: int = 16):
    """Build the two-level teacher CNN while keeping PyTorch optional."""
    torch, nn = _torch_modules()

    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_1 = _conv_block(nn, 3, base_channels)
            self.encoder_2 = _conv_block(nn, base_channels, base_channels * 2)
            self.bottleneck = _conv_block(nn, base_channels * 2, base_channels * 4)
            self.decoder_2 = _conv_block(nn, base_channels * 6, base_channels * 2)
            self.decoder_1 = _conv_block(nn, base_channels * 3, base_channels)
            self.output = nn.Conv2d(base_channels, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, inputs):
            feature_1 = self.encoder_1(inputs)
            feature_2 = self.encoder_2(self.pool(feature_1))
            bottleneck = self.bottleneck(self.pool(feature_2))
            up_2 = torch.nn.functional.interpolate(
                bottleneck, size=feature_2.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded_2 = self.decoder_2(torch.cat([up_2, feature_2], dim=1))
            up_1 = torch.nn.functional.interpolate(
                decoded_2, size=feature_1.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded_1 = self.decoder_1(torch.cat([up_1, feature_1], dim=1))
            return self.output(decoded_1)

    return TinyUNet()


def build_compact_unet(base_channels: int = 8):
    """Build a one-level student U-Net with four fewer spatial convolutions."""
    torch, nn = _torch_modules()

    class CompactUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _conv_block(nn, 3, base_channels)
            self.bottleneck = _conv_block(nn, base_channels, base_channels * 2)
            self.decoder = _conv_block(nn, base_channels * 3, base_channels)
            self.output = nn.Conv2d(base_channels, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, inputs):
            feature = self.encoder(inputs)
            bottleneck = self.bottleneck(self.pool(feature))
            upsampled = torch.nn.functional.interpolate(
                bottleneck, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = self.decoder(torch.cat([upsampled, feature], dim=1))
            return self.output(decoded)

    return CompactUNet()


def build_segmentation_model(architecture: str = "tiny_unet", base_channels: int = 16):
    builders = {
        "tiny_unet": build_tiny_unet,
        "compact_unet": build_compact_unet,
    }
    if architecture not in builders:
        raise ValueError(
            f"unsupported segmentation architecture '{architecture}'; "
            f"choose one of {sorted(builders)}"
        )
    if int(base_channels) < 2:
        raise ValueError("base_channels must be at least 2")
    return builders[architecture](base_channels=int(base_channels))


def load_segmentation_checkpoint(checkpoint_path, device="cpu"):
    """Load legacy Tiny U-Net and architecture-aware metadata checkpoints."""
    torch, _ = _torch_modules()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        model_config = payload.get("model_config", {})
        base_channels = int(model_config.get("base_channels", 16))
        architecture = str(model_config.get("architecture", "tiny_unet"))
        metadata = payload
    else:
        state_dict = payload
        base_channels = 16
        architecture = "tiny_unet"
        metadata = {"threshold": 0.5, "model_config": {"base_channels": base_channels}}
    model = build_segmentation_model(architecture, base_channels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata


def load_tiny_unet_checkpoint(checkpoint_path, device="cpu"):
    """Backward-compatible alias for callers written before architecture-aware checkpoints."""
    return load_segmentation_checkpoint(checkpoint_path, device=device)
