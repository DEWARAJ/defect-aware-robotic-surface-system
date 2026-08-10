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


def build_tiny_unet(base_channels: int = 16):
    """Build the Phase 2 CNN while keeping PyTorch optional for the runnable MVP."""
    torch, nn = _torch_modules()

    class ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
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

    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_1 = ConvBlock(3, base_channels)
            self.encoder_2 = ConvBlock(base_channels, base_channels * 2)
            self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
            self.decoder_2 = ConvBlock(base_channels * 6, base_channels * 2)
            self.decoder_1 = ConvBlock(base_channels * 3, base_channels)
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


def load_tiny_unet_checkpoint(checkpoint_path, device="cpu"):
    """Load both the v0.2 raw state dictionary and v0.3 metadata checkpoints."""
    torch, _ = _torch_modules()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        model_config = payload.get("model_config", {})
        base_channels = int(model_config.get("base_channels", 16))
        metadata = payload
    else:
        state_dict = payload
        base_channels = 16
        metadata = {"threshold": 0.5, "model_config": {"base_channels": base_channels}}
    model = build_tiny_unet(base_channels=base_channels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata
