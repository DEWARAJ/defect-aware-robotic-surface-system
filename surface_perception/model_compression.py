from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np

from .evaluate_real import evaluate_real_model
from .export_onnx import export_checkpoint
from .onnx_parity import verify_onnx_parity
from .torch_data import SegmentationManifestDataset
from .torch_model import build_segmentation_model, load_segmentation_checkpoint
from .train_real import _class_balance, _evaluate_thresholds, _manifest_hash


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for model compression. Install the optional ML dependencies."
        ) from exc
    return torch, nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_model(model) -> dict[str, int | float]:
    """Measure topology, serialized state size, and exact zero structure."""
    torch, nn = _torch_modules()
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    zero_parameters = sum(
        int(torch.count_nonzero(parameter == 0).item()) for parameter in parameters
    )
    convolution_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    spatial_layers = [
        module for module in convolution_layers if tuple(module.kernel_size) != (1, 1)
    ]
    inactive_filters = 0
    total_filters = 0
    for layer in convolution_layers:
        flattened = layer.weight.detach().reshape(layer.out_channels, -1)
        inactive_filters += int(torch.count_nonzero(flattened.abs().sum(dim=1) == 0).item())
        total_filters += int(layer.out_channels)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "convolution_layers": len(convolution_layers),
        "spatial_convolution_layers": len(spatial_layers),
        "serialized_state_dict_bytes": len(buffer.getvalue()),
        "zero_parameters": int(zero_parameters),
        "parameter_zero_fraction": float(zero_parameters / max(1, parameter_count)),
        "inactive_filters": int(inactive_filters),
        "total_filters": int(total_filters),
        "inactive_filter_fraction": float(inactive_filters / max(1, total_filters)),
    }


def structured_prune_filters(model, amount: float):
    """Return a copy with low-L1 convolution filters and paired BN channels disabled.

    The tensor shapes intentionally remain unchanged. This is an accuracy/sparsity ablation;
    dense runtimes should not be expected to accelerate it without a later graph compaction step.
    """
    torch, nn = _torch_modules()
    if not 0.0 <= amount < 1.0:
        raise ValueError("structured pruning amount must be in [0, 1)")
    pruned = copy.deepcopy(model)
    layers: list[dict] = []
    with torch.no_grad():
        for block_name, block in pruned.named_modules():
            if not isinstance(block, nn.Sequential):
                continue
            children = list(block.children())
            for index, convolution in enumerate(children):
                if not isinstance(convolution, nn.Conv2d) or convolution.out_channels < 2:
                    continue
                batch_norm = children[index + 1] if index + 1 < len(children) else None
                if not isinstance(batch_norm, nn.BatchNorm2d):
                    continue
                prune_count = int(round(convolution.out_channels * amount))
                prune_count = min(convolution.out_channels - 1, max(0, prune_count))
                if prune_count == 0:
                    continue
                importance = convolution.weight.abs().sum(dim=(1, 2, 3))
                indexes = torch.argsort(importance)[:prune_count]
                convolution.weight[indexes] = 0
                if convolution.bias is not None:
                    convolution.bias[indexes] = 0
                batch_norm.weight[indexes] = 0
                batch_norm.bias[indexes] = 0
                layers.append(
                    {
                        "layer": f"{block_name}.{index}",
                        "output_filters": int(convolution.out_channels),
                        "pruned_filters": prune_count,
                        "indexes": [int(value) for value in indexes.cpu().tolist()],
                    }
                )
    return pruned, {
        "method": "per-layer L1 structured output-filter pruning",
        "requested_fraction": float(amount),
        "shape_compacted": False,
        "layers": layers,
        "pruned_filters": sum(layer["pruned_filters"] for layer in layers),
    }


def supervised_segmentation_loss(logits, targets, positive_weight):
    torch, _ = _torch_modules()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight
    )
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def combined_distillation_loss(
    student_logits,
    targets,
    positive_weight,
    *,
    teacher_logits=None,
    alpha: float = 0.35,
    temperature: float = 2.0,
):
    torch, _ = _torch_modules()
    supervised = supervised_segmentation_loss(student_logits, targets, positive_weight)
    if teacher_logits is None or alpha == 0.0:
        return supervised, {
            "supervised": float(supervised.detach().item()),
            "distillation": 0.0,
        }
    if not 0.0 <= alpha < 1.0:
        raise ValueError("distillation alpha must be in [0, 1)")
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    soft_targets = torch.sigmoid(teacher_logits.detach() / temperature)
    distillation = torch.nn.functional.binary_cross_entropy_with_logits(
        student_logits / temperature, soft_targets
    ) * (temperature**2)
    total = (1.0 - alpha) * supervised + alpha * distillation
    return total, {
        "supervised": float(supervised.detach().item()),
        "distillation": float(distillation.detach().item()),
    }


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def train_compact_student(
    manifest_path: Path,
    output_dir: Path,
    *,
    teacher=None,
    image_size: int = 128,
    base_channels: int = 4,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    patience: int = 6,
    seed: int = 42,
    alpha: float = 0.35,
    temperature: float = 2.0,
) -> dict:
    torch, _ = _torch_modules()
    from torch.utils.data import DataLoader

    _seed_everything(seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_dataset = SegmentationManifestDataset(
        manifest_path, "train", image_size=image_size, augment=True, seed=seed
    )
    validation_dataset = SegmentationManifestDataset(
        manifest_path, "validation", image_size=image_size, augment=False, seed=seed
    )
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    balance = _class_balance(manifest_path, image_size)
    positive_weight = torch.tensor([balance["pos_weight"]], dtype=torch.float32, device=device)
    student = build_segmentation_model("compact_unet", base_channels).to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    effective_alpha = float(alpha if teacher is not None else 0.0)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    thresholds = np.linspace(0.20, 0.80, 25)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history: list[dict] = []
    best_iou = -1.0
    best_threshold = 0.5
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        training_dataset.set_epoch(epoch)
        student.train()
        loss_totals = {"total": 0.0, "supervised": 0.0, "distillation": 0.0}
        samples_seen = 0
        for batch in training_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                student_logits = student(images)
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                loss, parts = combined_distillation_loss(
                    student_logits,
                    targets,
                    positive_weight,
                    teacher_logits=teacher_logits,
                    alpha=effective_alpha,
                    temperature=temperature,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(images.shape[0])
            samples_seen += count
            loss_totals["total"] += float(loss.item()) * count
            loss_totals["supervised"] += parts["supervised"] * count
            loss_totals["distillation"] += parts["distillation"] * count

        threshold, validation = _evaluate_thresholds(
            student, validation_loader, device, thresholds, torch
        )
        validation_iou = float(validation["iou"])
        scheduler.step(validation_iou)
        epoch_report = {
            "epoch": epoch,
            "training_loss": {
                key: value / max(1, samples_seen) for key, value in loss_totals.items()
            },
            "validation": validation,
            "threshold": threshold,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True))
        if validation_iou > best_iou + 1e-6:
            best_iou = validation_iou
            best_threshold = threshold
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": student.state_dict(),
                    "threshold": best_threshold,
                    "model_config": {
                        "architecture": "compact_unet",
                        "base_channels": base_channels,
                    },
                    "training_config": {
                        "image_size": image_size,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "seed": seed,
                    },
                    "optimization": {
                        "mode": "knowledge_distillation" if teacher is not None else "supervised",
                        "teacher_guidance": teacher is not None,
                        "alpha": effective_alpha,
                        "temperature": float(temperature),
                    },
                    "manifest_sha256": _manifest_hash(manifest_path),
                    "best_validation_metrics": validation,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    saved_model, _ = load_segmentation_checkpoint(checkpoint_path, device="cpu")
    report = {
        "model": "compact_unet",
        "mode": "knowledge_distillation" if teacher is not None else "supervised",
        "device": str(device),
        "profile": profile_model(saved_model),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _manifest_hash(manifest_path),
        "class_balance": balance,
        "completed_epochs": len(history),
        "requested_epochs": int(epochs),
        "best_validation_iou": best_iou,
        "selected_threshold": best_threshold,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _save_pruned_teacher(
    teacher,
    metadata: dict,
    manifest_path: Path,
    output_dir: Path,
    *,
    amount: float,
    image_size: int,
    batch_size: int,
) -> tuple[Path, dict]:
    torch, _ = _torch_modules()
    from torch.utils.data import DataLoader

    model, pruning = structured_prune_filters(teacher, amount)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    validation_dataset = SegmentationManifestDataset(
        manifest_path, "validation", image_size=image_size, augment=False
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    threshold, validation = _evaluate_thresholds(
        model, validation_loader, device, np.linspace(0.20, 0.80, 25), torch
    )
    payload = {
        key: value for key, value in metadata.items() if key not in {"state_dict", "threshold"}
    }
    payload.update(
        {
            "state_dict": model.to("cpu").state_dict(),
            "threshold": threshold,
            "best_validation_metrics": validation,
            "optimization": {
                "mode": "structured_pruning",
                **pruning,
                "validation_metrics_after_pruning": validation,
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    torch.save(payload, checkpoint_path)
    report = {
        "mode": "structured_pruning",
        "selected_threshold": threshold,
        "validation": validation,
        "profile": profile_model(model),
        "pruning": pruning,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    (output_dir / "pruning_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checkpoint_path, report


def benchmark_onnx_candidates(
    model_paths: dict[str, Path],
    *,
    image_size: int,
    iterations: int = 100,
    warmup: int = 10,
    trials: int = 3,
    seed: int = 42,
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime is required for the compression benchmark.") from exc
    if iterations < 1 or warmup < 0 or trials < 1:
        raise ValueError("iterations/trials must be positive and warmup cannot be negative")
    available = set(ort.get_available_providers())
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    sessions = {
        name: ort.InferenceSession(str(path), providers=providers)
        for name, path in model_paths.items()
    }
    sample = np.random.default_rng(seed).random(
        (1, 3, image_size, image_size), dtype=np.float32
    )
    for session in sessions.values():
        input_name = session.get_inputs()[0].name
        for _ in range(warmup):
            session.run(None, {input_name: sample})
    rng = random.Random(seed)
    latencies: dict[str, list[float]] = {name: [] for name in sessions}
    trial_reports: list[dict] = []
    for trial in range(1, trials + 1):
        order = sorted(sessions)
        rng.shuffle(order)
        trial_report = {"trial": trial, "order": order, "latency_ms": {}}
        for name in order:
            session = sessions[name]
            input_name = session.get_inputs()[0].name
            values = []
            for _ in range(iterations):
                started = time.perf_counter()
                session.run(None, {input_name: sample})
                values.append((time.perf_counter() - started) * 1000.0)
            latencies[name].extend(values)
            trial_report["latency_ms"][name] = float(np.mean(values))
        trial_reports.append(trial_report)
    results = {}
    for name, values in latencies.items():
        mean = float(np.mean(values))
        results[name] = {
            "model": str(Path(model_paths[name]).resolve()),
            "model_bytes": int(Path(model_paths[name]).stat().st_size),
            "model_sha256": _sha256(model_paths[name]),
            "providers": sessions[name].get_providers(),
            "latency_ms": {
                "mean": mean,
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            },
            "fps_from_mean_latency": float(1000.0 / mean),
        }
    return {
        "method": "seeded randomized trial order with a shared input tensor",
        "image_size": int(image_size),
        "iterations_per_trial": int(iterations),
        "warmup_per_model": int(warmup),
        "trials": trial_reports,
        "results": results,
    }


def _evaluate_candidate(
    name: str,
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    image_size: int,
    parity_samples: int,
) -> tuple[dict, Path]:
    candidate_dir = output_dir / name
    evaluation = evaluate_real_model(
        manifest_path,
        checkpoint_path,
        candidate_dir / "evaluation",
        split="test",
        image_size=image_size,
        preview_count=4,
    )
    onnx_path = candidate_dir / "model.onnx"
    export_checkpoint(checkpoint_path, onnx_path, image_size=image_size)
    parity = verify_onnx_parity(
        checkpoint_path,
        onnx_path,
        manifest_path,
        split="test",
        samples=parity_samples,
        image_size=image_size,
    )
    (candidate_dir / "onnx_parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True), encoding="utf-8"
    )
    model, metadata = load_segmentation_checkpoint(checkpoint_path, device="cpu")
    validation = metadata.get("best_validation_metrics", {})
    if not validation and metadata.get("optimization", {}).get(
        "validation_metrics_after_pruning"
    ):
        validation = metadata["optimization"]["validation_metrics_after_pruning"]
    record = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_bytes": int(Path(checkpoint_path).stat().st_size),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_config": metadata.get("model_config", {}),
        "profile": profile_model(model),
        "validation": validation,
        "held_out_test": evaluation["aggregate"],
        "test_images": evaluation["samples"],
        "selected_threshold": evaluation["threshold"],
        "pytorch_latency_ms": evaluation["latency_ms"],
        "onnx_parity": {
            key: parity[key]
            for key in (
                "samples",
                "maximum_absolute_logit_error",
                "minimum_binary_mask_agreement",
                "pass",
            )
        },
    }
    return record, onnx_path


def _validate_config(config: dict) -> dict:
    defaults = {
        "schema_version": "1.0",
        "image_size": 128,
        "student_base_channels": 4,
        "student_epochs": 20,
        "batch_size": 8,
        "learning_rate": 0.001,
        "patience": 6,
        "seed": 42,
        "distillation_alpha": 0.35,
        "distillation_temperature": 2.0,
        "pruning_fraction": 0.25,
        "max_absolute_validation_iou_drop": 0.03,
        "max_absolute_test_iou_drop": 0.03,
        "onnx_iterations": 100,
        "onnx_warmup": 10,
        "onnx_trials": 3,
        "parity_samples": 10,
    }
    merged = {**defaults, **config}
    if str(merged["schema_version"]) != "1.0":
        raise ValueError("unsupported compression config schema_version")
    if int(merged["student_base_channels"]) < 2:
        raise ValueError("student_base_channels must be at least 2")
    if int(merged["student_epochs"]) < 1:
        raise ValueError("student_epochs must be positive")
    if not 0.0 <= float(merged["distillation_alpha"]) < 1.0:
        raise ValueError("distillation_alpha must be in [0, 1)")
    if float(merged["distillation_temperature"]) <= 0:
        raise ValueError("distillation_temperature must be positive")
    if not 0.0 <= float(merged["pruning_fraction"]) < 1.0:
        raise ValueError("pruning_fraction must be in [0, 1)")
    for key in ("max_absolute_validation_iou_drop", "max_absolute_test_iou_drop"):
        if float(merged[key]) < 0:
            raise ValueError(f"{key} cannot be negative")
    return merged


def load_compression_config(path: Path) -> dict:
    config_path = Path(path)
    payload = _validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    for key in ("manifest", "teacher_checkpoint", "output"):
        value = Path(payload[key])
        payload[key] = value if value.is_absolute() else (config_path.parent / value).resolve()
    return payload


def select_deployment_candidate(
    candidates: dict[str, dict],
    *,
    maximum_validation_iou_drop: float,
    maximum_test_iou_drop: float,
) -> dict:
    """Select on validation, then apply test once as a fail-safe qualification gate."""
    teacher_validation_iou = float(candidates["teacher"]["validation"]["iou"])
    teacher_test_iou = float(candidates["teacher"]["held_out_test"]["iou"])
    minimum_validation_iou = teacher_validation_iou - float(maximum_validation_iou_drop)
    minimum_test_iou = teacher_test_iou - float(maximum_test_iou_drop)
    compact_names = ("student_supervised", "student_distilled")
    for candidate in candidates.values():
        candidate["validation_quality_gate_pass"] = bool(
            float(candidate["validation"]["iou"]) >= minimum_validation_iou
        )
        candidate["held_out_quality_guardrail_pass"] = bool(
            float(candidate["held_out_test"]["iou"]) >= minimum_test_iou
        )
    eligible = [
        name for name in compact_names if candidates[name]["validation_quality_gate_pass"]
    ]
    if eligible:
        provisional = min(
            eligible,
            key=lambda name: (
                candidates[name]["profile"]["parameter_count"],
                -float(candidates[name]["validation"]["iou"]),
                candidates[name]["onnx_benchmark"]["latency_ms"]["mean"],
            ),
        )
        if candidates[provisional]["held_out_quality_guardrail_pass"]:
            selected = provisional
            rationale = (
                "smallest validation-eligible compact candidate; passed the held-out guardrail"
            )
        else:
            selected = "teacher"
            rationale = (
                f"{provisional} passed validation selection but failed the held-out guardrail"
            )
    else:
        provisional = "teacher"
        selected = "teacher"
        rationale = "compact candidates exceeded the configured validation IoU budget"
    return {
        "provisional_candidate_from_validation": provisional,
        "selected_candidate": selected,
        "maximum_absolute_validation_iou_drop": float(maximum_validation_iou_drop),
        "minimum_accepted_validation_iou": minimum_validation_iou,
        "maximum_absolute_test_iou_drop": float(maximum_test_iou_drop),
        "minimum_accepted_test_iou": minimum_test_iou,
        "rationale": rationale,
    }


def run_compression_study(config: dict) -> dict:
    config = _validate_config(config)
    manifest_path = Path(config["manifest"])
    teacher_checkpoint = Path(config["teacher_checkpoint"])
    output_dir = Path(config["output"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(f"teacher checkpoint does not exist: {teacher_checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher, teacher_metadata = load_segmentation_checkpoint(teacher_checkpoint, device="cpu")
    configured_size = int(
        teacher_metadata.get("training_config", {}).get("image_size", config["image_size"])
    )
    image_size = int(config.get("image_size") or configured_size)
    teacher_profile = profile_model(teacher)

    supervised_report = train_compact_student(
        manifest_path,
        output_dir / "student_supervised",
        image_size=image_size,
        base_channels=int(config["student_base_channels"]),
        epochs=int(config["student_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        patience=int(config["patience"]),
        seed=int(config["seed"]),
    )
    teacher_for_distillation, _ = load_segmentation_checkpoint(
        teacher_checkpoint, device="cpu"
    )
    distilled_report = train_compact_student(
        manifest_path,
        output_dir / "student_distilled",
        teacher=teacher_for_distillation,
        image_size=image_size,
        base_channels=int(config["student_base_channels"]),
        epochs=int(config["student_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        patience=int(config["patience"]),
        seed=int(config["seed"]),
        alpha=float(config["distillation_alpha"]),
        temperature=float(config["distillation_temperature"]),
    )
    pruned_checkpoint, pruning_report = _save_pruned_teacher(
        teacher,
        teacher_metadata,
        manifest_path,
        output_dir / "teacher_pruned",
        amount=float(config["pruning_fraction"]),
        image_size=image_size,
        batch_size=int(config["batch_size"]),
    )
    checkpoints = {
        "teacher": teacher_checkpoint,
        "student_supervised": Path(supervised_report["checkpoint"]),
        "student_distilled": Path(distilled_report["checkpoint"]),
        "teacher_pruned": pruned_checkpoint,
    }
    candidates: dict[str, dict] = {}
    onnx_paths: dict[str, Path] = {}
    for name, checkpoint in checkpoints.items():
        candidates[name], onnx_paths[name] = _evaluate_candidate(
            name,
            checkpoint,
            manifest_path,
            output_dir,
            image_size=image_size,
            parity_samples=int(config["parity_samples"]),
        )
    benchmark = benchmark_onnx_candidates(
        onnx_paths,
        image_size=image_size,
        iterations=int(config["onnx_iterations"]),
        warmup=int(config["onnx_warmup"]),
        trials=int(config["onnx_trials"]),
        seed=int(config["seed"]),
    )
    for name, result in benchmark["results"].items():
        candidates[name]["onnx_benchmark"] = result

    teacher_validation_iou = float(candidates["teacher"]["validation"]["iou"])
    teacher_test_iou = float(candidates["teacher"]["held_out_test"]["iou"])
    teacher_parameters = int(candidates["teacher"]["profile"]["parameter_count"])
    teacher_bytes = int(candidates["teacher"]["onnx_benchmark"]["model_bytes"])
    for name, candidate in candidates.items():
        candidate_iou = float(candidate["held_out_test"]["iou"])
        parameters = int(candidate["profile"]["parameter_count"])
        model_bytes = int(candidate["onnx_benchmark"]["model_bytes"])
        candidate["relative_to_teacher"] = {
            "validation_iou_delta": (
                float(candidate["validation"]["iou"]) - teacher_validation_iou
            ),
            "test_iou_delta": candidate_iou - teacher_test_iou,
            "test_iou_retention_fraction": (
                candidate_iou / teacher_test_iou if teacher_test_iou > 0 else None
            ),
            "parameter_reduction_fraction": 1.0 - parameters / teacher_parameters,
            "onnx_size_reduction_fraction": 1.0 - model_bytes / teacher_bytes,
            "mean_latency_speedup": (
                candidates["teacher"]["onnx_benchmark"]["latency_ms"]["mean"]
                / candidate["onnx_benchmark"]["latency_ms"]["mean"]
            ),
        }
    selection = select_deployment_candidate(
        candidates,
        maximum_validation_iou_drop=float(config["max_absolute_validation_iou_drop"]),
        maximum_test_iou_drop=float(config["max_absolute_test_iou_drop"]),
    )
    report = {
        "schema_version": "1.0",
        "project_version": "0.10.0",
        "study": "architecture reduction, structured pruning, and knowledge distillation",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _manifest_hash(manifest_path),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": _sha256(teacher_checkpoint),
        "teacher_profile_before_study": teacher_profile,
        "config": {
            key: value
            for key, value in config.items()
            if key not in {"manifest", "teacher_checkpoint", "output"}
        },
        "candidates": candidates,
        "benchmark_protocol": {
            key: value for key, value in benchmark.items() if key != "results"
        },
        "selection": selection,
        "ablation": {
            "supervised_vs_distilled_iou_delta": (
                candidates["student_distilled"]["held_out_test"]["iou"]
                - candidates["student_supervised"]["held_out_test"]["iou"]
            ),
            "pruned_vs_teacher_iou_delta": (
                candidates["teacher_pruned"]["held_out_test"]["iou"]
                - candidates["teacher"]["held_out_test"]["iou"]
            ),
            "pruning": pruning_report,
        },
        "limitations": [
            "The supervised-development MVTec protocol is not the official unsupervised benchmark.",
            "Structured pruning leaves dense tensor shapes unchanged and does not imply speedup.",
            "Latency is environment-specific and excludes camera, ROS, planning, and control.",
            "No production or aircraft-surface performance claim is made.",
        ],
    }
    report_path = output_dir / "compression_study.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare compact training, distillation, and structured pruning"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--student-base-channels", type=int)
    parser.add_argument("--student-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--onnx-iterations", type=int)
    parser.add_argument("--onnx-trials", type=int)
    parser.add_argument("--parity-samples", type=int)
    args = parser.parse_args()
    config = load_compression_config(args.config)
    for key in (
        "manifest",
        "teacher_checkpoint",
        "output",
        "image_size",
        "student_base_channels",
        "student_epochs",
        "batch_size",
        "onnx_iterations",
        "onnx_trials",
        "parity_samples",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    report = run_compression_study(config)
    print(json.dumps(report["selection"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
