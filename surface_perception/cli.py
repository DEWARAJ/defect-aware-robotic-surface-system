from __future__ import annotations

import argparse
import json
from pathlib import Path

from .coverage_demo import run_coverage_demo
from .pipeline import evaluate_model, train_baseline
from .real_data import build_segmentation_manifest
from .sim_data import (
    load_sim_config,
    render_sim_plan_preview,
    save_sim_run_plan,
    validate_replicator_dataset,
)
from .synthetic import generate_dataset


def _load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_summary(report: dict) -> None:
    aggregate = report["aggregate"]
    latency = report["latency_ms"]
    print(
        "test metrics | "
        f"IoU={aggregate['iou']:.3f} "
        f"F1={aggregate['f1']:.3f} "
        f"precision={aggregate['precision']:.3f} "
        f"recall={aggregate['recall']:.3f} "
        f"P95 latency={latency['p95']:.2f} ms"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial surface-defect segmentation MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a synthetic surface dataset")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--samples", type=int, default=120)
    generate.add_argument("--size", type=int, default=96)
    generate.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="train the learned baseline")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--report", type=Path, required=True)
    train.add_argument("--config", type=Path, default=Path("configs/mvp.json"))

    evaluate = subparsers.add_parser("evaluate", help="evaluate a trained model")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", default="test", choices=("train", "validation", "test"))
    evaluate.add_argument("--preview-count", type=int, default=8)

    run_all = subparsers.add_parser("run-all", help="generate, train, and evaluate the MVP")
    run_all.add_argument("--workspace", type=Path, default=Path("runs/mvp"))
    run_all.add_argument("--config", type=Path, default=Path("configs/mvp.json"))

    plan_demo = subparsers.add_parser(
        "plan-demo", help="generate a safe, defect-aware robotic coverage-path demonstration"
    )
    plan_demo.add_argument("--output", type=Path, default=Path("runs/coverage_demo"))
    plan_demo.add_argument("--size", type=int, default=384)
    plan_demo.add_argument("--seed", type=int, default=11)
    plan_demo.add_argument("--tool-radius", type=int, default=6)
    plan_demo.add_argument("--lane-spacing", type=int, default=9)
    plan_demo.add_argument("--min-segment-length", type=int, default=12)

    manifest = subparsers.add_parser(
        "build-manifest", help="validate MVTec/custom data and create deterministic splits"
    )
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--mvtec-root", type=Path)
    manifest.add_argument("--custom-root", type=Path)
    manifest.add_argument("--categories", nargs="*")
    manifest.add_argument("--seed", type=int, default=42)

    sim_plan = subparsers.add_parser(
        "sim-plan", help="validate an Isaac Sim config and create its deterministic capture plan"
    )
    sim_plan.add_argument(
        "--config", type=Path, default=Path("configs/isaac_sim_surface.json")
    )
    sim_plan.add_argument("--output", type=Path, required=True)
    sim_plan.add_argument("--preview", type=Path)

    validate_sim = subparsers.add_parser(
        "validate-sim", help="validate Isaac Sim Replicator outputs and write a manifest"
    )
    validate_sim.add_argument(
        "--config", type=Path, default=Path("configs/isaac_sim_surface.json")
    )
    validate_sim.add_argument("--dataset", type=Path, required=True)
    validate_sim.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        records = generate_dataset(args.output, args.samples, args.size, args.seed)
        print(f"generated {len(records)} samples in {args.output}")
        return

    if args.command == "train":
        config = _load_config(args.config)
        report = train_baseline(
            args.dataset,
            args.model,
            args.report,
            config["training"],
            int(config["seed"]),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "evaluate":
        report = evaluate_model(
            args.dataset, args.model, args.output, args.split, args.preview_count
        )
        _print_summary(report)
        return

    if args.command == "plan-demo":
        report = run_coverage_demo(
            args.output,
            size=args.size,
            seed=args.seed,
            tool_radius=args.tool_radius,
            lane_spacing=args.lane_spacing,
            min_segment_length=args.min_segment_length,
        )
        metrics = report["metrics"]
        print(
            "coverage plan | "
            f"coverage={metrics['coverage_fraction']:.3f} "
            f"reachable_coverage={metrics['reachable_coverage_fraction']:.3f} "
            f"defect_coverage={metrics['defect_coverage_fraction']:.3f} "
            f"protected_contact_pixels={metrics['protected_contact_pixels']} "
            f"segments={metrics['segments']}"
        )
        print(f"artifacts: {args.output.resolve()}")
        return

    if args.command == "build-manifest":
        payload = build_segmentation_manifest(
            args.output,
            mvtec_root=args.mvtec_root,
            custom_root=args.custom_root,
            categories=args.categories,
            seed=args.seed,
        )
        print(json.dumps(payload["report"], indent=2, sort_keys=True))
        print(f"manifest: {args.output.resolve()}")
        return

    if args.command == "sim-plan":
        sim_config = load_sim_config(args.config)
        plan = save_sim_run_plan(sim_config, args.output)
        if args.preview is not None:
            render_sim_plan_preview(sim_config, args.preview)
            print(f"preview: {args.preview.resolve()}")
        print(
            json.dumps(
                {"plan": str(args.output.resolve()), "split_counts": plan["split_counts"]},
                indent=2,
            )
        )
        return

    if args.command == "validate-sim":
        sim_config = load_sim_config(args.config)
        payload = validate_replicator_dataset(args.dataset, sim_config, args.output)
        print(json.dumps(payload["report"], indent=2, sort_keys=True))
        print(f"manifest: {args.output.resolve()}")
        return

    config = _load_config(args.config)
    workspace = args.workspace
    dataset_dir = workspace / "dataset"
    model_path = workspace / "models" / "baseline_model.json"
    training_report_path = workspace / "reports" / "training_report.json"
    dataset_config = config["dataset"]
    generate_dataset(
        dataset_dir,
        samples=int(dataset_config["samples"]),
        image_size=int(dataset_config["image_size"]),
        seed=int(config["seed"]),
        train_fraction=float(dataset_config["train_fraction"]),
        validation_fraction=float(dataset_config["validation_fraction"]),
    )
    train_baseline(
        dataset_dir,
        model_path,
        training_report_path,
        config["training"],
        int(config["seed"]),
    )
    report = evaluate_model(
        dataset_dir,
        model_path,
        workspace / "reports",
        split="test",
        preview_count=int(config["evaluation"]["preview_count"]),
    )
    _print_summary(report)
    print(f"artifacts: {workspace.resolve()}")


if __name__ == "__main__":
    main()
