"""Generate an aircraft-like surface dataset with NVIDIA Isaac Sim Replicator.

Run this script with Isaac Sim's Python interpreter. A dry run works with normal Python and
creates the deterministic capture contract plus a configuration preview without launching Isaac.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from surface_perception.sim_data import (  # noqa: E402
    load_sim_config,
    render_sim_plan_preview,
    save_sim_run_plan,
    validate_replicator_dataset,
    validate_sim_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate randomized surface data in Isaac Sim")
    parser.add_argument("--config", type=Path, default=Path("configs/isaac_sim_surface.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/isaac_sim_surface"))
    parser.add_argument("--frames", type=int, help="override the configured frame count")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and write the capture plan without launching Isaac Sim",
    )
    return parser


def _bounds(panel: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    size_x, size_y, size_z = panel["size_m"]
    margin = 0.08
    return (
        (-size_x / 2 + margin, -size_y / 2 + margin, size_z / 2 + 0.006),
        (size_x / 2 - margin, size_y / 2 - margin, size_z / 2 + 0.012),
    )


def _create_scene(rep, config: dict, output: Path) -> None:
    panel = config["panel"]
    size_x, size_y, size_z = panel["size_m"]
    lower, upper = _bounds(panel)

    panel_materials = rep.create.material_omnipbr(
        diffuse=rep.distribution.uniform(
            tuple(panel["diffuse_min"]), tuple(panel["diffuse_max"])
        ),
        roughness=rep.distribution.uniform(*panel["roughness"]),
        metallic=rep.distribution.uniform(*panel["metallic"]),
        count=24,
    )
    scratch_materials = rep.create.material_omnipbr(
        diffuse=rep.distribution.uniform((0.18, 0.04, 0.04), (0.48, 0.18, 0.12)),
        roughness=rep.distribution.uniform(0.35, 0.90),
        metallic=rep.distribution.uniform(0.05, 0.50),
        count=12,
    )
    corrosion_materials = rep.create.material_omnipbr(
        diffuse=rep.distribution.uniform((0.24, 0.08, 0.01), (0.72, 0.36, 0.08)),
        roughness=rep.distribution.uniform(0.55, 1.0),
        metallic=rep.distribution.uniform(0.0, 0.25),
        count=12,
    )
    pit_materials = rep.create.material_omnipbr(
        diffuse=rep.distribution.uniform((0.005, 0.005, 0.008), (0.12, 0.13, 0.15)),
        roughness=rep.distribution.uniform(0.25, 0.90),
        metallic=rep.distribution.uniform(0.15, 0.80),
        count=12,
    )

    surface = rep.create.cube(
        name="AircraftLikePanel",
        position=(0.0, 0.0, 0.0),
        scale=(size_x, size_y, size_z),
        semantics=[("class", "panel")],
    )
    with surface:
        rep.randomizer.materials(panel_materials)

    scratch = config["defects"]["scratch"]
    scratches = rep.create.cube(
        name="ScratchDefects",
        count=scratch["count"],
        semantics=[("class", "defect_scratch")],
        position=(0.0, 0.0, size_z / 2 + 0.007),
        scale=(scratch["length_m"][0], scratch["width_m"][0], 0.008),
    )
    corrosion = config["defects"]["corrosion"]
    corrosion_patches = rep.create.sphere(
        name="CorrosionDefects",
        count=corrosion["count"],
        semantics=[("class", "defect_corrosion")],
        position=(0.0, 0.0, size_z / 2 + 0.005),
        scale=(corrosion["length_m"][0], corrosion["width_m"][0], 0.006),
    )
    pit = config["defects"]["pit"]
    pits = rep.create.sphere(
        name="PitDefects",
        count=pit["count"],
        semantics=[("class", "defect_pit")],
        position=(0.0, 0.0, size_z / 2 + 0.002),
        scale=(pit["length_m"][0], pit["width_m"][0], 0.004),
    )

    protected = config["protected_objects"]
    fasteners = rep.create.cylinder(
        name="ProtectedFasteners",
        count=protected["fastener_count"],
        semantics=[("class", "protected_fastener")],
        position=(0.0, 0.0, size_z / 2 + 0.014),
        scale=(0.025, 0.025, 0.012),
    )
    seams = rep.create.cube(
        name="ProtectedSeams",
        count=protected["seam_count"],
        semantics=[("class", "protected_seam")],
        position=(0.0, 0.0, size_z / 2 + 0.009),
        scale=(0.012, size_y * 0.86, 0.010),
    )

    camera_config = config["camera"]
    camera = rep.create.camera(
        name="SurfaceInspectionCamera",
        position=(0.0, 0.0, camera_config["z_m"][1]),
        look_at=(0.0, 0.0, 0.0),
        focal_length=sum(camera_config["focal_length_mm"]) / 2,
    )
    render_product = rep.create.render_product(
        camera, tuple(config["resolution"]), name="SurfaceInspection"
    )

    lighting = config["lighting"]
    def randomized_inspection_lights():
        lights = rep.create.light(
            name="RandomizedInspectionLights",
            light_type="sphere",
            count=lighting["count"],
            position=rep.distribution.uniform(
                (-size_x, -size_y, 1.0), (size_x, size_y, 3.5)
            ),
            intensity=rep.distribution.uniform(*lighting["intensity"]),
            temperature=rep.distribution.uniform(*lighting["color_temperature_k"]),
            scale=rep.distribution.uniform(0.15, 0.55),
        )
        return lights.node

    rep.randomizer.register(randomized_inspection_lights)

    with rep.trigger.on_frame(num_frames=config["frames"]):
        with surface:
            rep.randomizer.materials(panel_materials)
        with scratches:
            rep.modify.pose(
                position=rep.distribution.uniform(lower, upper),
                rotation=rep.distribution.uniform((0.0, 0.0, -180.0), (0.0, 0.0, 180.0)),
                scale=rep.distribution.uniform(
                    (scratch["length_m"][0], scratch["width_m"][0], 0.004),
                    (scratch["length_m"][1], scratch["width_m"][1], 0.010),
                ),
            )
            rep.randomizer.materials(scratch_materials)
        with corrosion_patches:
            rep.modify.pose(
                position=rep.distribution.uniform(lower, upper),
                scale=rep.distribution.uniform(
                    (corrosion["length_m"][0], corrosion["width_m"][0], 0.003),
                    (corrosion["length_m"][1], corrosion["width_m"][1], 0.012),
                ),
            )
            rep.randomizer.materials(corrosion_materials)
        with pits:
            rep.modify.pose(
                position=rep.distribution.uniform(lower, upper),
                scale=rep.distribution.uniform(
                    (pit["length_m"][0], pit["width_m"][0], 0.002),
                    (pit["length_m"][1], pit["width_m"][1], 0.007),
                ),
            )
            rep.randomizer.materials(pit_materials)
        with fasteners:
            rep.modify.pose(position=rep.distribution.uniform(lower, upper))
        with seams:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (-size_x * 0.35, 0.0, upper[2]),
                    (size_x * 0.35, 0.0, upper[2]),
                )
            )
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (
                        camera_config["x_m"][0],
                        camera_config["y_m"][0],
                        camera_config["z_m"][0],
                    ),
                    (
                        camera_config["x_m"][1],
                        camera_config["y_m"][1],
                        camera_config["z_m"][1],
                    ),
                ),
                look_at=(0.0, 0.0, 0.0),
            )
            rep.modify.attribute(
                "focalLength", rep.distribution.uniform(*camera_config["focal_length_mm"])
            )
        rep.randomizer.randomized_inspection_lights()

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(output.resolve()),
        semantic_types=["class"],
        rgb=config["outputs"]["rgb"],
        semantic_segmentation=config["outputs"]["semantic_segmentation"],
        distance_to_image_plane=config["outputs"]["distance_to_image_plane"],
        normals=config["outputs"]["normals"],
        camera_params=config["outputs"]["camera_params"],
        bounding_box_2d_tight=True,
        colorize_semantic_segmentation=False,
        frame_padding=6,
    )
    writer.attach([render_product])
    rep.orchestrator.run()


def main() -> None:
    args = build_parser().parse_args()
    config = load_sim_config(args.config)
    if args.frames is not None:
        config["frames"] = args.frames
        config = validate_sim_config(config)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan_path = output / "capture_plan.json"
    preview_path = output / "capture_plan_preview.png"
    plan = save_sim_run_plan(config, plan_path)
    render_sim_plan_preview(config, preview_path)
    if args.dry_run:
        print(json.dumps({"plan": str(plan_path), "split_counts": plan["split_counts"]}, indent=2))
        print(f"preview: {preview_path}")
        return

    try:
        from isaacsim import SimulationApp
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Isaac Sim is not available. Run with Isaac Sim's python.bat/python.sh, "
            "or add --dry-run to validate the capture contract locally."
        ) from error

    simulation_app = SimulationApp(
        launch_config={"headless": args.headless, "renderer": config["renderer"]}
    )
    try:
        import omni.replicator.core as rep
        import omni.usd

        omni.usd.get_context().new_stage()
        rep.settings.set_stage_meters_per_unit(1.0)
        rep.orchestrator.set_capture_on_play(False)
        rep.set_global_seed(config["seed"])
        with rep.new_layer():
            _create_scene(rep, config, output)
    finally:
        simulation_app.close(wait_for_replicator=True)

    manifest_path = output / "validated_manifest.json"
    report = validate_replicator_dataset(output, config, manifest_path)
    print(json.dumps(report["report"], indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
