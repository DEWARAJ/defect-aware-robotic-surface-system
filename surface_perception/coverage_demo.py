from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .coverage import CoveragePlan, plan_coverage_path
from .workcell import WorkcellScene, make_workcell_scene


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)


def _semantic_overlay(scene: WorkcellScene) -> Image.Image:
    image = scene.image.astype(np.float32).copy()
    image[scene.sanding_mask] = image[scene.sanding_mask] * 0.72 + np.array([20, 155, 125]) * 0.28
    image[scene.protected_mask] = np.array([245, 180, 45])
    image[scene.defect_mask] = np.array([245, 65, 65])
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def _path_overlay(scene: WorkcellScene, plan: CoveragePlan) -> Image.Image:
    image = Image.fromarray(scene.image).copy()
    draw = ImageDraw.Draw(image)
    for segment in plan.segments:
        color = (255, 70, 180) if segment.priority else (40, 225, 235)
        draw.line([segment.start, segment.end], fill=color, width=2)
        draw.ellipse(
            (
                segment.start[0] - 2,
                segment.start[1] - 2,
                segment.start[0] + 2,
                segment.start[1] + 2,
            ),
            fill=color,
        )
    first = plan.segments[0].start
    last = plan.segments[-1].end
    draw.ellipse((first[0] - 5, first[1] - 5, first[0] + 5, first[1] + 5), fill=(80, 255, 100))
    draw.ellipse((last[0] - 5, last[1] - 5, last[0] + 5, last[1] + 5), fill=(255, 90, 70))
    return image


def _coverage_overlay(scene: WorkcellScene, plan: CoveragePlan) -> Image.Image:
    image = np.zeros_like(scene.image)
    image[scene.workpiece_mask] = np.array([37, 53, 68], dtype=np.uint8)
    image[plan.swept_mask & scene.sanding_mask] = np.array([35, 205, 165], dtype=np.uint8)
    image[scene.sanding_mask & ~plan.swept_mask] = np.array([215, 75, 65], dtype=np.uint8)
    image[scene.protected_mask] = np.array([245, 180, 45], dtype=np.uint8)
    image[scene.defect_mask & plan.swept_mask] = np.array([235, 75, 210], dtype=np.uint8)
    return Image.fromarray(image)


def _labeled_panel(image: Image.Image, title: str) -> Image.Image:
    heading = 34
    panel = Image.new("RGB", (image.width, image.height + heading), (8, 14, 21))
    panel.paste(image, (0, heading))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), title, fill=(225, 239, 245))
    return panel


def _save_preview(scene: WorkcellScene, plan: CoveragePlan, path: Path) -> None:
    panels = [
        _labeled_panel(Image.fromarray(scene.image), "RGB WORKCELL SCENE"),
        _labeled_panel(_semantic_overlay(scene), "GREEN SAND / YELLOW PROTECT / RED DEFECT"),
        _labeled_panel(_path_overlay(scene, plan), "CYAN STANDARD / MAGENTA REDUCED FEED"),
        _labeled_panel(_coverage_overlay(scene, plan), "COVERAGE + SAFETY RESULT"),
    ]
    gap = 8
    canvas = Image.new(
        "RGB",
        (panels[0].width * 2 + gap, panels[0].height * 2 + gap),
        (4, 8, 12),
    )
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width + gap, 0))
    canvas.paste(panels[2], (0, panels[0].height + gap))
    canvas.paste(panels[3], (panels[0].width + gap, panels[0].height + gap))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_coverage_demo(
    output_dir: Path,
    *,
    size: int = 384,
    seed: int = 42,
    tool_radius: int = 6,
    lane_spacing: int = 9,
    min_segment_length: int = 12,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = make_workcell_scene(size=size, seed=seed)
    plan = plan_coverage_path(
        scene.sanding_mask,
        scene.protected_mask,
        scene.defect_mask,
        tool_radius=tool_radius,
        lane_spacing=lane_spacing,
        min_segment_length=min_segment_length,
    )

    Image.fromarray(scene.image).save(output_dir / "workcell_scene.png")
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    _mask_image(scene.workpiece_mask).save(masks_dir / "workpiece.png")
    _mask_image(scene.sanding_mask).save(masks_dir / "sanding.png")
    _mask_image(scene.protected_mask).save(masks_dir / "protected.png")
    _mask_image(scene.defect_mask).save(masks_dir / "defect.png")
    _mask_image(plan.swept_mask).save(masks_dir / "swept_area.png")
    _save_preview(scene, plan, output_dir / "coverage_preview.png")

    report = {
        "scene": {"seed": seed, "size": size},
        "planner": {
            "type": "boustrophedon_raster",
            "safety_model": "tool-center erosion against panel boundary and protected regions",
            "priority_behavior": "0.55 feed scale on defect-intersecting lanes",
        },
        **plan.to_report(),
    }
    (output_dir / "coverage_plan.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "point", "x_px", "y_px", "x_norm", "y_norm", "feed_scale"])
        for segment in plan.segments:
            for label, point in (("start", segment.start), ("end", segment.end)):
                writer.writerow(
                    [
                        segment.sequence,
                        label,
                        point[0],
                        point[1],
                        point[0] / max(1, size - 1),
                        point[1] / max(1, size - 1),
                        segment.feed_scale,
                    ]
                )
    return report
