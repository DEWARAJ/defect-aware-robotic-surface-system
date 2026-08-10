from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .workcell import binary_dilate, binary_erode


@dataclass(frozen=True)
class CoverageSegment:
    sequence: int
    lane_index: int
    start: tuple[int, int]
    end: tuple[int, int]
    feed_scale: float
    priority: bool


@dataclass(frozen=True)
class CoveragePlan:
    segments: tuple[CoverageSegment, ...]
    safe_center_mask: np.ndarray
    swept_mask: np.ndarray
    metrics: dict[str, float | int]

    def to_report(self) -> dict:
        height, width = self.safe_center_mask.shape
        segment_reports = []
        for segment in self.segments:
            report = asdict(segment)
            report["start_normalized"] = [
                segment.start[0] / max(1, width - 1),
                segment.start[1] / max(1, height - 1),
            ]
            report["end_normalized"] = [
                segment.end[0] / max(1, width - 1),
                segment.end[1] / max(1, height - 1),
            ]
            segment_reports.append(report)
        return {"metrics": self.metrics, "segments": segment_reports}


def _contiguous_runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(row, dtype=np.int8), (1, 1))
    differences = np.diff(padded)
    starts = np.flatnonzero(differences == 1)
    ends = np.flatnonzero(differences == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _swept_mask(
    shape: tuple[int, int], segments: list[CoverageSegment], tool_radius: int
) -> np.ndarray:
    swept = np.zeros(shape, dtype=bool)
    for segment in segments:
        first_x = min(segment.start[0], segment.end[0])
        last_x = max(segment.start[0], segment.end[0])
        center_y = segment.start[1]
        for dy in range(-tool_radius, tool_radius + 1):
            y = center_y + dy
            if y < 0 or y >= shape[0]:
                continue
            cap_extent = int(np.floor(np.sqrt(tool_radius * tool_radius - dy * dy)))
            x0 = max(0, first_x - cap_extent)
            x1 = min(shape[1], last_x + cap_extent + 1)
            swept[y, x0:x1] = True
    return swept


def plan_coverage_path(
    sanding_mask: np.ndarray,
    protected_mask: np.ndarray,
    defect_mask: np.ndarray | None = None,
    *,
    tool_radius: int = 6,
    lane_spacing: int = 9,
    min_segment_length: int = 12,
    nominal_feed_pixels_per_second: float = 80.0,
) -> CoveragePlan:
    """Plan a safe boustrophedon raster path over a planar sanding mask."""
    sanding = np.asarray(sanding_mask, dtype=bool)
    protected = np.asarray(protected_mask, dtype=bool)
    if sanding.ndim != 2 or protected.shape != sanding.shape:
        raise ValueError("sanding and protected masks must be equally sized 2D arrays")
    if tool_radius < 1:
        raise ValueError("tool_radius must be at least one pixel")
    if lane_spacing < 1 or lane_spacing > tool_radius * 2:
        raise ValueError("lane_spacing must be between one and the tool diameter")
    if min_segment_length < 2:
        raise ValueError("min_segment_length must be at least two pixels")

    allowed = sanding & ~protected
    safe_centers = binary_erode(allowed, tool_radius)
    priority = np.zeros(sanding.shape, dtype=bool)
    defects = np.zeros(sanding.shape, dtype=bool)
    if defect_mask is not None:
        defects = np.asarray(defect_mask, dtype=bool)
        if defects.shape != sanding.shape:
            raise ValueError("defect mask must match the sanding mask")
        priority = binary_dilate(defects & allowed, tool_radius)

    valid_rows = np.flatnonzero(safe_centers.any(axis=1))
    if valid_rows.size == 0:
        raise ValueError("no sandable area remains after applying the tool safety radius")

    segments: list[CoverageSegment] = []
    forward = True
    lane_index = 0
    for y in range(int(valid_rows[0]), int(valid_rows[-1]) + 1, lane_spacing):
        runs = [
            run
            for run in _contiguous_runs(safe_centers[y])
            if run[1] - run[0] + 1 >= min_segment_length
        ]
        if not runs:
            continue
        ordered_runs = runs if forward else list(reversed(runs))
        for start_x, end_x in ordered_runs:
            priority_segment = bool(priority[y, start_x : end_x + 1].any())
            start = (start_x, y) if forward else (end_x, y)
            end = (end_x, y) if forward else (start_x, y)
            segments.append(
                CoverageSegment(
                    sequence=len(segments),
                    lane_index=lane_index,
                    start=start,
                    end=end,
                    feed_scale=0.55 if priority_segment else 1.0,
                    priority=priority_segment,
                )
            )
        forward = not forward
        lane_index += 1

    if not segments:
        raise ValueError("no valid coverage segments were found")

    reachable = binary_dilate(safe_centers, tool_radius) & allowed
    swept = _swept_mask(sanding.shape, segments, tool_radius)
    uncovered_reachable_defects = defects & reachable & ~swept
    cleanup_segments = 0
    if uncovered_reachable_defects.any():
        target_centers = binary_dilate(uncovered_reachable_defects, tool_radius) & safe_centers
        candidate_rows = np.flatnonzero(target_centers.any(axis=1))
        selected_rows: list[int] = []
        row_step = max(1, tool_radius)
        for row in candidate_rows:
            if not selected_rows or int(row) - selected_rows[-1] >= row_step:
                selected_rows.append(int(row))
        if candidate_rows.size and selected_rows[-1] != int(candidate_rows[-1]):
            selected_rows.append(int(candidate_rows[-1]))

        for y in selected_rows:
            for safe_start, safe_end in _contiguous_runs(safe_centers[y]):
                targets = np.flatnonzero(target_centers[y, safe_start : safe_end + 1])
                if targets.size == 0:
                    continue
                start_x = max(safe_start, safe_start + int(targets[0]) - tool_radius)
                end_x = min(safe_end, safe_start + int(targets[-1]) + tool_radius)
                shortfall = min_segment_length - (end_x - start_x + 1)
                if shortfall > 0:
                    extend_left = min(start_x - safe_start, (shortfall + 1) // 2)
                    start_x -= extend_left
                    end_x = min(safe_end, end_x + shortfall - extend_left)
                if end_x - start_x + 1 < min_segment_length:
                    continue
                cleanup_forward = cleanup_segments % 2 == 0
                start = (start_x, y) if cleanup_forward else (end_x, y)
                end = (end_x, y) if cleanup_forward else (start_x, y)
                segments.append(
                    CoverageSegment(
                        sequence=len(segments),
                        lane_index=lane_index,
                        start=start,
                        end=end,
                        feed_scale=0.45,
                        priority=True,
                    )
                )
                cleanup_segments += 1
                lane_index += 1
        swept = _swept_mask(sanding.shape, segments, tool_radius)

    sandable_pixels = int(allowed.sum())
    defect_pixels = (
        int((np.asarray(defect_mask, dtype=bool) & allowed).sum())
        if defect_mask is not None
        else 0
    )
    reachable_defects = (
        np.asarray(defect_mask, dtype=bool) & reachable
        if defect_mask is not None
        else np.zeros(sanding.shape, dtype=bool)
    )
    covered_sandable = int((swept & allowed).sum())
    covered_reachable = int((swept & reachable).sum())
    protected_contact = int((swept & protected).sum())
    covered_defects = (
        int((swept & np.asarray(defect_mask, dtype=bool) & allowed).sum())
        if defect_mask is not None
        else 0
    )

    process_length = 0.0
    transfer_length = 0.0
    estimated_seconds = 0.0
    previous_end: tuple[int, int] | None = None
    for segment in segments:
        length = float(np.hypot(segment.end[0] - segment.start[0], segment.end[1] - segment.start[1]))
        process_length += length
        estimated_seconds += length / (nominal_feed_pixels_per_second * segment.feed_scale)
        if previous_end is not None:
            transfer_length += float(
                np.hypot(segment.start[0] - previous_end[0], segment.start[1] - previous_end[1])
            )
        previous_end = segment.end

    metrics: dict[str, float | int] = {
        "segments": len(segments),
        "lanes": lane_index,
        "priority_segments": sum(segment.priority for segment in segments),
        "priority_cleanup_segments": cleanup_segments,
        "sandable_pixels": sandable_pixels,
        "coverage_fraction": covered_sandable / max(1, sandable_pixels),
        "reachable_sandable_pixels": int(reachable.sum()),
        "reachable_coverage_fraction": covered_reachable / max(1, int(reachable.sum())),
        "unreachable_safety_pixels": int((allowed & ~reachable).sum()),
        "defect_coverage_fraction": covered_defects / max(1, defect_pixels),
        "reachable_defect_coverage_fraction": int((swept & reachable_defects).sum())
        / max(1, int(reachable_defects.sum())),
        "unreachable_defect_pixels": int(
            (np.asarray(defect_mask, dtype=bool) & allowed & ~reachable).sum()
        )
        if defect_mask is not None
        else 0,
        "protected_contact_pixels": protected_contact,
        "process_path_pixels": process_length,
        "transfer_path_pixels": transfer_length,
        "estimated_process_seconds": estimated_seconds,
        "tool_radius_pixels": tool_radius,
        "lane_spacing_pixels": lane_spacing,
    }
    return CoveragePlan(tuple(segments), safe_centers, swept, metrics)
