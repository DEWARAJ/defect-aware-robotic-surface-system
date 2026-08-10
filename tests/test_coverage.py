import unittest

import numpy as np

from surface_perception.coverage import plan_coverage_path
from surface_perception.workcell import make_workcell_scene


class CoveragePlannerTest(unittest.TestCase):
    def test_plan_covers_surface_without_touching_protected_regions(self) -> None:
        scene = make_workcell_scene(192, seed=11)
        plan = plan_coverage_path(
            scene.sanding_mask,
            scene.protected_mask,
            scene.defect_mask,
            tool_radius=4,
            lane_spacing=6,
            min_segment_length=8,
        )
        self.assertGreater(len(plan.segments), 10)
        self.assertGreater(float(plan.metrics["coverage_fraction"]), 0.80)
        self.assertGreater(float(plan.metrics["reachable_coverage_fraction"]), 0.90)
        self.assertGreater(float(plan.metrics["defect_coverage_fraction"]), 0.85)
        self.assertGreater(float(plan.metrics["reachable_defect_coverage_fraction"]), 0.90)
        self.assertEqual(int(plan.metrics["protected_contact_pixels"]), 0)
        self.assertTrue(any(segment.priority for segment in plan.segments))

    def test_invalid_lane_spacing_is_rejected(self) -> None:
        mask = np.ones((32, 32), dtype=bool)
        with self.assertRaises(ValueError):
            plan_coverage_path(mask, np.zeros_like(mask), tool_radius=2, lane_spacing=5)


if __name__ == "__main__":
    unittest.main()
