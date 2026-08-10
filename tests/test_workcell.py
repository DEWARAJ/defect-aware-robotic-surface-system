import unittest

import numpy as np

from surface_perception.workcell import binary_dilate, binary_erode, make_workcell_scene


class WorkcellSceneTest(unittest.TestCase):
    def test_scene_is_deterministic_and_masks_are_consistent(self) -> None:
        first = make_workcell_scene(128, seed=19)
        second = make_workcell_scene(128, seed=19)
        self.assertTrue(np.array_equal(first.image, second.image))
        self.assertTrue(np.array_equal(first.defect_mask, second.defect_mask))
        self.assertTrue(np.all(first.sanding_mask <= first.workpiece_mask))
        self.assertFalse(np.any(first.sanding_mask & first.protected_mask))
        self.assertTrue(np.all(first.defect_mask <= first.sanding_mask))
        self.assertGreater(int(first.defect_mask.sum()), 0)

    def test_binary_morphology_respects_radius(self) -> None:
        mask = np.zeros((11, 11), dtype=bool)
        mask[3:8, 3:8] = True
        self.assertEqual(int(binary_erode(mask, 1).sum()), 9)
        self.assertEqual(int(binary_dilate(mask, 1).sum()), 45)


if __name__ == "__main__":
    unittest.main()
