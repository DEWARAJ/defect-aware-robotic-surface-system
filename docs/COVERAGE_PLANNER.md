# Coverage Planner Engineering Note

## Purpose

The planner demonstrates the software boundary between semantic perception and robotic surface
processing. It converts three aligned binary image masks into ordered sanding segments:

- `sanding_mask`: pixels the process is allowed to cover.
- `protected_mask`: ports, seams, fasteners, sensitive substrates, or other keep-out regions.
- `defect_mask`: regions requiring reduced feed or an additional local pass.

## Safety geometry

The algorithm erodes the allowed center-position mask by the circular tool radius. Every generated
segment is therefore planned in center space. It then reconstructs the swept capsule of every
segment and explicitly counts overlap with protected pixels. The deterministic reference scene
reports zero protected-region contact.

This is a geometric image-plane check, not a certified safety system. Mask uncertainty, camera
calibration error, robot repeatability, tool compliance, workpiece motion, and 3D collisions are
not represented.

## Planning behavior

1. Construct the sandable mask and subtract protected regions.
2. Erode the result by the configured circular tool radius.
3. Extract valid horizontal runs at the configured lane spacing.
4. Alternate run direction to create a boustrophedon path.
5. Mark segments near detected defects and reduce their feed scale.
6. Add local priority cleanup segments if reachable defect pixels remain uncovered.
7. Reconstruct the swept tool mask and publish coverage and safety measurements.

## Output contract

`coverage_plan.json` contains metrics and segment metadata. `trajectory.csv` contains normalized
image coordinates. The C++ ROS2 node publishes repeated groups of five floating-point values:

```text
start_x_normalized, start_y_normalized, end_x_normalized, end_y_normalized, feed_scale
```

## Required gates before hardware

- Calibrate camera intrinsics and extrinsics.
- Reconstruct or estimate the workpiece surface in 3D.
- Project image paths into a named robot coordinate frame.
- Use MoveIt 2 or an equivalent system for reachability and collision checking.
- Add tool orientation, velocity, acceleration, jerk, and force constraints.
- Include uncertainty margins around protected masks.
- Validate in simulation, then on instrumented noncritical test coupons.
- Require an independent emergency stop and supervisory safety layer.
