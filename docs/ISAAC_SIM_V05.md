# Isaac Sim Surface Digital Twin (v0.5)

## Objective

The v0.5 milestone extends the measured real-data segmentation pipeline with synthetic data from
an aircraft-like inspection surface. The generator varies camera pose, focal length, lighting,
metal appearance, defect pose, scale, and density. It requests aligned RGB, semantic segmentation,
metric depth, surface normals, and camera parameters from NVIDIA Isaac Sim Replicator.

This is aimed at a concrete engineering question: can controlled synthetic variation improve the
model's robustness to reflective surfaces, rare defects, and camera/lighting changes without
hiding the synthetic-to-real gap?

## Honest status

The generator, configuration contract, deterministic split planner, planning preview, output
discovery, and dataset validator are implemented and tested with local fixtures. A real Isaac Sim
render has **not yet been executed on this CPU-only development environment**. The included preview
is labeled as a pre-capture schematic; it is not presented as an Isaac Sim sensor frame.

The first GPU gate is a 50-frame smoke capture on an RTX system. That capture must pass the output
validator and receive visual inspection before the 1,200-frame run or any sim-to-real claim.

## Dataset contract

The versioned config is [`configs/isaac_sim_surface.json`](../configs/isaac_sim_surface.json).

| Item | Contract |
|---|---|
| Frames | 1,200 for the full capture; 50 for the first smoke test |
| Resolution | 640 x 480 |
| Seed | 42 |
| Splits | 70% train, 15% validation, 15% test, assigned deterministically |
| Defects | Scratch, corrosion, and pit |
| Protected geometry | Fasteners and seams |
| Outputs | RGB, semantic segmentation, distance-to-image-plane, normals, camera parameters |

The normalized downstream class IDs are `panel=1`, defects `10-12`, and protected geometry
`20-21`. Replicator may assign different raw IDs in an individual semantic image; the writer's
label metadata must be used to remap raw IDs into this normalized taxonomy before training.

## Validate without Isaac Sim

The dry run works with ordinary project Python and writes the exact capture/split plan plus a
configuration preview:

```powershell
python scripts/isaac_sim/generate_surface_dataset.py `
  --config configs/isaac_sim_surface.json `
  --output runs/isaac_sim_dry_run `
  --frames 12 `
  --dry-run
```

The same planning layer is available through the package CLI:

```powershell
python -m surface_perception.cli sim-plan `
  --config configs/isaac_sim_surface.json `
  --output runs/isaac_sim_plan/capture_plan.json `
  --preview runs/isaac_sim_plan/capture_plan_preview.png
```

## Run the first RTX capture

Use Isaac Sim's bundled Python interpreter so `SimulationApp` and `omni.replicator.core` are on the
module path. Replace the example path with the actual `python.bat` installed with Isaac Sim:

```powershell
.\scripts\run_isaac_sim.ps1 `
  -IsaacPython "C:\path\to\isaac-sim\python.bat" `
  -Frames 50 `
  -Output "runs/isaac_sim_smoke_50" `
  -Headless
```

The generator creates a new meter-based stage, procedural surface/defect/protected geometry,
randomization triggers, a camera render product, and a BasicWriter. It closes `SimulationApp` with
`wait_for_replicator=True`, then validates that every requested modality exists for every frame and
that raster dimensions match the config.

To re-run validation independently:

```powershell
python -m surface_perception.cli validate-sim `
  --config configs/isaac_sim_surface.json `
  --dataset runs/isaac_sim_smoke_50 `
  --output runs/isaac_sim_smoke_50/validated_manifest.json
```

## Acceptance gates

1. No missing, duplicate, or extra modality frames.
2. RGB, semantic, and normal images match 640 x 480.
3. Semantic label metadata covers all six configured classes.
4. RGB, depth, normals, and semantic masks are pixel-aligned in 20 random samples.
5. Geometry scale is visually credible in the meter-based stage.
6. No train/validation/test frame duplication; repeated seed produces the same split plan.
7. A real-only held-out test set remains untouched during synthetic-data tuning.

## Current limitations and next iteration

- The v0.5 scene is procedural and planar. The next geometry iteration should import a curved,
  riveted CAD/USD panel with realistic seams and material texture maps.
- Primitive defect geometry is useful for pipeline verification, not photorealistic defect truth.
- The first capture must verify current Isaac Sim API behavior and visual scale on the target GPU.
- Synthetic images do not establish real-world accuracy. Report real-only and synthetic-to-real
  ablations separately.
- A robot/inspection head, calibrated camera frame, reachability, collision checks, and closed-loop
  process execution remain future work.

## API references

- [Isaac Sim Replicator getting started](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_getting_started.html)
- [Scene-based synthetic dataset generation](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_scene_based_sdg.html)
- [Replicator randomizer examples](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html)
- [SimulationApp API](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.simulation_app/docs/index.html)
