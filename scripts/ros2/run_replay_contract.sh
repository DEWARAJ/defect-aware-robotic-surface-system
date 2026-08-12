#!/usr/bin/env bash
set -euo pipefail

fixture_dir="${1:?fixture output directory required}"
onnxruntime_root="${2:?ONNX Runtime root required}"
mkdir -p "${fixture_dir}"

node_log="${fixture_dir}/node.log"
planner_log="${fixture_dir}/planner.log"
verifier_log="${fixture_dir}/verifier.log"
result_json="${fixture_dir}/replay_result.json"
node_pid=""
planner_pid=""
verifier_pid=""

cleanup() {
  if [[ -n "${verifier_pid}" ]]; then
    kill "${verifier_pid}" 2>/dev/null || true
    wait "${verifier_pid}" 2>/dev/null || true
  fi
  if [[ -n "${node_pid}" ]]; then
    kill "${node_pid}" 2>/dev/null || true
    wait "${node_pid}" 2>/dev/null || true
  fi
  if [[ -n "${planner_pid}" ]]; then
    kill "${planner_pid}" 2>/dev/null || true
    wait "${planner_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 scripts/ros2/create_replay_fixture.py --output "${fixture_dir}" --size 32
export LD_LIBRARY_PATH="${onnxruntime_root}/lib:${LD_LIBRARY_PATH:-}"

ros2 run surface_perception_cpp realtime_inference_node --ros-args \
  -p model_path:="${fixture_dir}/replay_model.onnx" \
  -p input_width:=32 -p input_height:=32 \
  -p telemetry_window:=2 -p maximum_p95_ms:=1000.0 \
  >"${node_log}" 2>&1 &
node_pid=$!

ros2 run surface_perception_cpp coverage_planner_node --ros-args \
  -p tool_radius_pixels:=2 -p lane_spacing_pixels:=3 \
  -p minimum_segment_length_pixels:=3 \
  >"${planner_log}" 2>&1 &
planner_pid=$!

sleep 1
if ! kill -0 "${node_pid}" 2>/dev/null; then
  wait "${node_pid}" || true
  cat "${node_log}"
  exit 1
fi
if ! kill -0 "${planner_pid}" 2>/dev/null; then
  wait "${planner_pid}" || true
  cat "${planner_log}"
  exit 1
fi

timeout 25s python3 scripts/ros2/verify_replay.py \
  --output "${result_json}" --timeout-seconds 20 \
  --size 32 \
  >"${verifier_log}" 2>&1 &
verifier_pid=$!

sleep 2
ros2 bag play "${fixture_dir}/camera_bag" --rate 1.0
if ! wait "${verifier_pid}"; then
  cat "${node_log}"
  cat "${planner_log}"
  cat "${verifier_log}"
  exit 1
fi
verifier_pid=""
cat "${result_json}"
