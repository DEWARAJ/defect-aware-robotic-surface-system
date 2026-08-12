#!/usr/bin/env bash
set -euo pipefail

fixture_dir="${1:?fixture output directory required}"
onnxruntime_root="${2:?ONNX Runtime root required}"
mkdir -p "${fixture_dir}"

node_log="${fixture_dir}/node.log"
verifier_log="${fixture_dir}/verifier.log"
result_json="${fixture_dir}/replay_result.json"
node_pid=""
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
}
trap cleanup EXIT

python3 scripts/ros2/create_replay_fixture.py --output "${fixture_dir}" --size 8
export LD_LIBRARY_PATH="${onnxruntime_root}/lib:${LD_LIBRARY_PATH:-}"

ros2 run surface_perception_cpp realtime_inference_node --ros-args \
  -p model_path:="${fixture_dir}/replay_model.onnx" \
  -p input_width:=8 -p input_height:=8 \
  -p telemetry_window:=2 -p maximum_p95_ms:=1000.0 \
  >"${node_log}" 2>&1 &
node_pid=$!

timeout 25s python3 scripts/ros2/verify_replay.py \
  --output "${result_json}" --timeout-seconds 20 \
  >"${verifier_log}" 2>&1 &
verifier_pid=$!

sleep 3
ros2 bag play "${fixture_dir}/camera_bag" --rate 1.0
if ! wait "${verifier_pid}"; then
  cat "${node_log}"
  cat "${verifier_log}"
  exit 1
fi
verifier_pid=""
cat "${result_json}"
