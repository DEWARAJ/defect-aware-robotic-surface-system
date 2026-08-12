#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:-${repository_root}/configs/tensorrt_deployment_v09.json}"
output="${2:-${repository_root}/runs/tensorrt_v09}"

cd "${repository_root}"
python3 -m surface_perception.tensorrt_deployment \
  --config "${config}" \
  --output "${output}"
