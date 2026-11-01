#!/usr/bin/env bash
# Same-session LiteMoE/vLLM throughput on one GPU. Usage: bash tools/reproduce.sh [output-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-benchmarks/runs/$(date +%Y%m%d-%H%M%S)}
env=${REPRODUCE_ENV:-.venv-reproduce}
command -v nvidia-smi >/dev/null || { echo "An NVIDIA GPU is required" >&2; exit 1; }
# Some images ship venv without ensurepip, so the outer pip installs into the environment.
[ -x "$env/bin/python" ] || python3 -m venv --without-pip "$env"
python3 -m pip --python "$env/bin/python" install -q -e '.[vllm]'
python="$(cd "$env" && pwd)/bin/python"  # Keep the venv path: resolving the symlink escapes it.
"$python" -m tools.matched_run --output "$out" --inputs benchmarks --vllm-python "$python" \
    --mini-backend vllm --models 1b 3b --reuse-environments --skip-tests --max-seconds "${REPRODUCE_SECONDS:-3000}"
"$python" - "$out" <<'PY'
import json, math, sys
ratios = []
print(f"{'model':6}{'profile':12}{'litemoe tok/s':>12}{'vllm tok/s':>12}{'ratio':>9}")
for model in ('1b', '3b'):
    data = json.load(open(f'{sys.argv[1]}/comparison-{model}.json'))
    for row in data['rows']:
        print(f"{model:6}{row['profile']:12}{row['mini_tps']:12.1f}{row['vllm_tps']:12.1f}{row['mini_over_vllm']:9.1%}")
    ratios.append(data['geometric_mean_ratio'])
    print(f"{model:6}{'geomean':12}{'':24}{data['geometric_mean_ratio']:9.1%}")
print(f"{'all':6}{'geomean':12}{'':24}{math.sqrt(ratios[0] * ratios[1]):9.1%}")
PY
