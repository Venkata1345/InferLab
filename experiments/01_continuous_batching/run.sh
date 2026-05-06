#!/usr/bin/env bash
# Continuous batching impact experiment.
# Restarts vLLM twice (serial then vanilla), benches each at c=1,4,8,16,32,
# then aggregates results into results.json + throughput_vs_concurrency.png.
#
# Estimated time on A100: ~25 minutes (~11 min serial + ~7 min vanilla + restarts).
# Run from the project root:
#   bash experiments/01_continuous_batching/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/01_continuous_batching"
RESULTS_DIR="$EXP_DIR/results"
mkdir -p "$RESULTS_DIR"

cd "$REPO_ROOT"

stop_vllm() {
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 5
}

wait_for_server() {
    local timeout_s="${1:-180}"
    local i
    for i in $(seq 1 "$timeout_s"); do
        if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
            echo "   server up after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: vLLM did not come up within ${timeout_s}s" >&2
    return 1
}

run_config() {
    local cfg_path="$1"     # path to config script
    local tag="$2"          # tag for predictor-name (serial, vanilla)
    local log="$EXP_DIR/vllm-$tag.log"

    echo
    echo "=========================================="
    echo ">> Config: $tag  ($cfg_path)"
    echo "=========================================="

    stop_vllm
    nohup bash "$cfg_path" >"$log" 2>&1 &
    wait_for_server 180

    python -m cli.main bench vllm \
        --predictor-name "vllm-Qwen-3B-$tag" \
        --concurrency 1,4,8,16,32 \
        --n-requests 100 \
        --results-dir "$RESULTS_DIR"
}

run_config "service/configs/serial.sh"  "serial"
run_config "service/configs/vanilla.sh" "vanilla"

stop_vllm

echo
echo ">> Aggregating + plotting"
python "$EXP_DIR/aggregate.py"

echo
echo ">> Done."
echo "   results.json: $EXP_DIR/results.json"
echo "   plot:         $EXP_DIR/throughput_vs_concurrency.png"
echo "   raw benches:  $RESULTS_DIR/"
