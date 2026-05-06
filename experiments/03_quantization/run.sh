#!/usr/bin/env bash
# Quantization experiment (AWQ INT4 vs BF16 vanilla).
#
# Runs eval + bench on each config. Eval writes to eval/results/, bench
# writes to bench/results/ (under custom predictor names so neither config
# clobbers the other or the Part A baselines).
#
# Estimated A100 time: ~15 min (eval ~3 min/config; bench at c=1,16 is small;
# AWQ first-time download ~1.5 GB / ~2 min).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/03_quantization"
mkdir -p "$EXP_DIR"
cd "$REPO_ROOT"

stop_vllm() {
    pkill -9 -f vllm 2>/dev/null || true
    sleep 3
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+$' | xargs -r kill -9 2>/dev/null || true
    sleep 5
    local total free target i
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    target=$(( total * 90 / 100 ))
    for i in $(seq 1 30); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ "$free" -gt "$target" ]; then
            echo "   GPU free: ${free}MB / ${total}MB"
            return 0
        fi
        sleep 2
    done
    echo "WARNING: only ${free}MB free of ${total}MB after 60s" >&2
}

wait_for_server() {
    local timeout_s="${1:-300}"
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
    local cfg_path="$1"
    local tag="$2"            # e.g. "vanilla" or "AWQ"
    local boot_timeout="${3:-300}"
    local log="$EXP_DIR/vllm-$tag.log"
    local pred_name="vllm-Qwen-3B-$tag"

    echo
    echo "=========================================="
    echo ">> Config: $tag  ($cfg_path)"
    echo "=========================================="

    stop_vllm
    nohup bash "$cfg_path" >"$log" 2>&1 &
    wait_for_server "$boot_timeout"

    echo ">> Eval ($pred_name)"
    python -m cli.main eval vllm --predictor-name "$pred_name"

    echo ">> Bench ($pred_name) at c=1, 16"
    python -m cli.main bench vllm \
        --predictor-name "$pred_name" \
        --concurrency 1,16 \
        --n-requests 50
}

# Re-bench vanilla on this hardware so we have a clean apples-to-apples baseline
# (step 1's vanilla bench JSONs were lost in a Colab kernel reset).
run_config "service/configs/vanilla.sh" "vanilla" 180

# AWQ — first-run weight download takes ~2 min for the 1.5 GB shards
run_config "service/configs/awq.sh" "AWQ" 300

stop_vllm

echo
echo ">> Aggregating"
python "$EXP_DIR/aggregate.py"

echo
echo ">> Done. See $EXP_DIR/results.json"
