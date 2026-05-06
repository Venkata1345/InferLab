#!/usr/bin/env bash
# Speculative decoding experiment: K ∈ {2, 4, 8} at c=1.
#
# Three vLLM restarts (one per K). Each runs a short bench + scrapes /metrics
# for acceptance rate. K=4 also does a 25-record quality verify (spec dec is
# mathematically exact, so output should match vanilla eval).
#
# Estimated A100 time: ~10 min total.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/04_speculative_decoding"
mkdir -p "$EXP_DIR"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-3B-Instruct"

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

# Snapshot vLLM /metrics to a file (called after a bench so it captures the
# accumulated counters from the run we just did).
snapshot_metrics() {
    local out_path="$1"
    curl -sf http://localhost:8000/metrics > "$out_path" || echo "(metrics scrape failed)" > "$out_path"
}

run_at_k() {
    local k="$1"
    local with_eval="${2:-no}"   # "yes" -> also run a 25-record eval for quality verify
    local tag="K$k"
    local pred_name="vllm-Qwen-3B-spec-$tag"
    local log="$EXP_DIR/vllm-spec-$tag.log"
    local metrics_path="$EXP_DIR/metrics-spec-$tag.txt"

    echo
    echo "=========================================="
    echo ">> Spec decoding: K=$k  (with_eval=$with_eval)"
    echo "=========================================="

    stop_vllm
    SPEC_K="$k" nohup bash service/configs/spec.sh >"$log" 2>&1 &
    wait_for_server 240

    echo ">> Bench ($pred_name) at c=1, n=25"
    python -m cli.main bench vllm \
        --model "$MODEL" \
        --predictor-name "$pred_name" \
        --concurrency 1 \
        --n-requests 25

    if [ "$with_eval" = "yes" ]; then
        echo ">> Quality verify at K=$k (eval --limit 25)"
        python -m cli.main eval vllm \
            --model "$MODEL" \
            --predictor-name "$pred_name-quality" \
            --limit 25
    fi

    echo ">> Snapshot /metrics"
    snapshot_metrics "$metrics_path"
}

run_at_k 2 "no"
run_at_k 4 "yes"
run_at_k 8 "no"

stop_vllm

echo
echo ">> Aggregating"
python "$EXP_DIR/aggregate.py"

echo
echo ">> Done. See $EXP_DIR/results.json"
