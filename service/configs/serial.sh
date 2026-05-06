#!/usr/bin/env bash
# Serial config: vLLM with max_num_seqs=1 — only one request in flight at a time.
# Simulates the pre-continuous-batching era. Used by experiment 01 to measure
# the throughput delta that continuous batching provides.
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
DTYPE="${VLLM_DTYPE:-auto}"

echo ">> InferLab vLLM serve (config: serial / no continuous batching)"
echo "   model:       $MODEL"
echo "   max_num_seqs: 1 (no batching — one request at a time)"

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --max-num-seqs 1
