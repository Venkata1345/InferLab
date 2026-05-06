#!/usr/bin/env bash
# Part B baseline: vLLM with continuous batching enabled (default).
# Same flags as Part A's service/run.sh, with --max-num-seqs made explicit
# so the config self-documents the claim "this is continuous batching".
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
DTYPE="${VLLM_DTYPE:-auto}"

echo ">> InferLab vLLM serve (config: vanilla / continuous batching)"
echo "   model:       $MODEL"
echo "   max_num_seqs: 256 (default — continuous batching)"

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --max-num-seqs 256
