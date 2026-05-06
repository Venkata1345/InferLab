#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server with InferLab defaults.
# Defaults match service/vllm_server.py:DEFAULT_CONFIG. Override via env vars.
#
# Usage (Colab cell or local):
#   bash service/run.sh               # foreground (cell stays alive)
#   nohup bash service/run.sh > vllm.log 2>&1 &   # background (Colab pattern)
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
DTYPE="${VLLM_DTYPE:-auto}"

echo ">> InferLab vLLM serve"
echo "   model:    $MODEL"
echo "   endpoint: http://${HOST}:${PORT}/v1"
echo "   max_len:  $MAX_MODEL_LEN"
echo "   gpu_util: $GPU_MEM_UTIL"
echo "   dtype:    $DTYPE"

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE"
