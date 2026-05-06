#!/usr/bin/env bash
# AWQ INT4 config: Qwen 2.5 3B-Instruct AWQ-quantized.
# Weights ~1.5 GB on disk vs ~6 GB for BF16. vLLM AWQ kernels are fused
# matmuls — generally faster at small batch sizes (memory-bandwidth bound).
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"

echo ">> InferLab vLLM serve (config: AWQ INT4)"
echo "   model:       $MODEL"
echo "   quantization: awq"

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --quantization awq \
    --max-num-seqs 256
