#!/usr/bin/env bash
# Speculative decoding config: Qwen 2.5 3B verifier + Qwen 2.5 0.5B draft.
# Number of speculative tokens (K) is set via SPEC_K env var.
#
# How it works: the small draft model proposes K tokens per step; the verifier
# does a single forward pass to validate them. Accepted tokens are emitted;
# rejected ones cause a fallback to single-token verifier generation.
# Output is byte-for-byte identical to running the verifier alone.
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DRAFT="${VLLM_DRAFT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SPEC_K="${SPEC_K:-4}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"

echo ">> InferLab vLLM serve (config: speculative decoding K=$SPEC_K)"
echo "   target model: $MODEL"
echo "   draft model:  $DRAFT"
echo "   K (spec tok): $SPEC_K"

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$SPEC_K}"
