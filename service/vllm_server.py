"""vLLM server config — single source of truth for run.sh, Dockerfile, and tests.

We don't reimplement the HTTP server: vLLM ships its own OpenAI-compatible one
(`vllm serve`). This module just exposes the defaults so the launcher script
and the Python client agree on model name, port, and context size.

Override via environment variables (VLLM_MODEL, VLLM_PORT, etc.) — see run.sh.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VLLMConfig:
    model: str
    host: str
    port: int
    max_model_len: int  # Token cap. Our inputs are <500 tokens, output <500 — 4096 is generous.
    gpu_memory_utilization: (
        float  # Fraction of GPU mem given to weights + KV cache. 0.85 leaves headroom.
    )
    dtype: str  # 'auto' picks bfloat16 on Ampere+, float16 elsewhere.

    @property
    def base_url(self) -> str:
        # The OpenAI-compatible endpoint that vllm_caller and any external clients hit.
        return f"http://{self.host}:{self.port}/v1"


# Defaults sized for a 24GB L4 (Colab Pro) running Qwen 2.5 3B-Instruct.
# Same defaults work on A100 40GB with room to spare; on T4 16GB drop
# gpu_memory_utilization to ~0.75 if the model OOMs at boot.
DEFAULT_CONFIG = VLLMConfig(
    model="Qwen/Qwen2.5-3B-Instruct",
    host="0.0.0.0",
    port=8000,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    dtype="auto",
)
