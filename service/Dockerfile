# vLLM serving container for InferLab.
# Base: NVIDIA CUDA 12.1 runtime + Python 3.11. vLLM ships compiled wheels for CUDA 12.x.
#
# Build:  docker build -t inferlab-vllm -f service/Dockerfile .
# Run:    docker run --rm --gpus all -p 8000:8000 \
#           -v ~/.cache/huggingface:/cache/huggingface \
#           inferlab-vllm
#
# The container only runs `vllm serve` — data prep + eval + bench live on the
# host. That keeps the image lean and the responsibilities clean.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches when only source changes.
# Use `python -m pip` (not bare `pip`) — more reliable across base-image quirks.
COPY requirements.txt requirements-vllm.txt ./
RUN python -m pip install --no-cache-dir -r requirements-vllm.txt

# Only the serving entry point lives in the container — no Python source from
# our package needs to be importable by vllm.
COPY service/ ./service/

EXPOSE 8000
CMD ["bash", "service/run.sh"]
