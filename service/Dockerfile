# vLLM serving container for InferLab.
# Base: NVIDIA CUDA 12.1 runtime + Python 3.11. vLLM ships compiled wheels for CUDA 12.x.
#
# Build:  docker build -t inferlab-vllm -f service/Dockerfile .
# Run:    docker run --rm --gpus all -p 8000:8000 \
#           -v ~/.cache/huggingface:/cache/huggingface \
#           inferlab-vllm

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/local/bin/pip3 /usr/local/bin/pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so layer caches when only source changes.
COPY requirements.txt requirements-vllm.txt ./
RUN pip install --no-cache-dir -r requirements-vllm.txt

# Source last — most-frequently-changing layer.
COPY service/ ./service/
COPY data/build_dataset.py ./data/build_dataset.py
COPY data/__init__.py ./data/__init__.py

EXPOSE 8000
CMD ["bash", "service/run.sh"]
