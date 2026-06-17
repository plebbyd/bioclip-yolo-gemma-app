# Generic Sage GPU node build (amd64 + NVIDIA runtime). For Jetson Thor + CUDA,
# see Dockerfile.jetson-thor. The plugin runs on GPU by default; CUDA torch is
# installed here so it can use the host GPU via the NVIDIA container runtime.
#
# Build (amd64, CUDA 12.4 wheels — default):
#   docker build -t <registry>/bioclip2-vision:0.5.0 .
# CPU-only image (local dry-run / VW; pair with ALLOW_CPU=1 at runtime):
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t ... .
# Note: the CUDA torch wheels are x86_64-only. For arm64 GPU (Jetson) use
# Dockerfile.jetson-thor instead.
FROM python:3.12-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libopenblas0 \
    libpng16-16 \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch/torchvision first (CUDA build by default) so ultralytics /
# open_clip_torch reuse it instead of pulling a CPU-only wheel.
RUN pip3 install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL} \
 && pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
