# Generic Sage node build — multi-arch (linux/amd64 + linux/arm64), CPU torch from
# PyPI. Builds cleanly in ECR for both architectures. Inference runs on CPU
# (fine for one frame per run); it uses the GPU automatically if present.
# For GPU-accelerated in-container inference on Jetson Thor, see Dockerfile.jetson-thor.
FROM python:3.12-slim-bookworm

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

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
