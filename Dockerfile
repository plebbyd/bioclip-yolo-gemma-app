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

# Bake BioCLIP 2 weights + TreeOfLife-200M species embeddings into the image so the
# container needs no network at runtime (scheduled WES jobs frequently lack HF egress).
# HF_HUB_DISABLE_XET avoids the Xet CDN, which is often unreachable from build/edge nets.
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_XET=1
# Bake into the SAME cache_dir the runtime uses (detectors.py passes cache_dir=$HF_HOME
# to open_clip, which treats it as the cache root directly — no "hub/" segment).
RUN python3 -c "import os, open_clip; open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2', cache_dir=os.environ['HF_HOME'])" \
 && python3 -c "from huggingface_hub import hf_hub_download as d; import shutil, os; [shutil.move(d('imageomics/TreeOfLife-200M', f, repo_type='dataset', local_dir='/app/_emb'), '/app/'+os.path.basename(f)) for f in ('embeddings/txt_emb_bioclip-2.npy', 'embeddings/txt_emb_bioclip-2.json')]" \
 && rm -rf /app/_emb
# Models are now cached in the image; do not attempt any HF network calls at runtime.
ENV HF_HUB_OFFLINE=1

COPY . .

ENTRYPOINT ["python3", "main.py"]
