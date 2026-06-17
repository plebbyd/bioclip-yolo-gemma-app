# Jetson Thor (arm64 + CUDA) build.
#
# THOR_BASE MUST be an arm64 image that already provides a CUDA build of PyTorch
# AND torchvision for this Thor (that is why requirements.txt does not pin torch —
# it inherits them from the base). Use the same base your other working GPU
# containers on this node use (e.g. an NVIDIA L4T / JetPack PyTorch image).
#
# Build ON the Thor node so the image is arm64, then push to ECR:
#   docker build -f Dockerfile.jetson-thor --build-arg THOR_BASE=<your-thor-pytorch-base> \
#     -t registry.sagecontinuum.org/<namespace>/bioclip2-vision:0.8.0 .
#   docker login registry.sagecontinuum.org
#   docker push registry.sagecontinuum.org/<namespace>/bioclip2-vision:0.8.0
#
# (ECR's cloud builder can only build this if THOR_BASE is pullable from a registry;
#  otherwise build locally on the node and push the tag above.)

ARG THOR_BASE=jetson-thor-plugin-base:local
FROM ${THOR_BASE}

WORKDIR /app

# ffmpeg CLI + Python bindings (pywaggle vision / RTSP)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch/torchvision come from the base (CUDA build); pip should NOT replace them.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
