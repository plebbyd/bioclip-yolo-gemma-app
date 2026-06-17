# BioCLIP 2 Vision

## Science

Biodiversity monitoring at the edge requires recognizing *which organism* a camera
is seeing, not just that "an animal" is present. This plugin runs
[**BioCLIP 2**](https://huggingface.co/imageomics/bioclip-2) — a biology foundation
model from the [Imageomics Institute](https://imageomics.osu.edu/) (OSU) trained on
the **TreeOfLife-200M** dataset (>200M organism images) — directly on Sage/Waggle
nodes. BioCLIP 2 performs zero-shot classification across the full Linnean hierarchy
(Kingdom → Phylum → Class → Order → Family → Genus → Species) and exposes emergent
biological structure beyond species labels.

On each invocation the plugin captures one frame from a node camera (named camera or
RTSP stream), classifies the organism at a chosen taxonomic rank, localizes it with a
Grad-CAM heatmap (producing bounding boxes), and publishes structured results plus the
snapshot to the Sage data store. It is designed both as a **reusable ECR example**
("how do I run a CLIP-style foundation model on a node?") and as a **summer-camp
teaching artifact** that students can rebuild and tweak.

A flagship demo runs on **Woodstock-cam-01**, pointed at a hummingbird feeder, to
detect and classify visiting birds (e.g. family *Trochilidae*).

## Method

1. **Acquire** one frame via pywaggle `Camera` (RTSP stream → first frame, or named
   camera snapshot).
2. **Infer** with one of:
   - `bioclip2` (default) — `hf-hub:imageomics/bioclip-2` (OpenCLIP ViT-L/14) with the
     `txt_emb_bioclip-2` TreeOfLife-200M species text embeddings. Image features are
     matched against precomputed species text embeddings; per-species probabilities
     are aggregated to the requested rank.
   - `bioclip25` — `hf-hub:imageomics/bioclip-2.5-vith14` (ViT-H/14), the newest
     [BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14) model,
     with its own `txt_emb_bioclip-2.5-vith14` embeddings. Most accurate, but the
     largest/heaviest — best on capable GPU nodes.
   - `bioclip` — original `hf-hub:imageomics/bioclip` (ViT-B/16), for comparison.
   - `yolo` — Ultralytics YOLO11 for generic object detection.
3. **Localize** (BioCLIP variants) via Grad-CAM on a vision-transformer block,
   thresholding the heatmap into bounding boxes (full-frame fallback if unavailable).
4. **Publish** `vision.result_json`, `vision.backend`, `vision.mode`,
   `vision.detection_count` / `vision.caption_preview`, and upload `snapshot.jpg`.

Each BioCLIP version (ViT-B/16, ViT-L/14, ViT-H/14) lives in a **different embedding
space**, so each pairs its own image encoder with its own version-specific text
embeddings; they are not interchangeable.

## Hardware

The plugin runs on **CPU or GPU**: it uses a CUDA GPU when a CUDA-enabled torch is
present, otherwise CPU. Because it captures one frame per run, CPU is fine for periodic
schedules. The default image is multi-arch (amd64 + arm64) with CPU torch and builds in
ECR directly; for GPU-accelerated in-container inference on Jetson Thor, build
`Dockerfile.jetson-thor` (CUDA torch from a Jetson base) on the node. Set `REQUIRE_GPU=1`
to fail fast if no GPU is available. BioCLIP 2.5 Huge (ViT-H/14) is the heaviest model.

## Inference output

```json
{
  "backend": "bioclip2",
  "mode": "detect",
  "image_size": [1920, 1080],
  "camera": "woodstock-cam-01",
  "result": {
    "model": "bioclip2",
    "detections": [
      {"bbox": [812, 433, 1190, 905], "label": "Animalia Chordata Aves ... Trochilidae", "confidence": 0.9}
    ],
    "elapsed_ms": 540,
    "image_size": [1920, 1080]
  }
}
```

Set `BIOCLIP_DEBUG=1` (or pass `bioclip_debug`) to include a `bioclip_debug` block
with top lineages, probability mass, and Grad-CAM diagnostics.

## Arguments

See `sage.yaml` for the declared inputs. Most-used:

- `--backend` (`bioclip2` | `bioclip25` | `bioclip` | `yolo`), `--mode` (`detect` | `caption`)
- `--stream` + `--name` (RTSP) or `--camera` (named camera)
- `--bioclip-rank` (`Kingdom`…`Species`), `--bioclip-target-taxon`, `--bioclip-min-confidence`
- `--targets` (YOLO class filter)

## Build & run

```bash
docker build -t registry.sagecontinuum.org/<namespace>/bioclip2-vision:0.5.0 .
# Jetson Thor (CUDA): docker build -f Dockerfile.jetson-thor ...
```

First run downloads the model weights and the matching TreeOfLife-200M text
embeddings from Hugging Face. To avoid network access at the edge, bake the
version's `txt_emb_*.{npy,json}` into the image or mount them and set `BIOCLIP2_EMB_DIR`.

## Scheduling

Use `sesctl` with the included `job-explore.yaml` (a set of test nodes) and
`job-hummingbird-demo.yaml` (Woodstock-cam-01). Fill in the `TODO` placeholders
(node VSNs, RTSP URL, image namespace) first.

## References & credits

- BioCLIP 2: Gu et al., *"BioCLIP 2: Emergent Properties from Scaling Hierarchical
  Contrastive Learning"* (2025), [arXiv:2505.23883](https://arxiv.org/abs/2505.23883);
  model: <https://huggingface.co/imageomics/bioclip-2>; project:
  <https://imageomics.github.io/bioclip-2/>.
- BioCLIP 2.5 Huge (ViT-H/14): <https://huggingface.co/imageomics/bioclip-2.5-vith14>.
- TreeOfLife-200M dataset: <https://huggingface.co/datasets/imageomics/TreeOfLife-200M>.
- Developed with **Matthew Thompson (OSU / Imageomics)**, a BioCLIP 2 co-author
  (contact via Sage Slack), who is finalizing testing and teaching the summer camp.
