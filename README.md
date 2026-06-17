# BioCLIP 2 Vision — Sage / Waggle plugin

Capture one frame from a camera (named camera or RTSP stream), run a vision model,
publish JSON results to Sage (Beehive), and optionally upload the snapshot.

The **default backend is [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2)**
(`imageomics/bioclip-2`, ViT-L/14) from the Imageomics Institute — a biology
foundation model trained on TreeOfLife-200M that classifies organisms across the
full Linnean hierarchy (Kingdom → Species). Backends:

| Backend     | Model                                          | Use |
|-------------|------------------------------------------------|-----|
| `bioclip2`  | `hf-hub:imageomics/bioclip-2` (ViT-L/14)       | **Default.** Species/taxon classification + Grad-CAM localization |
| `bioclip25` | `hf-hub:imageomics/bioclip-2.5-vith14` (ViT-H/14) | Newest [BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14) — most accurate, **largest / heaviest** |
| `bioclip`   | `hf-hub:imageomics/bioclip` (ViT-B/16)         | Original BioCLIP, for comparison |
| `yolo`      | Ultralytics YOLO11                              | Generic object detection (people, vehicles, common animals) |

> **GPU required.** The plugin runs on CUDA by default and fails fast if no GPU is
> visible. Set `ALLOW_CPU=1` to permit (slow) CPU inference for local dry-runs.
> BioCLIP 2.5 Huge (ViT-H/14) is substantially larger than BioCLIP 2 — schedule it
> only on capable GPU nodes.

> Each BioCLIP version uses its **own** species text embeddings from the
> [`imageomics/TreeOfLife-200M`](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
> dataset (BioCLIP 2: `embeddings/txt_emb_bioclip-2.{npy,json}`; BioCLIP 2.5:
> `embeddings/txt_emb_bioclip-2.5-vith14.{npy,json}`). They download automatically
> on first run, or place them next to `main.py` (or set `BIOCLIP2_EMB_DIR`) to bake
> them into an image / avoid network access at the edge. The encoder and embeddings
> are version-specific and must not be mixed.

## What it publishes

For every run the plugin publishes to Sage:

- `vision.result_json` — full JSON payload (backend, mode, detections/caption, image size, source).
- `vision.backend`, `vision.mode` — strings.
- `vision.detection_count` (detect mode) or `vision.caption_preview` (caption mode).
- An uploaded `snapshot.jpg` (unless `--no-upload-snapshot` / `UPLOAD_SNAPSHOT=0`).

A detection looks like:

```json
{"bbox": [x1, y1, x2, y2], "label": "Aves Trochilidae (hummingbird)", "confidence": 0.91}
```

## Quick start (local / Virtual Waggle)

```bash
pip install -r requirements.txt

# BioCLIP 2 species detection from a named camera, logging to ./test-run
./test-app --bioclip-rank Species

# From an RTSP stream
python3 main.py --backend bioclip2 --mode detect \
  --stream "rtsp://USER:PASS@HOST:554/stream" --name woodstock-cam-01 \
  --bioclip-rank Species --bioclip-target-taxon Trochilidae

# From an HTTP snapshot endpoint (Reolink cmd=Snap) when RTSP is firewalled
python3 main.py --backend bioclip2 --mode detect \
  --snapshot-url "http://CAM_IP:PORT/cgi-bin/api.cgi?cmd=Snap&channel=0&user=USER&password=PASS" \
  --name woodstock-cam-01 --bioclip-rank Species --bioclip-target-taxon Trochilidae

# Caption mode (top-5 taxa at a rank)
python3 main.py --backend bioclip2 --mode caption --camera left --bioclip-rank Order
```

### Key arguments / environment variables

| Arg | Env | Default | Notes |
|-----|-----|---------|-------|
| `--backend` | `VISION_BACKEND` | `bioclip2` | `bioclip2` \| `bioclip25` \| `bioclip` \| `yolo` |
| `--mode` | `VISION_MODE` | `detect` | `detect` \| `caption` |
| `--snapshot-url` | `SNAPSHOT_URL` | — | HTTP JPEG endpoint (e.g. Reolink `cmd=Snap`); **takes priority** over `--stream`/`--camera`. Use when RTSP is firewalled. Credentials are not published. |
| `--stream` | `WAGGLE_STREAM` / `RTSP_URL` | — | RTSP URL / stream id (first frame) |
| `--name` | `WAGGLE_STREAM_NAME` | — | Logical stream name for the node UI |
| `--camera` | `WAGGLE_CAMERA` | `left` | Named camera when `--stream` is unset |
| `--bioclip-rank` | `BIOCLIP_RANK` | `Class` | `Kingdom`…`Species` |
| `--bioclip-target-taxon` | `BIOCLIP_TARGET_TAXON` | — | Filter, e.g. `Aves` or `Trochilidae` |
| `--bioclip-min-confidence` | `BIOCLIP_MIN_CONFIDENCE` | `0.1` | |
| `--targets` | `YOLO_TARGETS` | `*` | YOLO class filter |
| | `ALLOW_CPU` | unset | Set to `1` to allow CPU inference (default: GPU required) |
| | `BIOCLIP2_EMB_DIR` | — | Dir holding the species `txt_emb_*.{npy,json}` |
| | `HF_HOME` | — | Hugging Face cache dir |

## Build the container

The plugin requires a GPU. The default image installs CUDA `torch`/`torchvision`.

```bash
# amd64 + NVIDIA runtime (CUDA 12.4 wheels — default):
docker build -t registry.sagecontinuum.org/<namespace>/bioclip2-vision:0.5.0 .

# CPU-only image for local dry-runs / Virtual Waggle (pair with ALLOW_CPU=1):
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  -t registry.sagecontinuum.org/<namespace>/bioclip2-vision:0.5.0-cpu .

# Jetson Thor (arm64 CUDA) — build the Thor base first, then:
docker build -f Dockerfile.jetson-thor \
  -t registry.sagecontinuum.org/<namespace>/bioclip2-vision:0.5.0-thor .
```

The CUDA torch wheels are x86_64-only; for arm64 GPU (Jetson) use the Thor image,
whose base already provides CUDA torch (intentionally not pinned in `requirements.txt`).

## Deploy to the Edge Code Repository (ECR) & schedule

1. Push to a public GitHub repo, then register the app in the
   [Sage portal](https://portal.sagecontinuum.org/apps) (it reads `sage.yaml` and
   `ecr-meta/ecr-science-description.md`). Build the image in ECR.
2. Mark the app **public** so it can be scheduled.
3. Schedule with `sesctl` using the provided job specs:

```bash
# Explore/testing across a set of nodes
sesctl create --file-path job-explore.yaml
sesctl submit --job-id <ID>

# Woodstock-cam-01 hummingbird-feeder demo
sesctl create --file-path job-hummingbird-demo.yaml
sesctl submit --job-id <ID>
```

Both job files contain `TODO` placeholders (node VSNs, RTSP URL, image namespace) —
fill them in before submitting. See the
[Sage developer quick reference](https://sagecontinuum.org/docs/reference-guides/dev-quick-reference).

## Hummingbird-feeder demo (Woodstock-cam-01)

`job-hummingbird-demo.yaml` points the plugin at Woodstock-cam-01 (the hummingbird
feeder) and runs BioCLIP 2 every 10 minutes at `Species` rank with an optional
`Trochilidae` (hummingbird family) taxon filter. View live results and uploaded
snapshots on the node page in the [Sage portal](https://portal.sagecontinuum.org/).

On Woodstock's node the Reolink camera only exposes the **HTTP snapshot API**
(RTSP/554 is firewalled), so the demo captures via `--snapshot-url` (Reolink
`cmd=Snap`) rather than RTSP. Single-frame capture is all this plugin needs.
Keep the camera password out of the committed job file — inject it via the
`SNAPSHOT_URL` env / a Sage secret on the pod.

## Summer-camp exercise (rebuild / tweak)

This repo is intended as a hands-on exercise. Good things for students to change:

- **Pick a target taxon.** Set `--bioclip-target-taxon` (e.g. `Aves`, `Insecta`,
  `Trochilidae`) and `--bioclip-rank` and watch how detections change.
- **Compare models.** Run `--backend bioclip` vs `bioclip2` vs `bioclip25` on the
  same frame and compare the top taxa (and the speed/accuracy trade-off).
- **Tune confidence & localization.** Adjust `--bioclip-min-confidence` and read
  the `bioclip_debug` block (set `BIOCLIP_DEBUG=1`) to see how Grad-CAM boxes form.
- **Change the schedule.** Edit the `cronjob(...)` rule in the job YAML.

Example test images: use the
[BioCLIP 2 demo examples](https://huggingface.co/spaces/imageomics/bioclip-2-demo)
or any iNaturalist/feeder photo; point `--stream` at an MP4/RTSP source, or drop a
JPEG and adapt `main.py`'s `_acquire_frame` to read a file.

## Contact / collaboration

BioCLIP 2 testing is being finalized with **Matthew Thompson (OSU / Imageomics)**,
a BioCLIP 2 co-author (reach him via the Sage Slack). For Sage tokens and node
access see [sagecontinuum.org/docs/contact-us](https://sagecontinuum.org/docs/contact-us).

## License

MIT (this plugin). BioCLIP 2 and BioCLIP are MIT-licensed by the Imageomics Institute.
