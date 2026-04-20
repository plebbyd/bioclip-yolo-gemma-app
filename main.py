"""
Sage / Waggle plugin: capture one camera frame, run YOLO / BioCLIP / Gemma4
(via detectors module), publish JSON results and optionally upload the snapshot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import numpy as np
from PIL import Image
from waggle.data.vision import Camera
from waggle.plugin import Plugin

from detectors import available_models, caption, detect

# Map user-facing name "bioclip2" to the same backend as MSA detectors (OpenCLIP imageomics/bioclip).
_BACKEND_ALIASES = {"bioclip2": "bioclip"}


def _normalize_backend(name: str) -> str:
    k = (name or "").strip().lower()
    return _BACKEND_ALIASES.get(k, k)


def _numpy_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {a.shape}")
    return Image.fromarray(a[:, :, :3].astype(np.uint8, copy=False))


def _build_detect_kwargs(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if backend == "yolo":
        if args.targets:
            out["targets"] = args.targets
    elif backend == "bioclip":
        out["rank"] = args.bioclip_rank
        out["target_taxon"] = args.bioclip_target_taxon
        out["min_confidence"] = args.bioclip_min_confidence
    elif backend == "gemma4":
        out["target"] = args.gemma_targets
        if args.gemma_max_soft_tokens is not None:
            out["max_soft_tokens"] = args.gemma_max_soft_tokens
    return out


def _build_caption_kwargs(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if backend == "gemma4":
        if args.gemma_caption_prompt:
            out["prompt"] = args.gemma_caption_prompt
        if args.gemma_max_soft_tokens is not None:
            out["max_soft_tokens"] = args.gemma_max_soft_tokens
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera snapshot → vision model → publish to Sage (pywaggle)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("VISION_BACKEND", "yolo"),
        help="yolo | bioclip | bioclip2 | gemma4 (bioclip2 aliases bioclip)",
    )
    parser.add_argument(
        "--mode",
        choices=("detect", "caption"),
        default=os.environ.get("VISION_MODE", "detect"),
        help="detect: boxes/labels; caption: BioCLIP top taxa or Gemma4 description",
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("WAGGLE_CAMERA", "left"),
        help="Camera id passed to pywaggle Camera() (e.g. left, right)",
    )
    parser.add_argument(
        "--no-upload-snapshot",
        action="store_true",
        help="Do not save/upload JPEG (default: upload if UPLOAD_SNAPSHOT is unset or 1)",
    )
    parser.add_argument(
        "--targets",
        default=os.environ.get("YOLO_TARGETS", "*"),
        help="YOLO: comma-separated class names or *",
    )
    parser.add_argument(
        "--bioclip-rank",
        default=os.environ.get("BIOCLIP_RANK", "Class"),
        help="BioCLIP taxonomic rank for detection",
    )
    parser.add_argument(
        "--bioclip-target-taxon",
        default=os.environ.get("BIOCLIP_TARGET_TAXON", ""),
        help="BioCLIP optional taxon filter (whitespace/comma tokens)",
    )
    parser.add_argument(
        "--bioclip-min-confidence",
        type=float,
        default=float(os.environ.get("BIOCLIP_MIN_CONFIDENCE", "0.1")),
        help="BioCLIP minimum confidence",
    )
    parser.add_argument(
        "--gemma-targets",
        default=os.environ.get("GEMMA4_TARGETS", ""),
        help="Gemma4 detection: comma categories or empty/* for all",
    )
    parser.add_argument(
        "--gemma-max-soft-tokens",
        type=int,
        default=None,
        help="Gemma4 visual soft-token budget hint (70,140,280,...)",
    )
    parser.add_argument(
        "--gemma-caption-prompt",
        default=os.environ.get("GEMMA_CAPTION_PROMPT", ""),
        help="Gemma4 caption mode: custom prompt (optional)",
    )
    args = parser.parse_args()
    if args.gemma_max_soft_tokens is None:
        raw_ms = os.environ.get("GEMMA4_MAX_SOFT_TOKENS", "").strip()
        if raw_ms.isdigit():
            args.gemma_max_soft_tokens = int(raw_ms)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    backend = _normalize_backend(args.backend)
    if backend not in ("yolo", "bioclip", "gemma4"):
        logging.error("Unknown backend %r", args.backend)
        sys.exit(2)

    if args.mode == "caption" and backend == "yolo":
        logging.error("Caption mode requires bioclip or gemma4, not yolo")
        sys.exit(2)

    upload_snapshot = not args.no_upload_snapshot
    if os.environ.get("UPLOAD_SNAPSHOT", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        upload_snapshot = False

    avail = available_models()
    logging.info("Detector availability: %s", avail)
    if not avail.get(backend):
        logging.warning(
            "Backend %r not available at startup (%s). Run may fail — check deps and services.",
            backend,
            avail,
        )

    with Plugin() as plugin:
        with Camera(args.camera) as camera:
            snapshot = camera.snapshot()

        image = _numpy_rgb_to_pil(snapshot.data)
        ts = snapshot.timestamp

        payload: dict[str, Any] = {
            "backend": backend,
            "mode": args.mode,
            "camera": args.camera,
            "image_size": list(image.size),
        }

        if args.mode == "detect":
            kwargs = _build_detect_kwargs(args, backend)
            result = detect(image, model=backend, **kwargs)
            payload["result"] = result
            n = len(result.get("detections") or [])
            plugin.publish("vision.detection_count", float(n), timestamp=ts)
        else:
            kwargs = _build_caption_kwargs(args, backend)
            result = caption(image, model=backend, **kwargs)
            payload["result"] = result
            cap = result.get("caption") or ""
            if cap:
                plugin.publish("vision.caption_preview", cap[:512], timestamp=ts)

        out_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        logging.info("Publishing vision.result_json (%d bytes)", len(out_json))
        plugin.publish("vision.result_json", out_json, timestamp=ts)
        plugin.publish("vision.backend", backend, timestamp=ts)
        plugin.publish("vision.mode", args.mode, timestamp=ts)

        if upload_snapshot:
            path = "snapshot.jpg"
            snapshot.save(path)
            plugin.upload_file(path, timestamp=ts)
            logging.info("Uploaded %s", path)


if __name__ == "__main__":
    main()
