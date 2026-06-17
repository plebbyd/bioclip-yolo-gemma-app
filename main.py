"""
Sage / Waggle plugin: capture one camera frame, run BioCLIP 2.5 / BioCLIP 2 /
BioCLIP / YOLO (via the detectors module), publish JSON results and optionally
upload the snapshot.

Default backend is BioCLIP 2 (``imageomics/bioclip-2``), the demo/summer-camp model;
``bioclip25`` selects the newer, larger BioCLIP 2.5 Huge (``imageomics/bioclip-2.5-vith14``).
Inference uses the GPU when available, else CPU (set REQUIRE_GPU=1 to enforce GPU).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

# Hugging Face's Xet transfer protocol (cas-server.xethub.hf.co / *.cdn.hf.co) is
# often unreachable from edge nodes and silently stalls large downloads at 0 B/s.
# Default to the plain HTTPS CDN (huggingface.co) which works there; override by
# exporting HF_HUB_DISABLE_XET=0 on networks where Xet is reachable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from PIL import Image
from waggle.data.vision import Camera
from waggle.plugin import Plugin

from detectors import available_models, caption, detect

# Accept common spellings of the BioCLIP backend names.
_BACKEND_ALIASES = {
    "bioclip-2": "bioclip2",
    "bioclip_2": "bioclip2",
    "bioclipv2": "bioclip2",
    "bioclip2.5": "bioclip25",
    "bioclip-2.5": "bioclip25",
    "bioclip_2.5": "bioclip25",
    "bioclip-2.5-vith14": "bioclip25",
    "bioclip2.5huge": "bioclip25",
}
_VALID_BACKENDS = ("yolo", "bioclip", "bioclip2", "bioclip25")


class _HTTPFrame:
    """pywaggle-snapshot-like wrapper around a JPEG fetched over HTTP.

    Exposes the same interface the plugin uses from a pywaggle snapshot:
    ``.data`` (HxWx3 RGB uint8), ``.timestamp`` (ns), and ``.save(path)``.
    """

    def __init__(self, data: np.ndarray, timestamp: int, raw_bytes: bytes | None = None):
        self.data = data
        self.timestamp = timestamp
        self._raw = raw_bytes

    def save(self, path: str) -> None:
        if self._raw is not None and path.lower().endswith((".jpg", ".jpeg")):
            with open(path, "wb") as f:
                f.write(self._raw)
        else:
            Image.fromarray(self.data).save(path)


def _fresh_snapshot_url(url: str) -> str:
    """Add/refresh the Reolink ``rs`` cache-buster so each call returns a new frame.

    The original query is preserved verbatim (NOT re-encoded) so values like a
    password containing ``!`` are sent exactly as given — many camera CGIs do not
    URL-decode query params, so re-encoding ``!`` to ``%21`` would break auth.
    """
    parts = urllib.parse.urlsplit(url)
    items = [kv for kv in parts.query.split("&") if kv and not kv.startswith("rs=")]
    items.append(f"rs={time.time_ns()}")
    return urllib.parse.urlunsplit(parts._replace(query="&".join(items)))


def _acquire_frame_http(url: str, timeout: float = 15.0) -> tuple[_HTTPFrame, dict[str, Any]]:
    """Fetch one JPEG via an HTTP snapshot endpoint (e.g. Reolink ``cmd=Snap``).

    Credentials in the URL are NOT published — only ``source_kind`` is returned.
    """
    req_url = _fresh_snapshot_url(url)
    logging.info("Acquiring snapshot via HTTP (host=%s)", urllib.parse.urlsplit(req_url).hostname)
    with urllib.request.urlopen(req_url, timeout=timeout) as resp:  # noqa: S310
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read()
    if not raw:
        raise RuntimeError("HTTP snapshot endpoint returned no data")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"HTTP snapshot did not return a decodable image (content-type={ctype!r}, "
            f"{len(raw)} bytes; starts with {raw[:120]!r}). Check the URL/credentials "
            "and do not URL-encode the password."
        ) from exc
    data = np.asarray(image, dtype=np.uint8)
    return _HTTPFrame(data, time.time_ns(), raw_bytes=raw), {"source_kind": "http"}


def _acquire_frame(
    stream: str | None,
    camera_id: str | None,
    snapshot_url: str | None = None,
) -> tuple[object, dict[str, Any]]:
    """Grab one frame via HTTP snapshot URL, RTSP/stream URL, or named camera.

    HTTP path: fetch a JPEG from an HTTP snapshot endpoint (Reolink ``cmd=Snap``),
    useful when RTSP is firewalled. RTSP path matches temp-imagesampler / pywaggle:
    ``Camera(stream)`` and first frame from ``camera.stream()``. Named path:
    ``Camera(id)`` and ``snapshot()``.
    """
    if snapshot_url and str(snapshot_url).strip():
        return _acquire_frame_http(str(snapshot_url).strip())

    if stream and str(stream).strip():
        s = str(stream).strip()
        logging.info("Acquiring frame from stream: %s", s)
        with Camera(s) as camera:
            snapshot = None
            for snap in camera.stream():
                snapshot = snap
                break
        if snapshot is None:
            raise RuntimeError("camera.stream() produced no frame")
        return snapshot, {"source_kind": "stream", "stream": s}

    cid = (camera_id or os.environ.get("WAGGLE_CAMERA") or "left").strip()
    logging.info("Acquiring snapshot from camera id: %s", cid)
    with Camera(cid) as camera:
        snapshot = camera.snapshot()
    return snapshot, {"source_kind": "camera", "camera": cid}


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
    elif backend in ("bioclip", "bioclip2"):
        out["rank"] = args.bioclip_rank
        out["target_taxon"] = args.bioclip_target_taxon
        out["min_confidence"] = args.bioclip_min_confidence
    return out


def _build_caption_kwargs(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if backend in ("bioclip", "bioclip2"):
        out["rank"] = args.bioclip_rank
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
        default=os.environ.get("VISION_BACKEND", "bioclip2"),
        help="bioclip2 (default) | bioclip25 (newest, ViT-H/14, heaviest) | bioclip | yolo",
    )
    parser.add_argument(
        "--mode",
        choices=("detect", "caption"),
        default=os.environ.get("VISION_MODE", "detect"),
        help="detect: boxes/labels; caption: BioCLIP/BioCLIP 2 top taxa",
    )
    parser.add_argument(
        "--snapshot-url",
        default=os.environ.get("SNAPSHOT_URL", ""),
        help="HTTP JPEG snapshot endpoint (e.g. Reolink cmd=Snap). If set, takes "
        "priority over --stream/--camera. Useful when RTSP is firewalled. "
        "Credentials in the URL are not published.",
    )
    parser.add_argument(
        "--stream",
        default=os.environ.get("WAGGLE_STREAM") or os.environ.get("RTSP_URL") or "",
        help="RTSP URL or stream id for Camera(stream); if set, first frame from "
        "camera.stream() (see imagesampler). Overrides --camera for capture.",
    )
    parser.add_argument(
        "--name",
        default=os.environ.get("WAGGLE_STREAM_NAME") or "",
        help="Logical name for this stream (upload meta['camera'] for node UI; "
        "recommended with --stream).",
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("WAGGLE_CAMERA", "left"),
        help="Camera id when --stream is not set (e.g. left, lab_ptz)",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # When running outside WES there is no rabbitmq broker, so pywaggle's publisher
    # thread spams connection tracebacks. Silence that logger; on a real node the
    # broker resolves and these never fire.
    logging.getLogger("waggle.plugin.rabbitmq").setLevel(logging.CRITICAL)
    logging.getLogger("pika").setLevel(logging.CRITICAL)

    backend = _normalize_backend(args.backend)
    if backend not in _VALID_BACKENDS:
        logging.error("Unknown backend %r (choose from %s)", args.backend, ", ".join(_VALID_BACKENDS))
        sys.exit(2)

    if args.mode == "caption" and backend == "yolo":
        logging.error("Caption mode requires bioclip or bioclip2, not yolo")
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

    stream = (args.stream or "").strip()
    stream_name = (args.name or "").strip()
    snapshot_url = (args.snapshot_url or "").strip()

    with Plugin() as plugin:
        snapshot, source_info = _acquire_frame(
            stream if (stream and not snapshot_url) else None,
            args.camera if not (stream or snapshot_url) else None,
            snapshot_url=snapshot_url or None,
        )

        image = _numpy_rgb_to_pil(snapshot.data)
        ts = snapshot.timestamp

        payload: dict[str, Any] = {
            "backend": backend,
            "mode": args.mode,
            "image_size": list(image.size),
            **source_info,
        }
        if stream_name:
            payload["stream_name"] = stream_name
        if snapshot_url:
            payload["camera"] = stream_name or "http"
        elif not stream:
            payload["camera"] = source_info.get("camera", args.camera)
        else:
            payload["camera"] = stream_name or "rtsp"

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

        if args.mode == "detect":
            dets = result.get("detections") or []
            if dets:
                summary = ", ".join(
                    f"{d.get('label', '?')} ({float(d.get('confidence', 0.0)):.1%})"
                    for d in dets[:5]
                )
            else:
                summary = "no detections"
            logging.info("RESULT [%s] %s — %s", backend, args.mode, summary)
        else:
            logging.info("RESULT [%s] %s — %s", backend, args.mode,
                         result.get("caption") or "(empty)")

        if upload_snapshot:
            path = "snapshot.jpg"
            snapshot.save(path)
            # imagesampler sets meta["camera"] so the node page can attribute uploads.
            upload_meta: dict[str, str] = {}
            label = stream_name or (payload.get("camera") if isinstance(payload.get("camera"), str) else "")
            if label:
                upload_meta["camera"] = label
            elif stream:
                upload_meta["camera"] = "rtsp"
            if upload_meta:
                plugin.upload_file(path, timestamp=ts, meta=upload_meta)
            else:
                plugin.upload_file(path, timestamp=ts)
            logging.info("Uploaded %s meta=%s", path, upload_meta or None)


if __name__ == "__main__":
    main()
