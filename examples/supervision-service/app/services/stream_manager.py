"""Resident real-time stream analysis workers (Phase 3).

Each configured RTSP source gets one background worker thread that:

* opens the stream with OpenCV and **reconnects with exponential backoff** when
  it drops,
* applies **backpressure** by decoding every frame (to keep the buffer drained
  and latency low) but only running inference at the configured ``sample_fps``,
* **decouples display from inference**: a dedicated inference thread runs YOLO
  at the configured ``sample_fps`` and produces a fully annotated frame
  (boxes drawn on the exact frame they were computed from, so tracking stays
  correct). The display thread republishes that latest annotated frame up to
  ``_DISPLAY_MAX_FPS`` so the MJPEG preview stream stays smooth and alive,
* runs the shared :class:`~app.services.analytics.TrafficFrameAnalyzer` so live
  streams use the exact same analysis code as offline files,
* keeps the latest annotated JPEG (for the MJPEG preview endpoint) and the
  latest metric snapshot (for the WebSocket live feed),
* flushes completed-minute ``traffic_metrics`` and ``risk_assessments`` to the
  database keyed by the stream id.

The :class:`StreamManager` singleton owns the worker registry and is used by the
``streams`` router to start / stop / inspect streams.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# OpenCV polyline / contour compatibility shim
# ---------------------------------------------------------------------------
# On some OpenCV builds (notably 4.x) ``cv2.polylines`` / ``cv2.fillPoly`` /
# ``cv2.drawContours`` require the polygon / contour point arrays to be
# 3-dimensional ``(N, 1, 2)``. Passing the more common 2-dimensional
# ``(N, 2)`` raises ``Both input arrays must be (arrays of) 3-dimensional
# vectors, ...``. The bundled ``supervision`` annotators (e.g.
# ``TraceAnnotator``) as well as our own lane drawing use ``(N, 2)`` arrays,
# so we transparently coerce 2-D point arrays to 3-D here. This keeps the live
# preview working across OpenCV versions without forking the dependency.
for _CV2_FN in ("polylines", "fillPoly", "drawContours"):
    _cv2_orig = getattr(cv2, _CV2_FN)

    def _make_cv2_shim(_orig):
        def _cv2_shim(img, pts, *args, **kwargs):
            if isinstance(pts, (list, tuple)):
                coerced = []
                for _p in pts:
                    _arr = np.asarray(_p)
                    if _arr.ndim == 2:
                        _arr = _arr.reshape(-1, 1, 2)
                    coerced.append(_arr)
                pts = coerced
            return _orig(img, pts, *args, **kwargs)

        return _cv2_shim

    setattr(cv2, _CV2_FN, _make_cv2_shim(_cv2_orig))

from app.config import DEFAULT_SPEED_WEIGHTS
from app.db.analytics_repository import (
    insert_environment_reading,
    insert_risk_assessments,
    insert_traffic_metrics,
)
from app.db.stream_repository import StreamSourceRecord, update_stream_status
from app.services.analytics import (
    TrafficFrameAnalyzer,
    compute_risk,
    risk_results_to_rows,
)
from app.services.hardware import torch_device

logger = logging.getLogger(__name__)

# RTSP transport candidates tried in order. Some servers only serve over UDP
# (VLC auto-negotiates, but OpenCV's ffmpeg defaults can differ), while others
# require TCP through NAT/firewalls.
_TRANSPORTS = ("tcp", "udp", None)
_PROBE_TIMEOUT_S = 6.0


def _apply_ffmpeg_options(transport: str | None) -> None:
    """Point OpenCV's bundled ffmpeg at the requested RTSP transport.

    Clearing the variable lets ffmpeg pick its own default (often UDP), which some
    RTSP servers require even though VLC negotiates transparently.
    """
    if transport:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


def _build_capture(
    url: str, transport: str | None, timeout_ms: int = 6000
) -> "cv2.VideoCapture":
    """Open a capture tuned for RTSP: chosen transport, bounded timeouts, tiny buffer."""
    _apply_ffmpeg_options(transport)
    cap = cv2.VideoCapture(url)
    for prop, value in (
        (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms),
        (cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
    ):
        try:
            cap.set(prop, value)
        except Exception:  # pragma: no cover - property may be unsupported
            pass
    return cap


class FrameSource:
    """Unified frame source so the worker is agnostic to the decoder backend."""

    backend: str = "unknown"

    def read(self) -> tuple[bool, Any]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class _OpenCvFrameSource(FrameSource):
    backend = "opencv"

    def __init__(self, cap: "cv2.VideoCapture") -> None:
        self.cap = cap

    def read(self) -> tuple[bool, Any]:
        return self.cap.read()

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:  # pragma: no cover
            pass


class _PyavFrameSource(FrameSource):
    backend = "pyav"

    def __init__(self, container: Any) -> None:
        self.container = container
        self.stream = container.streams.video[0]
        self._gen = self.container.decode(self.stream)

    def read(self) -> tuple[bool, Any]:
        try:
            frame = next(self._gen)
        except Exception:  # pragma: no cover - decode end / glitch
            return False, None
        try:
            arr = frame.to_ndarray(format="bgr24")
        except Exception:  # pragma: no cover
            return False, None
        if arr is None or arr.size == 0:
            return False, None
        return True, arr

    def release(self) -> None:
        try:
            self.container.close()
        except Exception:  # pragma: no cover
            pass


def _try_opencv(
    url: str, transport: str | None, deadline: float
) -> tuple[FrameSource, Any] | None:
    """Open ``url`` with OpenCV and read one frame using ``transport``."""
    timeout_ms = max(2000, min(int((deadline - time.monotonic()) * 1000), 6000))
    cap = _build_capture(url, transport, timeout_ms=timeout_ms)
    if not cap.isOpened():
        cap.release()
        return None
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            return _OpenCvFrameSource(cap), frame
        time.sleep(0.05)
    cap.release()
    return None


def _try_pyav(
    url: str, transport: str | None, deadline: float
) -> tuple[FrameSource, Any] | None:
    """Open ``url`` with PyAV (full ffmpeg, supports HEVC) and grab one frame."""
    try:
        import av  # noqa: F401  (only available if the user installed it)
    except ImportError:
        return None
    opts = {"rtsp_transport": transport} if transport else {}
    try:
        container = av.open(url, options=opts)
    except Exception as exc:  # pragma: no cover - connection failure
        logger.debug("PyAV open failed (%s): %s", transport, exc)
        return None
    source = _PyavFrameSource(container)
    ok, frame = source.read()
    if ok and frame is not None:
        return source, frame
    source.release()
    return None


def _connect_source(
    url: str, probe_timeout_s: float = _PROBE_TIMEOUT_S
) -> tuple[FrameSource, Any, str]:
    """Connect to a stream and return ``(source, first_frame, backend)``.

    Probes OpenCV then PyAV, each across TCP/UDP/default transports, returning the
    first combination that yields a frame. Raises ``RuntimeError`` with actionable
    diagnostics when nothing works.
    """
    deadline = time.monotonic() + probe_timeout_s
    for backend_fn, name in ((_try_opencv, "opencv"), (_try_pyav, "pyav")):
        for transport in _TRANSPORTS:
            result = backend_fn(url, transport, deadline)
            if result is not None:
                source, frame = result
                logger.info("Stream connected via %s/%s", name, transport or "default")
                return source, frame, source.backend
    raise RuntimeError(
        "已连接视频流但未能读取到画面帧。请确认：\n"
        "  (1) 该地址当前有视频正在推送；\n"
        "  (2) 编码格式受支持——若为 H.265/HEVC，OpenCV 自带 ffmpeg 通常无法解码，"
        "请在当前环境安装 PyAV（pip install av），系统会自动回退到 PyAV 解码；\n"
        "  (3) 必要时在服务端切换 TCP/UDP 传输方式。"
    )

_MAX_BACKOFF_S = 10.0
_JPEG_QUALITY = 80
# Upper bound for the MJPEG display path. The display thread republishes the
# latest inference-annotated frame up to this rate so the preview stays smooth
# and the connection does not time out, even though inference itself only runs
# at the (much lower) ``sample_fps``.
_DISPLAY_MAX_FPS = 25.0
# If inference is very slow, re-publish the same annotated frame on this
# cadence so MJPEG clients do not drop the stream while waiting for the next
# inference result.
_KEEPALIVE_DT = 0.5


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


@dataclass
class StreamRuntime:
    """Live runtime state for a single stream worker."""

    record: StreamSourceRecord
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    latest_jpeg: bytes | None = None
    frame_seq: int = 0
    latest_snapshot: dict | None = None
    status: str = "starting"
    error: str | None = None
    fps_actual: float = 0.0
    frames_processed: int = 0
    started_at: str = field(default_factory=_now_iso)

    def public_status(self) -> dict:
        return {
            "id": self.record.id,
            "name": self.record.name,
            "url": self.record.url,
            "status": self.status,
            "error": self.error,
            "fps_actual": round(self.fps_actual, 1),
            "frames_processed": self.frames_processed,
            "started_at": self.started_at,
            "lane_count": self.record.lane_count,
        }


@dataclass
class _StreamShared:
    """Cross-thread state shared between a stream's display and inference loops."""

    analyzer: "TrafficFrameAnalyzer | None" = None
    latest_raw: Any = None
    # Fully annotated frame produced by the inference thread for ``latest_raw``.
    # The display thread republishes this so the boxes always sit on the exact
    # frame they were computed on (tracking stays correct instead of lagging
    # behind the live video).
    latest_annotated: Any = None
    # Monotonic counter bumped every time ``latest_annotated`` is refreshed, so
    # the display thread can tell when a genuinely new frame is available.
    annot_seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class StreamManager:
    """Registry and lifecycle controller for resident stream workers."""

    def __init__(self) -> None:
        self._runtimes: dict[str, StreamRuntime] = {}
        self._lock = threading.Lock()

    def start(self, record: StreamSourceRecord) -> StreamRuntime:
        """Start (or restart) the worker for ``record`` and return its runtime."""
        with self._lock:
            existing = self._runtimes.get(record.id)
            if existing and existing.thread and existing.thread.is_alive():
                return existing
            runtime = StreamRuntime(record=record)
            thread = threading.Thread(
                target=self._run, args=(runtime,), name=f"stream-{record.id[:8]}", daemon=True
            )
            runtime.thread = thread
            self._runtimes[record.id] = runtime
            thread.start()
            return runtime

    def stop(self, stream_id: str, timeout: float = 8.0) -> bool:
        """Signal a worker to stop and wait briefly for it to finish."""
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            return False
        runtime.stop_event.set()
        if runtime.thread and runtime.thread.is_alive():
            runtime.thread.join(timeout=timeout)
        return True

    def get(self, stream_id: str) -> StreamRuntime | None:
        return self._runtimes.get(stream_id)

    def is_running(self, stream_id: str) -> bool:
        runtime = self._runtimes.get(stream_id)
        return bool(runtime and runtime.thread and runtime.thread.is_alive())

    def stop_all(self) -> None:
        """Stop every running worker (used on application shutdown)."""
        for stream_id in list(self._runtimes.keys()):
            self.stop(stream_id, timeout=3.0)

    def _run(self, runtime: StreamRuntime) -> None:
        record = runtime.record
        device = torch_device()
        try:
            model = YOLO(str(DEFAULT_SPEED_WEIGHTS)).to(device)
            if device == "cuda":
                model.half()
        except Exception as exc:  # pragma: no cover - model load guard
            runtime.status = "error"
            runtime.error = f"模型加载失败: {exc}"
            update_stream_status(record.id, "error", runtime.error)
            return

        # Decouple inference from display: a dedicated thread runs YOLO at
        # ``sample_fps`` and updates the shared analyzer, while this loop only
        # decodes frames, re-annotates them with the cached detections and
        # publishes at the display rate. That keeps the preview smooth even when
        # inference is the bottleneck.
        shared = _StreamShared()
        infer_stop = threading.Event()
        infer_thread = threading.Thread(
            target=self._inference_loop,
            args=(runtime, model, device, shared, infer_stop),
            name=f"infer-{record.id[:8]}",
            daemon=True,
        )

        source: FrameSource | None = None
        backoff = 1.0
        display_dt = 1.0 / _DISPLAY_MAX_FPS
        last_display = 0.0
        ema_dt: float | None = None
        last_out = 0.0
        last_shown_seq = -1

        # The inference thread must be started so it can create the analyzer,
        # run YOLO and produce the annotated frame that the display loop
        # republishes. Without this the preview would only ever show the raw
        # frame (no lanes, no boxes, no risk data).
        infer_thread.start()

        def publish(annotated: Any, now: float, count: bool = True) -> None:
            """Encode ``annotated`` and expose it to the MJPEG consumers.

            ``count`` controls whether this publish advances the fps counter and
            ``last_out`` timestamp. Keep-alive republishes of the same (already
            shown) frame pass ``count=False`` so the reported fps reflects the
            true inference/annotation rate rather than duplicate frames.
            """
            nonlocal last_out, ema_dt
            ok_enc, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
            )
            if not ok_enc:
                return
            runtime.latest_jpeg = buf.tobytes()
            runtime.frame_seq += 1
            if count and last_out:
                dt = now - last_out
                ema_dt = dt if ema_dt is None else 0.8 * ema_dt + 0.2 * dt
                runtime.fps_actual = 1.0 / ema_dt if ema_dt else 0.0
            if count:
                last_out = now

        runtime.status = "running"
        update_stream_status(record.id, "running")

        try:
            while not runtime.stop_event.is_set():
                if source is None:
                    try:
                        source, frame, _ = _connect_source(record.url)
                    except RuntimeError as exc:
                        runtime.status = "reconnecting"
                        runtime.error = str(exc)
                        update_stream_status(record.id, "reconnecting", runtime.error)
                        if runtime.stop_event.wait(backoff):
                            break
                        backoff = min(backoff * 2, _MAX_BACKOFF_S)
                        continue
                    backoff = 1.0
                    runtime.status = "running"
                    runtime.error = None
                    update_stream_status(record.id, "running")
                else:
                    ok, frame = source.read()
                    if not ok or frame is None:
                        runtime.status = "reconnecting"
                        update_stream_status(
                            record.id, "reconnecting", "视频流中断，正在重连…"
                        )
                        source.release()
                        source = None
                        if runtime.stop_event.wait(backoff):
                            break
                        backoff = min(backoff * 2, _MAX_BACKOFF_S)
                        continue

                now = time.monotonic()
                # Hand the freshest raw frame to the inference thread.
                with shared.lock:
                    shared.latest_raw = frame
                    annotated = shared.latest_annotated
                    annot_seq = shared.annot_seq

                # Publish the most recent *inference-annotated* frame. Boxes are
                # drawn on the exact frame they were computed from, so they track
                # vehicles correctly instead of lagging behind the live video.
                # A new annotated frame arrives only at the inference rate; in
                # between we republish it (up to _DISPLAY_MAX_FPS) to keep the
                # stream smooth, and on a slower keep-alive cadence if inference
                # is very slow, so MJPEG clients don't drop the connection.
                if now - last_display < display_dt:
                    continue
                try:
                    if annot_seq != last_shown_seq:
                        if annotated is not None:
                            publish(annotated, now, count=True)
                        else:
                            publish(frame, now, count=False)
                        last_shown_seq = annot_seq
                    elif now - last_display >= _KEEPALIVE_DT:
                        if annotated is not None:
                            publish(annotated, now, count=False)
                        else:
                            publish(frame, now, count=False)
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "Stream %s display frame error: %s", record.id, exc
                    )
                last_display = now
        except Exception as exc:  # pragma: no cover - worker crash guard
            runtime.status = "error"
            runtime.error = str(exc)
            update_stream_status(record.id, "error", str(exc))
            logger.exception("Stream worker %s crashed", record.id)
        finally:
            infer_stop.set()
            infer_thread.join(timeout=5.0)
            if source is not None:
                source.release()
            if shared.analyzer is not None:
                try:
                    remaining = shared.analyzer.build_metric_rows(record.id)
                    if remaining:
                        insert_traffic_metrics(remaining)
                except Exception:  # pragma: no cover - flush guard
                    logger.debug("Final metric flush failed for %s", record.id)
            if runtime.status != "error":
                runtime.status = "stopped"
                update_stream_status(record.id, "stopped")


    def _inference_loop(
        self,
        runtime: StreamRuntime,
        model: Any,
        device: str,
        shared: "_StreamShared",
        stop_event: threading.Event,
    ) -> None:
        """Run YOLO + analysis at ``sample_fps`` off the display thread.

        Grabs the latest raw frame published by the decode/display loop, runs
        detection/tracking/metrics, and updates the shared analyzer. Completed
        minutes are flushed to the database and the live risk snapshot is pushed
        for the WebSocket feed. Keeping this work off the display thread is what
        prevents slow inference from stalling the MJPEG preview.
        """
        record = runtime.record
        target_dt = 1.0 / max(record.sample_fps, 1.0)
        last_proc = 0.0
        last_flush_minute = -1

        while not stop_event.is_set() and not runtime.stop_event.is_set():
            now = time.monotonic()
            if now - last_proc < target_dt:
                stop_event.wait(min(0.02, target_dt - (now - last_proc)))
                continue

            with shared.lock:
                frame = shared.latest_raw
            if frame is None or frame.size == 0:
                stop_event.wait(0.02)
                last_proc = time.monotonic()
                continue

            if shared.analyzer is None:
                h, w = frame.shape[:2]
                shared.analyzer = TrafficFrameAnalyzer(
                    source_points=record.source_points,
                    target_width=record.target_width,
                    target_height=record.target_height,
                    fps=record.sample_fps,
                    resolution_wh=(w, h),
                    lane_count=record.lane_count,
                    confidence_threshold=record.confidence_threshold,
                )

            try:
                result = model(
                    frame,
                    verbose=False,
                    conf=record.confidence_threshold,
                    iou=record.iou_threshold,
                    device=device,
                )[0]
                annotated, snapshot = shared.analyzer.process(frame, result)
            except Exception as exc:  # pragma: no cover - per-frame guard
                logger.warning("Stream %s frame error: %s", record.id, exc)
                last_proc = time.monotonic()
                continue

            # Hand the fully annotated frame to the display thread. The boxes
            # are drawn on ``frame`` (the exact frame YOLO saw), so when the
            # display loop republishes this the detections stay aligned with the
            # vehicles instead of lagging behind the live video.
            with shared.lock:
                shared.latest_annotated = annotated
                shared.annot_seq += 1

            runtime.frames_processed += 1

            detected = shared.analyzer.latest_environment
            eff_weather = record.weather or (detected.weather if detected else None)
            eff_road = record.road_condition or (
                detected.road_condition if detected else None
            )

            current_minute = snapshot.minute_bucket
            if current_minute > last_flush_minute:
                rows = shared.analyzer.pop_completed_metric_rows(
                    record.id, current_minute
                )
                if rows:
                    insert_traffic_metrics(rows)
                    results = compute_risk(
                        rows, shared.analyzer.lane_count, eff_weather, eff_road
                    )
                    insert_risk_assessments(
                        risk_results_to_rows(record.id, _now_iso(), results)
                    )
                agg = shared.analyzer.aggregate_environment()
                if agg is not None:
                    insert_environment_reading(
                        upload_id=record.id,
                        observed_at=_now_iso(),
                        weather=agg.weather,
                        road_condition=agg.road_condition,
                        visibility=agg.visibility,
                        source="auto",
                        details=agg.to_dict(),
                        model=agg.model,
                    )
                last_flush_minute = current_minute

            live_rows = shared.analyzer.build_metric_rows(record.id)
            live_results = compute_risk(
                live_rows, shared.analyzer.lane_count, eff_weather, eff_road
            )
            overall = next((r for r in live_results if r.lane_id is None), None)
            snapshot.risk = {
                "overall": (
                    {
                        "risk_level": overall.risk_level,
                        "risk_score": overall.risk_score,
                        "probability": overall.probability,
                        "factors": overall.factors,
                    }
                    if overall
                    else None
                ),
                "lanes": [
                    {
                        "lane_id": r.lane_id,
                        "risk_level": r.risk_level,
                        "risk_score": r.risk_score,
                    }
                    for r in live_results
                    if r.lane_id is not None
                ],
            }
            snapshot_dict = snapshot.to_dict()
            snapshot_dict["weather"] = eff_weather
            snapshot_dict["road_condition"] = eff_road
            snapshot_dict["visibility"] = (
                detected.visibility if detected else record.visibility
            )
            runtime.latest_snapshot = snapshot_dict

            last_proc = time.monotonic()


# Module-level singleton used across the app.
stream_manager = StreamManager()


def grab_preview_frame(url: str, timeout_s: float = 12.0) -> tuple[bytes, int, int]:
    """Grab a single frame from a stream URL for calibration.

    Args:
        url: The RTSP / video stream URL.
        timeout_s: Best-effort wall-clock budget to obtain a frame.

    Returns:
        A ``(jpeg_bytes, width, height)`` tuple.

    Raises:
        RuntimeError: If the stream cannot be opened or no frame is read.
    """
    source, frame, _ = _connect_source(url, probe_timeout_s=min(timeout_s, 12))
    try:
        h, w = frame.shape[:2]
        ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok_enc:
            raise RuntimeError("画面帧编码失败。")
        return buf.tobytes(), w, h
    finally:
        source.release()
