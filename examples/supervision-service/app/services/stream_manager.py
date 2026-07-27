"""Resident real-time stream analysis workers (Phase 3).

Each configured RTSP source gets one background worker thread that:

* opens the stream with OpenCV and **reconnects with exponential backoff** when
  it drops,
* applies **backpressure** by decoding every frame (to keep the buffer drained
  and latency low) but only running inference at the configured ``sample_fps``,
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
from ultralytics import YOLO

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


@dataclass
class StreamRuntime:
    """Live runtime state for a single stream worker."""

    record: StreamSourceRecord
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    latest_jpeg: bytes | None = None
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

        analyzer: TrafficFrameAnalyzer | None = None
        source: FrameSource | None = None
        backoff = 1.0
        target_dt = 1.0 / max(record.sample_fps, 1.0)
        last_proc = 0.0
        last_flush_minute = -1
        ema_dt: float | None = None

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
                # Backpressure: keep decoding to drain the buffer, but only run
                # the (expensive) inference at the configured sample rate.
                if now - last_proc < target_dt:
                    continue
                if last_proc:
                    dt = now - last_proc
                    ema_dt = dt if ema_dt is None else 0.8 * ema_dt + 0.2 * dt
                    runtime.fps_actual = 1.0 / ema_dt if ema_dt else 0.0
                last_proc = now

                if analyzer is None:
                    h, w = frame.shape[:2]
                    analyzer = TrafficFrameAnalyzer(
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
                    annotated, snapshot = analyzer.process(frame, result)
                except Exception as exc:  # pragma: no cover - per-frame guard
                    logger.warning("Stream %s frame error: %s", record.id, exc)
                    continue

                runtime.frames_processed += 1

                ok_enc, buf = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
                )
                if ok_enc:
                    runtime.latest_jpeg = buf.tobytes()

                # Phase 4: operator override wins, otherwise use the live
                # auto-recognized environment for risk scoring.
                detected = analyzer.latest_environment
                eff_weather = record.weather or (detected.weather if detected else None)
                eff_road = record.road_condition or (
                    detected.road_condition if detected else None
                )

                # Flush finished minutes to the database.
                current_minute = snapshot.minute_bucket
                if current_minute > last_flush_minute:
                    rows = analyzer.pop_completed_metric_rows(record.id, current_minute)
                    if rows:
                        insert_traffic_metrics(rows)
                        results = compute_risk(
                            rows, analyzer.lane_count, eff_weather, eff_road
                        )
                        insert_risk_assessments(
                            risk_results_to_rows(record.id, _now_iso(), results)
                        )
                    # Persist the auto-recognized environment for the minute that
                    # just completed.
                    agg = analyzer.aggregate_environment()
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

                # Attach a lightweight live risk summary for the WebSocket push.
                live_rows = analyzer.build_metric_rows(record.id)
                live_results = compute_risk(
                    live_rows, analyzer.lane_count, eff_weather, eff_road
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
        except Exception as exc:  # pragma: no cover - worker crash guard
            runtime.status = "error"
            runtime.error = str(exc)
            update_stream_status(record.id, "error", str(exc))
            logger.exception("Stream worker %s crashed", record.id)
        finally:
            if source is not None:
                source.release()
            if analyzer is not None:
                try:
                    remaining = analyzer.build_metric_rows(record.id)
                    if remaining:
                        insert_traffic_metrics(remaining)
                except Exception:  # pragma: no cover - flush guard
                    logger.debug("Final metric flush failed for %s", record.id)
            if runtime.status != "error":
                runtime.status = "stopped"
                update_stream_status(record.id, "stopped")


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
