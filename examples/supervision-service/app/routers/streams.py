"""Real-time stream ingestion endpoints (Phase 3).

Exposes CRUD for configured RTSP sources plus the live channels the dashboard
consumes: a one-shot calibration ``preview-frame``, an ``mjpeg`` annotated video
feed, and a ``ws`` WebSocket that pushes metric snapshots. Offline file analysis
is untouched and continues to live under the ``analyze`` router.
"""

import asyncio
import base64

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.db.stream_repository import (
    create_stream_source,
    delete_stream_source,
    get_stream_source,
    list_stream_sources,
)
from app.services.stream_manager import grab_preview_frame, stream_manager

router = APIRouter(prefix="/api/v1/streams", tags=["streams"])

WEATHER_LABELS = {"晴", "晴天", "多云", "阴", "雨", "下雨", "雪", "下雪", "雾", "霾"}
ROAD_LABELS = {"干燥", "潮湿", "湿", "积水", "积雪", "雪", "结冰", "冰"}


class SourcePoint(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class PreviewFrameRequest(BaseModel):
    url: str = Field(min_length=1, description="RTSP / 视频流地址")


class StreamCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=1)
    source_points: list[SourcePoint] = Field(min_length=4, max_length=4)
    target_width: float = Field(gt=0)
    target_height: float = Field(gt=0)
    lane_count: int = Field(default=3, ge=1, le=8)
    weather: str | None = None
    road_condition: str | None = None
    visibility: str | None = None
    sample_fps: float = Field(default=10.0, gt=0, le=30)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


def _status_payload(stream_id: str) -> dict:
    record = get_stream_source(stream_id)
    if record is None:
        raise HTTPException(status_code=404, detail="未找到该视频流。")
    payload = record.to_dict()
    runtime = stream_manager.get(stream_id)
    if runtime is not None:
        payload["runtime"] = runtime.public_status()
        payload["status"] = runtime.status
        payload["error_message"] = runtime.error
    payload["running"] = stream_manager.is_running(stream_id)
    return payload


@router.post("/preview-frame")
def preview_frame(request: PreviewFrameRequest) -> dict:
    """Grab a single frame from the stream so the UI can calibrate on it."""
    try:
        jpeg, width, height = grab_preview_frame(request.url)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    encoded = base64.b64encode(jpeg).decode("ascii")
    return {
        "width": width,
        "height": height,
        "image": f"data:image/jpeg;base64,{encoded}",
    }


@router.post("", status_code=201)
def create_stream(request: StreamCreateRequest) -> dict:
    """Create a stream source and immediately start its analysis worker."""
    if request.weather and request.weather not in WEATHER_LABELS:
        raise HTTPException(status_code=400, detail=f"未知天气标签: {request.weather}")
    if request.road_condition and request.road_condition not in ROAD_LABELS:
        raise HTTPException(status_code=400, detail=f"未知路面标签: {request.road_condition}")

    record = create_stream_source(
        name=request.name,
        url=request.url,
        source_points=[[p.x, p.y] for p in request.source_points],
        target_width=request.target_width,
        target_height=request.target_height,
        lane_count=request.lane_count,
        weather=request.weather,
        road_condition=request.road_condition,
        visibility=request.visibility,
        sample_fps=request.sample_fps,
        confidence_threshold=request.confidence_threshold,
        iou_threshold=request.iou_threshold,
    )
    stream_manager.start(record)
    return _status_payload(record.id)


@router.get("")
def list_streams() -> dict:
    """List all configured streams with their current runtime status."""
    records = list_stream_sources()
    items = []
    for record in records:
        payload = record.to_dict()
        runtime = stream_manager.get(record.id)
        if runtime is not None:
            payload["status"] = runtime.status
            payload["error_message"] = runtime.error
            payload["runtime"] = runtime.public_status()
        payload["running"] = stream_manager.is_running(record.id)
        items.append(payload)
    return {"items": items, "total": len(items)}


@router.get("/{stream_id}")
def get_stream(stream_id: str) -> dict:
    """Return a single stream's configuration and status."""
    return _status_payload(stream_id)


@router.post("/{stream_id}/start")
def start_stream(stream_id: str) -> dict:
    """Start (or restart) a previously configured stream's worker."""
    record = get_stream_source(stream_id)
    if record is None:
        raise HTTPException(status_code=404, detail="未找到该视频流。")
    stream_manager.start(record)
    return _status_payload(stream_id)


@router.post("/{stream_id}/stop")
def stop_stream(stream_id: str) -> dict:
    """Stop a running stream's worker."""
    record = get_stream_source(stream_id)
    if record is None:
        raise HTTPException(status_code=404, detail="未找到该视频流。")
    stream_manager.stop(stream_id)
    return _status_payload(stream_id)


@router.delete("/{stream_id}", status_code=204)
def remove_stream(stream_id: str) -> None:
    """Stop and delete a stream plus its accumulated analytics."""
    if get_stream_source(stream_id) is None:
        raise HTTPException(status_code=404, detail="未找到该视频流。")
    stream_manager.stop(stream_id)
    delete_stream_source(stream_id)


@router.get("/{stream_id}/snapshot")
def stream_snapshot(stream_id: str) -> dict:
    """Return the latest metric snapshot (polling fallback for the WebSocket)."""
    runtime = stream_manager.get(stream_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="该视频流未在运行。")
    return {"status": runtime.status, "snapshot": runtime.latest_snapshot}


@router.get("/{stream_id}/mjpeg")
def stream_mjpeg(stream_id: str) -> StreamingResponse:
    """Stream the annotated frames as multipart MJPEG for the live preview."""
    runtime = stream_manager.get(stream_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="该视频流未在运行。")

    boundary = "frame"

    async def frames():
        # A connected MJPEG consumer is also an active viewer, so register it
        # (see the WebSocket handler) to keep VLM recognition alive while the
        # stream is being watched; unregister on disconnect.
        stream_manager.add_viewer(stream_id)
        try:
            # Push a frame as soon as the worker publishes a new one (tracked by
            # ``frame_seq``). A fixed-interval sleep would beat against the actual
            # frame production rate and cause duplicated/skipped frames, which the
            # browser renders as visible stutter.
            last_seq = -1
            while stream_manager.is_running(stream_id):
                seq = runtime.frame_seq
                jpeg = runtime.latest_jpeg
                if jpeg is not None and seq != last_seq:
                    last_seq = seq
                    yield (
                        b"--" + boundary.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                    stream_manager.mark_viewer_heartbeat(stream_id)
                else:
                    await asyncio.sleep(0.01)
        finally:
            stream_manager.remove_viewer(stream_id)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@router.websocket("/{stream_id}/ws")
async def stream_ws(websocket: WebSocket, stream_id: str) -> None:
    """Push live metric snapshots to the dashboard over a WebSocket.

    Opening a socket registers the client as an active viewer of the stream,
    which (when ``SV_VLM_ONLY_WHEN_VIEWED`` is on) keeps the VLM environment
    recognition running while someone is watching and lets it idle when the
    last viewer disconnects.
    """
    await websocket.accept()
    runtime = stream_manager.get(stream_id)
    if runtime is None:
        await websocket.send_json({"type": "error", "detail": "该视频流未在运行。"})
        await websocket.close()
        return
    stream_manager.add_viewer(stream_id)
    try:
        while True:
            if not stream_manager.is_running(stream_id):
                await websocket.send_json({"type": "stopped", "status": runtime.status})
                break
            payload = {
                "type": "snapshot",
                "status": runtime.status,
                "fps_actual": round(runtime.fps_actual, 1),
                "snapshot": runtime.latest_snapshot,
            }
            await websocket.send_json(payload)
            stream_manager.mark_viewer_heartbeat(stream_id)
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return
    except Exception:  # pragma: no cover - transport guard
        return
    finally:
        stream_manager.remove_viewer(stream_id)
