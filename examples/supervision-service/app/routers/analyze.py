import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU
from app.db.repository import create_processing_job
from app.schemas.records import JobCreatedResponse
from app.services.job_runner import run_analyze_job
from app.services.upload_service import resolve_upload

router = APIRouter(prefix="/api/v1/videos", tags=["analyze"])

# Allowed weather / road-condition labels (Chinese UI labels accepted).
WEATHER_LABELS = {"晴", "晴天", "多云", "阴", "雨", "下雨", "雪", "下雪", "雾", "霾"}
ROAD_LABELS = {"干燥", "潮湿", "湿", "积水", "积雪", "雪", "结冰", "冰"}


class SourcePoint(BaseModel):
    """A single calibration point in video pixel coordinates."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class SourcePointsPayload(BaseModel):
    """Four-point road surface calibration payload."""

    points: list[SourcePoint] = Field(min_length=4, max_length=4)


class LanePolygonPayload(BaseModel):
    """A single lane polygon expressed as pixel [x, y] points."""

    points: list[SourcePoint]


def _parse_json_points(raw: str | None) -> list[list[int]] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="lane_points 不是合法 JSON。") from exc
    if not isinstance(data, list) or len(data) < 1:
        raise HTTPException(status_code=400, detail="lane_points 必须是一个车道多边形数组。")
    lanes: list[list[list[int]]] = []
    for poly in data:
        if not isinstance(poly, list) or len(poly) < 3:
            raise HTTPException(status_code=400, detail="每条车道多边形至少需要 3 个点。")
        lanes.append([[int(p["x"]), int(p["y"])] for p in poly])
    return lanes


def _parse_source_points(raw_points: str) -> list[list[int]]:
    try:
        payload = SourcePointsPayload.model_validate({"points": json.loads(raw_points)})
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="source_points 必须是恰好四个 {x, y} 点的 JSON。",
        ) from exc
    return [[point.x, point.y] for point in payload.points]


@router.post("/analyze", response_model=JobCreatedResponse, status_code=202)
def analyze_traffic(
    background_tasks: BackgroundTasks,
    source_points: Annotated[
        str,
        Form(description='JSON 四点路面标定: [{"x":1,"y":2}, ...]'),
    ],
    target_width: Annotated[float, Form(gt=0, description="路面宽度(米)")],
    target_height: Annotated[float, Form(gt=0, description="路面长度(米)")],
    lane_count: Annotated[int, Form(ge=1, le=8, description="自动划分车道数")] = 3,
    lane_points: Annotated[
        str | None,
        Form(description="可选: 自定义车道多边形 JSON (覆盖 lane_count)"),
    ] = None,
    weather: Annotated[str | None, Form(description="天气: 晴/雨/雪/雾…")] = None,
    road_condition: Annotated[str | None, Form(description="路面: 干燥/潮湿/积雪/结冰…")] = None,
    visibility: Annotated[str | None, Form(description="能见度: 好/中/差")] = None,
    file: Annotated[UploadFile | None, File(description="输入视频文件")] = None,
    upload_id: Annotated[str | None, Form(description="复用已有上传")] = None,
    server_video: Annotated[str | None, Form(description="使用内置示例视频")] = None,
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_CONFIDENCE,
    iou_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_IOU,
) -> JobCreatedResponse:
    """Submit a comprehensive traffic analysis job (speed + flow + headway + risk)."""
    upload_record = resolve_upload(file, upload_id, server_video)
    parsed_points = _parse_source_points(source_points)
    lanes = _parse_json_points(lane_points)

    if weather and weather not in WEATHER_LABELS:
        raise HTTPException(status_code=400, detail=f"未知天气标签: {weather}")
    if road_condition and road_condition not in ROAD_LABELS:
        raise HTTPException(status_code=400, detail=f"未知路面标签: {road_condition}")

    job = create_processing_job(
        upload_id=upload_record.id,
        job_type="analyze",
        parameters={
            "source_points": parsed_points,
            "target_width": target_width,
            "target_height": target_height,
            "lane_count": lane_count,
            "lanes": lanes,
            "weather": weather,
            "road_condition": road_condition,
            "visibility": visibility,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
        },
    )

    background_tasks.add_task(
        run_analyze_job,
        job.id,
        upload_record.id,
        Path(upload_record.file_path),
        parsed_points,
        target_width,
        target_height,
        lane_count,
        lanes,
        weather,
        road_condition,
        visibility,
        confidence_threshold,
        iou_threshold,
    )

    return JobCreatedResponse(job_id=job.id, upload_id=upload_record.id)
