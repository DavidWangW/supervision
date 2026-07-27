from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.db.analytics_repository import (
    latest_environment_reading,
    list_environment_readings,
    list_risk_assessments,
    list_traffic_metrics,
    list_vehicle_tracks,
)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/traffic-metrics")
def get_traffic_metrics(
    upload_id: Annotated[str, Query(description="上传记录 ID")]
) -> dict:
    """Return per-lane, per-minute aggregated traffic metrics for an upload."""
    rows = list_traffic_metrics(upload_id)
    if not rows:
        raise HTTPException(status_code=404, detail="未找到该视频的分析结果。")
    return {"upload_id": upload_id, "metrics": rows}


@router.get("/vehicle-tracks")
def get_vehicle_tracks(
    upload_id: Annotated[str, Query(description="上传记录 ID")]
) -> dict:
    """Return summarized vehicle trajectories for an upload."""
    rows = list_vehicle_tracks(upload_id)
    if not rows:
        raise HTTPException(status_code=404, detail="未找到该视频的车辆轨迹。")
    return {"upload_id": upload_id, "tracks": rows}


@router.get("/risk")
def get_risk_assessments(
    upload_id: Annotated[str, Query(description="上传记录 ID")]
) -> dict:
    """Return rule-based risk assessments (per lane + overall) for an upload."""
    rows = list_risk_assessments(upload_id)
    if not rows:
        raise HTTPException(status_code=404, detail="未找到该视频的风险评估。")
    return {"upload_id": upload_id, "risk": rows}


@router.get("/environment")
def get_environment_readings(
    upload_id: Annotated[str, Query(description="上传记录 ID")]
) -> dict:
    """Return environment readings (weather / road condition) for an upload."""
    rows = list_environment_readings(upload_id)
    return {"upload_id": upload_id, "readings": rows}


@router.get("/environment/latest")
def get_latest_environment(
    upload_id: Annotated[str, Query(description="上传记录 ID")]
) -> dict:
    """Return the most recent (auto-detected) environment reading for an upload."""
    reading = latest_environment_reading(upload_id)
    if reading is None:
        return {"upload_id": upload_id, "reading": None}
    return {"upload_id": upload_id, "reading": reading}
