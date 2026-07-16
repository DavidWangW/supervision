import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU
from app.db.repository import create_processing_job
from app.schemas.records import JobCreatedResponse
from app.services.job_runner import run_speed_job
from app.services.upload_service import resolve_upload

router = APIRouter(prefix="/api/v1/videos", tags=["speed"])


class SourcePoint(BaseModel):
    """A single calibration point in video pixel coordinates."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class SourcePointsPayload(BaseModel):
    """Four-point road surface calibration payload."""

    points: list[SourcePoint] = Field(min_length=4, max_length=4)


def _parse_source_points(raw_points: str) -> list[list[int]]:
    try:
        payload = SourcePointsPayload.model_validate({"points": json.loads(raw_points)})
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="source_points must be JSON with exactly four {x, y} objects.",
        ) from exc

    return [[point.x, point.y] for point in payload.points]


@router.post("/speed-estimate", response_model=JobCreatedResponse, status_code=202)
def speed_estimate_video(
    background_tasks: BackgroundTasks,
    source_points: Annotated[
        str,
        Form(description='JSON array of four points, e.g. [{"x":1,"y":2}, ...]'),
    ],
    target_width: Annotated[float, Form(gt=0, description="Road width in meters")],
    target_height: Annotated[float, Form(gt=0, description="Road length in meters")],
    file: Annotated[UploadFile | None, File(description="Input video file")] = None,
    upload_id: Annotated[str | None, Form(description="Reuse a previous upload instead of uploading again")] = None,
    server_video: Annotated[
        str | None,
        Form(description="Use a bundled sample video instead of uploading one"),
    ] = None,
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_CONFIDENCE,
    iou_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_IOU,
) -> JobCreatedResponse:
    """Submit a speed estimation job and process it in the background."""
    upload_record = resolve_upload(file, upload_id, server_video)

    parsed_points = _parse_source_points(source_points)
    job = create_processing_job(
        upload_id=upload_record.id,
        job_type="speed",
        parameters={
            "source_points": parsed_points,
            "target_width": target_width,
            "target_height": target_height,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
        },
    )

    background_tasks.add_task(
        run_speed_job,
        job.id,
        Path(upload_record.file_path),
        parsed_points,
        target_width,
        target_height,
        confidence_threshold,
        iou_threshold,
    )

    return JobCreatedResponse(job_id=job.id, upload_id=upload_record.id)
