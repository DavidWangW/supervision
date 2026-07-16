from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU
from app.db.repository import create_processing_job
from app.schemas.records import JobCreatedResponse
from app.services.job_runner import run_track_job
from app.services.upload_service import preview_path_for, resolve_upload, save_upload_file
from app.services.video_encoding import build_browser_preview

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])


class PreviewResponse(BaseModel):
    """Upload metadata plus a browser-playable preview URL."""

    upload_id: str
    preview_url: str


@router.post("/preview", response_model=PreviewResponse, status_code=201)
def preview_uploaded_video(
    file: Annotated[UploadFile, File(description="Input video file (mp4, mov, avi, etc.)")],
) -> PreviewResponse:
    """Upload a video and return a browser-playable preview URL.

    Browsers cannot decode many containers (e.g. ``.avi``) natively, so the
    source is transcoded to an H.264 MP4. The returned ``upload_id`` can be
    reused by the processing endpoints to avoid uploading the file twice.

    Args:
        file: Incoming multipart upload.

    Returns:
        The created upload id and its preview URL.
    """
    try:
        upload_record = save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    build_browser_preview(
        Path(upload_record.file_path),
        preview_path_for(upload_record.id),
    )

    return PreviewResponse(
        upload_id=upload_record.id,
        preview_url=f"/api/v1/records/uploads/{upload_record.id}/preview",
    )


@router.post("/track", response_model=JobCreatedResponse, status_code=202)
def track_uploaded_video(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile | None, File(description="Input video file (mp4, mov, avi, etc.)")] = None,
    upload_id: Annotated[str | None, Form(description="Reuse a previous upload instead of uploading again")] = None,
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_CONFIDENCE,
    iou_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_IOU,
) -> JobCreatedResponse:
    """Submit a tracking job and process it in the background."""
    upload_record = resolve_upload(file, upload_id)

    job = create_processing_job(
        upload_id=upload_record.id,
        job_type="track",
        parameters={
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
        },
    )

    background_tasks.add_task(
        run_track_job,
        job.id,
        Path(upload_record.file_path),
        confidence_threshold,
        iou_threshold,
    )

    return JobCreatedResponse(job_id=job.id, upload_id=upload_record.id)
