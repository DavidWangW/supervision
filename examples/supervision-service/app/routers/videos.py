from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    EXTENSION_MIME_TYPES,
    PREVIEW_DIR,
    SERVER_VIDEOS_DIR,
)
from app.db.repository import create_processing_job
from app.schemas.records import JobCreatedResponse
from app.services.job_runner import run_track_job
from app.services.upload_service import (
    preview_path_for,
    resolve_upload,
    safe_server_path,
    save_upload_file,
)
from app.services.video_encoding import build_browser_preview

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])


class PreviewResponse(BaseModel):
    """Upload metadata plus a browser-playable preview URL."""

    upload_id: str
    preview_url: str


class ServerVideoItem(BaseModel):
    """Metadata for a bundled sample video offered by the server."""

    name: str
    size: int
    preview_url: str
    download_url: str


@router.get("/server", response_model=list[ServerVideoItem])
def list_server_videos() -> list[ServerVideoItem]:
    """List bundled sample videos the user can process without uploading.

    Returns:
        Sample videos found in ``SERVER_VIDEOS_DIR`` (empty if the folder is
        missing or has no supported files).
    """
    if not SERVER_VIDEOS_DIR.is_dir():
        return []

    items: list[ServerVideoItem] = []
    for path in sorted(SERVER_VIDEOS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            name = path.name
            items.append(
                ServerVideoItem(
                    name=name,
                    size=path.stat().st_size,
                    preview_url=f"/api/v1/videos/server/preview?name={quote(name)}",
                    download_url=f"/api/v1/videos/server/file?name={quote(name)}",
                )
            )
    return items


@router.get("/server/preview")
def server_video_preview(name: str) -> FileResponse:
    """Stream a browser-playable preview of a bundled sample video.

    Many samples use containers/codecs HTML5 ``<video>`` cannot decode (e.g.
    ``.avi``), so the source is transcoded to an H.264 MP4 and cached.

    Args:
        name: File name of the bundled sample video.

    Returns:
        A browser-playable MP4 preview.
    """
    source = safe_server_path(name)
    preview_path = build_browser_preview(
        source,
        PREVIEW_DIR / f"server_{source.stem}_preview.mp4",
    )
    return FileResponse(
        path=preview_path,
        media_type="video/mp4",
        filename=f"{source.stem}_preview.mp4",
    )


@router.get("/server/file")
def server_video_file(name: str) -> FileResponse:
    """Download the original bundled sample video file.

    Args:
        name: File name of the bundled sample video.

    Returns:
        The original sample file with a sensible content type.
    """
    source = safe_server_path(name)
    content_type = EXTENSION_MIME_TYPES.get(source.suffix.lower(), "video/mp4")
    return FileResponse(
        path=source,
        media_type=content_type,
        filename=source.name,
    )


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
    server_video: Annotated[
        str | None,
        Form(description="Use a bundled sample video instead of uploading one"),
    ] = None,
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_CONFIDENCE,
    iou_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_IOU,
) -> JobCreatedResponse:
    """Submit a tracking job and process it in the background."""
    upload_record = resolve_upload(file, upload_id, server_video)

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
