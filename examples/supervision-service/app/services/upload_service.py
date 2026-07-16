from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    EXTENSION_MIME_TYPES,
    SERVER_VIDEOS_DIR,
    UPLOAD_DIR,
)
from app.db.repository import UploadRecord, create_upload, get_upload


def save_upload_file(file: UploadFile) -> UploadRecord:
    """Persist an uploaded video under ``uploads/`` and store metadata.

    Args:
        file: Incoming multipart upload.

    Returns:
        Created upload database record.

    Raises:
        ValueError: If filename is missing, file is empty, or the extension is
            not a supported video format.
    """
    if not file.filename:
        raise ValueError("Filename is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{suffix or 'unknown'}'. "
            f"Supported: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}."
        )

    content = file.file.read()
    if not content:
        raise ValueError("Uploaded file is empty.")

    stored_filename = f"{uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / stored_filename
    upload_path.write_bytes(content)

    content_type = file.content_type or EXTENSION_MIME_TYPES.get(suffix, "video/mp4")

    return create_upload(
        original_filename=file.filename,
        stored_filename=stored_filename,
        file_path=upload_path,
        file_size=len(content),
        content_type=content_type,
    )


def preview_path_for(upload_id: str) -> Path:
    """Return the cached browser-playable preview path for an upload.

    Args:
        upload_id: Upload record identifier.

    Returns:
        Path to the cached preview MP4 (may not exist until generated).
    """
    from app.config import PREVIEW_DIR

    return PREVIEW_DIR / f"{upload_id}.mp4"


def resolve_upload(
    file: Optional[UploadFile],
    upload_id: Optional[str],
    server_video: Optional[str] = None,
) -> UploadRecord:
    """Return an existing upload record or persist a new file upload.

    Resolution priority: a bundled ``server_video`` name, a previously stored
    ``upload_id``, or a freshly uploaded ``file``.

    Args:
        file: Incoming multipart upload (optional if ``upload_id`` is given).
        upload_id: Identifier of a previously stored upload.
        server_video: Name of a bundled sample video under ``SERVER_VIDEOS_DIR``.

    Returns:
        The resolved upload record.

    Raises:
        HTTPException: If no source is usable or the referenced record/file is
            missing.
    """
    if server_video:
        return server_video_record(server_video)

    if upload_id:
        record = get_upload(upload_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Upload record not found.")
        return record

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Either file, upload_id, or server_video is required.",
        )

    try:
        return save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def safe_server_path(name: str) -> Path:
    """Resolve a bundled sample video name to a safe absolute path.

    Guards against path traversal by ensuring the resolved path stays inside
    ``SERVER_VIDEOS_DIR``.

    Args:
        name: File name of the bundled sample video.

    Returns:
        Absolute path to the sample video.

    Raises:
        HTTPException: If the name escapes the sample directory or the file is
            missing.
    """
    source = (SERVER_VIDEOS_DIR / name).resolve()
    root = SERVER_VIDEOS_DIR.resolve()
    if source != root and root not in source.parents:
        raise HTTPException(status_code=400, detail="Invalid server video name.")
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Server video not found.")
    return source


def server_video_record(name: str) -> UploadRecord:
    """Create an upload record pointing at a bundled sample video.

    The record lets the processing endpoints treat a server-side sample exactly
    like a user upload (it is persisted so it appears in history with the real
    filename).

    Args:
        name: File name of the bundled sample video.

    Returns:
        A freshly created upload record referencing the sample file.
    """
    source = safe_server_path(name)
    suffix = source.suffix.lower()
    content_type = EXTENSION_MIME_TYPES.get(suffix, "video/mp4")
    return create_upload(
        original_filename=source.name,
        stored_filename=source.name,
        file_path=source,
        file_size=source.stat().st_size,
        content_type=content_type,
    )

