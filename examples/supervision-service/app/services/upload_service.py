from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.config import ALLOWED_VIDEO_EXTENSIONS, EXTENSION_MIME_TYPES, UPLOAD_DIR
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
) -> UploadRecord:
    """Return an existing upload record or persist a new file upload.

    Args:
        file: Incoming multipart upload (optional if ``upload_id`` is given).
        upload_id: Identifier of a previously stored upload.

    Returns:
        The resolved upload record.

    Raises:
        HTTPException: If neither argument is usable or the record is missing.
    """
    if upload_id:
        record = get_upload(upload_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Upload record not found.")
        return record

    if file is None:
        raise HTTPException(status_code=400, detail="Either file or upload_id is required.")

    try:
        return save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

