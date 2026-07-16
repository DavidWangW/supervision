import subprocess
from pathlib import Path

import imageio_ffmpeg

from app.config import PREVIEW_DIR


def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _transcode_to_mp4(source_path: Path, target_path: Path) -> None:
    """Re-encode ``source_path`` into a browser-playable H.264 MP4."""
    ffmpeg = _ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(target_path),
        ],
        check=True,
        capture_output=True,
    )


def ensure_browser_playable(video_path: Path) -> Path:
    """Transcode a video to H.264 so browsers can play it in HTML5 video.

    OpenCV VideoSink defaults to ``mp4v``, which most browsers cannot decode.
    This re-encodes the file in place using the bundled ffmpeg from imageio-ffmpeg.

    Args:
        video_path: Path to the video file to transcode.

    Returns:
        The same path, now containing a browser-compatible MP4.

    Examples:
        >>> ensure_browser_playable(Path("output.mp4"))  # doctest: +SKIP
        PosixPath('output.mp4')
    """
    temp_path = video_path.with_name(f"{video_path.stem}_web{video_path.suffix}")
    _transcode_to_mp4(video_path, temp_path)

    video_path.unlink(missing_ok=True)
    temp_path.rename(video_path)
    return video_path


def build_browser_preview(source_path: Path, preview_path: Path | None = None) -> Path:
    """Produce a browser-playable MP4 preview of an arbitrary source video.

    The result is cached on disk keyed by ``preview_path`` and only regenerated
    when the source file is newer than the cached preview. This lets the web UI
    preview formats that HTML5 ``<video>`` cannot decode natively (e.g. ``.avi``,
    ``.mkv``, ``.wmv``).

    Args:
        source_path: Original uploaded video.
        preview_path: Optional destination. Defaults to
            ``PREVIEW_DIR / "{source_path.stem}_preview.mp4"``.

    Returns:
        Path to the browser-playable MP4 preview.

    Examples:
        >>> build_browser_preview(Path("clip.avi"))  # doctest: +SKIP
        PosixPath('previews/clip_preview.mp4')
    """
    if preview_path is None:
        preview_path = PREVIEW_DIR / f"{source_path.stem}_preview.mp4"

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    if (
        preview_path.is_file()
        and source_path.is_file()
        and preview_path.stat().st_mtime >= source_path.stat().st_mtime
    ):
        return preview_path

    _transcode_to_mp4(source_path, preview_path)
    return preview_path
