import logging
import subprocess
from pathlib import Path

import imageio_ffmpeg

logger = logging.getLogger(__name__)

# Preferred hardware H.264 encoders, most specific first. The first one that
# is compiled into the bundled ffmpeg and supported by the local GPU wins;
# otherwise transcoding falls back to the CPU ``libx264`` encoder.
_HW_ENCODERS = (
    "h264_nvenc",        # NVIDIA (Windows / Linux)
    "h264_amf",          # AMD (Windows / Linux)
    "h264_videotoolbox",  # macOS / Apple Silicon & Intel
    "h264_vaapi",        # Linux (Intel / AMD via open-source stack)
    "h264_qsv",          # Intel QuickSync (Windows / Linux)
)

# Extra arguments required per hardware encoder. Encoders that accept a plain
# software-decoded frame (yuv420p) as input only need a speed/quality preset.
# VAAPI/QSV need an explicit hardware upload filter and device.
_HW_ENCODER_EXTRA_ARGS: dict[str, list[str]] = {
    "h264_nvenc": ["-preset", "p1", "-profile:v", "high"],
    "h264_amf": ["-quality", "speed"],
    "h264_videotoolbox": ["-profile:v", "high"],
    "h264_qsv": ["-preset", "veryfast"],
    "h264_vaapi": [
        "-vaapi_device",
        "/dev/dri/renderD128",
        "-vf",
        "format=nv12,hwupload",
    ],
}


def cuda_available() -> bool:
    """Return ``True`` when a CUDA-capable GPU is usable by PyTorch."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - torch import guard
        logger.debug("CUDA availability check failed: %s", exc)
        return False


def torch_device() -> str:
    """Return the best available torch device, ``"cuda"`` or ``"cpu"``."""
    return "cuda" if cuda_available() else "cpu"


def _available_ffmpeg_encoders() -> set[str]:
    """List encoder names exposed by the bundled ffmpeg binary."""
    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:  # pragma: no cover - ffmpeg probe guard
        logger.debug("ffmpeg encoder probe failed: %s", exc)
        return set()

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        # Encoder lines look like: " V..... libx264  ..."
        if len(parts) >= 2 and parts[0].startswith("V") and "." in parts[0]:
            encoders.add(parts[1])
    return encoders


def detect_video_hw_encoder() -> str | None:
    """Return a usable hardware H.264 encoder name, or ``None`` for CPU.

    The result only reflects which encoders are compiled into ffmpeg; the
    actual GPU may still be unavailable at runtime, in which case callers
    must fall back to ``libx264``.

    Returns:
        Encoder name such as ``"h264_nvenc"``, or ``None`` when no hardware
        encoder is available.
    """
    available = _available_ffmpeg_encoders()
    for encoder in _HW_ENCODERS:
        if encoder in available:
            logger.info("Hardware video encoder selected: %s", encoder)
            return encoder
    logger.info("No hardware video encoder available; using CPU (libx264).")
    return None


def video_encoder_args(encoder: str | None) -> tuple[str, list[str]]:
    """Build ffmpeg ``-c:v`` args plus extras for ``encoder``.

    Args:
        encoder: Hardware encoder name, or ``None``/``"libx264"`` for CPU.

    Returns:
        A ``(codec, extra_args)`` tuple ready to splice into an ffmpeg command.
    """
    if not encoder or encoder == "libx264":
        return "libx264", ["-pix_fmt", "yuv420p"]

    extra = list(_HW_ENCODER_EXTRA_ARGS.get(encoder, []))
    # Force a browser-playable pixel format when the encoder accepts one.
    if "-pix_fmt" not in extra and encoder not in ("h264_vaapi", "h264_qsv"):
        extra = ["-pix_fmt", "yuv420p", *extra]
    return encoder, extra
