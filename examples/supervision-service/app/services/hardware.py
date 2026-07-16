import itertools
import logging
import subprocess
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

import imageio_ffmpeg
import numpy as np

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

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


def resolve_batch_size(device: str, requested: int) -> int:
    """Resolve the effective inference batch size for ``device``.

    Batched inference only helps on the GPU. On CPU it offers little benefit
    (and increases latency/memory), so the batch is forced to 1. On CUDA the
    requested size is capped to a safe value derived from available VRAM to
    avoid out-of-memory errors on smaller cards.

    Args:
        device: Target device, ``"cuda"`` or ``"cpu"``.
        requested: Desired batch size (e.g. from config).

    Returns:
        The batch size to use, always ``>= 1``.

    Examples:
        >>> resolve_batch_size("cpu", 16)
        1
        >>> resolve_batch_size("cuda", 0) >= 1
        True
    """
    requested = max(1, requested)
    if device != "cuda":
        return 1

    try:
        import torch

        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        # Roughly ~0.7 GB headroom per frame at high resolution for the x-model.
        vram_cap = max(1, int(free_gb / 0.7))
        return min(requested, vram_cap)
    except Exception as exc:  # pragma: no cover - VRAM probe guard
        logger.debug("VRAM probe failed, using requested batch size: %s", exc)
        return requested


def _cycle(frames: list[np.ndarray], index: int) -> np.ndarray:
    """Return ``frames[index % len(frames)]`` cycling the list safely."""
    return frames[index % len(frames)]


def auto_detect_batch_size(
    model: object,
    sample_frames: list[np.ndarray],
    device: str,
    max_batch: int,
    conf: float = 0.3,
    iou: float = 0.7,
    warmup_runs: int = 1,
    probe_runs: int = 3,
    min_improvement: float = 0.03,
) -> int:
    """Pick the GPU batch size with the highest measured inference throughput.

    Runs a short untimed warmup, then for each candidate batch size (powers of
    two up to ``max_batch``) measures frames/sec over ``probe_runs`` timed
    batches. Because throughput saturates as the batch grows, the search stops
    early as soon as a larger batch no longer improves throughput by at least
    ``min_improvement`` (fraction), keeping probing cheap even on long videos.

    Args:
        model: A loaded Ultralytics ``YOLO`` model (already moved to ``device``).
        sample_frames: A few representative frames used to build probe batches;
            they are cycled as needed, so any small number of frames works.
        device: ``"cuda"`` or ``"cpu"`` (auto-detection is only useful on CUDA).
        max_batch: Upper bound, e.g. a VRAM-derived cap from ``resolve_batch_size``.
        conf: Confidence threshold forwarded to the model.
        iou: IOU threshold forwarded to the model.
        warmup_runs: Untimed batches run before measurement to stabilize CUDA.
        probe_runs: Number of timed batches averaged per candidate.
        min_improvement: Relative throughput gain required to keep growing.

    Returns:
        The batch size that maximizes throughput, always ``>= 1``.

    Examples:
        >>> import numpy as np
        >>> frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(8)]
        >>> auto_detect_batch_size(None, frames, "cpu", 8)  # doctest: +SKIP
        1
    """
    if not sample_frames or device != "cuda" or max_batch < 1:
        return 1

    candidates: list[int] = []
    b = 1
    while b <= max_batch:
        candidates.append(b)
        b *= 2
    if not candidates or candidates[-1] != max_batch:
        candidates.append(max_batch)

    # Warm up CUDA kernels / memory pools at the largest candidate first.
    for _ in range(max(1, warmup_runs)):
        warm = [_cycle(sample_frames, i) for i in range(candidates[-1])]
        try:
            model(warm, verbose=False, conf=conf, iou=iou, device=device)
        except Exception as exc:  # pragma: no cover - probe guard
            logger.debug("Auto batch warmup failed, falling back to %s: %s", max_batch, exc)
            return max_batch

    def _measure(size: int) -> float:
        start = time.perf_counter()
        for r in range(probe_runs):
            batch = [_cycle(sample_frames, r * size + i) for i in range(size)]
            model(batch, verbose=False, conf=conf, iou=iou, device=device)
        if device == "cuda":
            import torch

            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return (size * probe_runs) / elapsed if elapsed > 0 else 0.0

    best_batch = candidates[0]
    best_tp = _measure(best_batch)
    for c in candidates[1:]:
        tp = _measure(c)
        if tp >= best_tp * (1.0 + min_improvement):
            best_tp, best_batch = tp, c
        else:
            break
    return best_batch


def resolve_effective_batch(
    device: str,
    model: object,
    frame_generator: Iterator[np.ndarray],
    requested: int,
    conf: float = 0.3,
    iou: float = 0.7,
    probe_frames: int = 64,
) -> tuple[int, Iterator[np.ndarray]]:
    """Resolve the batch size to use, auto-detecting when ``requested == 0``.

    When ``requested`` is a positive number the VRAM-capped value from
    ``resolve_batch_size`` is returned. When ``requested == 0`` (the auto
    sentinel) and a CUDA GPU is present, a handful of frames are peeked from
    ``frame_generator``, used to probe throughput, and then re-injected so the
    caller still processes every frame exactly once and in order.

    Args:
        device: ``"cuda"`` or ``"cpu"``.
        model: Loaded YOLO model already on ``device``.
        frame_generator: Iterator yielding video frames (numpy arrays).
        requested: Desired batch size, or ``0`` to auto-detect.
        conf: Confidence threshold (forwarded to the auto-detection probe).
        iou: IOU threshold (forwarded to the auto-detection probe).
        probe_frames: Max frames peeked for the auto-detection probe.

    Returns:
        A ``(effective_batch_size, frame_generator)`` tuple. The generator is
        the original one (CPU/non-auto paths) or a re-chained one (auto path).

    Examples:
        >>> import numpy as np
        >>> gen = iter([np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(3)])
        >>> size, _ = resolve_effective_batch("cpu", None, gen, 16)
        >>> size
        1
    """
    if device != "cuda" or requested != 0:
        return resolve_batch_size(device, requested), frame_generator

    max_batch = resolve_batch_size(device, 128)
    need = min(max(64, max_batch), 256)
    samples: list[np.ndarray] = []
    for _ in range(need):
        try:
            samples.append(next(frame_generator))
        except StopIteration:
            break

    if not samples:
        return 1, frame_generator

    detected = auto_detect_batch_size(model, samples, device, max_batch, conf, iou)
    logger.info(
        "Auto-detected optimal batch size: %s (VRAM cap=%s, probe frames=%s)",
        detected,
        max_batch,
        len(samples),
    )
    return detected, itertools.chain(samples, frame_generator)


def iter_batches(iterable: Iterable[_T], batch_size: int) -> Iterator[list[_T]]:
    """Yield consecutive lists of up to ``batch_size`` items from ``iterable``.

    Args:
        iterable: Any iterable (e.g. a video frame generator).
        batch_size: Maximum number of items per yielded batch (``>= 1``).

    Yields:
        Lists containing up to ``batch_size`` items, in order. The final batch
        may be shorter.

    Examples:
        >>> list(iter_batches(range(5), 2))
        [[0, 1], [2, 3], [4]]
    """
    batch_size = max(1, batch_size)
    batch: list[_T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
