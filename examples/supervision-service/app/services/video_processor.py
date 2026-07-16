import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import supervision as sv

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_GPU_BATCH_SIZE,
    DEFAULT_IOU,
    DEFAULT_WEIGHTS,
)
from app.services.hardware import (
    iter_batches,
    resolve_effective_batch,
    torch_device,
)
from app.services.label_map import build_chinese_labels, resolve_chinese_font
from app.services.video_encoding import ensure_browser_playable

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


def draw_chinese_labels(
    scene: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    font_path: str | None,
    font_size: int = 24,
) -> np.ndarray:
    """Draw detection labels on a frame using a Chinese-capable font.

    supervision 0.29.x ``LabelAnnotator`` renders text via OpenCV's Hershey font,
    which cannot display Chinese glyphs and exposes no ``font_path`` option. This
    helper uses Pillow instead so localized labels render correctly regardless of
    the installed supervision version.

    Args:
        scene: BGR frame as a numpy array.
        detections: Detections produced by the tracker.
        labels: Chinese label per detection, aligned with ``detections``.
        font_path: Path to a ``.ttf``/``.ttc`` font; ``None`` falls back to the
            default PIL bitmap font.
        font_size: Font size in pixels.

    Returns:
        The annotated frame in BGR order.
    """
    if not labels or detections.xyxy.shape[0] == 0:
        return scene

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for (x1, y1, x2, y2), label in zip(detections.xyxy, labels):
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        box_x = int(x1)
        box_y = int(y1) - text_height - 10
        if box_y < 0:
            box_y = int(y1) + 6
        bg_x1 = box_x
        bg_y1 = box_y
        bg_x2 = box_x + text_width + 12
        bg_y2 = box_y + text_height + 8

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(20, 20, 20))
        draw.text((bg_x1 + 6, bg_y1 + 4), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def track_video(
    source_video_path: Path,
    target_video_path: Path,
    weights_path: Path = DEFAULT_WEIGHTS,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU,
    on_progress: ProgressCallback | None = None,
    batch_size: int = DEFAULT_GPU_BATCH_SIZE,
) -> Path:
    """Run YOLO detection and ByteTrack on a video file."""
    device = torch_device()
    model = YOLO(str(weights_path)).to(device)
    if device == "cuda":
        # Half precision roughly doubles GPU throughput with negligible accuracy loss.
        model.half()
    logger.info(
        "track_video: device=%s, weights=%s, half_precision=%s, requested_batch_size=%s",
        device,
        weights_path,
        device == "cuda",
        batch_size,
    )
    video_info = sv.VideoInfo.from_video_path(str(source_video_path))
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    chinese_font = resolve_chinese_font()
    label_font_size = max(12, round(min(video_info.resolution_wh) / 40))

    class_names = model.names
    frame_generator = sv.get_video_frames_generator(str(source_video_path))
    effective_batch, frame_generator = resolve_effective_batch(
        device=device,
        model=model,
        frame_generator=frame_generator,
        requested=batch_size,
        conf=confidence_threshold,
        iou=iou_threshold,
    )
    logger.info("track_video: effective_batch_size=%s", effective_batch)
    total_frames = video_info.total_frames

    if on_progress and total_frames:
        on_progress(0, total_frames)

    target_video_path.parent.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    with sv.VideoSink(str(target_video_path), video_info) as sink:
        for frames in iter_batches(frame_generator, effective_batch):
            # Batched inference maximizes GPU utilization; tracking, annotation
            # and writing stay strictly per-frame to preserve temporal order.
            batch_results = model(
                frames,
                verbose=False,
                conf=confidence_threshold,
                iou=iou_threshold,
                device=device,
            )

            for frame, results in zip(frames, batch_results):
                frame_index += 1
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)

                labels = (
                    build_chinese_labels(class_names, detections.class_id)
                    if detections.class_id is not None
                    else []
                )

                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections,
                )
                annotated_frame = draw_chinese_labels(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels,
                    font_path=chinese_font,
                    font_size=label_font_size,
                )
                sink.write_frame(frame=annotated_frame)

                if on_progress and total_frames:
                    on_progress(frame_index, total_frames)

    if on_progress and total_frames:
        on_progress(total_frames, total_frames)

    ensure_browser_playable(target_video_path)
    return target_video_path


def build_output_path(prefix: str = "tracked") -> Path:
    """Create a unique output video path."""
    from uuid import uuid4

    from app.config import OUTPUT_DIR

    return OUTPUT_DIR / f"{prefix}_{uuid4().hex[:8]}.mp4"
