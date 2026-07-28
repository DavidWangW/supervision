"""Environment recognition endpoints (Phase 4).

Exposes the visual recognition logic so the UI can (a) classify a single frame
the operator points at, and (b) learn the supported label vocabulary. The heavy
lifting lives in :mod:`app.services.environment`; this router is only transport.
"""

import base64
import logging
import re

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.config import ENV_BACKEND, VLM_MODEL
from app.services.environment import (
    DEFAULT_MODEL,
    ROAD_LABELS,
    VISIBILITY_LABELS,
    WEATHER_LABELS,
    SceneClassifier,
    create_scene_classifier,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/environment", tags=["environment"])


class DetectRequest(BaseModel):
    """A single image frame to classify (data URL or raw base64)."""

    image: str = Field(..., description="data URL 或原始 base64 的 JPEG/PNG 图像")
    backend: str | None = Field(
        None,
        description="识别后端: heuristic | vlm。缺省时使用服务配置 (SV_ENV_BACKEND)。",
    )


class DetectResponse(BaseModel):
    """Classification outcome mirroring :class:`EnvironmentResult`."""

    weather: str
    weather_probs: dict[str, float]
    road_condition: str
    road_probs: dict[str, float]
    visibility: str
    visibility_probs: dict[str, float]
    is_night: bool
    model: str
    features: dict


def _decode_image(data_url: str) -> np.ndarray:
    """Decode a data URL / raw base64 string into a BGR frame."""
    encoded = data_url.split(",", 1)[-1] if "," in data_url else data_url
    encoded = re.sub(r"\s+", "", encoded)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail="图像 base64 编码无效。") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="无法解码图像，请提供有效的 JPEG/PNG。")
    return frame


@router.post("/detect", response_model=DetectResponse)
async def detect_environment(request: DetectRequest) -> DetectResponse:
    """Classify weather / road condition / visibility from one image frame.

    The backend follows the service configuration (``SV_ENV_BACKEND``) unless
    the request pins one explicitly. VLM classification is executed in a
    thread pool (a single call takes ~10 s) so the event loop stays free.
    When the VLM backend was *implicitly* selected via configuration, failures
    silently fall back to the heuristic; an explicitly requested ``vlm``
    backend surfaces the failure as HTTP 502 instead.
    """
    backend = (request.backend or ENV_BACKEND).strip().lower()
    if backend not in ("heuristic", "vlm", "hybrid"):
        raise HTTPException(status_code=400, detail=f"未知识别后端: {backend!r}")

    frame = _decode_image(request.image)
    if backend in ("vlm", "hybrid"):
        classifier = create_scene_classifier("vlm")
        try:
            result = await run_in_threadpool(classifier.classify, frame)
            return DetectResponse(**result.to_dict())
        except Exception as exc:
            if request.backend is not None:
                raise HTTPException(
                    status_code=502, detail=f"VLM 环境识别失败: {exc}"
                ) from exc
            logger.warning("VLM 环境识别失败,回退启发式: %s", exc)
    result = SceneClassifier().classify(frame)
    return DetectResponse(**result.to_dict())


@router.get("/labels")
def environment_labels() -> dict:
    """Return the candidate label sets for the UI dropdowns."""
    active_model = (
        DEFAULT_MODEL if ENV_BACKEND == "heuristic" else f"vlm:{VLM_MODEL}"
    )
    return {
        "weather": WEATHER_LABELS,
        "road": ROAD_LABELS,
        "visibility": VISIBILITY_LABELS,
        "model": active_model,
        "backend": ENV_BACKEND,
    }
