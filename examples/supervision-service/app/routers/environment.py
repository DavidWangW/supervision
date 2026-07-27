"""Environment recognition endpoints (Phase 4).

Exposes the visual recognition logic so the UI can (a) classify a single frame
the operator points at, and (b) learn the supported label vocabulary. The heavy
lifting lives in :mod:`app.services.environment`; this router is only transport.
"""

import base64
import re

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.environment import (
    DEFAULT_MODEL,
    ROAD_LABELS,
    VISIBILITY_LABELS,
    WEATHER_LABELS,
    SceneClassifier,
)

router = APIRouter(prefix="/api/v1/environment", tags=["environment"])


class DetectRequest(BaseModel):
    """A single image frame to classify (data URL or raw base64)."""

    image: str = Field(..., description="data URL 或原始 base64 的 JPEG/PNG 图像")


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
def detect_environment(request: DetectRequest) -> DetectResponse:
    """Classify weather / road condition / visibility from one image frame."""
    frame = _decode_image(request.image)
    result = SceneClassifier().classify(frame)
    return DetectResponse(**result.to_dict())


@router.get("/labels")
def environment_labels() -> dict:
    """Return the candidate label sets for the UI dropdowns."""
    return {
        "weather": WEATHER_LABELS,
        "road": ROAD_LABELS,
        "visibility": VISIBILITY_LABELS,
        "model": DEFAULT_MODEL,
    }
