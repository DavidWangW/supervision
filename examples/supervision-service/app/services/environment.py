"""Weather / road-surface visual recognition (Phase 4).

This module is the **business logic** for classifying the ambient environment
of a traffic scene directly from video frames:

* ``weather``      - 晴 / 多云 / 雨 / 雪 / 雾
* ``road_condition`` - 干燥 / 潮湿 / 积水 / 积雪 / 结冰
* ``visibility``   - 好 / 中 / 差

Because the dedicated recognition model is not trained yet (it will later use a
fine-tuned classifier, possibly reusing the ``yolo26x`` detection backbone), the
current implementation is a transparent **classical-CV heuristic**
(``SceneClassifier``). Every design decision is isolated behind a single
``classify(frame)`` boundary so the heuristic can be swapped for a trained model
without touching the call sites in ``analytics`` / ``stream_manager``.

The classifier returns an :class:`EnvironmentResult` carrying the dominant label
per dimension plus a full probability distribution, so the UI can show
confidence and an operator can override the auto-detection.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

#: Candidate labels per dimension. Kept in sync with the UI option lists and the
#: risk engine's normalization maps (``app.services.risk_engine``).
WEATHER_LABELS = ["晴", "多云", "雨", "雪", "雾"]
ROAD_LABELS = ["干燥", "潮湿", "积水", "积雪", "结冰"]
VISIBILITY_LABELS = ["好", "中", "差"]

#: Identifier stored alongside each reading so a future model swap is auditable.
DEFAULT_MODEL = "heuristic-v1"


@dataclass
class SceneFeatures:
    """Compact numeric description of a single frame's visual scene."""

    brightness: float
    contrast: float
    fog_index: float
    rain_index: float
    snow_index: float
    specular_index: float
    is_night: bool


@dataclass
class EnvironmentResult:
    """Classification outcome for one frame (or an aggregated window)."""

    weather: str
    weather_probs: dict[str, float]
    road_condition: str
    road_probs: dict[str, float]
    visibility: str
    visibility_probs: dict[str, float]
    is_night: bool
    features: SceneFeatures
    model: str = DEFAULT_MODEL

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (used for DB + API payloads)."""
        payload = asdict(self)
        payload["features"] = asdict(self.features)
        return payload


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    """Convert raw per-label scores into a normalized probability distribution."""
    keys = list(scores.keys())
    values = np.array([float(scores[k]) for k in keys], dtype=float)
    values -= float(values.max())
    exp = np.exp(values)
    probs = exp / float(exp.sum())
    return {k: round(float(p), 4) for k, p in zip(keys, probs)}


class SceneClassifier:
    """Recognize weather / road condition / visibility from a BGR frame.

    The public API is :meth:`classify` (frame in, :class:`EnvironmentResult` out).
    Internally it first extracts :class:`SceneFeatures` and then maps them to
    labels; :meth:`classify_features` lets callers classify pre-computed
    features (e.g. an exponential moving average over a clip).
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    # -- feature extraction -------------------------------------------------
    def extract_features(self, frame: np.ndarray) -> SceneFeatures:
        """Compute the numeric scene features used by the heuristic.

        Args:
            frame: A BGR ``np.ndarray`` (H, W, 3), e.g. a decoded video frame.

        Returns:
            A :class:`SceneFeatures` summary of brightness, contrast, haze,
            rain, snow and road-specular indicators.

        Raises:
            ValueError: If the frame is empty or has no color channels.
        """
        if frame is None or frame.size == 0 or frame.ndim != 3:
            raise ValueError("无法从空帧提取环境特征。")
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        brightness = float(gray.mean()) / 255.0
        # Laplacian variance is a cheap, scale-aware contrast / sharpness proxy.
        # Use CV_32F to match the float32 ``gray`` (avoids an AVX2 dtype mismatch).
        contrast = float(cv2.Laplacian(gray, cv2.CV_32F).var()) / 1000.0

        # Dark-channel prior: hazy frames have a high minimum-per-patch channel
        # value (light is scattered back uniformly).
        min_ch = rgb.min(axis=2)
        dark = cv2.erode(
            min_ch, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        )
        fog_index = float(dark.mean()) / 255.0

        # Rain shows as many small high-frequency bright specks relative to a
        # smoothed (rain-free) version of the frame.
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        rain_index = float((cv2.absdiff(gray, blur) > 40).mean())

        # Snow: large fraction of near-white pixels (falling snow + bright scene).
        white = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] > 200)
        snow_index = float(white.mean())

        # Wet / icy road surfaces produce sharp specular highlights in the lower
        # (road) region of the frame, where the camera looks down at tarmac.
        road = gray[int(h * 0.45) :, :]
        road_blur = cv2.GaussianBlur(road, (0, 0), sigmaX=5)
        spec_mask = (road > 0.55 * 255) & (cv2.absdiff(road, road_blur) > 25)
        specular_index = float(spec_mask.mean())

        is_night = brightness < 0.18
        return SceneFeatures(
            brightness=brightness,
            contrast=contrast,
            fog_index=fog_index,
            rain_index=rain_index,
            snow_index=snow_index,
            specular_index=specular_index,
            is_night=is_night,
        )

    # -- label mapping ------------------------------------------------------
    def classify_features(self, f: SceneFeatures) -> EnvironmentResult:
        """Map extracted features to labels + probabilities.

        The scoring is intentionally a transparent weighted sum so each factor
        is explainable; a future trained model only needs to replace this body
        while keeping the same :class:`EnvironmentResult` contract.
        """
        weather_scores = {
            "晴": f.contrast * 1.0 - f.fog_index * 1.5 - f.rain_index * 6
            - f.snow_index * 6 + (0.4 if not f.is_night else -0.2),
            "多云": (0.6 - abs(f.brightness - 0.45)) - f.fog_index * 1.0
            - f.rain_index * 4 - f.snow_index * 4 + f.contrast * 0.3,
            "雨": f.rain_index * 30 - f.fog_index * 0.5 - f.snow_index * 5,
            "雪": f.snow_index * 40 - f.rain_index * 2,
            "雾": f.fog_index * 15 - f.contrast * 2 - f.rain_index * 1,
        }
        if f.is_night and weather_scores["雨"] <= 0 and weather_scores["雪"] <= 0:
            # At night with no precipitation, treat the scene as clear (night).
            weather_scores["晴"] = max(weather_scores["晴"], 0.8)

        road_scores = {
            "干燥": 0.6 - f.specular_index * 6 - f.snow_index * 8,
            "潮湿": f.specular_index * 10 - 0.1,
            "积水": max(0.0, f.specular_index - 0.10) * 30,
            "积雪": f.snow_index * 35,
            # Ice: specular highlights on a very smooth (low-contrast) surface.
            "结冰": f.specular_index
            * 18
            * (0.5 + (1 - min(f.contrast / 0.05, 1)) * 0.5),
        }

        visibility_scores = {
            "好": 1.0 - f.fog_index * 3 - f.rain_index * 8 - f.snow_index * 12,
            "中": 0.3 + f.fog_index * 1.5 + f.rain_index * 4 + f.snow_index * 6,
            "差": f.fog_index * 3 + f.rain_index * 10 + f.snow_index * 14,
        }

        weather = max(weather_scores, key=weather_scores.get)
        road = max(road_scores, key=road_scores.get)
        visibility = max(visibility_scores, key=visibility_scores.get)
        return EnvironmentResult(
            weather=weather,
            weather_probs=_softmax(weather_scores),
            road_condition=road,
            road_probs=_softmax(road_scores),
            visibility=visibility,
            visibility_probs=_softmax(visibility_scores),
            is_night=f.is_night,
            features=f,
            model=self.model,
        )

    def classify(self, frame: np.ndarray, detections: object | None = None) -> EnvironmentResult:
        """Classify a single frame.

        Args:
            frame: A BGR ``np.ndarray`` frame.
            detections: Optional detection result (unused by the heuristic today,
                reserved for a future model that conditions on vehicles/objects).

        Returns:
            An :class:`EnvironmentResult` for the frame.
        """
        return self.classify_features(self.extract_features(frame))


def create_scene_classifier(backend: str | None = None):
    """Build the scene classifier selected by ``backend`` / ``SV_ENV_BACKEND``.

    Args:
        backend: ``"heuristic"`` or ``"vlm"``. ``None`` (default) resolves to
            the configured ``app.config.ENV_BACKEND``; ``"hybrid"`` maps to the
            VLM classifier here because the hybrid behaviour (heuristic
            per-frame + background VLM sampling) lives in the analyzer, not in
            the classifier itself.

    Returns:
        A :class:`SceneClassifier` or
        :class:`~app.services.vlm_environment.VLMSceneClassifier` instance,
        both exposing ``classify(frame) -> EnvironmentResult``.

    Example:
        >>> classifier = create_scene_classifier("heuristic")
        >>> classifier.model
        'heuristic-v1'
    """
    # Imported lazily: config for env vars, vlm_environment to avoid a
    # circular import (it imports this module's label vocabulary).
    from app import config

    resolved = (backend or config.ENV_BACKEND).strip().lower()
    if resolved in ("vlm", "hybrid"):
        from app.services.vlm_environment import VLMSceneClassifier

        return VLMSceneClassifier(
            base_url=config.VLM_BASE_URL,
            api_key=config.VLM_API_KEY,
            model=config.VLM_MODEL,
            timeout=config.VLM_TIMEOUT,
        )
    return SceneClassifier()
