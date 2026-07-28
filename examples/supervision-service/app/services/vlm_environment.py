"""VLM-based weather / road-surface recognition (heuristic-v1 replacement).

This module provides :class:`VLMSceneClassifier`, a drop-in alternative to the
classical-CV :class:`~app.services.environment.SceneClassifier`. It sends a
downscaled JPEG of the frame to an OpenAI-compatible ``/chat/completions``
endpoint (e.g. LMStudio serving ``gemma-4-26b-a4b-it``) and parses the
constrained-JSON answer back into the same :class:`EnvironmentResult` contract,
so call sites in ``analytics`` / ``routers.environment`` remain untouched.

Design notes (validated against a local LMStudio deployment):

* ``temperature=0`` makes the answer fully deterministic for a given frame.
* The request bypasses any system proxy (LAN endpoint) via an empty
  ``ProxyHandler`` — a system proxy previously caused opaque 502 responses.
* The prompt restricts labels to the exact vocabulary used by the risk engine
  (``WEATHER_LABELS`` / ``ROAD_LABELS`` / ``VISIBILITY_LABELS``); any
  out-of-vocabulary answer raises ``ValueError`` so callers can fall back to
  the heuristic.
* ``EnvironmentResult.features`` is still filled by the local heuristic
  extractor: it is essentially free and keeps the DB payload shape identical.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from app.services.environment import (
    ROAD_LABELS,
    TRAFFIC_LABELS,
    VISIBILITY_LABELS,
    WEATHER_LABELS,
    EnvironmentResult,
    SceneClassifier,
)

logger = logging.getLogger(__name__)

#: Longest frame side sent to the VLM. 512 px keeps vision-token cost (and
#: thus prompt-prefill time, the dominant cost on a local 26B model) low;
#: weather / road-surface are coarse cues so this resolution is sufficient.
DEFAULT_MAX_SIDE = 512

_JSON_BLOB_RE = re.compile(r"\{.*\}", re.DOTALL)

#: Built-in fallback prompt, used only when the editable prompt file
#: (``app.config.VLM_PROMPT_FILE``) cannot be read. Labels are inlined so this
#: string is self-sufficient; the file version uses ``__*_OPTIONS__``
#: placeholders that are filled from the shared vocabulary constants.
DEFAULT_VLM_PROMPT = (
    "你是高速公路监控环境识别助手。请观察图片,只输出JSON,"
    '格式: {"weather": "晴|多云|雨|雪|雾", '
    '"road_condition": "干燥|潮湿|积水|积雪|结冰", '
    '"visibility": "好|中|差", '
    '"traffic_condition": "畅通|缓行|拥堵|严重拥堵", '
    '"is_night": true/false, '
    '"description": "一句话描述"}'
)


def _one_hot(labels: list[str], chosen: str) -> dict[str, float]:
    """Build a degenerate probability distribution for a single chosen label."""
    return {label: (1.0 if label == chosen else 0.0) for label in labels}


class VLMSceneClassifier:
    """Recognize weather / road condition / visibility via a vision LLM.

    Implements the same ``classify(frame) -> EnvironmentResult`` boundary as
    :class:`~app.services.environment.SceneClassifier` so the two are
    interchangeable behind :func:`~app.services.environment.create_scene_classifier`.

    Args:
        base_url: OpenAI-compatible API root, e.g. ``http://2.0.0.1:1234/v1``.
        api_key: Bearer token; omitted from the request when empty.
        model: Upstream model identifier, e.g. ``gemma-4-26b-a4b-it``.
        timeout: Per-request timeout in seconds.
        temperature: Sampling temperature (0 = deterministic, recommended).
        max_tokens: Completion budget (covers reasoning + JSON answer).
        max_side: Longest image side after downscaling.

    Example:
        >>> classifier = VLMSceneClassifier(
        ...     base_url="http://2.0.0.1:1234/v1",
        ...     api_key="",
        ...     model="gemma-4-26b-a4b-it",
        ... )
        >>> # result = classifier.classify(frame)  # requires a live endpoint
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.0,
        # Budget covers hidden reasoning + the JSON answer. Local measurements
        # saw ~530-740 reasoning tokens, but occasional runs exceed 1024, which
        # yields an empty visible answer — hence the generous default.
        max_tokens: int = 2048,
        max_side: int = DEFAULT_MAX_SIDE,
        prompt_file: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model
        #: Audit identifier stored on every reading (DB ``model`` column).
        self.model = f"vlm:{model}"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_side = max_side
        self._feature_extractor = SceneClassifier()
        # Editable instruction prompt (re-read from disk on each request, cached
        # by mtime) so operators can tune recognition wording without a restart.
        self.prompt_file = Path(prompt_file) if prompt_file else None
        self._prompt_cache: tuple[float, str] | None = None
        self._prompt = self._load_prompt()
        # Bypass any system proxy: the endpoint is a LAN/localhost service and
        # proxied requests were observed to fail with opaque 502 responses.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )

    # -- prompt loading -------------------------------------------------------
    def _inject_labels(self, text: str) -> str:
        """Fill the ``__*_OPTIONS__`` placeholders with the shared vocabularies."""
        return (
            text.replace("__WEATHER_OPTIONS__", "、".join(WEATHER_LABELS))
            .replace("__ROAD_OPTIONS__", "、".join(ROAD_LABELS))
            .replace("__VISIBILITY_OPTIONS__", "、".join(VISIBILITY_LABELS))
            .replace("__TRAFFIC_OPTIONS__", "、".join(TRAFFIC_LABELS))
        )

    def _load_prompt(self) -> str:
        """Return the instruction prompt, preferring the editable config file.

        The configured file (``prompt_file``) is re-read on every call but
        cached by its modification time, so editing the file takes effect on
        the next request without restarting the service. Falls back to the
        built-in :data:`DEFAULT_VLM_PROMPT` when the file is missing/unreadable.
        """
        raw: str | None = None
        if self.prompt_file is not None:
            try:
                mtime = self.prompt_file.stat().st_mtime
                if self._prompt_cache is not None and self._prompt_cache[0] == mtime:
                    return self._prompt_cache[1]
                raw = self.prompt_file.read_text(encoding="utf-8")
                self._prompt_cache = (mtime, raw)
            except OSError as exc:
                logger.warning("读取 VLM 提示词文件失败,回退内置默认提示词: %s", exc)
        if raw is None:
            raw = DEFAULT_VLM_PROMPT
        return self._inject_labels(raw)

    # -- frame encoding -------------------------------------------------------
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Downscale ``frame`` and return it as a base64 JPEG string."""
        h, w = frame.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("无法将帧编码为 JPEG。")
        return base64.b64encode(buf.tobytes()).decode()

    # -- transport ------------------------------------------------------------
    def _request(self, image_b64: str) -> str:
        """POST the classification request; return the raw assistant content."""
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # S310: the URL scheme is constrained to the operator-configured
        # http(s) endpoint (SV_VLM_BASE_URL); no file:/custom schemes possible.
        request = urllib.request.Request(  # noqa: S310
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode())
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"VLM 响应格式异常: {data!r:.200}") from exc
        if not content:
            raise ValueError("VLM 返回了空正文(可能思考 token 耗尽)。")
        return content

    # -- parsing --------------------------------------------------------------
    @staticmethod
    def _parse_labels(content: str) -> dict:
        """Extract and validate the JSON label payload from ``content``.

        Handles answers wrapped in Markdown code fences or surrounded by prose.

        Raises:
            ValueError: If no JSON object is found or a label is outside the
                supported vocabulary.
        """
        match = _JSON_BLOB_RE.search(content)
        if match is None:
            raise ValueError(f"VLM 输出中未找到 JSON: {content[:200]!r}")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM JSON 解析失败: {content[:200]!r}") from exc

        weather = parsed.get("weather")
        road = parsed.get("road_condition")
        visibility = parsed.get("visibility")
        traffic = parsed.get("traffic_condition")
        if weather not in WEATHER_LABELS:
            raise ValueError(f"天气标签越界: {weather!r}")
        if road not in ROAD_LABELS:
            raise ValueError(f"路面标签越界: {road!r}")
        if visibility not in VISIBILITY_LABELS:
            raise ValueError(f"能见度标签越界: {visibility!r}")
        if traffic not in TRAFFIC_LABELS:
            raise ValueError(f"交通状况标签越界: {traffic!r}")
        return parsed

    # -- public API -----------------------------------------------------------
    def classify(
        self, frame: np.ndarray, detections: object | None = None
    ) -> EnvironmentResult:
        """Classify a single BGR frame via the VLM.

        Args:
            frame: A BGR ``np.ndarray`` frame.
            detections: Unused; kept for interface parity with
                :class:`SceneClassifier`.

        Returns:
            An :class:`EnvironmentResult` whose labels come from the VLM and
            whose ``features`` come from the local heuristic extractor.

        Raises:
            ValueError: On empty frames, malformed responses or
                out-of-vocabulary labels.
            OSError / urllib.error.URLError: On transport failures; callers
                are expected to fall back to the heuristic classifier.
        """
        features = self._feature_extractor.extract_features(frame)
        content = self._request(self._encode_frame(frame))
        parsed = self._parse_labels(content)
        is_night = parsed.get("is_night")
        # Cap the free-text description so a chatty model cannot bloat the DB row.
        description = (parsed.get("description") or "")[:200]
        return EnvironmentResult(
            weather=parsed["weather"],
            weather_probs=_one_hot(WEATHER_LABELS, parsed["weather"]),
            road_condition=parsed["road_condition"],
            road_probs=_one_hot(ROAD_LABELS, parsed["road_condition"]),
            visibility=parsed["visibility"],
            visibility_probs=_one_hot(VISIBILITY_LABELS, parsed["visibility"]),
            is_night=bool(is_night) if is_night is not None else features.is_night,
            features=features,
            traffic_condition=parsed["traffic_condition"],
            traffic_probs=_one_hot(TRAFFIC_LABELS, parsed["traffic_condition"]),
            description=description,
            model=self.model,
        )
