"""Tests for VLM traffic-accident (交通肇事) detection.

The VLM classifier should only flag ``has_accident`` when the evidence is
unambiguous, and must leave ``accident_desc`` empty whenever the accident is
uncertain. These tests exercise the parsing / coercion logic without hitting
the network or the vision model.
"""

import json

import numpy as np
import pytest

from app.services import environment as env_mod
from app.services.analytics import TrafficFrameAnalyzer
from app.services.environment import EnvironmentResult, SceneFeatures
from app.services.vlm_environment import (
    VLMSceneClassifier,
    _as_bool,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        ("false", False),
        ("是", True),
        ("否", False),
        ("yes", True),
        (None, False),
        ("maybe", False),
        ("", False),
    ],
)
def test_as_bool(value, expected) -> None:
    # Only an explicit, unambiguous *true* signal returns True; everything else
    # (including ambiguous strings) is treated as False so we never surface a
    # speculative accident notice.
    assert _as_bool(value) is expected


def _make_classifier() -> VLMSceneClassifier:
    classifier = VLMSceneClassifier(
        base_url="http://example/v1",
        api_key="",
        model="gemma-4-26b-a4b-it",
        max_side=16,
    )
    return classifier


def _classify_with_payload(payload: dict) -> EnvironmentResult:
    classifier = _make_classifier()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    # Bypass the network + frame encoding so we only test parsing logic.
    classifier._encode_frame = lambda f: "BASE64"  # type: ignore[assignment]
    classifier._request = lambda b64: json.dumps(payload)  # type: ignore[assignment]
    return classifier.classify(frame)


def test_classify_reports_clear_accident(monkeypatch) -> None:
    # A clearly visible accident must be reported with a non-empty summary.
    result = _classify_with_payload(
        {
            "weather": "晴",
            "road_condition": "干燥",
            "visibility": "好",
            "traffic_condition": "严重拥堵",
            "is_night": False,
            "has_accident": True,
            "accident_desc": "小轿车追尾货车，占用最右侧车道",
            "description": "前方车辆排队",
        }
    )
    assert result.has_accident is True
    assert "追尾" in result.accident_desc
    assert result.accident_desc == "小轿车追尾货车，占用最右侧车道"


def test_classify_suppresses_desc_when_not_accident(monkeypatch) -> None:
    # When the model says no accident (here with a stray, non-empty desc), the
    # summary must be force-cleared so the dashboard shows nothing.
    result = _classify_with_payload(
        {
            "weather": "雨",
            "road_condition": "潮湿",
            "visibility": "中",
            "traffic_condition": "缓行",
            "is_night": False,
            "has_accident": False,
            "accident_desc": "疑似事故但不确定",
            "description": "雨天缓行",
        }
    )
    assert result.has_accident is False
    assert result.accident_desc == ""


def test_classify_suppresses_desc_when_uncertain(monkeypatch) -> None:
    # An ambiguous / missing signal must never produce an accident notice.
    result = _classify_with_payload(
        {
            "weather": "晴",
            "road_condition": "干燥",
            "visibility": "好",
            "traffic_condition": "畅通",
            "is_night": False,
            # has_accident omitted entirely -> treat as uncertain.
            "description": "路面畅通",
        }
    )
    assert result.has_accident is False
    assert result.accident_desc == ""


def test_merge_vlm_carries_accident(monkeypatch) -> None:
    # The merged environment result must inherit the VLM's accident fields.
    monkeypatch.setattr("app.services.analytics.ENV_BACKEND", "heuristic")
    analyzer = TrafficFrameAnalyzer(
        source_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        target_width=10.0,
        target_height=10.0,
        fps=10.0,
        resolution_wh=(64, 64),
    )
    vlm_result = EnvironmentResult(
        weather="晴",
        weather_probs={"晴": 1.0},
        road_condition="干燥",
        road_probs={"干燥": 1.0},
        visibility="好",
        visibility_probs={"好": 1.0},
        is_night=False,
        features=SceneFeatures(0.5, 0.5, 0.1, 0.0, 0.0, 0.0, False),
        traffic_condition="严重拥堵",
        traffic_probs={"严重拥堵": 1.0},
        description="事故画面",
        has_accident=True,
        accident_desc="两车追尾占用应急车道",
        model="vlm:gemma",
    )
    analyzer._vlm_environment = vlm_result

    heuristic = env_mod.SceneClassifier().classify_features(
        SceneFeatures(0.5, 0.5, 0.1, 0.0, 0.0, 0.0, False)
    )
    merged = analyzer._merge_vlm(heuristic)
    assert merged.has_accident is True
    assert merged.accident_desc == "两车追尾占用应急车道"
