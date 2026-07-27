"""Unit tests for the heuristic weather / road-surface classifier (Phase 4).

The classifier is deterministic given either a frame or an explicit
``SceneFeatures`` blob, so we test both the feature extraction and the
feature->label mapping without needing a trained model.
"""

import numpy as np

from app.services.environment import SceneClassifier, SceneFeatures


def _gray_frame(value: int) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def test_extract_features_returns_typed_features() -> None:
    cls = SceneClassifier()
    features = cls.extract_features(_gray_frame(128))
    assert isinstance(features, SceneFeatures)
    assert 0.0 <= features.brightness <= 1.0
    assert isinstance(features.is_night, bool)


def test_night_detected_from_dark_frame() -> None:
    cls = SceneClassifier()
    result = cls.classify(_gray_frame(15))
    assert result.is_night is True


def test_rain_label_from_features() -> None:
    cls = SceneClassifier()
    features = SceneFeatures(
        brightness=0.5,
        contrast=0.0,
        fog_index=0.0,
        rain_index=0.5,
        snow_index=0.0,
        specular_index=0.0,
        is_night=False,
    )
    result = cls.classify_features(features)
    assert result.weather == "雨"
    assert result.weather_probs["雨"] > result.weather_probs["晴"]


def test_ice_road_from_features() -> None:
    cls = SceneClassifier()
    features = SceneFeatures(
        brightness=0.5,
        contrast=0.01,
        fog_index=0.0,
        rain_index=0.0,
        snow_index=0.0,
        specular_index=0.15,
        is_night=False,
    )
    result = cls.classify_features(features)
    assert result.road_condition == "结冰"


def test_to_dict_serializes_payload() -> None:
    cls = SceneClassifier()
    result = cls.classify(_gray_frame(128))
    payload = result.to_dict()
    assert payload["weather"] in {"晴", "多云", "雨", "雪", "雾"}
    assert "features" in payload
    assert isinstance(payload["features"], dict)
