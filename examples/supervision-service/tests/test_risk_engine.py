"""Unit tests for the rule-based traffic risk scoring card (Phase 5).

These are pure-logic tests: no DB, no model inference.
"""

from app.services.risk_engine import (
    RiskResult,
    TrafficRiskInput,
    assess,
    score_single,
)


def test_clear_dry_safe_conditions_low_risk() -> None:
    result = score_single(
        TrafficRiskInput(
            lane_id=0,
            avg_speed_kmh=80,
            avg_headway_s=3.0,
            min_headway_s=2.5,
            density_per_km=15,
            truck_ratio=0.1,
            weather="晴",
            road_condition="干燥",
        )
    )
    assert result.risk_level == "低"
    assert result.risk_score == 0.0
    assert 0.0 <= result.probability <= 1.0


def test_snow_and_ice_yields_high_risk() -> None:
    result = score_single(
        TrafficRiskInput(lane_id=0, weather="雪", road_condition="结冰")
    )
    assert result.risk_score == 70.0
    assert result.risk_level == "高"
    assert any("雪" in f for f in result.factors)
    assert any("结冰" in f for f in result.factors)


def test_min_headway_extreme_boosts_score() -> None:
    result = score_single(TrafficRiskInput(lane_id=0, min_headway_s=0.8))
    assert result.risk_score >= 25.0
    assert any("追尾" in f for f in result.factors)


def test_density_thresholds_are_monotonic() -> None:
    low = score_single(TrafficRiskInput(lane_id=0, density_per_km=20)).risk_score
    mid = score_single(TrafficRiskInput(lane_id=0, density_per_km=30)).risk_score
    high = score_single(TrafficRiskInput(lane_id=0, density_per_km=50)).risk_score
    assert mid > low
    assert high > mid


def test_truck_ratio_contributes() -> None:
    base = score_single(TrafficRiskInput(lane_id=0)).risk_score
    heavy = score_single(TrafficRiskInput(lane_id=0, truck_ratio=0.5)).risk_score
    assert heavy == base + 10.0


def test_chinese_labels_normalized() -> None:
    result = score_single(
        TrafficRiskInput(lane_id=0, weather="下雨", road_condition="湿")
    )
    assert result.risk_score == 15.0 + 10.0
    assert result.weather == "下雨"


def test_assess_appends_overall_road_assessment() -> None:
    lane_inputs = [
        TrafficRiskInput(lane_id=0, weather="晴", road_condition="干燥", avg_speed_kmh=90),
        TrafficRiskInput(lane_id=1, weather="晴", road_condition="干燥", avg_speed_kmh=90),
    ]
    results = assess(lane_inputs)
    assert len(results) == 3
    assert results[-1].lane_id is None
    assert all(isinstance(r, RiskResult) for r in results)


def test_score_extremes_clamped_and_probability_in_range() -> None:
    very_bad = score_single(
        TrafficRiskInput(
            lane_id=0,
            weather="雪",
            road_condition="结冰",
            density_per_km=70,
            min_headway_s=0.5,
            truck_ratio=0.5,
            avg_speed_kmh=120,
        )
    )
    assert very_bad.risk_score <= 100.0
    assert very_bad.risk_level == "极高"
    assert 0.0 <= very_bad.probability <= 1.0
