"""Unit tests for the structured analytics persistence layer (Phase 0).

Uses the ``temp_db`` fixture to redirect the SQLite file to an isolated temp
database so the real ``data/supervision.db`` is never touched.
"""

from app.db import analytics_repository as repo
from app.db.analytics_repository import (
    RiskAssessmentRow,
    TrafficMetricRow,
    VehicleTrackRow,
)


def test_traffic_metrics_roundtrip(temp_db) -> None:
    rows = [
        TrafficMetricRow("u1", 0, 0, 10, 90.0, 3.0, 2.0, 15.0, 0.1, 5),
        TrafficMetricRow("u1", 1, 1, 8, 80.0, 2.5, 1.8, 20.0, 0.2, 6),
    ]
    repo.insert_traffic_metrics(rows)
    got = repo.list_traffic_metrics("u1")
    assert len(got) == 2
    assert got[0]["lane_id"] == 0
    assert got[0]["avg_speed_kmh"] == 90.0
    assert repo.list_traffic_metrics("missing") == []


def test_vehicle_tracks_roundtrip(temp_db) -> None:
    rows = [
        VehicleTrackRow("u1", 1, "卡车", "卡车", 0.0, 5.0, 80.0, 100.0, 0),
    ]
    repo.insert_vehicle_tracks(rows)
    got = repo.list_vehicle_tracks("u1")
    assert got[0]["track_id"] == 1
    assert got[0]["avg_speed_kmh"] == 80.0


def test_environment_reading_roundtrip_and_latest(temp_db) -> None:
    repo.insert_environment_reading(
        upload_id="u1",
        observed_at="2026-01-01T00:00:00",
        weather="雪",
        road_condition="结冰",
        visibility="差",
        source="auto",
        details={"weather": "雪"},
        model="heuristic-v1",
    )
    got = repo.list_environment_readings("u1")
    assert got[0]["weather"] == "雪"
    assert got[0]["details"] == {"weather": "雪"}
    latest = repo.latest_environment_reading("u1")
    assert latest["road_condition"] == "结冰"


def test_risk_assessments_roundtrip_and_latest(temp_db) -> None:
    rows = [
        RiskAssessmentRow(
            "u1",
            "2026-01-01T00:00:00",
            0,
            "高",
            70.0,
            0.9,
            ["天气：雪（+30）"],
            "雪",
            "结冰",
        ),
        RiskAssessmentRow(
            "u1",
            "2026-01-01T00:00:00",
            None,
            "高",
            72.0,
            0.92,
            ["综合"],
            "雪",
            "结冰",
        ),
    ]
    repo.insert_risk_assessments(rows)
    got = repo.list_risk_assessments("u1")
    assert len(got) == 2
    # Ordered by lane_id; the lane_id=None overall row sorts first in SQLite.
    factors_list = [g["factors"] for g in got]
    assert ["天气：雪（+30）"] in factors_list
    assert ["综合"] in factors_list
    latest = repo.latest_risk_assessments("u1")
    assert len(latest) == 2


def test_empty_queries_return_empty(temp_db) -> None:
    assert repo.list_traffic_metrics("x") == []
    assert repo.latest_risk_assessments("x") == []
    assert repo.latest_environment_reading("x") is None
