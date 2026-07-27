"""Unit tests for the live stream-source persistence (Phase 3).

Verifies CRUD plus the cascade cleanup of accumulated analytics when a stream
source is deleted.
"""

from app.db import analytics_repository as repo
from app.db import stream_repository as srepo
from app.db.analytics_repository import (
    RiskAssessmentRow,
    TrafficMetricRow,
)


def _sample_points() -> list[list[int]]:
    return [[0, 0], [1, 0], [1, 1], [0, 1]]


def test_stream_source_crud(temp_db) -> None:
    record = srepo.create_stream_source(
        name="cam1",
        url="rtsp://example/stream",
        source_points=_sample_points(),
        target_width=10,
        target_height=50,
        lane_count=3,
    )
    assert record.id
    assert record.status == "stopped"

    fetched = srepo.get_stream_source(record.id)
    assert fetched.id == record.id
    assert fetched.lane_count == 3

    assert srepo.list_stream_sources()[0].id == record.id

    srepo.update_stream_status(record.id, "running")
    assert srepo.get_stream_source(record.id).status == "running"

    assert srepo.delete_stream_source(record.id) is True
    assert srepo.get_stream_source(record.id) is None


def test_delete_cascades_analytics(temp_db) -> None:
    record = srepo.create_stream_source(
        name="cam2",
        url="rtsp://example/stream2",
        source_points=_sample_points(),
        target_width=10,
        target_height=50,
    )
    sid = record.id
    repo.insert_traffic_metrics(
        [TrafficMetricRow(sid, 0, 0, 5, 80.0, 3.0, 2.0, 10.0, 0.1, 3)]
    )
    repo.insert_risk_assessments(
        [
            RiskAssessmentRow(
                sid, "2026-01-01T00:00:00", 0, "低", 5.0, 0.1, [], "晴", "干燥"
            )
        ]
    )

    assert srepo.delete_stream_source(sid) is True
    assert repo.list_traffic_metrics(sid) == []
    assert repo.latest_risk_assessments(sid) == []
