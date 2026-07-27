"""Unit tests for the unified traffic analysis pipeline (Phases 0 + 1).

Covers the geometry helpers, the per-minute metric aggregation math and the
risk aggregation entry point. No YOLO weights are loaded by
``TrafficFrameAnalyzer.__init__``, so these run without a GPU/model.
"""

import numpy as np

from app.db.analytics_repository import TrafficMetricRow
from app.services import analytics
from app.services.analytics import (
    LaneLiveMetric,
    LiveSnapshot,
    RoadViewTransformer,
    TrafficFrameAnalyzer,
    _LaneAccumulator,
    compute_risk,
    derive_lane_polygons,
)


def _make_analyzer(lane_count: int = 2) -> TrafficFrameAnalyzer:
    return TrafficFrameAnalyzer(
        source_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        target_width=10,
        target_height=50,
        fps=25,
        resolution_wh=(100, 100),
        lane_count=lane_count,
    )


def test_road_view_transformer_roundtrip() -> None:
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    tgt = analytics.build_target_array(10, 50)
    transformer = RoadViewTransformer(src, tgt)
    pts = np.array([[50, 50], [20, 30]], dtype=np.float32)
    forward = transformer.transform_points(pts)
    back = transformer.inverse_transform_points(forward)
    np.testing.assert_allclose(back, pts, atol=1e-3)


def test_derive_lane_polygons_shape() -> None:
    src = [[0, 0], [100, 0], [100, 100], [0, 100]]
    polygons = derive_lane_polygons(src, 10, 50, 3)
    assert len(polygons) == 3
    for poly in polygons:
        assert len(poly) == 4


def test_compute_risk_safe_lanes() -> None:
    rows = [
        TrafficMetricRow("u1", 0, 0, 10, 90.0, 3.0, 2.0, 15.0, 0.1, 5),
        TrafficMetricRow("u1", 1, 0, 12, 80.0, 2.5, 1.8, 20.0, 0.2, 6),
    ]
    results = compute_risk(rows, lane_count=2, weather="晴", road_condition="干燥")
    # Two per-lane results plus one aggregated whole-road result.
    assert len(results) == 3
    assert results[-1].lane_id is None
    assert all(r.risk_level == "低" for r in results[:2])


def test_compute_risk_high_risk_lane() -> None:
    rows = [
        TrafficMetricRow("u2", 0, 0, 8, 100.0, 0.8, 0.7, 70.0, 0.5, 4),
    ]
    results = compute_risk(rows, lane_count=1, weather="雪", road_condition="结冰")
    assert results[0].risk_level in {"高", "极高"}


def test_build_metric_rows_aggregation() -> None:
    analyzer = _make_analyzer()
    analyzer.accumulators[(0, 0)] = _LaneAccumulator(
        flow_total=10,
        speed_sum=200.0,
        speed_n=2,
        headway_sum=4.0,
        headway_n=2,
        min_headway=1.5,
        density_sum=6.0,
        density_n=2,
        truck_count=1,
        vehicle_set={1, 2, 3},
    )
    rows = analyzer.build_metric_rows("u1")
    assert len(rows) == 1
    row = rows[0]
    assert row.lane_id == 0 and row.minute_bucket == 0
    assert row.flow_count == 10
    assert row.avg_speed_kmh == 100.0
    assert row.avg_headway_s == 2.0
    assert row.min_headway_s == 1.5
    assert row.density_per_km == 3.0
    assert row.vehicle_count == 3
    assert abs(row.truck_ratio - 0.333) < 1e-9


def test_pop_completed_metric_rows_flushes_only_finished_minutes() -> None:
    analyzer = _make_analyzer()
    analyzer.accumulators[(0, 0)] = _LaneAccumulator(flow_total=5, vehicle_set={1})
    analyzer.accumulators[(1, 0)] = _LaneAccumulator(flow_total=7, vehicle_set={2})

    flushed = analyzer.pop_completed_metric_rows("u", current_minute=1)

    assert len(flushed) == 1
    assert flushed[0].minute_bucket == 0
    assert (1, 0) in analyzer.accumulators
    assert (0, 0) not in analyzer.accumulators


def test_live_snapshot_to_dict() -> None:
    snapshot = LiveSnapshot(
        frame_index=1,
        t_sec=1.0,
        minute_bucket=0,
        total_vehicles=2,
        lanes=[LaneLiveMetric(lane_id=0, avg_speed_kmh=80.0)],
    )
    payload = snapshot.to_dict()
    assert payload["frame_index"] == 1
    assert payload["lanes"][0]["avg_speed_kmh"] == 80.0
