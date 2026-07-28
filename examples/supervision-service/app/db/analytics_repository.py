"""Data-access helpers for the traffic analytics tables.

These functions persist and query the structured outputs produced by the
unified analysis pipeline: per-lane per-minute ``traffic_metrics``, summarized
``vehicle_tracks``, environment readings, and rule-based ``risk_assessments``.

The module deliberately mirrors ``app.db.repository`` in style (plain
``sqlite3`` + ``row_factory`` rows) so it can be swapped for PostgreSQL /
TimescaleDB later without touching the call sites.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sqlite3

from app.db.database import _now_iso, get_connection


@dataclass
class TrafficMetricRow:
    """One per-lane, per-minute aggregated traffic measurement."""

    upload_id: str
    lane_id: int
    minute_bucket: int
    flow_count: int
    avg_speed_kmh: float | None
    avg_headway_s: float | None
    min_headway_s: float | None
    density_per_km: float | None
    truck_ratio: float | None
    vehicle_count: int


@dataclass
class VehicleTrackRow:
    """Summarized trajectory for a single tracked vehicle."""

    upload_id: str
    track_id: int
    class_name: str | None
    vehicle_type: str | None
    first_seen: float
    last_seen: float
    avg_speed_kmh: float | None
    max_speed_kmh: float | None
    lane_id: int | None


@dataclass
class RiskAssessmentRow:
    """A rule-based risk assessment for an upload (optionally per lane)."""

    upload_id: str | None
    assessed_at: str
    lane_id: int | None
    risk_level: str
    risk_score: float
    probability: float | None
    factors: list[str]
    weather: str | None
    road_condition: str | None


def insert_traffic_metrics(rows: Sequence[TrafficMetricRow]) -> None:
    """Bulk insert aggregated per-lane traffic metrics."""
    if not rows:
        return
    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO traffic_metrics (
                upload_id, lane_id, minute_bucket, flow_count, avg_speed_kmh,
                avg_headway_s, min_headway_s, density_per_km, truck_ratio,
                vehicle_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.upload_id,
                    r.lane_id,
                    r.minute_bucket,
                    r.flow_count,
                    r.avg_speed_kmh,
                    r.avg_headway_s,
                    r.min_headway_s,
                    r.density_per_km,
                    r.truck_ratio,
                    r.vehicle_count,
                    _now_iso(),
                )
                for r in rows
            ],
        )
        connection.commit()


def insert_vehicle_tracks(rows: Sequence[VehicleTrackRow]) -> None:
    """Bulk insert summarized vehicle tracks."""
    if not rows:
        return
    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO vehicle_tracks (
                upload_id, track_id, class_name, vehicle_type, first_seen,
                last_seen, avg_speed_kmh, max_speed_kmh, lane_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.upload_id,
                    r.track_id,
                    r.class_name,
                    r.vehicle_type,
                    r.first_seen,
                    r.last_seen,
                    r.avg_speed_kmh,
                    r.max_speed_kmh,
                    r.lane_id,
                    _now_iso(),
                )
                for r in rows
            ],
        )
        connection.commit()


def insert_environment_reading(
    *,
    upload_id: str | None,
    observed_at: str,
    weather: str | None,
    road_condition: str | None,
    visibility: str | None,
    source: str,
    details: dict | None = None,
    model: str | None = None,
) -> None:
    """Insert a single environment reading (weather / road condition).

    ``details`` carries the full :class:`EnvironmentResult` payload (label
    probabilities + raw features) and ``model`` records which recognizer
    produced it, so a later model swap remains auditable.
    """
    import json

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO environment_readings (
                upload_id, observed_at, weather, road_condition, visibility,
                source, details_json, model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                observed_at,
                weather,
                road_condition,
                visibility,
                source,
                json.dumps(details, ensure_ascii=False) if details else None,
                model,
                _now_iso(),
            ),
        )
        connection.commit()


def insert_risk_assessments(rows: Sequence[RiskAssessmentRow]) -> None:
    """Bulk insert rule-based risk assessments."""
    if not rows:
        return
    import json

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO risk_assessments (
                upload_id, assessed_at, lane_id, risk_level, risk_score,
                probability, factors_json, weather, road_condition, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.upload_id,
                    r.assessed_at,
                    r.lane_id,
                    r.risk_level,
                    r.risk_score,
                    r.probability,
                    json.dumps(r.factors, ensure_ascii=False),
                    r.weather,
                    r.road_condition,
                    _now_iso(),
                )
                for r in rows
            ],
        )
        connection.commit()


def list_traffic_metrics(upload_id: str) -> list[dict[str, Any]]:
    """Return all traffic metrics for an upload ordered by lane then minute."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT lane_id, minute_bucket, flow_count, avg_speed_kmh,
                   avg_headway_s, min_headway_s, density_per_km, truck_ratio,
                   vehicle_count
            FROM traffic_metrics
            WHERE upload_id = ?
            ORDER BY lane_id, minute_bucket
            """,
            (upload_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_vehicle_tracks(upload_id: str) -> list[dict[str, Any]]:
    """Return all summarized vehicle tracks for an upload."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT track_id, class_name, vehicle_type, first_seen, last_seen,
                   avg_speed_kmh, max_speed_kmh, lane_id
            FROM vehicle_tracks
            WHERE upload_id = ?
            ORDER BY avg_speed_kmh DESC
            """,
            (upload_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_risk_assessments(upload_id: str) -> list[dict[str, Any]]:
    """Return all risk assessments for an upload, parsing the factors JSON."""
    import json

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, lane_id, risk_level, risk_score, probability,
                   factors_json, weather, road_condition, assessed_at
            FROM risk_assessments
            WHERE upload_id = ?
            ORDER BY lane_id
            """,
            (upload_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["factors"] = json.loads(item["factors_json"])
        except (json.JSONDecodeError, TypeError):
            item["factors"] = []
        result.append(item)
    return result


def latest_risk_assessments(upload_id: str) -> list[dict[str, Any]]:
    """Return only the most recent batch of risk assessments for an upload/stream.

    Long-running streams accumulate one risk batch per minute; the live console
    only needs the latest, so this filters to rows sharing the max ``assessed_at``.
    """
    import json

    with get_connection() as connection:
        row = connection.execute(
            "SELECT MAX(assessed_at) AS latest FROM risk_assessments WHERE upload_id = ?",
            (upload_id,),
        ).fetchone()
        if row is None or row["latest"] is None:
            return []
        rows = connection.execute(
            """
            SELECT id, lane_id, risk_level, risk_score, probability,
                   factors_json, weather, road_condition, assessed_at
            FROM risk_assessments
            WHERE upload_id = ? AND assessed_at = ?
            ORDER BY lane_id
            """,
            (upload_id, row["latest"]),
        ).fetchall()
    result = []
    for r in rows:
        item = dict(r)
        try:
            item["factors"] = json.loads(item["factors_json"])
        except (json.JSONDecodeError, TypeError):
            item["factors"] = []
        result.append(item)
    return result


def list_environment_readings(upload_id: str) -> list[dict[str, Any]]:
    """Return environment readings for an upload."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, observed_at, weather, road_condition, visibility, source,
                   details_json, model
            FROM environment_readings
            WHERE upload_id = ?
            ORDER BY observed_at
            """,
            (upload_id,),
        ).fetchall()
    return [_row_to_environment(row) for row in rows]


def _row_to_environment(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an ``environment_readings`` row into a JSON-friendly dict."""
    import json

    try:
        details = json.loads(row["details_json"]) if row["details_json"] else None
    except (json.JSONDecodeError, TypeError):
        details = None
    item = {
        "id": row["id"],
        "observed_at": row["observed_at"],
        "weather": row["weather"],
        "road_condition": row["road_condition"],
        "visibility": row["visibility"],
        "source": row["source"],
        "model": row["model"],
        # Surface the VLM-only dimensions (stored inside ``details_json``) at the
        # top level so the UI can read them without drilling into ``details``.
        "traffic_condition": details.get("traffic_condition") if details else None,
        "description": details.get("description") if details else None,
        "details": details,
    }
    return item


def latest_environment_reading(upload_id: str) -> dict[str, Any] | None:
    """Return the most recent environment reading (auto-detected preferred)."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, observed_at, weather, road_condition, visibility, source,
                   details_json, model
            FROM environment_readings
            WHERE upload_id = ?
            ORDER BY observed_at DESC, id DESC
            LIMIT 1
            """,
            (upload_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_environment(row)
