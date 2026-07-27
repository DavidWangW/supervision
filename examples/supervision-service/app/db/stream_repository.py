"""Persistence for configured RTSP / live stream sources.

A ``stream_source`` row stores everything needed to (re)start a resident
analysis worker for one camera: its RTSP URL, the four-point road calibration,
lane split and optional environment labels. Structured analytics produced by a
running stream are written to the shared ``traffic_metrics`` / ``risk_assessments``
tables keyed by the stream id (reusing the ``upload_id`` column), so the existing
analytics query endpoints work for streams as well as uploaded files.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.db.database import _now_iso, get_connection


@dataclass
class StreamSourceRecord:
    """A configured live stream source."""

    id: str
    name: str
    url: str
    source_points: list[list[int]]
    target_width: float
    target_height: float
    lane_count: int
    weather: str | None
    road_condition: str | None
    visibility: str | None
    sample_fps: float
    confidence_threshold: float
    iou_threshold: float
    status: str
    error_message: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "source_points": self.source_points,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "lane_count": self.lane_count,
            "weather": self.weather,
            "road_condition": self.road_condition,
            "visibility": self.visibility,
            "sample_fps": self.sample_fps,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def create_stream_source(
    *,
    name: str,
    url: str,
    source_points: list[list[int]],
    target_width: float,
    target_height: float,
    lane_count: int = 3,
    weather: str | None = None,
    road_condition: str | None = None,
    visibility: str | None = None,
    sample_fps: float = 10.0,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> StreamSourceRecord:
    """Insert a new stream source (initially stopped) and return it."""
    stream_id = uuid4().hex
    now = _now_iso()
    record = StreamSourceRecord(
        id=stream_id,
        name=name,
        url=url,
        source_points=source_points,
        target_width=target_width,
        target_height=target_height,
        lane_count=lane_count,
        weather=weather,
        road_condition=road_condition,
        visibility=visibility,
        sample_fps=sample_fps,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        status="stopped",
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO stream_sources (
                id, name, url, source_points, target_width, target_height,
                lane_count, weather, road_condition, visibility, sample_fps,
                confidence_threshold, iou_threshold, status, error_message,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.name,
                record.url,
                json.dumps(record.source_points),
                record.target_width,
                record.target_height,
                record.lane_count,
                record.weather,
                record.road_condition,
                record.visibility,
                record.sample_fps,
                record.confidence_threshold,
                record.iou_threshold,
                record.status,
                record.error_message,
                record.created_at,
                record.updated_at,
            ),
        )
        connection.commit()
    return record


def update_stream_status(
    stream_id: str, status: str, error_message: str | None = None
) -> None:
    """Update a stream's runtime status and optional error message."""
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE stream_sources
            SET status = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error_message, _now_iso(), stream_id),
        )
        connection.commit()


def get_stream_source(stream_id: str) -> StreamSourceRecord | None:
    """Return a single stream source by id."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM stream_sources WHERE id = ?", (stream_id,)
        ).fetchone()
    return _row_to_record(row) if row is not None else None


def list_stream_sources() -> list[StreamSourceRecord]:
    """Return all configured stream sources, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM stream_sources ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def delete_stream_source(stream_id: str) -> bool:
    """Delete a stream source and its accumulated analytics rows."""
    record = get_stream_source(stream_id)
    if record is None:
        return False
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM traffic_metrics WHERE upload_id = ?", (stream_id,)
        )
        connection.execute(
            "DELETE FROM risk_assessments WHERE upload_id = ?", (stream_id,)
        )
        connection.execute("DELETE FROM stream_sources WHERE id = ?", (stream_id,))
        connection.commit()
    return True


def _row_to_record(row: Any) -> StreamSourceRecord:
    return StreamSourceRecord(
        id=row["id"],
        name=row["name"],
        url=row["url"],
        source_points=json.loads(row["source_points"]),
        target_width=row["target_width"],
        target_height=row["target_height"],
        lane_count=row["lane_count"],
        weather=row["weather"],
        road_condition=row["road_condition"],
        visibility=row["visibility"],
        sample_fps=row["sample_fps"],
        confidence_threshold=row["confidence_threshold"],
        iou_threshold=row["iou_threshold"],
        status=row["status"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
