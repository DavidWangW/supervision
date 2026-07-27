import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import DATABASE_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _migrate(connection: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(processing_jobs)").fetchall()
    }
    if "progress" not in columns:
        connection.execute(
            "ALTER TABLE processing_jobs ADD COLUMN progress INTEGER NOT NULL DEFAULT 0"
        )
    if "current_frame" not in columns:
        connection.execute(
            "ALTER TABLE processing_jobs ADD COLUMN current_frame INTEGER NOT NULL DEFAULT 0"
        )
    if "total_frames" not in columns:
        connection.execute(
            "ALTER TABLE processing_jobs ADD COLUMN total_frames INTEGER NOT NULL DEFAULT 0"
        )

    env_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(environment_readings)").fetchall()
    }
    if "details_json" not in env_columns:
        connection.execute(
            "ALTER TABLE environment_readings ADD COLUMN details_json TEXT"
        )
    if "model" not in env_columns:
        connection.execute(
            "ALTER TABLE environment_readings ADD COLUMN model TEXT"
        )


def init_db() -> None:
    """Create database tables if they do not exist."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                content_type TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS processing_jobs (
                id TEXT PRIMARY KEY,
                upload_id TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                output_path TEXT,
                parameters TEXT NOT NULL,
                error_message TEXT,
                progress INTEGER NOT NULL DEFAULT 0,
                current_frame INTEGER NOT NULL DEFAULT 0,
                total_frames INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (upload_id) REFERENCES uploads(id)
            );

            CREATE INDEX IF NOT EXISTS idx_uploads_created_at
                ON uploads(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_jobs_upload_id
                ON processing_jobs(upload_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at
                ON processing_jobs(created_at DESC);

            CREATE TABLE IF NOT EXISTS vehicle_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                class_name TEXT,
                vehicle_type TEXT,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                avg_speed_kmh REAL,
                max_speed_kmh REAL,
                lane_id INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (upload_id) REFERENCES uploads(id)
            );

            CREATE TABLE IF NOT EXISTS traffic_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT NOT NULL,
                lane_id INTEGER NOT NULL,
                minute_bucket INTEGER NOT NULL,
                flow_count INTEGER NOT NULL DEFAULT 0,
                avg_speed_kmh REAL,
                avg_headway_s REAL,
                min_headway_s REAL,
                density_per_km REAL,
                truck_ratio REAL,
                vehicle_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (upload_id) REFERENCES uploads(id)
            );

            CREATE TABLE IF NOT EXISTS environment_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT,
                observed_at TEXT NOT NULL,
                weather TEXT,
                road_condition TEXT,
                visibility TEXT,
                source TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS accident_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT,
                location TEXT,
                accident_type TEXT,
                weather TEXT,
                road_condition TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT,
                assessed_at TEXT NOT NULL,
                lane_id INTEGER,
                risk_level TEXT NOT NULL,
                risk_score REAL NOT NULL,
                probability REAL,
                factors_json TEXT,
                weather TEXT,
                road_condition TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS stream_sources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                source_points TEXT NOT NULL,
                target_width REAL NOT NULL,
                target_height REAL NOT NULL,
                lane_count INTEGER NOT NULL DEFAULT 3,
                weather TEXT,
                road_condition TEXT,
                visibility TEXT,
                sample_fps REAL NOT NULL DEFAULT 10,
                confidence_threshold REAL NOT NULL DEFAULT 0.3,
                iou_threshold REAL NOT NULL DEFAULT 0.7,
                status TEXT NOT NULL DEFAULT 'stopped',
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_upload
                ON traffic_metrics(upload_id, lane_id, minute_bucket);
            CREATE INDEX IF NOT EXISTS idx_tracks_upload
                ON vehicle_tracks(upload_id);
            CREATE INDEX IF NOT EXISTS idx_risk_upload
                ON risk_assessments(upload_id);
            CREATE INDEX IF NOT EXISTS idx_env_upload
                ON environment_readings(upload_id);
            CREATE INDEX IF NOT EXISTS idx_stream_created
                ON stream_sources(created_at DESC);
            """
        )
        _migrate(connection)
        connection.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row factory enabled."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()
