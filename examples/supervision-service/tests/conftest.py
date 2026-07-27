"""Shared pytest fixtures for the supervision-service test-suite.

The analytics DB is a single SQLite file pointed to by ``app.db.database.DATABASE_PATH``.
We redirect that module-level path to an isolated temp file per test so the
structured-data tests never touch the real ``data/supervision.db``.
"""

import sys
from pathlib import Path

import pytest

# Make the ``app`` package importable regardless of the invocation cwd.
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.db import database as db_module  # noqa: E402


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the analytics DB at a fresh temp file and create the schema.

    Yields the temp DB path. Each test gets its own file, so suites are isolated.
    """
    db_path = tmp_path / "test_supervision.db"
    monkeypatch.setattr(db_module, "DATABASE_PATH", db_path)
    db_module.init_db()
    return db_path
