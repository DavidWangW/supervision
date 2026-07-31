"""Tests for the viewer-aware VLM request gating.

The (expensive) VLM environment recognition should only be issued while at
least one client is actively watching a stream. These tests cover the two
pieces of that logic without touching the network or the vision model:

* :func:`_vlm_should_run` - the pure gate decision (viewers + grace period).
* :meth:`StreamManager.add_viewer` / :meth:`StreamManager.remove_viewer` - the
  active-viewer counter plus the grace-period arming.
* :meth:`TrafficFrameAnalyzer._maybe_submit_vlm` - honours the ``_vlm_allowed``
  flag so no background VLM thread is spawned when nobody is watching.
"""

import threading
import time
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.analytics import LiveSnapshot, TrafficFrameAnalyzer
from app.services.stream_manager import (
    StreamManager,
    StreamRuntime,
    _vlm_should_run,
)


def _make_runtime() -> StreamRuntime:
    return StreamRuntime(record=types.SimpleNamespace(id="s1", lane_count=3))


def test_vlm_should_run_no_recent_heartbeat() -> None:
    # No client has pulled a frame for a long time -> VLM must idle.
    rt = _make_runtime()
    rt.last_viewer_heartbeat = time.monotonic() - 1000
    assert _vlm_should_run(rt) is False


def test_vlm_should_run_with_recent_heartbeat() -> None:
    # A live consumer just received traffic -> VLM should run.
    rt = _make_runtime()
    rt.last_viewer_heartbeat = time.monotonic()
    assert _vlm_should_run(rt) is True


def test_vlm_should_run_disabled(monkeypatch) -> None:
    # When gating is disabled the heartbeat is ignored entirely.
    monkeypatch.setattr(
        "app.services.stream_manager.VLM_ONLY_WHEN_VIEWED", False
    )
    rt = _make_runtime()
    rt.last_viewer_heartbeat = time.monotonic() - 1000
    assert _vlm_should_run(rt) is True


def test_add_and_remove_viewer_counting() -> None:
    manager = StreamManager()
    rt = _make_runtime()
    manager._runtimes["s1"] = rt
    # Force the stream to be considered running for the test.
    manager.is_running = lambda sid: True  # type: ignore[assignment]

    assert manager.add_viewer("s1") is True
    assert rt.active_viewers == 1

    manager.add_viewer("s1")
    assert rt.active_viewers == 2

    manager.remove_viewer("s1")
    assert rt.active_viewers == 1
    # Still watched -> no grace timestamp yet.
    assert rt.last_viewer_left_at is None

    manager.remove_viewer("s1")
    assert rt.active_viewers == 0
    assert rt.last_viewer_left_at is not None

    # Removing from zero never goes negative.
    manager.remove_viewer("s1")
    assert rt.active_viewers == 0


def test_add_viewer_when_not_running() -> None:
    manager = StreamManager()
    rt = _make_runtime()
    manager._runtimes["s1"] = rt
    manager.is_running = lambda sid: False  # type: ignore[assignment]

    assert manager.add_viewer("s1") is False
    assert rt.active_viewers == 0


def test_analyzer_vlm_allowed_gate(monkeypatch) -> None:
    # Avoid touching the real VLM endpoint during construction.
    monkeypatch.setattr("app.services.analytics.ENV_BACKEND", "heuristic")
    analyzer = TrafficFrameAnalyzer(
        source_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        target_width=10.0,
        target_height=10.0,
        fps=10.0,
        resolution_wh=(64, 64),
    )
    classifier = MagicMock()
    analyzer._vlm_classifier = classifier
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    # Gate open: a background VLM thread should kick off the classifier.
    analyzer._vlm_allowed = True
    analyzer._vlm_inflight = False
    analyzer._maybe_submit_vlm(frame)
    for _ in range(50):
        if classifier.classify.called:
            break
        time.sleep(0.01)
    assert classifier.classify.called is True
    # Let the daemon thread settle before the next assertion.
    for _ in range(50):
        if analyzer._vlm_inflight is False:
            break
        time.sleep(0.01)

    # Gate closed: no new request should be scheduled.
    classifier.classify.reset_mock()
    analyzer._vlm_allowed = False
    analyzer._vlm_inflight = False
    analyzer._maybe_submit_vlm(frame)
    assert classifier.classify.called is False


def test_analyzer_vlm_active_flag(monkeypatch) -> None:
    # The flag that drives the "paused" notice must reflect the viewer gate,
    # and must never report paused in pure-heuristic mode (no VLM to pause).
    monkeypatch.setattr("app.services.analytics.ENV_BACKEND", "heuristic")
    analyzer = TrafficFrameAnalyzer(
        source_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        target_width=10.0,
        target_height=10.0,
        fps=10.0,
        resolution_wh=(64, 64),
    )

    # Heuristic mode (no classifier): always active, no spurious notice.
    assert analyzer._vlm_active_flag() is True

    # VLM backend present.
    analyzer._vlm_classifier = MagicMock()
    analyzer._vlm_allowed = True
    assert analyzer._vlm_active_flag() is True

    # VLM backend present but gate closed (nobody watching) -> paused.
    analyzer._vlm_allowed = False
    assert analyzer._vlm_active_flag() is False


def test_livesnapshot_carries_vlm_active() -> None:
    snapshot = LiveSnapshot(
        frame_index=1,
        t_sec=1.0,
        minute_bucket=0,
        total_vehicles=0,
        vlm_active=False,
    )
    payload = snapshot.to_dict()
    assert payload["vlm_active"] is False

    snapshot.vlm_active = True
    assert snapshot.to_dict()["vlm_active"] is True
