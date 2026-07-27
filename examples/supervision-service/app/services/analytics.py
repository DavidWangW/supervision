"""Unified traffic analysis pipeline (Phase 0 + Phase 1 + Phase 3).

The per-frame analysis logic (detection + multi-object tracking, bird's-eye
speed, per-lane flow / headway / density and annotation) lives in
:class:`TrafficFrameAnalyzer`. A frame and its (already inferred) detection
result go in; an annotated frame plus a live metric snapshot come out, and the
same accumulators feed the structured analytics tables.

Both entry points share that one analyzer, so the offline **file** pipeline
(:func:`analyze_traffic_video`) and the **real-time stream** worker (see
``app.services.stream_manager``) run identical analysis code instead of
maintaining two copies. Adding a new metric only requires touching the analyzer.
"""

import logging
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_GPU_BATCH_SIZE,
    DEFAULT_IOU,
    DEFAULT_SPEED_WEIGHTS,
    OUTPUT_DIR,
)
from app.db.analytics_repository import (
    RiskAssessmentRow,
    TrafficMetricRow,
    VehicleTrackRow,
    insert_environment_reading,
    insert_risk_assessments,
    insert_traffic_metrics,
    insert_vehicle_tracks,
)
from app.services.environment import (
    EnvironmentResult,
    SceneClassifier,
    SceneFeatures,
)
from app.services.hardware import (
    iter_batches,
    resolve_effective_batch,
    torch_device,
)
from app.services.risk_engine import RiskResult, TrafficRiskInput, assess
from app.services.video_encoding import ensure_browser_playable

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]

# COCO vehicle class ids used by the detection model.
VEHICLE_CLASS_IDS = (2, 3, 5, 7)
VEHICLE_TYPE_BY_CLASS = {2: "小轿车", 3: "摩托车", 5: "公交车", 7: "卡车"}


class RoadViewTransformer:
    """Perspective transformer mapping image pixels to bird's-eye meters.

    Unlike the simpler transformer in ``speed_processor``, this one also exposes
    an inverse transform so we can derive lane polygons in pixel space from a
    lane split defined in the calibrated bird's-eye plane.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
        self.inv_m = np.linalg.inv(self.m)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.inv_m)
        return transformed.reshape(-1, 2)


def build_target_array(target_width: float, target_height: float) -> np.ndarray:
    """Build the bird's-eye target quadrilateral for the perspective transform."""
    width = max(target_width, 1.0)
    height = max(target_height, 1.0)
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def derive_lane_polygons(
    source_points: list[list[int]],
    target_width: float,
    target_height: float,
    lane_count: int,
) -> list[list[list[int]]]:
    """Split the calibrated road into ``lane_count`` pixel-space polygons.

    The road is divided into equal vertical strips in the bird's-eye plane and
    each strip is projected back to image pixels via the inverse perspective
    transform. This lets the UI specify lanes with a single integer instead of
    a custom polygon-drawing widget.

    Args:
        source_points: Four [x, y] road-surface calibration points.
        target_width: Calibrated road width in meters.
        target_height: Calibrated road length in meters.
        lane_count: Number of equal lanes to derive.

    Returns:
        A list of lane polygons, each a list of [x, y] pixel points.
    """
    lane_count = max(1, int(lane_count))
    source = np.array(source_points, dtype=np.float32)
    target = build_target_array(target_width, target_height)
    transformer = RoadViewTransformer(source, target)
    width = target_width

    lanes: list[list[list[int]]] = []
    for i in range(lane_count):
        x0 = i * width / lane_count
        x1 = (i + 1) * width / lane_count
        bev_quad = np.array(
            [[x0, 0], [x1, 0], [x1, target_height - 1], [x0, target_height - 1]],
            dtype=np.float32,
        )
        pixel = transformer.inverse_transform_points(bev_quad)
        lanes.append([[int(p[0]), int(p[1])] for p in pixel])
    return lanes


def _lane_counting_line(polygon: list[list[int]]) -> tuple[sv.Point, sv.Point]:
    """Return the bottom edge of a lane polygon as a counting line.

    The two polygon vertices with the largest y (lowest on screen) form the
    far counting line. Ordering by x keeps the line orientation consistent.
    """
    pts = sorted(polygon, key=lambda p: (-p[1], p[0]))
    bottom = pts[:2]
    bottom = sorted(bottom, key=lambda p: p[0])
    return sv.Point(x=bottom[0][0], y=bottom[0][1]), sv.Point(
        x=bottom[1][0], y=bottom[1][1]
    )


@dataclass
class _LaneAccumulator:
    flow_total: int = 0
    speed_sum: float = 0.0
    speed_n: int = 0
    headway_sum: float = 0.0
    headway_n: int = 0
    min_headway: float = float("inf")
    density_sum: float = 0.0
    density_n: int = 0
    truck_count: int = 0
    vehicle_set: set[int] = field(default_factory=set)


@dataclass
class LaneLiveMetric:
    """Instantaneous per-lane measurement for the live dashboard."""

    lane_id: int
    vehicle_count: int = 0
    flow_total: int = 0
    avg_speed_kmh: float | None = None
    min_headway_s: float | None = None
    density: float | None = None
    truck_ratio: float | None = None


@dataclass
class LiveSnapshot:
    """A real-time snapshot pushed to subscribers each processed frame."""

    frame_index: int
    t_sec: float
    minute_bucket: int
    total_vehicles: int
    lanes: list[LaneLiveMetric] = field(default_factory=list)
    risk: dict | None = None
    environment: dict | None = None

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "t_sec": self.t_sec,
            "minute_bucket": self.minute_bucket,
            "total_vehicles": self.total_vehicles,
            "lanes": [
                {
                    "lane_id": m.lane_id,
                    "vehicle_count": m.vehicle_count,
                    "flow_total": m.flow_total,
                    "avg_speed_kmh": m.avg_speed_kmh,
                    "min_headway_s": m.min_headway_s,
                    "density": m.density,
                    "truck_ratio": m.truck_ratio,
                }
                for m in self.lanes
            ],
            "risk": self.risk,
            "environment": self.environment,
        }


class TrafficFrameAnalyzer:
    """Stateful per-frame traffic analyzer shared by file and stream pipelines.

    Feed frames one at a time via :meth:`process` together with their already
    inferred Ultralytics ``result``. The analyzer maintains tracking state,
    speed history and per-lane / per-minute accumulators, returning an annotated
    frame plus a :class:`LiveSnapshot`. Persist the accumulated results with
    :meth:`build_metric_rows` / :meth:`build_track_rows` (offline file mode) or
    :meth:`pop_completed_metric_rows` (streaming, flush per finished minute).
    """

    def __init__(
        self,
        *,
        source_points: list[list[int]],
        target_width: float,
        target_height: float,
        fps: float,
        resolution_wh: tuple[int, int],
        lanes: list[list[list[int]]] | None = None,
        lane_count: int = 3,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
    ) -> None:
        if len(source_points) != 4:
            raise ValueError("Exactly four source points are required.")

        self.lane_polygons = lanes or derive_lane_polygons(
            source_points, target_width, target_height, lane_count
        )
        self.lane_count = len(self.lane_polygons)

        source = np.array(source_points, dtype=np.float32)
        target = build_target_array(target_width, target_height)
        self.transformer = RoadViewTransformer(source, target)
        self.polygon_zones = [
            sv.PolygonZone(polygon=np.array(p, dtype=np.float32))
            for p in self.lane_polygons
        ]
        self.line_counters = [
            sv.LineZone(start=start, end=end)
            for start, end in (_lane_counting_line(p) for p in self.lane_polygons)
        ]

        self.fps = max(float(fps), 1.0)
        self.lane_length_m = max(float(target_height), 1.0)
        self.byte_track = sv.ByteTrack(
            frame_rate=int(self.fps),
            track_activation_threshold=confidence_threshold,
        )

        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        self._thickness = thickness
        self._text_scale = text_scale
        self._box = sv.BoxAnnotator(thickness=thickness)
        self._label = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self._trace = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=int(self.fps * 2),
            position=sv.Position.BOTTOM_CENTER,
        )
        self._lane_color = (255, 179, 0)

        self._coordinates: defaultdict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=int(self.fps))
        )
        self.track_speeds: dict[int, list[float]] = defaultdict(list)
        self.track_class: dict[int, str] = {}
        self.track_first: dict[int, float] = {}
        self.track_last: dict[int, float] = {}
        self.track_lane: dict[int, Counter] = defaultdict(Counter)

        self.accumulators: dict[tuple[int, int], _LaneAccumulator] = defaultdict(
            _LaneAccumulator
        )
        self._prev_flow = [0] * self.lane_count
        self.frame_index = 0

        # Phase 4: per-frame environment recognition. The classifier is shared
        # across file and stream pipelines; ``latest_environment`` holds the most
        # recent result and ``_env_features_ema`` a smoothed feature window so
        # :meth:`aggregate_environment` can report a stable clip-level reading.
        self.scene_classifier = SceneClassifier()
        self.latest_environment: EnvironmentResult | None = None
        self._env_features_ema: SceneFeatures | None = None

    def draw_lanes(self, scene: np.ndarray) -> np.ndarray:
        """Draw lane polygons and labels onto ``scene`` (in place safe copy)."""
        for lane_id in range(self.lane_count):
            poly = np.array(self.lane_polygons[lane_id], dtype=np.int32)
            cv2.polylines(
                scene,
                [poly],
                isClosed=True,
                color=self._lane_color,
                thickness=max(1, self._thickness - 1),
            )
            cv2.putText(
                scene,
                f"L{lane_id + 1}",
                tuple(self.lane_polygons[lane_id][0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, self._text_scale * 0.8),
                self._lane_color,
                2,
            )
        return scene

    def _update_environment(self, frame: np.ndarray) -> dict | None:
        """Classify the frame's environment and update smoothed state.

        Returns the latest environment payload dict (or ``None`` on the very
        first failure before any result exists).
        """
        try:
            env = self.scene_classifier.classify(frame)
        except Exception as exc:  # pragma: no cover - frame-level guard
            logger.warning("Environment classification failed: %s", exc)
            return self.latest_environment.to_dict() if self.latest_environment else None

        self.latest_environment = env
        f = env.features
        if self._env_features_ema is None:
            self._env_features_ema = f
        else:
            a = 0.05
            prev = self._env_features_ema
            self._env_features_ema = SceneFeatures(
                brightness=self._ema(prev.brightness, f.brightness, a),
                contrast=self._ema(prev.contrast, f.contrast, a),
                fog_index=self._ema(prev.fog_index, f.fog_index, a),
                rain_index=self._ema(prev.rain_index, f.rain_index, a),
                snow_index=self._ema(prev.snow_index, f.snow_index, a),
                specular_index=self._ema(prev.specular_index, f.specular_index, a),
                is_night=prev.is_night and f.is_night,
            )
        return env.to_dict()

    @staticmethod
    def _ema(prev: float, new: float, alpha: float) -> float:
        """Exponential moving average step."""
        return prev * (1.0 - alpha) + new * alpha

    def aggregate_environment(self) -> EnvironmentResult | None:
        """Return a stable clip/window-level environment reading.

        Uses the smoothed feature window when available so a single odd frame
        (e.g. a truck's headlights) does not flip the whole-scene label.
        """
        if self._env_features_ema is None:
            return self.latest_environment
        return self.scene_classifier.classify_features(self._env_features_ema)

    def process(
        self, frame: np.ndarray, result: object
    ) -> tuple[np.ndarray, LiveSnapshot]:
        """Analyze one frame given its inferred detection ``result``.

        Args:
            frame: The raw BGR frame.
            result: Ultralytics result for this frame.

        Returns:
            A ``(annotated_frame, live_snapshot)`` tuple.
        """
        self.frame_index += 1
        fi = self.frame_index
        t_sec = fi / self.fps
        minute_bucket = int(t_sec // 60)

        # Phase 4: recognize weather / road / visibility from the raw frame.
        env_dict = self._update_environment(frame)

        detections = sv.Detections.from_ultralytics(result)
        if detections.class_id is not None:
            mask = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
            detections = detections[mask]
        detections = self.byte_track.update_with_detections(detections=detections)

        lane_live = [LaneLiveMetric(lane_id=i) for i in range(self.lane_count)]

        if detections.tracker_id is None or len(detections) == 0:
            annotated = self.draw_lanes(frame.copy())
            annotated = self._box.annotate(scene=annotated, detections=detections)
            return annotated, LiveSnapshot(
                frame_index=fi,
                t_sec=round(t_sec, 1),
                minute_bucket=minute_bucket,
                total_vehicles=0,
                lanes=lane_live,
                environment=env_dict,
            )

        # --- Speed (bird's-eye y displacement over time) ---
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        bev_points = self.transformer.transform_points(points).astype(int)
        speed_by_id: dict[int, float] = {}
        for tid, (_, y) in zip(detections.tracker_id, bev_points):
            self._coordinates[tid].append(y)
            if len(self._coordinates[tid]) >= self.fps / 2:
                dist = abs(self._coordinates[tid][-1] - self._coordinates[tid][0])
                speed = dist / (len(self._coordinates[tid]) / self.fps) * 3.6
            else:
                speed = 0.0
            speed_by_id[int(tid)] = speed

        class_by_id: dict[int, int] = {}
        if detections.class_id is not None:
            for tid, cid in zip(detections.tracker_id, detections.class_id):
                class_by_id[int(tid)] = int(cid)

        total_vehicles = int(len(detections.tracker_id))

        # --- Assign vehicles to lanes & compute per-lane metrics ---
        for lane_id in range(self.lane_count):
            in_mask = self.polygon_zones[lane_id].trigger(detections)
            if not in_mask.any():
                continue
            in_det = detections[in_mask]
            counter = self.accumulators[(minute_bucket, lane_id)]

            # Flow: count crossings of this lane's far line.
            self.line_counters[lane_id].trigger(in_det)
            current_flow = (
                self.line_counters[lane_id].in_count
                + self.line_counters[lane_id].out_count
            )
            counter.flow_total += max(0, current_flow - self._prev_flow[lane_id])
            self._prev_flow[lane_id] = current_flow

            counter.vehicle_set.update(int(tid) for tid in in_det.tracker_id)
            counter.density_sum += int(in_det.tracker_id.shape[0])
            counter.density_n += 1

            lane_bev = bev_points[in_mask]
            lane_ids = in_det.tracker_id
            lane_speeds = [speed_by_id[int(t)] for t in lane_ids]
            order = np.argsort(lane_bev[:, 1])
            frame_speed_sum = 0.0
            frame_speed_n = 0
            frame_min_hw = float("inf")
            frame_truck = 0
            for rank, idx in enumerate(order):
                tid = int(lane_ids[idx])
                spd = lane_speeds[idx]
                if spd > 0:
                    counter.speed_sum += spd
                    counter.speed_n += 1
                    frame_speed_sum += spd
                    frame_speed_n += 1
                vtype = VEHICLE_TYPE_BY_CLASS.get(class_by_id.get(tid, -1), "其他")
                if vtype in ("卡车", "公交车"):
                    counter.truck_count += 1
                    frame_truck += 1
                self.track_lane[tid][lane_id] += 1

                # time headway to the vehicle ahead (larger y)
                if rank + 1 < len(order):
                    ahead = order[rank + 1]
                    spacing = float(np.linalg.norm(lane_bev[idx] - lane_bev[ahead]))
                    rear_speed = lane_speeds[idx]
                    if rear_speed > 0:
                        hw = spacing / (rear_speed / 3.6)
                        counter.headway_sum += hw
                        counter.headway_n += 1
                        counter.min_headway = min(counter.min_headway, hw)
                        frame_min_hw = min(frame_min_hw, hw)

            n_in = int(len(lane_ids))
            live = lane_live[lane_id]
            live.vehicle_count = n_in
            live.flow_total = counter.flow_total
            live.avg_speed_kmh = (
                round(frame_speed_sum / frame_speed_n, 1) if frame_speed_n else None
            )
            live.min_headway_s = (
                round(frame_min_hw, 2) if frame_min_hw != float("inf") else None
            )
            live.density = round(n_in / (self.lane_length_m / 1000.0), 1)
            live.truck_ratio = round(frame_truck / n_in, 3) if n_in else None

        # --- Per-track summary bookkeeping ---
        for tid in detections.tracker_id:
            tid_i = int(tid)
            spd = speed_by_id[tid_i]
            self.track_speeds[tid_i].append(spd)
            self.track_first.setdefault(tid_i, t_sec)
            self.track_last[tid_i] = t_sec
            if detections.class_id is not None:
                cid = class_by_id.get(tid_i, -1)
                self.track_class[tid_i] = VEHICLE_TYPE_BY_CLASS.get(cid, "其他")

        # --- Annotation ---
        labels = [
            f"#{tid} {int(speed_by_id[int(tid)])} km/h"
            for tid in detections.tracker_id
        ]
        annotated = self.draw_lanes(frame.copy())
        annotated = self._trace.annotate(scene=annotated, detections=detections)
        annotated = self._box.annotate(scene=annotated, detections=detections)
        annotated = self._label.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        snapshot = LiveSnapshot(
            frame_index=fi,
            t_sec=round(t_sec, 1),
            minute_bucket=minute_bucket,
            total_vehicles=total_vehicles,
            lanes=lane_live,
            environment=env_dict,
        )
        return annotated, snapshot

    def _metric_row(
        self, upload_id: str, minute_bucket: int, lane_id: int, acc: _LaneAccumulator
    ) -> TrafficMetricRow:
        avg_speed = acc.speed_sum / acc.speed_n if acc.speed_n else None
        avg_headway = acc.headway_sum / acc.headway_n if acc.headway_n else None
        min_headway = acc.min_headway if acc.min_headway != float("inf") else None
        density = acc.density_sum / acc.density_n if acc.density_n else None
        veh_n = len(acc.vehicle_set)
        truck_ratio = (acc.truck_count / veh_n) if veh_n else None
        return TrafficMetricRow(
            upload_id=upload_id,
            lane_id=lane_id,
            minute_bucket=minute_bucket,
            flow_count=acc.flow_total,
            avg_speed_kmh=round(avg_speed, 1) if avg_speed is not None else None,
            avg_headway_s=round(avg_headway, 2) if avg_headway is not None else None,
            min_headway_s=round(min_headway, 2) if min_headway is not None else None,
            density_per_km=round(density, 1) if density is not None else None,
            truck_ratio=round(truck_ratio, 3) if truck_ratio is not None else None,
            vehicle_count=veh_n,
        )

    def build_metric_rows(self, upload_id: str) -> list[TrafficMetricRow]:
        """Build metric rows for every accumulated (minute, lane) bucket."""
        return [
            self._metric_row(upload_id, minute_bucket, lane_id, acc)
            for (minute_bucket, lane_id), acc in sorted(self.accumulators.items())
        ]

    def pop_completed_metric_rows(
        self, upload_id: str, current_minute: int
    ) -> list[TrafficMetricRow]:
        """Return and drop metric rows for minute buckets before ``current_minute``.

        Used by the streaming worker to flush a finished minute to the database
        without keeping every bucket in memory for the lifetime of the stream.
        """
        completed = sorted(
            key for key in self.accumulators if key[0] < current_minute
        )
        rows = [
            self._metric_row(upload_id, mb, lane_id, self.accumulators[(mb, lane_id)])
            for (mb, lane_id) in completed
        ]
        for key in completed:
            del self.accumulators[key]
        return rows

    def build_track_rows(self, upload_id: str) -> list[VehicleTrackRow]:
        """Build summarized vehicle-track rows for every observed track id."""
        rows: list[VehicleTrackRow] = []
        for tid, speeds in self.track_speeds.items():
            valid = [s for s in speeds if s > 0]
            rows.append(
                VehicleTrackRow(
                    upload_id=upload_id,
                    track_id=tid,
                    class_name=self.track_class.get(tid),
                    vehicle_type=self.track_class.get(tid),
                    first_seen=round(self.track_first.get(tid, 0.0), 2),
                    last_seen=round(self.track_last.get(tid, 0.0), 2),
                    avg_speed_kmh=round(sum(valid) / len(valid), 1) if valid else None,
                    max_speed_kmh=round(max(valid), 1) if valid else None,
                    lane_id=(
                        self.track_lane[tid].most_common(1)[0][0]
                        if self.track_lane[tid]
                        else None
                    ),
                )
            )
        return rows


def compute_risk(
    metric_rows: list[TrafficMetricRow],
    lane_count: int,
    weather: str | None,
    road_condition: str | None,
) -> list[RiskResult]:
    """Aggregate metric rows into per-lane inputs and run the risk engine."""
    lane_inputs: list[TrafficRiskInput] = []
    for lane_id in range(lane_count):
        lane_metrics = [m for m in metric_rows if m.lane_id == lane_id]
        min_headway = min(
            (m.min_headway_s for m in lane_metrics if m.min_headway_s is not None),
            default=None,
        )
        lane_inputs.append(
            TrafficRiskInput(
                lane_id=lane_id,
                avg_speed_kmh=_mean([m.avg_speed_kmh for m in lane_metrics]),
                avg_headway_s=_mean([m.avg_headway_s for m in lane_metrics]),
                min_headway_s=min_headway,
                density_per_km=_mean([m.density_per_km for m in lane_metrics]),
                truck_ratio=_mean([m.truck_ratio for m in lane_metrics]),
                weather=weather,
                road_condition=road_condition,
            )
        )
    return assess(lane_inputs)


def risk_results_to_rows(
    upload_id: str | None,
    assessed_at: str,
    results: list[RiskResult],
) -> list[RiskAssessmentRow]:
    """Convert :class:`RiskResult` objects to persistable rows."""
    return [
        RiskAssessmentRow(
            upload_id=upload_id,
            assessed_at=assessed_at,
            lane_id=r.lane_id,
            risk_level=r.risk_level,
            risk_score=r.risk_score,
            probability=r.probability,
            factors=r.factors,
            weather=r.weather,
            road_condition=r.road_condition,
        )
        for r in results
    ]


def analyze_traffic_video(
    source_video_path: Path,
    target_video_path: Path,
    source_points: list[list[int]],
    target_width: float,
    target_height: float,
    upload_id: str,
    lanes: list[list[list[int]]] | None = None,
    lane_count: int = 3,
    weather: str | None = None,
    road_condition: str | None = None,
    visibility: str | None = None,
    weights_path: Path = DEFAULT_SPEED_WEIGHTS,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU,
    on_progress: ProgressCallback | None = None,
    batch_size: int = DEFAULT_GPU_BATCH_SIZE,
) -> Path:
    """Run the full traffic analysis on a video file and persist results.

    Args:
        source_video_path: Input video path.
        target_video_path: Output annotated video path.
        source_points: Four [x, y] road-surface calibration points.
        target_width: Calibrated road width in meters.
        target_height: Calibrated road length in meters.
        upload_id: Upload record id used to tag persisted rows.
        lanes: Explicit lane polygons (pixel space). Derived from ``lane_count``
            when omitted.
        lane_count: Number of equal lanes to derive when ``lanes`` is omitted.
        weather: Optional weather label for the environment reading.
        road_condition: Optional road-condition label.
        visibility: Optional visibility label.
        weights_path: YOLO weights path.
        confidence_threshold: Detection confidence threshold.
        iou_threshold: Detection IOU threshold.

    Returns:
        Path to the annotated output video.
    """
    video_info = sv.VideoInfo.from_video_path(str(source_video_path))
    device = torch_device()
    model = YOLO(str(weights_path)).to(device)
    if device == "cuda":
        model.half()

    analyzer = TrafficFrameAnalyzer(
        source_points=source_points,
        target_width=target_width,
        target_height=target_height,
        fps=video_info.fps,
        resolution_wh=video_info.resolution_wh,
        lanes=lanes,
        lane_count=lane_count,
        confidence_threshold=confidence_threshold,
    )

    frame_generator = sv.get_video_frames_generator(str(source_video_path))
    effective_batch, frame_generator = resolve_effective_batch(
        device=device,
        model=model,
        frame_generator=frame_generator,
        requested=batch_size,
        conf=confidence_threshold,
        iou=iou_threshold,
    )

    total_frames = video_info.total_frames
    if on_progress and total_frames:
        on_progress(0, total_frames)

    frame_index = 0
    with sv.VideoSink(str(target_video_path), video_info) as sink:
        for frames in iter_batches(frame_generator, effective_batch):
            batch_results = model(
                frames,
                verbose=False,
                conf=confidence_threshold,
                iou=iou_threshold,
                device=device,
            )
            for frame, result in zip(frames, batch_results):
                frame_index += 1
                annotated, _ = analyzer.process(frame, result)
                sink.write_frame(annotated)
                if on_progress and total_frames:
                    on_progress(frame_index, total_frames)

    if on_progress and total_frames:
        on_progress(total_frames, total_frames)

    # --- Persist structured analytics ---
    metric_rows = analyzer.build_metric_rows(upload_id)
    track_rows = analyzer.build_track_rows(upload_id)
    insert_traffic_metrics(metric_rows)
    insert_vehicle_tracks(track_rows)

    from datetime import datetime, timezone

    # Phase 4: persist the auto-recognized environment for the whole clip and
    # feed it into risk scoring (an explicit manual override still wins).
    env = analyzer.aggregate_environment()
    if env is not None:
        observed_at = (
            datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        )
        insert_environment_reading(
            upload_id=upload_id,
            observed_at=observed_at,
            weather=env.weather,
            road_condition=env.road_condition,
            visibility=env.visibility,
            source="auto",
            details=env.to_dict(),
            model=env.model,
        )

    eff_weather = weather or (env.weather if env else None)
    eff_road = road_condition or (env.road_condition if env else None)
    results = compute_risk(metric_rows, analyzer.lane_count, eff_weather, eff_road)
    assessed_at = (
        datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    )
    insert_risk_assessments(risk_results_to_rows(upload_id, assessed_at, results))

    ensure_browser_playable(target_video_path)
    return target_video_path


def _mean(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None and v == v]
    return sum(finite) / len(finite) if finite else None


def build_analytics_output_path(prefix: str = "analyze") -> Path:
    """Create a unique output path for analysis results."""
    return OUTPUT_DIR / f"{prefix}_{uuid4().hex[:8]}.mp4"
