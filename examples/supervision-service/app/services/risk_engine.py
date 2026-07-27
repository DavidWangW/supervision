"""Rule-based traffic accident risk scoring.

This is the **phase-5 cold-start** model: an explainable expert scoring card.
It converts per-lane / per-minute traffic metrics plus environment readings
into a 0-100 risk score, a 4-level risk grade, an estimated accident
probability, and a human-readable list of contributing factors.

The design deliberately favors transparency over accuracy so highway
management can trust *why* a risk level was assigned. When enough historical
``accident_events`` accumulate, this module can later be replaced by a
gradient-boosted or time-series model without changing its call signature.
"""

from dataclasses import dataclass, field

# Normalization maps so the engine accepts Chinese labels from the UI as well
# as English keywords from programmatic callers.
_WEATHER_MAP: dict[str, str] = {
    "晴": "clear",
    "晴天": "clear",
    "clear": "clear",
    "多云": "cloudy",
    "阴": "cloudy",
    "cloudy": "cloudy",
    "雨": "rain",
    "下雨": "rain",
    "rain": "rain",
    "雪": "snow",
    "下雪": "snow",
    "snow": "snow",
    "雾": "fog",
    "fog": "fog",
    "霾": "haze",
    "haze": "haze",
}

_ROAD_MAP: dict[str, str] = {
    "干燥": "dry",
    "dry": "dry",
    "潮湿": "wet",
    "湿": "wet",
    "wet": "wet",
    "积水": "flood",
    "flood": "flood",
    "积雪": "snow",
    "雪": "snow",
    "snow": "snow",
    "结冰": "ice",
    "冰": "ice",
    "ice": "ice",
}

# Points added to the risk score for each normalized condition.
_WEATHER_POINTS: dict[str, int] = {
    "clear": 0,
    "cloudy": 0,
    "haze": 10,
    "fog": 20,
    "rain": 15,
    "snow": 30,
}
_ROAD_POINTS: dict[str, int] = {
    "dry": 0,
    "wet": 10,
    "flood": 30,
    "snow": 25,
    "ice": 40,
}

# Chinese labels used in the produced factors / risk levels.
_RISK_LEVELS = ("低", "中", "高", "极高")


@dataclass
class TrafficRiskInput:
    """Inputs for a single risk assessment (one lane or the whole road)."""

    lane_id: int | None
    avg_speed_kmh: float | None = None
    avg_headway_s: float | None = None
    min_headway_s: float | None = None
    density_per_km: float | None = None
    truck_ratio: float | None = None
    weather: str | None = None
    road_condition: str | None = None


@dataclass
class RiskResult:
    """Outcome of a single risk assessment."""

    lane_id: int | None
    risk_level: str
    risk_score: float
    probability: float
    factors: list[str] = field(default_factory=list)
    weather: str | None = None
    road_condition: str | None = None


def _normalize(value: str | None, mapping: dict[str, str]) -> str | None:
    if not value:
        return None
    return mapping.get(value.strip().lower())


def _safe(v: float | None, default: float = 0.0) -> float:
    return v if v is not None and v == v else default  # noqa: PLR0124 (NaN guard)


def score_single(input: TrafficRiskInput) -> RiskResult:
    """Score a single lane / whole-road input with the expert rule card.

    Args:
        input: Traffic and environment metrics for one scope.

    Returns:
        A :class:`RiskResult` with level, score, estimated probability and the
        list of factors that drove the score (for explainability).
    """
    score = 0.0
    factors: list[str] = []

    weather = _normalize(input.weather, _WEATHER_MAP)
    road = _normalize(input.road_condition, _ROAD_MAP)

    if weather is not None:
        w_points = _WEATHER_POINTS.get(weather, 0)
        if w_points:
            score += w_points
            factors.append(f"天气：{input.weather}（+{w_points}）")
    if road is not None:
        r_points = _ROAD_POINTS.get(road, 0)
        if r_points:
            score += r_points
            factors.append(f"路面：{input.road_condition}（+{r_points}）")

    # Headway (time gap to the leading vehicle). The single strongest
    # behavioral crash predictor on highways. Only score it when a real
    # measurement exists: ``None`` means "no vehicles / not computed this
    # minute" and must not be treated as a 0-second (extreme) gap.
    min_hw = input.min_headway_s
    avg_hw = input.avg_headway_s
    if min_hw is not None:
        if min_hw < 1.0:
            score += 25
            factors.append(f"最小车头时距 {min_hw:.1f}s < 1.0s，追尾风险极高（+25）")
        elif min_hw < 1.5:
            score += 18
            factors.append(f"最小车头时距 {min_hw:.1f}s，偏危险（+18）")
        elif min_hw < 2.0:
            score += 12
            factors.append(f"最小车头时距 {min_hw:.1f}s，偏近（+12）")
        elif avg_hw is not None and avg_hw > 6.0:
            # Excessively large gaps can indicate stop-and-go / congestion onset.
            score += 5
            factors.append(f"平均车头时距 {avg_hw:.1f}s，疑似拥堵（+5）")

    # Density (vehicles per km). Higher density amplifies any hazard.
    density = _safe(input.density_per_km)
    if density > 60:
        score += 20
        factors.append(f"密度 {density:.0f} 辆/km，严重拥堵（+20）")
    elif density > 40:
        score += 12
        factors.append(f"密度 {density:.0f} 辆/km，较拥堵（+12）")
    elif density > 25:
        score += 6
        factors.append(f"密度 {density:.0f} 辆/km，车流偏密（+6）")

    # Average speed: very high speeds compound poor road/weather conditions.
    speed = _safe(input.avg_speed_kmh)
    if speed > 110 and (road in ("snow", "ice", "wet", "flood") or weather in ("rain", "snow", "fog")):
        score += 10
        factors.append(f"恶劣天气下平均车速 {speed:.0f}km/h 仍偏高（+10）")
    elif speed > 120:
        score += 6
        factors.append(f"平均车速 {speed:.0f}km/h 偏高（+6）")

    # Heavy-vehicle proportion raises severity and instability.
    truck_ratio = _safe(input.truck_ratio)
    if truck_ratio > 0.4:
        score += 10
        factors.append(f"货车占比 {truck_ratio * 100:.0f}% 偏高（+10）")
    elif truck_ratio > 0.25:
        score += 5
        factors.append(f"货车占比 {truck_ratio * 100:.0f}%（+5）")

    score = max(0.0, min(100.0, score))

    # Map to a 4-level grade and a logistic accident probability.
    if score < 25:
        level = _RISK_LEVELS[0]
    elif score < 50:
        level = _RISK_LEVELS[1]
    elif score < 75:
        level = _RISK_LEVELS[2]
    else:
        level = _RISK_LEVELS[3]

    probability = round(1.0 / (1.0 + __import__("math").exp(-(score - 45) / 12)), 3)
    if not factors:
        factors.append("各项指标平稳，无明显风险因子。")

    return RiskResult(
        lane_id=input.lane_id,
        risk_level=level,
        risk_score=round(score, 1),
        probability=probability,
        factors=factors,
        weather=input.weather,
        road_condition=input.road_condition,
    )


def assess(
    lane_inputs: list[TrafficRiskInput],
) -> list[RiskResult]:
    """Assess risk for several lanes plus an overall road-level assessment.

    The overall assessment aggregates the mean of the per-lane metrics so the
    management console always has a single headline number alongside the
    per-lane breakdown.

    Args:
        lane_inputs: One :class:`TrafficRiskInput` per lane (may be empty).

    Returns:
        A list of :class:`RiskResult`, where the final element (``lane_id=None``)
        is the aggregated whole-road assessment.
    """
    results = [score_single(item) for item in lane_inputs]

    if lane_inputs:
        overall = TrafficRiskInput(
            lane_id=None,
            avg_speed_kmh=_mean([i.avg_speed_kmh for i in lane_inputs]),
            avg_headway_s=_mean([i.avg_headway_s for i in lane_inputs]),
            min_headway_s=(
                min(_finite([i.min_headway_s for i in lane_inputs]))
                if any(i.min_headway_s is not None for i in lane_inputs)
                else None
            ),
            density_per_km=_mean([i.density_per_km for i in lane_inputs]),
            truck_ratio=_mean([i.truck_ratio for i in lane_inputs]),
            weather=lane_inputs[0].weather,
            road_condition=lane_inputs[0].road_condition,
        )
        results.append(score_single(overall))

    return results


def _mean(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None and v == v]
    return sum(finite) / len(finite) if finite else None


def _finite(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None and v == v]
