from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median


class ObservationRoute(str, Enum):
    RAW = "RAW"
    ROBUST = "ROBUST"
    DELAY_AWARE = "DELAY_AWARE"


@dataclass(frozen=True)
class ObservationRouteState:
    raw: float
    output: float
    route: ObservationRoute
    median_reference: float
    mad_scale: float
    is_outlier: bool


class ObservationRouter:
    """Adaptive, inspectable observation pre-processing router.

    RAW is the default path and adds no smoothing. ROBUST is activated only for
    impulsive samples detected against a rolling median/MAD reference. If a
    sensor delay is explicitly configured, DELAY_AWARE marks the sample so the
    runtime can apply temporal credit assignment instead of blindly filtering.
    """

    def __init__(
        self,
        window: int = 5,
        outlier_z: float = 4.0,
        innovation_clip: float = 0.05,
        sensor_delay: int = 0,
    ) -> None:
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be an odd integer >= 3")
        if outlier_z <= 0:
            raise ValueError("outlier_z must be positive")
        if innovation_clip <= 0:
            raise ValueError("innovation_clip must be positive")
        if sensor_delay < 0:
            raise ValueError("sensor_delay must be non-negative")

        self.window = int(window)
        self.outlier_z = float(outlier_z)
        self.innovation_clip = float(innovation_clip)
        self.sensor_delay = int(sensor_delay)
        self.history: deque[float] = deque(maxlen=max(window, sensor_delay + 3))
        self.previous_output: float | None = None

    def process(self, observation: float) -> ObservationRouteState:
        raw = float(observation)

        prior = list(self.history)
        if prior:
            med = float(median(prior))
            deviations = [abs(x - med) for x in prior]
            mad = float(median(deviations)) if deviations else 0.0
        else:
            med = raw
            mad = 0.0

        robust_scale = max(1e-6, 1.4826 * mad)
        threshold = self.outlier_z * robust_scale + 1e-3
        is_outlier = len(prior) >= 3 and abs(raw - med) > threshold

        if self.sensor_delay > 0:
            route = ObservationRoute.DELAY_AWARE
            output = raw
        elif is_outlier:
            route = ObservationRoute.ROBUST
            candidate = med
            if self.previous_output is None:
                output = candidate
            else:
                innovation = candidate - self.previous_output
                clipped = max(
                    -self.innovation_clip,
                    min(self.innovation_clip, innovation),
                )
                output = self.previous_output + clipped
        else:
            route = ObservationRoute.RAW
            output = raw

        self.history.append(raw)
        self.previous_output = float(output)

        return ObservationRouteState(
            raw=raw,
            output=float(output),
            route=route,
            median_reference=med,
            mad_scale=robust_scale,
            is_outlier=is_outlier,
        )
