from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class RobustObservationState:
    raw: float
    filtered: float
    innovation: float
    clipped_innovation: float
    delayed_reference: float | None


class RobustObservationLayer:
    """Small, inspectable pre-processing layer for noisy/delayed observations.

    Features:
    - short median window for impulsive outlier suppression;
    - innovation clipping to limit one-sample shocks;
    - explicit delayed reference history for experiments with known sensor delay.
    """

    def __init__(
        self,
        median_window: int = 3,
        innovation_clip: float = 0.05,
        sensor_delay: int = 0,
    ) -> None:
        if median_window <= 0 or median_window % 2 == 0:
            raise ValueError("median_window must be a positive odd integer")
        if innovation_clip <= 0:
            raise ValueError("innovation_clip must be positive")
        if sensor_delay < 0:
            raise ValueError("sensor_delay must be non-negative")

        self.window = deque(maxlen=median_window)
        self.innovation_clip = float(innovation_clip)
        self.sensor_delay = int(sensor_delay)
        self.previous_filtered: float | None = None
        self.filtered_history: deque[float] = deque(maxlen=max(2, sensor_delay + 2))

    def process(self, observation: float) -> RobustObservationState:
        raw = float(observation)
        self.window.append(raw)
        med = float(median(self.window))

        if self.previous_filtered is None:
            innovation = 0.0
            clipped = 0.0
            filtered = med
        else:
            innovation = med - self.previous_filtered
            clipped = max(-self.innovation_clip, min(self.innovation_clip, innovation))
            filtered = self.previous_filtered + clipped

        self.filtered_history.append(filtered)
        delayed_reference: float | None = None
        if self.sensor_delay > 0 and len(self.filtered_history) > self.sensor_delay:
            delayed_reference = list(self.filtered_history)[-(self.sensor_delay + 1)]

        self.previous_filtered = filtered
        return RobustObservationState(
            raw=raw,
            filtered=filtered,
            innovation=innovation,
            clipped_innovation=clipped,
            delayed_reference=delayed_reference,
        )
