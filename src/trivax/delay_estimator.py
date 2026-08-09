from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class DelayEstimate:
    delay: int | None
    score: float
    stable: bool
    samples: int


class DelayEstimator:
    """Lightweight delay estimator using lagged action/observation correlation.

    The estimator works on first differences and searches lags in [0, max_delay].
    It is intentionally small and inspectable. Confidence is based on the best
    absolute correlation and its margin over the second-best lag.
    """

    def __init__(
        self,
        max_delay: int = 12,
        window: int = 128,
        min_samples: int = 32,
        min_abs_correlation: float = 0.25,
        min_margin: float = 0.05,
    ) -> None:
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        if window < max(8, min_samples):
            raise ValueError("window must be >= min_samples and at least 8")
        if min_samples < 8:
            raise ValueError("min_samples must be at least 8")
        if not 0.0 <= min_abs_correlation <= 1.0:
            raise ValueError("min_abs_correlation must be in [0, 1]")
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be in [0, 1]")

        self.max_delay = int(max_delay)
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.min_abs_correlation = float(min_abs_correlation)
        self.min_margin = float(min_margin)

        self.actions: deque[float] = deque(maxlen=window + max_delay + 2)
        self.observations: deque[float] = deque(maxlen=window + max_delay + 2)
        self.previous_action: float | None = None
        self.previous_observation: float | None = None

    @staticmethod
    def _corr(xs: list[float], ys: list[float]) -> float:
        if len(xs) != len(ys) or len(xs) < 2:
            return 0.0
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        dx = [x - mx for x in xs]
        dy = [y - my for y in ys]
        num = sum(a * b for a, b in zip(dx, dy))
        den = sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
        return 0.0 if den == 0.0 else num / den

    def update(self, action: float, observation: float) -> DelayEstimate:
        action = float(action)
        observation = float(observation)

        if self.previous_action is not None:
            self.actions.append(action - self.previous_action)
            self.observations.append(observation - float(self.previous_observation))

        self.previous_action = action
        self.previous_observation = observation
        return self.estimate()

    def estimate(self) -> DelayEstimate:
        n = min(len(self.actions), len(self.observations))
        if n < self.min_samples:
            return DelayEstimate(None, 0.0, False, n)

        da = list(self.actions)[-self.window :]
        dy = list(self.observations)[-self.window :]
        scores: list[tuple[int, float]] = []

        for lag in range(self.max_delay + 1):
            if len(da) <= lag + 2:
                continue
            xs = da[: len(da) - lag]
            ys = dy[lag:]
            corr = abs(self._corr(xs, ys))
            scores.append((lag, corr))

        if not scores:
            return DelayEstimate(None, 0.0, False, n)

        scores.sort(key=lambda item: item[1], reverse=True)
        best_lag, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else 0.0
        stable = (
            best_score >= self.min_abs_correlation
            and (best_score - second_score) >= self.min_margin
        )
        return DelayEstimate(best_lag if stable else None, best_score, stable, n)
