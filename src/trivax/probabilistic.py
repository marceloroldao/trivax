from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp


class ProbabilisticRegime(str, Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    STABILIZE = "STABILIZE"


@dataclass(frozen=True)
class ProbabilisticState:
    observation: float
    delta: float
    volatility: float
    coherence: float
    reversal_rate: float
    p_search: float
    p_track: float
    p_stabilize: float
    regime: ProbabilisticRegime
    direction: int
    effective_step: float


class ProbabilisticRegimeController:
    """TRIVAX v0.3 experimental statistical regime controller.

    The controller keeps lightweight exponentially-weighted statistics and
    converts them into normalized regime scores. No neural network or fitted
    model is required. The regime with highest posterior-like score selects the
    control gain while all intermediate probabilities remain inspectable.
    """

    def __init__(
        self,
        initial_action: float = 0.5,
        step_size: float = 0.05,
        min_action: float = 0.0,
        max_action: float = 1.0,
        alpha: float = 0.15,
        track_gain: float = 1.45,
        stabilize_gain: float = 0.45,
        search_gain: float = 1.0,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if min_action >= max_action:
            raise ValueError("min_action must be smaller than max_action")
        if not min_action <= initial_action <= max_action:
            raise ValueError("initial_action must be within action bounds")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

        self.action = float(initial_action)
        self.base_step = float(step_size)
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.alpha = float(alpha)
        self.track_gain = float(track_gain)
        self.stabilize_gain = float(stabilize_gain)
        self.search_gain = float(search_gain)

        self.previous_observation: float | None = None
        self.direction = 1
        self.coherence = 0.5
        self.volatility = 0.0
        self.reversal_rate = 0.0
        self.effective_step = self.base_step
        self.regime = ProbabilisticRegime.SEARCH

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    @staticmethod
    def _softmax(a: float, b: float, c: float) -> tuple[float, float, float]:
        m = max(a, b, c)
        ea, eb, ec = exp(a - m), exp(b - m), exp(c - m)
        z = ea + eb + ec
        return ea / z, eb / z, ec / z

    def step(self, observation: float) -> tuple[float, ProbabilisticState]:
        observation = float(observation)
        first = self.previous_observation is None
        delta = 0.0 if first else observation - float(self.previous_observation)

        if not first:
            reversal = 1.0 if delta < 0.0 else 0.0
            if reversal:
                self.direction *= -1

            a = self.alpha
            self.coherence = (1.0 - a) * self.coherence + a * (1.0 - reversal)
            self.volatility = (1.0 - a) * self.volatility + a * abs(delta)
            self.reversal_rate = (1.0 - a) * self.reversal_rate + a * reversal

        magnitude = abs(delta)
        scale = max(self.volatility, 1e-6)
        normalized_change = min(3.0, magnitude / scale) if not first else 0.0

        score_track = 2.2 * self.coherence + 0.8 * normalized_change - 2.0 * self.reversal_rate
        score_stabilize = 2.5 * self.reversal_rate + 0.8 * (1.0 - self.coherence) - 0.35 * normalized_change
        score_search = 1.0 + 0.6 * (1.0 - self.coherence) + 0.2 * normalized_change

        p_search, p_track, p_stabilize = self._softmax(
            score_search, score_track, score_stabilize
        )

        if p_track >= p_search and p_track >= p_stabilize:
            self.regime = ProbabilisticRegime.TRACK
            gain = self.track_gain
        elif p_stabilize >= p_search:
            self.regime = ProbabilisticRegime.STABILIZE
            gain = self.stabilize_gain
        else:
            self.regime = ProbabilisticRegime.SEARCH
            gain = self.search_gain

        self.effective_step = self.base_step * gain
        self.action = self._clip(self.action + self.direction * self.effective_step)
        self.previous_observation = observation

        state = ProbabilisticState(
            observation=observation,
            delta=delta,
            volatility=self.volatility,
            coherence=self.coherence,
            reversal_rate=self.reversal_rate,
            p_search=p_search,
            p_track=p_track,
            p_stabilize=p_stabilize,
            regime=self.regime,
            direction=self.direction,
            effective_step=self.effective_step,
        )
        return self.action, state
