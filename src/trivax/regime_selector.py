from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median

from .adaptive_baseline import AdaptiveHillClimber, AdaptiveHillState
from .runtime_v2 import TrivaxRuntimeV2, RuntimeV2State


class RegimeMode(str, Enum):
    ADAPTIVE = "adaptive"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class RegimeSelectorState:
    mode: RegimeMode
    score: float
    delay: int | None
    delay_score: float
    delay_stable: bool
    local_noise: float
    local_speed: float
    adaptive_action: float
    temporal_action: float
    action: float
    switched: bool


class TrivaxRegimeSelector:
    """Online selector between lightweight adaptive control and temporal credit.

    The selector uses only online observables. It does not receive scenario
    labels or true delay. Temporal mode is favored by stable non-zero delay,
    sufficient apparent dynamics, and moderate local noise. Hysteresis and a
    minimum dwell time prevent sample-to-sample mode chatter.
    """

    def __init__(
        self,
        adaptive: AdaptiveHillClimber | None = None,
        temporal: TrivaxRuntimeV2 | None = None,
        *,
        window: int = 17,
        enter_threshold: float = 0.64,
        exit_threshold: float = 0.42,
        min_dwell: int = 24,
        delay_score_floor: float = 0.45,
    ) -> None:
        if window < 5:
            raise ValueError("window must be >= 5")
        if not 0.0 <= exit_threshold < enter_threshold <= 1.0:
            raise ValueError("require 0 <= exit < enter <= 1")
        if min_dwell <= 0:
            raise ValueError("min_dwell must be positive")

        self.adaptive = adaptive or AdaptiveHillClimber()
        self.temporal = temporal or TrivaxRuntimeV2()
        self.window = int(window)
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.min_dwell = int(min_dwell)
        self.delay_score_floor = float(delay_score_floor)

        self.mode = RegimeMode.ADAPTIVE
        self.action = float(self.adaptive.action)
        self._observations: deque[float] = deque(maxlen=self.window)
        self._steps_in_mode = 0

    def _features(self) -> tuple[float, float]:
        vals = list(self._observations)
        if len(vals) < 5:
            return 0.0, 0.0
        diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        abs_diffs = [abs(v) for v in diffs]
        local_speed = median(abs_diffs)
        med = median(diffs)
        mad = median(abs(v - med) for v in diffs)
        return float(local_speed), float(1.4826 * mad)

    def _temporal_score(self, state: RuntimeV2State, speed: float, noise: float) -> float:
        delay = state.estimated_delay
        if delay is None or delay <= 0 or not state.delay_stable:
            delay_term = 0.0
        else:
            delay_strength = max(0.0, min(1.0, float(state.delay_score)))
            if delay_strength < self.delay_score_floor:
                delay_term = 0.0
            else:
                delay_term = min(1.0, 0.45 + 0.08 * min(7, int(delay)) + 0.35 * delay_strength)

        # Dynamic term: near-zero motion does not justify temporal complexity.
        dynamic_term = max(0.0, min(1.0, speed / 0.004))

        # Noise penalty: temporal map showed loss of advantage at high noise.
        snr_like = speed / (noise + 1e-9)
        noise_term = max(0.0, min(1.0, (snr_like - 0.7) / 2.3))

        return max(0.0, min(1.0, 0.58 * delay_term + 0.24 * dynamic_term + 0.18 * noise_term))

    def step(self, observation: float) -> tuple[float, RegimeSelectorState]:
        obs = float(observation)
        self._observations.append(obs)

        adaptive_action, _adaptive_state = self.adaptive.step(obs)
        temporal_action, temporal_state = self.temporal.step(obs)
        speed, noise = self._features()
        score = self._temporal_score(temporal_state, speed, noise)

        switched = False
        if self._steps_in_mode >= self.min_dwell:
            if self.mode is RegimeMode.ADAPTIVE and score >= self.enter_threshold:
                self.mode = RegimeMode.TEMPORAL
                self._steps_in_mode = 0
                switched = True
            elif self.mode is RegimeMode.TEMPORAL and score <= self.exit_threshold:
                self.mode = RegimeMode.ADAPTIVE
                self._steps_in_mode = 0
                switched = True

        # Synchronize chosen action back into both controllers to limit drift
        # between their latent action states during long dwell periods.
        chosen = temporal_action if self.mode is RegimeMode.TEMPORAL else adaptive_action
        self.action = float(chosen)
        self.adaptive.action = self.action
        self.temporal.action = self.action
        self.temporal.controller.action = self.action

        self._steps_in_mode += 1
        state = RegimeSelectorState(
            mode=self.mode,
            score=float(score),
            delay=temporal_state.estimated_delay,
            delay_score=float(temporal_state.delay_score),
            delay_stable=bool(temporal_state.delay_stable),
            local_noise=float(noise),
            local_speed=float(speed),
            adaptive_action=float(adaptive_action),
            temporal_action=float(temporal_action),
            action=self.action,
            switched=switched,
        )
        return self.action, state
