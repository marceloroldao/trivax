from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveState:
    observation: float
    delta: float
    coherence: float
    direction: int
    effective_step: float


class CoherenceAdaptiveController:
    """Experimental TRIVAX controller with coherence-modulated step size.

    This variant is intentionally isolated from the v0.1 reference controller.
    Low coherence reduces the effective step near reversals; sustained agreement
    permits a larger step. It must be benchmarked before promotion into core.
    """

    def __init__(
        self,
        initial_action: float = 0.5,
        step_size: float = 0.05,
        min_action: float = 0.0,
        max_action: float = 1.0,
        coherence_alpha: float = 0.2,
        min_gain: float = 0.25,
        gain_span: float = 1.25,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if min_action >= max_action:
            raise ValueError("min_action must be smaller than max_action")
        if not min_action <= initial_action <= max_action:
            raise ValueError("initial_action must be within action bounds")
        if not 0.0 < coherence_alpha <= 1.0:
            raise ValueError("coherence_alpha must be in (0, 1]")
        if min_gain <= 0.0:
            raise ValueError("min_gain must be positive")
        if gain_span < 0.0:
            raise ValueError("gain_span must be non-negative")

        self.action = float(initial_action)
        self.base_step = float(step_size)
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.coherence_alpha = float(coherence_alpha)
        self.min_gain = float(min_gain)
        self.gain_span = float(gain_span)

        self.direction = 1
        self.previous_observation: float | None = None
        self.coherence = 0.5
        self.effective_step = self.base_step

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    def step(self, observation: float) -> tuple[float, AdaptiveState]:
        observation = float(observation)
        if self.previous_observation is None:
            delta = 0.0
        else:
            delta = observation - self.previous_observation
            agreement = 1.0 if delta >= 0.0 else 0.0
            if delta < 0.0:
                self.direction *= -1
            a = self.coherence_alpha
            self.coherence = (1.0 - a) * self.coherence + a * agreement

        gain = self.min_gain + self.gain_span * self.coherence
        self.effective_step = self.base_step * gain
        self.action = self._clip(self.action + self.direction * self.effective_step)
        self.previous_observation = observation

        state = AdaptiveState(
            observation=observation,
            delta=delta,
            coherence=self.coherence,
            direction=self.direction,
            effective_step=self.effective_step,
        )
        return self.action, state
