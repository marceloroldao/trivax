from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveHillState:
    observation: float
    delta_observation: float
    direction: int
    step_size: float
    action: float


class AdaptiveHillClimber:
    """Independent adaptive scalar hill-climbing baseline.

    This baseline is intentionally non-resolutive. It adapts its step size from
    recent success/failure only, providing a stronger comparator than fixed-step
    perturb-and-observe while remaining lightweight and auditable.
    """

    def __init__(
        self,
        initial_action: float = 0.5,
        step_size: float = 0.01,
        min_step: float = 0.001,
        max_step: float = 0.05,
        grow: float = 1.08,
        shrink: float = 0.55,
        min_action: float = 0.0,
        max_action: float = 1.0,
    ) -> None:
        if not 0.0 < min_step <= step_size <= max_step:
            raise ValueError("require 0 < min_step <= step_size <= max_step")
        if grow <= 1.0:
            raise ValueError("grow must be > 1")
        if not 0.0 < shrink < 1.0:
            raise ValueError("shrink must be in (0, 1)")
        self.action = float(initial_action)
        self.step_size = float(step_size)
        self.min_step = float(min_step)
        self.max_step = float(max_step)
        self.grow = float(grow)
        self.shrink = float(shrink)
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.direction = 1
        self.previous_observation: float | None = None

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    def step(self, observation: float) -> tuple[float, AdaptiveHillState]:
        observation = float(observation)
        delta = 0.0
        if self.previous_observation is not None:
            delta = observation - self.previous_observation
            if delta >= 0.0:
                self.step_size = min(self.max_step, self.step_size * self.grow)
            else:
                self.direction *= -1
                self.step_size = max(self.min_step, self.step_size * self.shrink)

        self.action = self._clip(self.action + self.direction * self.step_size)
        self.previous_observation = observation
        return self.action, AdaptiveHillState(
            observation=observation,
            delta_observation=delta,
            direction=self.direction,
            step_size=self.step_size,
            action=self.action,
        )
