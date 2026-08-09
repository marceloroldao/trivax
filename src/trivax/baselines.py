from __future__ import annotations


class PerturbAndObserve:
    """Deterministic fixed-step perturb-and-observe baseline."""

    def __init__(
        self,
        initial_action: float = 0.5,
        step_size: float = 0.05,
        min_action: float = 0.0,
        max_action: float = 1.0,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if min_action >= max_action:
            raise ValueError("min_action must be smaller than max_action")
        if not min_action <= initial_action <= max_action:
            raise ValueError("initial_action must be within action bounds")

        self.action = float(initial_action)
        self.step_size = float(step_size)
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.direction = 1
        self.previous_observation: float | None = None

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    def step(self, observation: float) -> float:
        observation = float(observation)
        if self.previous_observation is not None:
            if observation - self.previous_observation < 0.0:
                self.direction *= -1
        self.previous_observation = observation
        self.action = self._clip(self.action + self.direction * self.step_size)
        return self.action
