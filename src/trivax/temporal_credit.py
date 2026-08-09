from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalCreditState:
    observation: float
    delay: int
    credited_action: float | None
    previous_credited_action: float | None
    delta_observation: float
    delta_action: float
    local_slope: float | None
    direction: int
    action: float


class HistoricalCreditController:
    """Continuous controller with explicit delayed temporal credit assignment.

    The current observation is attributed to the action that was issued `delay`
    samples earlier. The local response slope is estimated from first
    differences:

        slope_t = (y_t - y_{t-1}) / (a_{t-d} - a_{t-d-1})

    The sign of this slope selects the current search direction while actions
    continue to be issued every cycle. This avoids the conservatism of holding
    the actuator for delay+1 samples.

    This module is experimental and assumes a scalar action and scalar objective.
    """

    def __init__(
        self,
        initial_action: float = 0.5,
        step_size: float = 0.01,
        delay: int = 0,
        min_action: float = 0.0,
        max_action: float = 1.0,
        slope_deadband: float = 1e-12,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if delay < 0:
            raise ValueError("delay must be non-negative")
        if min_action >= max_action:
            raise ValueError("min_action must be smaller than max_action")
        if not min_action <= initial_action <= max_action:
            raise ValueError("initial_action must be within action bounds")
        if slope_deadband < 0:
            raise ValueError("slope_deadband must be non-negative")

        self.action = float(initial_action)
        self.step_size = float(step_size)
        self.delay = int(delay)
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.slope_deadband = float(slope_deadband)

        self.direction = 1
        self.previous_observation: float | None = None
        self.action_history: list[float] = []

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    def set_delay(self, delay: int) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self.delay = int(delay)

    def step(self, observation: float) -> tuple[float, TemporalCreditState]:
        observation = float(observation)

        # Record the action currently applied at time t before deciding a_{t+1}.
        self.action_history.append(self.action)
        t = len(self.action_history) - 1

        credited_action: float | None = None
        previous_credited_action: float | None = None
        delta_observation = 0.0
        delta_action = 0.0
        local_slope: float | None = None

        if self.previous_observation is not None and t - self.delay >= 1:
            credited_action = self.action_history[t - self.delay]
            previous_credited_action = self.action_history[t - self.delay - 1]
            delta_observation = observation - self.previous_observation
            delta_action = credited_action - previous_credited_action

            if abs(delta_action) > self.slope_deadband:
                local_slope = delta_observation / delta_action
                self.direction = 1 if local_slope >= 0.0 else -1

        self.action = self._clip(
            self.action + self.direction * self.step_size
        )
        self.previous_observation = observation

        state = TemporalCreditState(
            observation=observation,
            delay=self.delay,
            credited_action=credited_action,
            previous_credited_action=previous_credited_action,
            delta_observation=delta_observation,
            delta_action=delta_action,
            local_slope=local_slope,
            direction=self.direction,
            action=self.action,
        )
        return self.action, state
