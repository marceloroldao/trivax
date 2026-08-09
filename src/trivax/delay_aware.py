from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DelayAwareState:
    observation: float
    held_action: float
    update_applied: bool
    hold_counter: int


class DelayAwareController:
    """Temporal credit-assignment wrapper for controllers with delayed feedback.

    The wrapped controller is updated once per (delay + 1) observations and the
    previously selected action is held in between. This prevents rapidly issuing
    new actions while the available feedback still corresponds to older actions.
    """

    def __init__(self, controller, sensor_delay: int) -> None:
        if sensor_delay < 0:
            raise ValueError("sensor_delay must be non-negative")
        if not hasattr(controller, "step") or not hasattr(controller, "action"):
            raise TypeError("controller must expose step() and action")
        self.controller = controller
        self.sensor_delay = int(sensor_delay)
        self.hold_period = self.sensor_delay + 1
        self.counter = 0
        self.action = float(controller.action)

    def step(self, observation: float) -> tuple[float, DelayAwareState]:
        update_applied = self.counter % self.hold_period == 0
        if update_applied:
            result = self.controller.step(float(observation))
            self.action = float(result[0] if isinstance(result, tuple) else result)

        state = DelayAwareState(
            observation=float(observation),
            held_action=self.action,
            update_applied=update_applied,
            hold_counter=self.counter % self.hold_period,
        )
        self.counter += 1
        return self.action, state
