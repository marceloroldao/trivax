from __future__ import annotations

from dataclasses import dataclass
from math import sin
from typing import Callable

from .core import TrivaxController


@dataclass
class ScalarPlant:
    """Simple bounded plant with a moving optimum for controller evaluation."""

    optimum: float = 0.70
    curvature: float = 4.0
    drift_amplitude: float = 0.10
    drift_period: float = 200.0

    def optimum_at(self, t: int) -> float:
        return self.optimum + self.drift_amplitude * sin(2.0 * 3.141592653589793 * t / self.drift_period)

    def evaluate(self, action: float, t: int) -> float:
        target = self.optimum_at(t)
        return 1.0 - self.curvature * (action - target) ** 2


def run_closed_loop(
    controller: TrivaxController,
    plant: ScalarPlant,
    steps: int = 500,
) -> list[dict[str, float]]:
    if steps <= 0:
        raise ValueError("steps must be positive")

    records: list[dict[str, float]] = []
    action = controller.action

    for t in range(steps):
        observation = plant.evaluate(action, t)
        next_action, state = controller.step(observation)
        target = plant.optimum_at(t)
        error = abs(action - target)

        records.append(
            {
                "t": float(t),
                "action": float(action),
                "next_action": float(next_action),
                "observation": float(observation),
                "target": float(target),
                "abs_error": float(error),
                "coherence": float(state.coherence),
                "direction": float(state.direction),
            }
        )
        action = next_action

    return records
