from __future__ import annotations

import json
import random
from collections import deque
from math import pi, sin
from statistics import mean, median

from trivax.baselines import PerturbAndObserve
from trivax.probabilistic import ProbabilisticRegimeController
from trivax.runtime_v2 import TrivaxRuntimeV2
from trivax.runtime_v3 import TrivaxRuntimeV3
from trivax.temporal_credit import HistoricalCreditController


class CompositeAdversarialPlant:
    """Moving, delayed, noisy, nonstationary scalar plant.

    The benchmark intentionally combines several previously isolated stressors:
    - moving optimum;
    - time-varying feedback delay;
    - Gaussian observation noise;
    - impulsive outliers;
    - abrupt optimum shift;
    - abrupt curvature change.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.action_history = deque([0.5] * 32, maxlen=96)

    @staticmethod
    def delay_at(t: int) -> int:
        if t < 400:
            return 2
        if t < 800:
            return 5
        return 1

    @staticmethod
    def optimum_at(t: int) -> float:
        value = 0.65 + 0.08 * sin(2.0 * pi * t / 180.0)
        if t >= 700:
            value += 0.08
        return max(0.10, min(0.90, value))

    @staticmethod
    def curvature_at(t: int) -> float:
        return 4.0 if t < 700 else 7.0

    def evaluate(self, action: float, t: int) -> float:
        self.action_history.append(float(action))
        delay = self.delay_at(t)
        history = list(self.action_history)
        delayed_action = history[-(delay + 1)]
        target = self.optimum_at(t)
        observation = 1.0 - self.curvature_at(t) * (delayed_action - target) ** 2
        observation += self.rng.gauss(0.0, 0.01)
        if self.rng.random() < 0.01:
            observation += self.rng.choice((-1.0, 1.0)) * self.rng.uniform(0.12, 0.30)
        return observation


def step_with_state(controller, observation: float):
    result = controller.step(observation)
    if isinstance(result, tuple):
        return float(result[0]), result[1]
    return float(result), None


def run(controller, seed: int, steps: int = 1200, oracle_delay: bool = False):
    plant = CompositeAdversarialPlant(seed=seed)
    action = float(controller.action)
    errors = []
    final_delay = None

    for t in range(steps):
        observation = plant.evaluate(action, t)
        if oracle_delay and hasattr(controller, "set_delay"):
            controller.set_delay(plant.delay_at(t))
        action, state = step_with_state(controller, observation)
        errors.append(abs(action - plant.optimum_at(t)))

        if state is not None:
            final_delay = getattr(state, "accepted_delay", final_delay)
            if final_delay is None:
                final_delay = getattr(state, "estimated_delay", final_delay)

    return {
        "mean_abs_error": mean(errors),
        "median_abs_error": median(errors),
        "tail_mean_abs_error": mean(errors[-300:]),
        "max_abs_error": max(errors),
        "final_estimated_delay": final_delay,
    }


def benchmark(seeds=range(20)):
    rows = []
    for seed in seeds:
        controllers = {
            "trivax_runtime_v3": TrivaxRuntimeV3(),
            "trivax_runtime_v2": TrivaxRuntimeV2(),
            "historical_credit_oracle_delay": HistoricalCreditController(step_size=0.01),
            "trivax_v0_3": ProbabilisticRegimeController(step_size=0.01),
            "perturb_and_observe": PerturbAndObserve(step_size=0.01),
        }
        for name, controller in controllers.items():
            row = {"seed": seed, "controller": name}
            row.update(
                run(
                    controller,
                    seed=seed,
                    oracle_delay=(name == "historical_credit_oracle_delay"),
                )
            )
            rows.append(row)
    return rows


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
