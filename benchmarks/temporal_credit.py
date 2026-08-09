from __future__ import annotations

import json
import random
from collections import deque
from math import pi, sin
from statistics import mean

from trivax.baselines import PerturbAndObserve
from trivax.delay_aware import DelayAwareController
from trivax.probabilistic import ProbabilisticRegimeController
from trivax.temporal_credit import HistoricalCreditController


class DelayedSinusoidalPlant:
    def __init__(self, delay=0, center=0.70, amplitude=0.10, period=200.0, curvature=4.0):
        self.delay = delay
        self.center = center
        self.amplitude = amplitude
        self.period = period
        self.curvature = curvature
        self.actions = deque([0.5] * (delay + 1), maxlen=delay + 1)

    def optimum_at(self, t):
        return self.center + self.amplitude * sin(2.0 * pi * t / self.period)

    def evaluate(self, action, t):
        self.actions.append(float(action))
        effective_action = self.actions[0]
        target = self.optimum_at(t)
        return 1.0 - self.curvature * (effective_action - target) ** 2


def run(controller, delay, steps=1200, noise_sigma=0.0, seed=0):
    rng = random.Random(seed)
    plant = DelayedSinusoidalPlant(delay=delay)
    action = float(controller.action)
    errors = []

    for t in range(steps):
        observation = plant.evaluate(action, t)
        if noise_sigma:
            observation += rng.gauss(0.0, noise_sigma)
        result = controller.step(observation)
        action = float(result[0] if isinstance(result, tuple) else result)
        errors.append(abs(action - plant.optimum_at(t)))

    return mean(errors[-300:])


def benchmark(delays=(0, 1, 2, 4, 7, 10), seeds=range(20), noise_sigma=0.005):
    rows = []
    for delay in delays:
        for seed in seeds:
            controllers = {
                "temporal_credit": HistoricalCreditController(step_size=0.01, delay=delay),
                "delay_hold": DelayAwareController(
                    ProbabilisticRegimeController(step_size=0.01),
                    sensor_delay=delay,
                ),
                "trivax_v0_3": ProbabilisticRegimeController(step_size=0.01),
                "perturb_and_observe": PerturbAndObserve(step_size=0.01),
            }
            for name, controller in controllers.items():
                rows.append(
                    {
                        "delay": delay,
                        "seed": seed,
                        "controller": name,
                        "tail_mean_abs_error": run(
                            controller,
                            delay=delay,
                            noise_sigma=noise_sigma,
                            seed=seed,
                        ),
                    }
                )
    return rows


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
