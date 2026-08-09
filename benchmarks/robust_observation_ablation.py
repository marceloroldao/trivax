from __future__ import annotations

import json
import random
from collections import deque
from math import pi, sin
from statistics import mean

from trivax.probabilistic import ProbabilisticRegimeController
from trivax.robust_observation import RobustObservationLayer


class SinusoidalPlant:
    def optimum_at(self, t):
        return 0.70 + 0.10 * sin(2.0 * pi * t / 200.0)

    def evaluate(self, action, t):
        target = self.optimum_at(t)
        return 1.0 - 4.0 * (action - target) ** 2


def run(mode, *, seed=0, delay=0, outlier_probability=0.0, steps=1000):
    rng = random.Random(seed)
    plant = SinusoidalPlant()
    controller = ProbabilisticRegimeController(step_size=0.01)
    layer = RobustObservationLayer(median_window=3, innovation_clip=0.05)
    queue = deque()
    action = controller.action
    errors = []

    for t in range(steps):
        obs = plant.evaluate(action, t)
        if outlier_probability and rng.random() < outlier_probability:
            obs += rng.choice((-1.0, 1.0)) * 0.30

        queue.append(obs)
        delivered = queue[0] if len(queue) > delay else obs
        if len(queue) > delay:
            queue.popleft()

        if mode == "raw":
            control_obs = delivered
        elif mode == "robust":
            control_obs = layer.process(delivered).filtered
        else:
            raise ValueError(mode)

        action, _ = controller.step(control_obs)
        errors.append(abs(action - plant.optimum_at(t)))

    return mean(errors)


def benchmark(seeds=range(20)):
    scenarios = {
        "clean": dict(delay=0, outlier_probability=0.0),
        "outliers": dict(delay=0, outlier_probability=0.03),
        "delay_4": dict(delay=4, outlier_probability=0.0),
    }
    results = []
    for scenario, kwargs in scenarios.items():
        for mode in ("raw", "robust"):
            values = [run(mode, seed=seed, **kwargs) for seed in seeds]
            results.append({
                "scenario": scenario,
                "mode": mode,
                "mean_abs_error": mean(values),
            })
    return results


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
