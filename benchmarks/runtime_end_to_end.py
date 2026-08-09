from __future__ import annotations

import json
import random
from collections import deque
from math import pi, sin
from statistics import mean

from trivax.baselines import PerturbAndObserve
from trivax.probabilistic import ProbabilisticRegimeController
from trivax.runtime import TrivaxRuntime


class DelayedSinusoidalPlant:
    def __init__(self, delay=0, noise_sigma=0.002):
        self.delay = int(delay)
        self.noise_sigma = float(noise_sigma)

    def target(self, t):
        return 0.70 + 0.10 * sin(2.0 * pi * t / 200.0)

    def evaluate(self, action, t):
        target = self.target(t)
        return 1.0 - 4.0 * (action - target) ** 2


def controller_step(controller, observation):
    result = controller.step(observation)
    return result[0] if isinstance(result, tuple) else result


def run(controller_name, delay, seed=0, steps=1200):
    rng = random.Random(seed)
    plant = DelayedSinusoidalPlant(delay=delay)

    if controller_name == "runtime":
        controller = TrivaxRuntime(
            controller=ProbabilisticRegimeController(step_size=0.01)
        )
    elif controller_name == "v0_3":
        controller = ProbabilisticRegimeController(step_size=0.01)
    elif controller_name == "po":
        controller = PerturbAndObserve(step_size=0.01)
    else:
        raise ValueError(controller_name)

    action = 0.5
    action_history = deque([action] * (delay + 1), maxlen=delay + 1)
    errors = []
    estimated_delays = []

    for t in range(steps):
        delayed_action = action_history[0]
        observation = plant.evaluate(delayed_action, t)
        observation += rng.gauss(0.0, plant.noise_sigma)

        action = controller_step(controller, observation)
        action_history.append(action)
        errors.append(abs(action - plant.target(t)))

        if controller_name == "runtime":
            estimated_delays.append(controller.estimated_delay)

    return {
        "controller": controller_name,
        "delay": delay,
        "seed": seed,
        "tail_mean_abs_error": mean(errors[300:]),
        "final_estimated_delay": estimated_delays[-1] if estimated_delays else None,
    }


def benchmark(delays=(0, 1, 2, 4, 7), seeds=range(20)):
    rows = []
    for delay in delays:
        for seed in seeds:
            for controller_name in ("runtime", "v0_3", "po"):
                rows.append(run(controller_name, delay, seed))
    return rows


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
