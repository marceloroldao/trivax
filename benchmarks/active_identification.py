from __future__ import annotations

import json
import random
from collections import deque
from math import pi, sin
from statistics import mean, median

from trivax.runtime_v3 import TrivaxRuntimeV3
from trivax.runtime_v4 import TrivaxRuntimeV4


class LowExcitationDelayedPlant:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.actions = deque([0.5] * 64, maxlen=128)

    @staticmethod
    def delay_at(t: int) -> int:
        if t < 500:
            return 4
        return 7

    @staticmethod
    def optimum_at(t: int) -> float:
        return 0.66 + 0.025 * sin(2.0 * pi * t / 260.0)

    def evaluate(self, action: float, t: int) -> float:
        self.actions.append(float(action))
        delayed = list(self.actions)[-(self.delay_at(t) + 1)]
        target = self.optimum_at(t)
        y = 1.0 - 4.0 * (delayed - target) ** 2
        y += self.rng.gauss(0.0, 0.006)
        if self.rng.random() < 0.005:
            y += self.rng.choice((-1.0, 1.0)) * self.rng.uniform(0.10, 0.20)
        return y


def run(runtime, seed: int, steps: int = 1000):
    plant = LowExcitationDelayedPlant(seed)
    action = float(runtime.action)
    errors = []
    effort = []
    accepted = []
    probes = 0

    for t in range(steps):
        y = plant.evaluate(action, t)
        next_action, state = runtime.step(y)
        errors.append(abs(action - plant.optimum_at(t)))
        effort.append(abs(next_action - action))
        accepted.append(getattr(state, "accepted_delay", None))
        probe_state = getattr(state, "probe_state", None)
        if probe_state is not None:
            probes = probe_state.probe_count
        action = next_action

    correct = [
        int(d == plant.delay_at(t)) if d is not None else 0
        for t, d in enumerate(accepted)
    ]
    return {
        "mean_abs_error": mean(errors),
        "median_abs_error": median(errors),
        "tail_mean_abs_error": mean(errors[-250:]),
        "mean_control_effort": mean(effort),
        "delay_accuracy": mean(correct),
        "final_delay": accepted[-1],
        "probe_count": probes,
    }


def benchmark(seeds=range(20)):
    rows = []
    for seed in seeds:
        for name, runtime in {
            "runtime_v3_passive": TrivaxRuntimeV3(),
            "runtime_v4_active": TrivaxRuntimeV4(),
        }.items():
            row = {"seed": seed, "runtime": name}
            row.update(run(runtime, seed))
            rows.append(row)
    return rows


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
