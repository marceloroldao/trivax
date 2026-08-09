from __future__ import annotations

import json
import random
from collections import deque
from math import pi, sin
from statistics import mean

from trivax.runtime_v4 import TrivaxRuntimeV4
from trivax.runtime_v5 import TrivaxRuntimeV5


class LowExcitationPlant:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.history = deque([0.5] * 32, maxlen=96)

    @staticmethod
    def delay_at(t: int) -> int:
        return 4 if t < 600 else 7

    @staticmethod
    def optimum_at(t: int) -> float:
        return 0.62 + 0.04 * sin(2.0 * pi * t / 260.0)

    def evaluate(self, action: float, t: int) -> float:
        self.history.append(float(action))
        delayed = list(self.history)[-(self.delay_at(t) + 1)]
        target = self.optimum_at(t)
        y = 1.0 - 4.0 * (delayed - target) ** 2
        y += self.rng.gauss(0.0, 0.008)
        if self.rng.random() < 0.008:
            y += self.rng.choice((-1.0, 1.0)) * self.rng.uniform(0.10, 0.22)
        return y


def run(runtime, seed: int, steps: int = 1200):
    plant = LowExcitationPlant(seed)
    action = runtime.action
    errors = []
    effort = []
    correct = []
    probes = 0

    for t in range(steps):
        obs = plant.evaluate(action, t)
        previous = action
        action, state = runtime.step(obs)
        errors.append(abs(action - plant.optimum_at(t)))
        effort.append(abs(action - previous))
        accepted = getattr(state, "accepted_delay", None)
        correct.append(1.0 if accepted == plant.delay_at(t) else 0.0)
        if hasattr(state, "probe_state"):
            probes += int(state.probe_state.applied)
        elif hasattr(state, "voi_state"):
            probes += int(state.voi_state.applied)

    return {
        "mean_abs_error": mean(errors),
        "tail_mean_abs_error": mean(errors[-300:]),
        "mean_control_effort": mean(effort),
        "delay_accuracy": mean(correct),
        "probe_count": probes,
    }


def benchmark(seeds=range(20)):
    rows = []
    for seed in seeds:
        for name, runtime in {
            "runtime_v4_fixed_probe": TrivaxRuntimeV4(),
            "runtime_v5_voi_probe": TrivaxRuntimeV5(),
        }.items():
            row = {"seed": seed, "controller": name}
            row.update(run(runtime, seed))
            rows.append(row)
    return rows


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
