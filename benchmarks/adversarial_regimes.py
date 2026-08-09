from __future__ import annotations

import json
import math
import random
from statistics import mean

from trivax.baselines import PerturbAndObserve
from trivax.core import TrivaxController
from trivax.experimental import CoherenceAdaptiveController, RegimeAdaptiveController
from trivax.probabilistic import ProbabilisticRegimeController


class AdversarialPlant:
    def __init__(self, kind: str):
        self.kind = kind
        self.last_action = 0.5

    def optimum_at(self, t: int) -> float:
        if self.kind == "slow_drift":
            return 0.25 + 0.00055 * t
        if self.kind == "repeated_steps":
            return [0.25, 0.80, 0.35, 0.75][(t // 200) % 4]
        if self.kind in {"outliers", "delay"}:
            return 0.65 + 0.10 * math.sin(2.0 * math.pi * t / 200.0)
        if self.kind == "hysteresis":
            return 0.62 + 0.08 * math.sin(2.0 * math.pi * t / 220.0)
        if self.kind == "curvature_shift":
            return 0.30 if t < 400 else 0.78
        raise ValueError(f"unknown plant kind: {self.kind}")

    def curvature_at(self, t: int) -> float:
        if self.kind == "curvature_shift":
            return 2.0 if t < 400 else 12.0
        return 4.0

    def evaluate(self, action: float, t: int) -> float:
        target = self.optimum_at(t)
        effective_action = action
        if self.kind == "hysteresis":
            direction_bias = 0.0
            if action > self.last_action:
                direction_bias = 0.015
            elif action < self.last_action:
                direction_bias = -0.015
            effective_action = 0.8 * action + 0.2 * self.last_action + direction_bias
            self.last_action = action
        return 1.0 - self.curvature_at(t) * (effective_action - target) ** 2


def controller_step(controller, observation: float) -> float:
    result = controller.step(observation)
    return result[0] if isinstance(result, tuple) else result


def make_controllers(step_size: float):
    return {
        "trivax_v0_1": TrivaxController(step_size=step_size),
        "trivax_coherence_adaptive": CoherenceAdaptiveController(step_size=step_size),
        "trivax_v0_2_regime": RegimeAdaptiveController(step_size=step_size),
        "trivax_v0_3_probabilistic": ProbabilisticRegimeController(step_size=step_size),
        "perturb_and_observe": PerturbAndObserve(step_size=step_size),
    }


def run(controller, kind: str, seed: int, steps: int = 900, delay_steps: int = 5):
    plant = AdversarialPlant(kind)
    rng = random.Random(seed)
    action = controller.action
    errors = []
    observation_history = []

    for t in range(steps):
        observation = plant.evaluate(action, t)

        if kind == "outliers":
            observation += rng.gauss(0.0, 0.005)
            if rng.random() < 0.02:
                observation += rng.choice((-1.0, 1.0)) * rng.uniform(0.08, 0.25)

        if kind == "delay":
            observation_history.append(observation)
            delayed_index = max(0, len(observation_history) - 1 - delay_steps)
            observation = observation_history[delayed_index]

        action = controller_step(controller, observation)
        errors.append(abs(action - plant.optimum_at(t)))

    return {
        "mean_abs_error": mean(errors),
        "tail_mean_abs_error": mean(errors[-200:]),
        "max_abs_error": max(errors),
    }


def benchmark(step_sizes=(0.005, 0.01, 0.02, 0.05), seeds=range(20)):
    kinds = (
        "slow_drift",
        "repeated_steps",
        "outliers",
        "delay",
        "hysteresis",
        "curvature_shift",
    )
    rows = []
    for kind in kinds:
        for step_size in step_sizes:
            for seed in seeds:
                for controller_name, controller in make_controllers(step_size).items():
                    row = {
                        "scenario": kind,
                        "controller": controller_name,
                        "step_size": step_size,
                        "seed": seed,
                    }
                    row.update(run(controller, kind, seed))
                    rows.append(row)
    return rows


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
