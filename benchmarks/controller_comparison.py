from __future__ import annotations

import json
import random
from math import pi, sin
from statistics import mean, median

from trivax.baselines import PerturbAndObserve
from trivax.core import TrivaxController
from trivax.experimental import CoherenceAdaptiveController


class StepPlant:
    def __init__(self, before=0.25, after=0.80, change_step=300, curvature=4.0):
        self.before = before
        self.after = after
        self.change_step = change_step
        self.curvature = curvature

    def optimum_at(self, t):
        return self.before if t < self.change_step else self.after

    def evaluate(self, action, t):
        target = self.optimum_at(t)
        return 1.0 - self.curvature * (action - target) ** 2


class SinusoidalPlant:
    def __init__(self, center=0.70, amplitude=0.10, period=200.0, curvature=4.0):
        self.center = center
        self.amplitude = amplitude
        self.period = period
        self.curvature = curvature

    def optimum_at(self, t):
        return self.center + self.amplitude * sin(2.0 * pi * t / self.period)

    def evaluate(self, action, t):
        target = self.optimum_at(t)
        return 1.0 - self.curvature * (action - target) ** 2


def controller_step(controller, observation):
    result = controller.step(observation)
    return result[0] if isinstance(result, tuple) else result


def run(controller, plant, steps=700, noise_sigma=0.0, seed=0):
    rng = random.Random(seed)
    action = controller.action
    errors = []
    effort = []

    for t in range(steps):
        observation = plant.evaluate(action, t)
        if noise_sigma:
            observation += rng.gauss(0.0, noise_sigma)

        next_action = controller_step(controller, observation)
        errors.append(abs(action - plant.optimum_at(t)))
        effort.append(abs(next_action - action))
        action = next_action

    return errors, effort


def settling_time(errors, start, threshold=0.03, window=10):
    for i in range(start, len(errors) - window + 1):
        if all(e < threshold for e in errors[i : i + window]):
            return i - start
    return None


def summarize(errors, effort, settling_start=None):
    out = {
        "mean_abs_error": mean(errors),
        "median_abs_error": median(errors),
        "tail_mean_abs_error": mean(errors[-100:]),
        "max_abs_error": max(errors),
        "mean_control_effort": mean(effort),
    }
    if settling_start is not None:
        out["settling_steps"] = settling_time(errors, settling_start)
    return out


def make_controllers(step_size):
    return {
        "trivax_v0_1": TrivaxController(step_size=step_size),
        "trivax_coherence_adaptive": CoherenceAdaptiveController(step_size=step_size),
        "perturb_and_observe": PerturbAndObserve(step_size=step_size),
    }


def benchmark(step_sizes=(0.005, 0.01, 0.02, 0.05), seeds=range(10)):
    results = []
    scenarios = [
        ("sinusoidal_clean", lambda: SinusoidalPlant(), 1000, 0.0, None),
        ("sinusoidal_noisy", lambda: SinusoidalPlant(), 1000, 0.01, None),
        ("step_clean", lambda: StepPlant(), 700, 0.0, 300),
        ("step_noisy", lambda: StepPlant(), 700, 0.01, 300),
    ]

    for scenario_name, plant_factory, steps, noise_sigma, settling_start in scenarios:
        for step_size in step_sizes:
            for seed in seeds:
                for controller_name, controller in make_controllers(step_size).items():
                    errors, effort = run(
                        controller,
                        plant_factory(),
                        steps=steps,
                        noise_sigma=noise_sigma,
                        seed=seed,
                    )
                    row = {
                        "scenario": scenario_name,
                        "controller": controller_name,
                        "step_size": step_size,
                        "seed": seed,
                    }
                    row.update(summarize(errors, effort, settling_start))
                    results.append(row)
    return results


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
