"""Out-of-distribution holdout benchmark for TRIVAX.

The holdout deliberately differs from tuning benchmarks in waveform, delay
schedule, noise structure, curvature transitions and outlier timing.
"""
from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean, median

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.baselines import PerturbAndObserve
from trivax.runtime_v2 import TrivaxRuntimeV2
from trivax.runtime_v5 import TrivaxRuntimeV5


def target(t: int) -> float:
    # Triangular-ish target plus slow drift; not used in training/sweeps.
    period = 260
    phase = (t % period) / period
    tri = 4.0 * abs(phase - 0.5) - 1.0
    value = 0.58 + 0.07 * tri + 0.00003 * t
    return max(0.15, min(0.85, value))


def true_delay(t: int) -> int:
    if t < 500:
        return 3
    if t < 950:
        return 9
    if t < 1450:
        return 0
    return 6


def curvature(t: int) -> float:
    if t < 800:
        return 2.5
    if t < 1300:
        return 6.5
    return 3.8


def run(controller, seed: int, steps: int = 1900) -> dict[str, float]:
    rng = random.Random(seed)
    action = float(controller.action)
    history: deque[float] = deque([action] * 32, maxlen=64)
    errors: list[float] = []
    effort: list[float] = []
    delay_hits: list[float] = []
    prev = action

    for t in range(steps):
        d = true_delay(t)
        delayed_action = list(history)[-(d + 1)]
        optimum = target(t)
        y = 1.0 - curvature(t) * (delayed_action - optimum) ** 2
        # Heteroskedastic + occasional Laplace-like impulsive noise.
        sigma = 0.003 if t < 900 else 0.008
        y += rng.gauss(0.0, sigma)
        if rng.random() < 0.008:
            y += (1 if rng.random() < 0.5 else -1) * rng.expovariate(12.0)

        result = controller.step(y)
        action = float(result[0] if isinstance(result, tuple) else result)
        history.append(action)
        errors.append(abs(action - optimum))
        effort.append(abs(action - prev))
        prev = action
        accepted = getattr(controller, "estimated_delay", None)
        if accepted is None and isinstance(result, tuple) and len(result) > 1:
            state = result[1]
            accepted = getattr(state, "accepted_delay", None)
        delay_hits.append(float(accepted == d))

    return {
        "mae": mean(errors),
        "median_ae": median(errors),
        "tail_mae": mean(errors[-400:]),
        "max_ae": max(errors),
        "effort": mean(effort),
        "delay_accuracy": mean(delay_hits[300:]),
    }


def main() -> None:
    seeds = range(40)
    factories = {
        "trivax_runtime_v5": TrivaxRuntimeV5,
        "trivax_runtime_v2": TrivaxRuntimeV2,
        "adaptive_hill_climber": AdaptiveHillClimber,
        "perturb_and_observe": PerturbAndObserve,
    }

    print("controller,mae,median_ae,tail_mae,max_ae,effort,delay_accuracy")
    for name, factory in factories.items():
        rows = [run(factory(), seed) for seed in seeds]
        print(
            name,
            f'{mean(r["mae"] for r in rows):.6f}',
            f'{mean(r["median_ae"] for r in rows):.6f}',
            f'{mean(r["tail_mae"] for r in rows):.6f}',
            f'{mean(r["max_ae"] for r in rows):.6f}',
            f'{mean(r["effort"] for r in rows):.6f}',
            f'{mean(r["delay_accuracy"] for r in rows):.4f}',
            sep=",",
        )


if __name__ == "__main__":
    main()
