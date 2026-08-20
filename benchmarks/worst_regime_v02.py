"""Worst-regime robustness benchmark for TRIVAX v0.2.

This benchmark complements mean-MAE evaluations by reporting the worst regime
across a deliberately diverse grid. It is not a tuning benchmark.
"""
from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.regime_selector import TrivaxRegimeSelector
from trivax.runtime_v2 import TrivaxRuntimeV2


def target(t: int, speed: str) -> float:
    if speed == "slow":
        return 0.50 + 0.035 * math.sin(t / 95.0)
    if speed == "fast":
        return 0.50 + 0.075 * math.sin(t / 24.0)
    return 0.56 if (t // 110) % 2 else 0.43


def run(kind: str, seed: int, delay: int, speed: str, sigma: float, curvature: float) -> float:
    rng = random.Random(seed * 991 + delay * 17 + int(sigma * 1e6))
    if kind == "selector":
        ctrl = TrivaxRegimeSelector()
    elif kind == "adaptive":
        ctrl = AdaptiveHillClimber()
    elif kind == "temporal":
        ctrl = TrivaxRuntimeV2()
    else:
        raise ValueError(kind)

    action = float(ctrl.action)
    history: deque[float] = deque([action] * 24, maxlen=32)
    errors: list[float] = []

    for t in range(1500):
        applied = list(history)[-(delay + 1)]
        optimum = target(t, speed)
        y = 1.0 - curvature * (applied - optimum) ** 2
        y += rng.gauss(0.0, sigma)
        if rng.random() < 0.004:
            y += rng.choice((-0.08, 0.08))
        action, _ = ctrl.step(y)
        action = float(action)
        history.append(action)
        errors.append(abs(action - optimum))

    return mean(errors[250:])


def main() -> None:
    delays = (0, 2, 5, 9)
    speeds = ("slow", "fast", "step")
    sigmas = (0.0015, 0.0035, 0.0060)
    curvatures = (1.0, 4.0)
    kinds = ("selector", "adaptive", "temporal")
    seeds = range(12)

    print("controller,worst_mae,median_regime_mae,regimes_won,total_regimes")
    regime_scores: dict[str, list[float]] = {k: [] for k in kinds}
    wins = {k: 0 for k in kinds}
    total = 0

    for delay in delays:
        for speed in speeds:
            for sigma in sigmas:
                for curvature in curvatures:
                    total += 1
                    scores = {}
                    for kind in kinds:
                        mae = mean(run(kind, seed, delay, speed, sigma, curvature) for seed in seeds)
                        scores[kind] = mae
                        regime_scores[kind].append(mae)
                    best = min(scores, key=scores.get)
                    wins[best] += 1

    for kind in kinds:
        vals = sorted(regime_scores[kind])
        median_regime = vals[len(vals) // 2]
        print(
            f"{kind},{max(vals):.8f},{median_regime:.8f},{wins[kind]},{total}"
        )


if __name__ == "__main__":
    main()
