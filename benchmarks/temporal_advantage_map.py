"""Map regimes where temporal credit beats a strong adaptive hill climber.

This benchmark is descriptive: it searches for regions of operating conditions
where explicit delay attribution is useful. It does not tune controller
parameters per cell.
"""
from __future__ import annotations

import csv
import math
import random
from collections import deque
from statistics import mean

from trivax.runtime_v2 import TrivaxRuntimeV2
from trivax.adaptive_baseline import AdaptiveHillClimber


def target(t: int, mode: str) -> float:
    if mode == "slow":
        return 0.50 + 0.05 * math.sin(t / 140.0)
    if mode == "fast":
        return 0.50 + 0.08 * math.sin(t / 35.0)
    if mode == "steps":
        return (0.42, 0.58, 0.47, 0.62)[(t // 450) % 4]
    raise ValueError(mode)


def delay_at(t: int, spec: tuple[int, ...]) -> int:
    if len(spec) == 1:
        return spec[0]
    segment = max(1, 1800 // len(spec))
    return spec[min(len(spec) - 1, t // segment)]


def run(seed: int, controller_name: str, delay_spec: tuple[int, ...], noise: float,
        outlier_rate: float, motion: str, curvature: float) -> float:
    rng = random.Random(seed)
    if controller_name == "trivax_v2":
        controller = TrivaxRuntimeV2()
    elif controller_name == "adaptive_hill":
        controller = AdaptiveHillClimber()
    else:
        raise ValueError(controller_name)

    max_delay = max(delay_spec)
    action0 = float(controller.action)
    history: deque[float] = deque([action0] * (max_delay + 3), maxlen=max_delay + 3)
    errors: list[float] = []

    for t in range(1800):
        d = delay_at(t, delay_spec)
        delayed_action = list(history)[-(d + 1)]
        optimum = target(t, motion)
        observation = 1.0 - curvature * (delayed_action - optimum) ** 2
        observation += rng.gauss(0.0, noise)
        if outlier_rate > 0.0 and rng.random() < outlier_rate:
            observation += rng.choice((-0.10, 0.10))

        if controller_name == "trivax_v2":
            action, _ = controller.step(observation)
        else:
            action, _ = controller.step(observation)
        history.append(float(action))
        errors.append(abs(float(action) - optimum))

    return mean(errors[-600:])


def main() -> None:
    delay_specs = {
        "d0": (0,),
        "d2": (2,),
        "d5": (5,),
        "d9": (9,),
        "d2_7_1": (2, 7, 1),
        "d1_9_3_6": (1, 9, 3, 6),
    }
    noises = (0.0, 0.002, 0.006)
    outliers = (0.0, 0.01)
    motions = ("slow", "fast", "steps")
    curvatures = (0.7, 1.8)
    seeds = range(12)

    writer = csv.writer(__import__("sys").stdout)
    writer.writerow([
        "delay_regime", "noise", "outlier_rate", "motion", "curvature",
        "trivax_tail_mae", "adaptive_tail_mae", "relative_gain", "winner"
    ])

    wins = {"trivax_v2": 0, "adaptive_hill": 0, "tie": 0}
    for dname, dspec in delay_specs.items():
        for noise in noises:
            for outlier_rate in outliers:
                for motion in motions:
                    for curvature in curvatures:
                        tv = mean(run(s, "trivax_v2", dspec, noise, outlier_rate, motion, curvature) for s in seeds)
                        ah = mean(run(s, "adaptive_hill", dspec, noise, outlier_rate, motion, curvature) for s in seeds)
                        relative_gain = (ah - tv) / ah if ah > 0 else 0.0
                        if abs(tv - ah) <= 1e-6:
                            winner = "tie"
                        elif tv < ah:
                            winner = "trivax_v2"
                        else:
                            winner = "adaptive_hill"
                        wins[winner] += 1
                        writer.writerow([
                            dname, noise, outlier_rate, motion, curvature,
                            f"{tv:.8f}", f"{ah:.8f}", f"{relative_gain:.8f}", winner
                        ])

    print(f"# summary,trivax_wins={wins['trivax_v2']},adaptive_wins={wins['adaptive_hill']},ties={wins['tie']}")


if __name__ == "__main__":
    main()
