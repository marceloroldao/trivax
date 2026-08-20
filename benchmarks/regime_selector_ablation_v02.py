"""TRIVAX v0.2 regime-selector ablation benchmark.

Compares the full selector against variants with one observable term removed.
The benchmark is intentionally deterministic per seed and uses the same plant
for every variant.
"""
from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.regime_selector import TrivaxRegimeSelector


def target(t: int) -> float:
    if t < 600:
        return 0.48 + 0.035 * math.sin(t / 95.0)
    if t < 1200:
        return 0.56 if (t // 90) % 2 == 0 else 0.43
    return 0.50 + 0.065 * math.sin(t / 24.0)


def delay(t: int) -> int:
    if t < 450:
        return 0
    if t < 900:
        return 5
    if t < 1350:
        return 2
    return 8


def noise_sigma(t: int) -> float:
    if t < 1000:
        return 0.0015
    if t < 1500:
        return 0.0030
    return 0.0055


def run(seed: int, selector: TrivaxRegimeSelector) -> dict[str, float]:
    rng = random.Random(seed)
    hist: deque[float] = deque([selector.action] * 20, maxlen=20)
    errors: list[float] = []
    effort: list[float] = []
    prev = selector.action
    last_state = None

    for t in range(1900):
        d = delay(t)
        applied = list(hist)[-(d + 1)]
        optimum = target(t)
        curvature = 1.0 if t < 1100 else 1.55
        obs = 1.0 - curvature * (applied - optimum) ** 2
        obs += rng.gauss(0.0, noise_sigma(t))
        if t in (510, 1010, 1495, 1710):
            obs += rng.choice((-0.10, 0.10))
        action, last_state = selector.step(obs)
        hist.append(action)
        errors.append(abs(action - optimum))
        effort.append(abs(action - prev))
        prev = action

    assert last_state is not None
    return {
        "mae": mean(errors),
        "tail_mae": mean(errors[-400:]),
        "effort": mean(effort),
        "duty": last_state.temporal_duty_fraction,
        "switches": float(last_state.switch_count),
    }


def factory(name: str) -> TrivaxRegimeSelector:
    if name == "full":
        return TrivaxRegimeSelector()
    if name == "no_delay":
        return TrivaxRegimeSelector(delay_weight=0.0, dynamic_weight=0.24, noise_weight=0.18)
    if name == "no_dynamic":
        return TrivaxRegimeSelector(delay_weight=0.58, dynamic_weight=0.0, noise_weight=0.18)
    if name == "no_noise":
        return TrivaxRegimeSelector(delay_weight=0.58, dynamic_weight=0.24, noise_weight=0.0)
    if name == "low_hysteresis":
        return TrivaxRegimeSelector(enter_threshold=0.56, exit_threshold=0.52, min_dwell=8)
    raise ValueError(name)


def main() -> None:
    names = ("full", "no_delay", "no_dynamic", "no_noise", "low_hysteresis")
    seeds = range(30)
    print("variant,mae,tail_mae,effort,temporal_duty,switches")
    for name in names:
        rows = [run(seed, factory(name)) for seed in seeds]
        print(
            f"{name},"
            f"{mean(r['mae'] for r in rows):.8f},"
            f"{mean(r['tail_mae'] for r in rows):.8f},"
            f"{mean(r['effort'] for r in rows):.8f},"
            f"{mean(r['duty'] for r in rows):.6f},"
            f"{mean(r['switches'] for r in rows):.3f}"
        )


if __name__ == "__main__":
    main()
