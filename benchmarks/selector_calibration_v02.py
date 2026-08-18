"""Calibration-only sweep for TRIVAX v0.2 regime selector.

This benchmark is intentionally separated from holdout evaluation. It searches
selector thresholds and dwell time on calibration scenarios only. Results must
not be treated as final generalization evidence.
"""
from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.regime_selector import TrivaxRegimeSelector


def target(t: int, variant: int) -> float:
    if variant == 0:
        return 0.5 + 0.08 * math.sin(t / 70.0)
    if variant == 1:
        return 0.52 + 0.05 * math.sin(t / 35.0) + 0.015 * math.sin(t / 9.0)
    return 0.48 + (0.06 if (t // 180) % 2 else -0.06)


def delay(t: int, variant: int) -> int:
    schedules = (
        (0, 3, 6, 2),
        (2, 5, 1, 7),
        (1, 4, 8, 3),
    )
    seq = schedules[variant]
    return seq[min(3, t // 500)]


def run(seed: int, variant: int, enter: float, exit_: float, dwell: int) -> dict[str, float]:
    rng = random.Random(seed * 1009 + variant * 9176)
    selector = TrivaxRegimeSelector(
        enter_threshold=enter,
        exit_threshold=exit_,
        min_dwell=dwell,
    )
    history: deque[float] = deque([selector.action] * 20, maxlen=20)
    errors: list[float] = []
    switches = 0
    last_state = None

    for t in range(2000):
        d = delay(t, variant)
        a = list(history)[-(d + 1)]
        optimum = target(t, variant)
        curvature = 1.0 + 0.35 * (variant == 1) + 0.55 * (t >= 1100 and variant == 2)
        y = 1.0 - curvature * (a - optimum) ** 2
        sigma = (0.0015, 0.0025, 0.0035)[variant]
        y += rng.gauss(0.0, sigma)
        if t in (430, 910, 1470) and variant != 0:
            y += rng.choice((-0.08, 0.08))

        action, state = selector.step(y)
        history.append(action)
        errors.append(abs(action - optimum))
        switches += int(state.switched)
        last_state = state

    assert last_state is not None
    return {
        "mae": mean(errors),
        "tail_mae": mean(errors[-400:]),
        "switches": float(switches),
        "temporal_duty": float(last_state.temporal_duty_fraction),
    }


def main() -> None:
    enters = (0.58, 0.64, 0.70)
    exits = (0.34, 0.42, 0.50)
    dwells = (16, 24, 40)
    seeds = range(8)
    rows = []

    for enter in enters:
        for exit_ in exits:
            if exit_ >= enter:
                continue
            for dwell in dwells:
                results = [run(s, v, enter, exit_, dwell) for v in range(3) for s in seeds]
                row = {
                    "enter": enter,
                    "exit": exit_,
                    "dwell": dwell,
                    "mae": mean(r["mae"] for r in results),
                    "tail_mae": mean(r["tail_mae"] for r in results),
                    "switches": mean(r["switches"] for r in results),
                    "temporal_duty": mean(r["temporal_duty"] for r in results),
                }
                # Penalize excessive switching and always-temporal behavior.
                row["score"] = (
                    row["tail_mae"]
                    + 0.35 * row["mae"]
                    + 0.0008 * row["switches"]
                    + 0.015 * max(0.0, row["temporal_duty"] - 0.75)
                )
                rows.append(row)

    rows.sort(key=lambda r: r["score"])
    print("rank,enter,exit,dwell,score,tail_mae,mae,switches,temporal_duty")
    for rank, row in enumerate(rows[:15], 1):
        print(
            f'{rank},{row["enter"]:.2f},{row["exit"]:.2f},{row["dwell"]},'
            f'{row["score"]:.6f},{row["tail_mae"]:.6f},{row["mae"]:.6f},'
            f'{row["switches"]:.3f},{row["temporal_duty"]:.4f}'
        )


if __name__ == "__main__":
    main()
