"""Blind calibration-to-holdout evaluation for TRIVAX v0.2.

The selector configuration is chosen exclusively on calibration scenarios.
Only after the best calibration configuration is frozen is it evaluated on a
separate holdout distribution. This script is intended to detect tuning
fragility and should not be used to tune against the holdout result.
"""
from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.regime_selector import TrivaxRegimeSelector
from trivax.runtime_v2 import TrivaxRuntimeV2

ENTERS = (0.58, 0.64, 0.70)
EXITS = (0.34, 0.42, 0.50)
DWELLS = (16, 24, 40)
DEFAULT = (0.64, 0.42, 24)


def calibration_target(t: int, variant: int) -> float:
    if variant == 0:
        return 0.50 + 0.08 * math.sin(t / 70.0)
    if variant == 1:
        return 0.52 + 0.05 * math.sin(t / 35.0) + 0.015 * math.sin(t / 9.0)
    return 0.48 + (0.06 if (t // 180) % 2 else -0.06)


def calibration_delay(t: int, variant: int) -> int:
    schedules = ((0, 3, 6, 2), (2, 5, 1, 7), (1, 4, 8, 3))
    return schedules[variant][min(3, t // 500)]


def run_calibration(seed: int, variant: int, config: tuple[float, float, int]) -> dict[str, float]:
    enter, exit_, dwell = config
    rng = random.Random(seed * 1009 + variant * 9176)
    ctrl = TrivaxRegimeSelector(enter_threshold=enter, exit_threshold=exit_, min_dwell=dwell)
    history: deque[float] = deque([ctrl.action] * 20, maxlen=20)
    errors: list[float] = []
    last = None

    for t in range(2000):
        d = calibration_delay(t, variant)
        applied = list(history)[-(d + 1)]
        optimum = calibration_target(t, variant)
        curvature = 1.0 + 0.35 * (variant == 1) + 0.55 * (t >= 1100 and variant == 2)
        y = 1.0 - curvature * (applied - optimum) ** 2
        y += rng.gauss(0.0, (0.0015, 0.0025, 0.0035)[variant])
        if t in (430, 910, 1470) and variant != 0:
            y += rng.choice((-0.08, 0.08))
        action, last = ctrl.step(y)
        history.append(action)
        errors.append(abs(action - optimum))

    assert last is not None
    return {
        "mae": mean(errors),
        "tail": mean(errors[-400:]),
        "switches": float(last.switch_count),
        "duty": float(last.temporal_duty_fraction),
    }


def calibration_score(config: tuple[float, float, int]) -> float:
    rows = [run_calibration(s, v, config) for v in range(3) for s in range(8)]
    mae = mean(r["mae"] for r in rows)
    tail = mean(r["tail"] for r in rows)
    switches = mean(r["switches"] for r in rows)
    duty = mean(r["duty"] for r in rows)
    return tail + 0.35 * mae + 0.0008 * switches + 0.015 * max(0.0, duty - 0.75)


def holdout_target(t: int) -> float:
    period = 310
    phase = (t % period) / period
    triangle = 4.0 * abs(phase - 0.5) - 1.0
    drift = 0.00002 * t
    return max(0.15, min(0.85, 0.54 + 0.085 * triangle + drift))


def holdout_delay(t: int) -> int:
    if t < 430:
        return 4
    if t < 870:
        return 10
    if t < 1330:
        return 1
    if t < 1760:
        return 7
    return 3


def holdout_curvature(t: int) -> float:
    if t < 700:
        return 1.8
    if t < 1450:
        return 5.4
    return 2.7


def run_holdout(kind: str, seed: int, config: tuple[float, float, int] | None = None) -> dict[str, float]:
    rng = random.Random(700001 + seed)
    if kind == "selector":
        assert config is not None
        enter, exit_, dwell = config
        ctrl = TrivaxRegimeSelector(enter_threshold=enter, exit_threshold=exit_, min_dwell=dwell)
    elif kind == "adaptive":
        ctrl = AdaptiveHillClimber()
    elif kind == "temporal":
        ctrl = TrivaxRuntimeV2()
    else:
        raise ValueError(kind)

    action = float(ctrl.action)
    history: deque[float] = deque([action] * 32, maxlen=64)
    errors: list[float] = []
    effort: list[float] = []
    previous = action
    last = None

    for t in range(2200):
        d = holdout_delay(t)
        applied = list(history)[-(d + 1)]
        optimum = holdout_target(t)
        y = 1.0 - holdout_curvature(t) * (applied - optimum) ** 2
        sigma = 0.002 if t < 900 else (0.0045 if t < 1700 else 0.007)
        y += rng.gauss(0.0, sigma)
        if rng.random() < 0.006:
            y += rng.choice((-1.0, 1.0)) * rng.expovariate(15.0)

        action, last = ctrl.step(y)
        action = float(action)
        history.append(action)
        errors.append(abs(action - optimum))
        effort.append(abs(action - previous))
        previous = action

    duty = float(getattr(last, "temporal_duty_fraction", 0.0)) if last is not None else 0.0
    switches = float(getattr(last, "switch_count", 0.0)) if last is not None else 0.0
    return {
        "mae": mean(errors),
        "tail": mean(errors[-500:]),
        "effort": mean(effort),
        "duty": duty,
        "switches": switches,
    }


def main() -> None:
    configs = [
        (enter, exit_, dwell)
        for enter in ENTERS
        for exit_ in EXITS
        if exit_ < enter
        for dwell in DWELLS
    ]
    ranked = sorted((calibration_score(c), c) for c in configs)
    best_score, selected = ranked[0]

    print("selection_metric,value")
    print(f"calibration_score,{best_score:.8f}")
    print(f"selected_enter,{selected[0]:.2f}")
    print(f"selected_exit,{selected[1]:.2f}")
    print(f"selected_dwell,{selected[2]}")
    print()

    variants = {
        "selector_calibrated": ("selector", selected),
        "selector_default": ("selector", DEFAULT),
        "always_adaptive": ("adaptive", None),
        "always_temporal": ("temporal", None),
    }
    seeds = range(40)
    print("controller,mae,tail_mae,effort,temporal_duty,switches")
    for name, (kind, config) in variants.items():
        rows = [run_holdout(kind, seed, config) for seed in seeds]
        print(
            f"{name},"
            f"{mean(r['mae'] for r in rows):.8f},"
            f"{mean(r['tail'] for r in rows):.8f},"
            f"{mean(r['effort'] for r in rows):.8f},"
            f"{mean(r['duty'] for r in rows):.6f},"
            f"{mean(r['switches'] for r in rows):.3f}"
        )


if __name__ == "__main__":
    main()
