"""Information-matched blind benchmark for TRIVAX v0.2.

Primary peer group: all controllers receive only the scalar objective value and
their own internal action history. Hyperparameters are tuned on calibration
scenarios, frozen, and then evaluated on a separate holdout distribution.
"""
from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.baselines import PerturbAndObserve
from trivax.classic_baselines import ExtremumSeekingController
from trivax.regime_selector import TrivaxRegimeSelector
from trivax.runtime_v2 import TrivaxRuntimeV2

CAL_STEPS = 1500
HOLDOUT_STEPS = 2200


def cal_target(t: int, variant: int) -> float:
    if variant == 0:
        return 0.50 + 0.06 * math.sin(t / 65.0)
    if variant == 1:
        return 0.50 + (0.07 if (t // 140) % 2 else -0.07)
    return 0.53 + 0.045 * math.sin(t / 31.0) + 0.018 * math.sin(t / 11.0)


def cal_delay(t: int, variant: int) -> int:
    schedules = ((0, 3, 6), (2, 5, 1), (1, 7, 4))
    return schedules[variant][min(2, t // 500)]


def holdout_target(t: int) -> float:
    period = 290
    phase = (t % period) / period
    tri = 4.0 * abs(phase - 0.5) - 1.0
    return max(0.15, min(0.85, 0.55 + 0.082 * tri + 0.000018 * t))


def holdout_delay(t: int) -> int:
    if t < 430:
        return 4
    if t < 860:
        return 9
    if t < 1310:
        return 1
    if t < 1760:
        return 7
    return 3


def make_controller(kind: str, cfg: tuple[float, ...]):
    if kind == "pno":
        return PerturbAndObserve(step_size=cfg[0])
    if kind == "adaptive":
        return AdaptiveHillClimber(step_size=cfg[0], grow=cfg[1], shrink=cfg[2])
    if kind == "es":
        return ExtremumSeekingController(amplitude=cfg[0], gain=cfg[1], omega=cfg[2])
    if kind == "selector":
        return TrivaxRegimeSelector(enter_threshold=cfg[0], exit_threshold=cfg[1], min_dwell=int(cfg[2]))
    if kind == "temporal":
        return TrivaxRuntimeV2()
    raise ValueError(kind)


def step_controller(ctrl, kind: str, y: float) -> float:
    result = ctrl.step(y)
    if kind == "pno":
        return float(result)
    return float(result[0])


def run_cal(kind: str, cfg: tuple[float, ...], seed: int, variant: int) -> dict[str, float]:
    rng = random.Random(100003 * seed + 7919 * variant)
    ctrl = make_controller(kind, cfg)
    action = float(ctrl.action)
    hist: deque[float] = deque([action] * 24, maxlen=48)
    errors: list[float] = []
    effort: list[float] = []
    prev = action

    for t in range(CAL_STEPS):
        d = cal_delay(t, variant)
        applied = list(hist)[-(d + 1)]
        optimum = cal_target(t, variant)
        curvature = (1.3, 2.6, 4.2)[variant]
        y = 1.0 - curvature * (applied - optimum) ** 2
        y += rng.gauss(0.0, (0.0015, 0.0025, 0.0035)[variant])
        if variant and t in (470, 980, 1330):
            y += rng.choice((-0.08, 0.08))
        action = step_controller(ctrl, kind, y)
        hist.append(action)
        errors.append(abs(action - optimum))
        effort.append(abs(action - prev))
        prev = action

    return {"mae": mean(errors), "tail": mean(errors[-300:]), "effort": mean(effort)}


def calibration_score(kind: str, cfg: tuple[float, ...]) -> float:
    rows = [run_cal(kind, cfg, seed, variant) for variant in range(3) for seed in range(6)]
    return (
        mean(r["tail"] for r in rows)
        + 0.35 * mean(r["mae"] for r in rows)
        + 0.08 * mean(r["effort"] for r in rows)
    )


def grids() -> dict[str, list[tuple[float, ...]]]:
    return {
        "pno": [(s,) for s in (0.005, 0.01, 0.02, 0.04)],
        "adaptive": [
            (s, g, sh)
            for s in (0.005, 0.01, 0.02)
            for g in (1.04, 1.08)
            for sh in (0.45, 0.55, 0.70)
        ],
        "es": [
            (amp, gain, omega)
            for amp in (0.005, 0.015, 0.03)
            for gain in (0.005, 0.018, 0.04)
            for omega in (0.20, 0.37, 0.55)
        ],
        "selector": [
            (enter, exit_, float(dwell))
            for enter in (0.58, 0.64, 0.70)
            for exit_ in (0.34, 0.42, 0.50)
            if exit_ < enter
            for dwell in (16, 24, 40)
        ],
        "temporal": [tuple()],
    }


def select_configs() -> dict[str, tuple[float, ...]]:
    selected: dict[str, tuple[float, ...]] = {}
    for kind, options in grids().items():
        if kind == "temporal":
            selected[kind] = tuple()
            continue
        ranked = sorted((calibration_score(kind, cfg), cfg) for cfg in options)
        selected[kind] = ranked[0][1]
    return selected


def run_holdout(kind: str, cfg: tuple[float, ...], seed: int) -> dict[str, float]:
    rng = random.Random(900001 + seed)
    ctrl = make_controller(kind, cfg)
    action = float(ctrl.action)
    hist: deque[float] = deque([action] * 32, maxlen=64)
    errors: list[float] = []
    effort: list[float] = []
    prev = action

    for t in range(HOLDOUT_STEPS):
        d = holdout_delay(t)
        applied = list(hist)[-(d + 1)]
        optimum = holdout_target(t)
        curvature = 1.7 if t < 700 else (5.0 if t < 1500 else 2.8)
        y = 1.0 - curvature * (applied - optimum) ** 2
        sigma = 0.002 if t < 850 else (0.004 if t < 1650 else 0.007)
        y += rng.gauss(0.0, sigma)
        if rng.random() < 0.006:
            y += rng.choice((-1.0, 1.0)) * rng.expovariate(14.0)
        action = step_controller(ctrl, kind, y)
        hist.append(action)
        errors.append(abs(action - optimum))
        effort.append(abs(action - prev))
        prev = action

    return {
        "mae": mean(errors),
        "tail": mean(errors[-500:]),
        "effort": mean(effort),
        "max_error": max(errors),
    }


def main() -> None:
    selected = select_configs()
    print("selected_controller,config")
    for kind, cfg in selected.items():
        print(f"{kind},{cfg}")
    print()

    print("controller,mae,tail_mae,effort,max_error,win_rate")
    seeds = range(40)
    per_kind: dict[str, list[dict[str, float]]] = {
        kind: [run_holdout(kind, cfg, seed) for seed in seeds]
        for kind, cfg in selected.items()
    }

    winners = []
    for i, _seed in enumerate(seeds):
        winner = min(per_kind, key=lambda k: per_kind[k][i]["mae"])
        winners.append(winner)

    for kind, rows in per_kind.items():
        print(
            f"{kind},"
            f"{mean(r['mae'] for r in rows):.8f},"
            f"{mean(r['tail'] for r in rows):.8f},"
            f"{mean(r['effort'] for r in rows):.8f},"
            f"{mean(r['max_error'] for r in rows):.8f},"
            f"{mean(float(w == kind) for w in winners):.6f}"
        )


if __name__ == "__main__":
    main()
