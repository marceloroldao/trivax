from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.classic_baselines import PIDController, ExtremumSeekingController, SimpleModelPredictiveController
from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.runtime_v2 import TrivaxRuntimeV2
from trivax.regime_selector import TrivaxRegimeSelector

STEPS = 2400


def target(t: int) -> float:
    if t < 600:
        return 0.50 + 0.025 * math.sin(t / 95.0)
    if t < 1200:
        return 0.58 if (t // 100) % 2 else 0.42
    if t < 1800:
        return 0.50 + 0.07 * math.sin(t / 27.0)
    return 0.53 + 0.035 * math.sin(t / 60.0)


def delay(t: int) -> int:
    if t < 600:
        return 0
    if t < 1200:
        return 4
    if t < 1800:
        return 8
    return 2


def curvature(t: int) -> float:
    if t < 800:
        return 1.2
    if t < 1600:
        return 3.8
    return 2.0


def run(name: str, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    if name == "pid":
        ctrl = PIDController(kp=0.10, ki=0.001, kd=0.03)
    elif name == "extremum_seeking":
        ctrl = ExtremumSeekingController()
    elif name == "simple_mpc":
        ctrl = SimpleModelPredictiveController()
    elif name == "adaptive":
        ctrl = AdaptiveHillClimber()
    elif name == "temporal":
        ctrl = TrivaxRuntimeV2()
    elif name == "selector":
        ctrl = TrivaxRegimeSelector()
    else:
        raise ValueError(name)

    action = float(ctrl.action)
    hist: deque[float] = deque([action] * 32, maxlen=64)
    errors: list[float] = []
    effort: list[float] = []
    previous = action

    for t in range(STEPS):
        d = delay(t)
        applied = list(hist)[-(d + 1)]
        optimum = target(t)
        y = 1.0 - curvature(t) * (applied - optimum) ** 2
        sigma = 0.0015 if t < 1600 else 0.0045
        y += rng.gauss(0.0, sigma)
        if t in (810, 1460, 2110):
            y += rng.choice((-0.08, 0.08))

        if name == "pid":
            action, _ = ctrl.step(y, setpoint=1.0)
        else:
            action, _ = ctrl.step(y)
        action = float(action)
        hist.append(action)
        errors.append(abs(action - optimum))
        effort.append(abs(action - previous))
        previous = action

    return {
        "mae": mean(errors),
        "tail": mean(errors[-400:]),
        "effort": mean(effort),
        "worst": max(errors),
    }


def main() -> None:
    names = ("selector", "temporal", "adaptive", "pid", "extremum_seeking", "simple_mpc")
    seeds = range(40)
    print("controller,mae,tail_mae,effort,max_error")
    for name in names:
        rows = [run(name, s) for s in seeds]
        print(
            f"{name},"
            f"{mean(r['mae'] for r in rows):.8f},"
            f"{mean(r['tail'] for r in rows):.8f},"
            f"{mean(r['effort'] for r in rows):.8f},"
            f"{mean(r['worst'] for r in rows):.8f}"
        )


if __name__ == "__main__":
    main()
