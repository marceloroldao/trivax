"""TRIVAX v0.2 selector benchmark with oracle regret and switching cost.

The offline oracle chooses the lower-error isolated controller independently
for each pre-defined evaluation segment. The online selector never receives
segment labels or true delay.
"""
from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.regime_selector import TrivaxRegimeSelector, RegimeMode
from trivax.runtime_v2 import TrivaxRuntimeV2

STEPS = 2400
SEGMENTS = ((0, 600), (600, 1200), (1200, 1800), (1800, 2400))


def target(t: int) -> float:
    if t < 600:
        return 0.48 + 0.018 * math.sin(t / 85.0)
    if t < 1200:
        return 0.58 if (t // 90) % 2 else 0.40
    if t < 1800:
        return 0.50 + 0.065 * math.sin(t / 25.0)
    return 0.52 + 0.03 * math.sin(t / 55.0)


def delay(t: int) -> int:
    if t < 600:
        return 0
    if t < 1200:
        return 5
    if t < 1800:
        return 2
    return 8


def noise_sigma(t: int) -> float:
    if t < 1800:
        return 0.0015
    return 0.0055


def observation(action_history: deque[float], t: int, rng: random.Random) -> float:
    d = delay(t)
    delayed_action = list(action_history)[-(d + 1)]
    optimum = target(t)
    curvature = 1.1 if t < 1200 else 1.6
    y = 1.0 - curvature * (delayed_action - optimum) ** 2
    y += rng.gauss(0.0, noise_sigma(t))
    if t in (875, 1540, 2110):
        y += rng.choice((-0.10, 0.10))
    return y


def run_controller(kind: str, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    if kind == "adaptive":
        ctrl = AdaptiveHillClimber()
    elif kind == "temporal":
        ctrl = TrivaxRuntimeV2()
    elif kind == "selector":
        ctrl = TrivaxRegimeSelector()
    else:
        raise ValueError(kind)

    action = float(ctrl.action)
    history: deque[float] = deque([action] * 16, maxlen=16)
    errors: list[float] = []
    effort: list[float] = []
    modes: list[str] = []
    switch_count = 0
    temporal_duty = 0.0
    previous_action = action

    for t in range(STEPS):
        y = observation(history, t, rng)
        action, state = ctrl.step(y)
        history.append(float(action))
        errors.append(abs(float(action) - target(t)))
        effort.append(abs(float(action) - previous_action))
        previous_action = float(action)

        if kind == "selector":
            modes.append(str(state.mode.value))
            switch_count = int(state.switch_count)
            temporal_duty = float(state.temporal_duty_fraction)

    segment_mae = [mean(errors[a:b]) for a, b in SEGMENTS]
    return {
        "mae": mean(errors),
        "tail_mae": mean(errors[-400:]),
        "effort": mean(effort),
        "segment_mae": segment_mae,
        "switch_count": float(switch_count),
        "temporal_duty": float(temporal_duty),
    }


def main() -> None:
    seeds = range(40)
    rows = []
    for seed in seeds:
        adaptive = run_controller("adaptive", seed)
        temporal = run_controller("temporal", seed)
        selector = run_controller("selector", seed)

        oracle_segment = [
            min(adaptive["segment_mae"][i], temporal["segment_mae"][i])
            for i in range(len(SEGMENTS))
        ]
        oracle_mae = mean(oracle_segment)
        selector_regret = selector["mae"] - oracle_mae
        rows.append({
            "adaptive_mae": adaptive["mae"],
            "temporal_mae": temporal["mae"],
            "selector_mae": selector["mae"],
            "oracle_mae": oracle_mae,
            "selector_regret": selector_regret,
            "selector_tail": selector["tail_mae"],
            "selector_effort": selector["effort"],
            "switch_count": selector["switch_count"],
            "temporal_duty": selector["temporal_duty"],
        })

    print("metric,value")
    for key in (
        "adaptive_mae",
        "temporal_mae",
        "selector_mae",
        "oracle_mae",
        "selector_regret",
        "selector_tail",
        "selector_effort",
        "switch_count",
        "temporal_duty",
    ):
        print(f"{key},{mean(row[key] for row in rows):.8f}")

    better_than_adaptive = mean(float(r["selector_mae"] < r["adaptive_mae"]) for r in rows)
    better_than_temporal = mean(float(r["selector_mae"] < r["temporal_mae"]) for r in rows)
    print(f"selector_win_rate_vs_adaptive,{better_than_adaptive:.8f}")
    print(f"selector_win_rate_vs_temporal,{better_than_temporal:.8f}")


if __name__ == "__main__":
    main()
