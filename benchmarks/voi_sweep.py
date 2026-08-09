"""Parameter sweep for TRIVAX V5 value-of-information probing.

This benchmark is intentionally dependency-free. It explores how information
weight, control-cost weight and decision threshold trade identification effort
against control perturbation. It is a screening experiment, not a claim of
optimality.
"""
from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.runtime_v5 import TrivaxRuntimeV5
from trivax.value_of_information import ValueOfInformationProbePolicy


def target(t: int) -> float:
    return 0.50 + 0.06 * math.sin(t / 110.0)


def true_delay(t: int) -> int:
    if t < 700:
        return 4
    if t < 1400:
        return 7
    return 2


def run(seed: int, info_weight: float, cost_weight: float, min_value: float) -> dict[str, float]:
    rng = random.Random(seed)
    policy = ValueOfInformationProbePolicy(
        amplitude=0.005,
        trigger_steps=24,
        min_interval=32,
        max_probes=32,
        information_weight=info_weight,
        cost_weight=cost_weight,
        min_value=min_value,
    )
    runtime = TrivaxRuntimeV5(probe_policy=policy)
    history: deque[float] = deque([runtime.action] * 16, maxlen=16)
    errors: list[float] = []
    effort: list[float] = []
    delay_hits: list[float] = []
    probes = 0
    previous_action = runtime.action

    for t in range(2100):
        d = true_delay(t)
        delayed_action = list(history)[-(d + 1)]
        optimum = target(t)
        curvature = 1.0 if t < 1050 else 1.7
        observation = 1.0 - curvature * (delayed_action - optimum) ** 2
        observation += rng.gauss(0.0, 0.002)
        if t in (420, 1080, 1730):
            observation += rng.choice((-0.12, 0.12))

        action, state = runtime.step(observation)
        history.append(action)
        errors.append(abs(action - optimum))
        effort.append(abs(action - previous_action))
        previous_action = action
        probes += int(state.probe_state.applied)
        delay_hits.append(float(state.accepted_delay == d))

    return {
        "mae": mean(errors),
        "tail_mae": mean(errors[-500:]),
        "effort": mean(effort),
        "delay_accuracy": mean(delay_hits[300:]),
        "probes": float(probes),
    }


def main() -> None:
    info_weights = (0.5, 1.0, 2.0)
    cost_weights = (5.0, 20.0, 80.0)
    min_values = (0.02, 0.05, 0.10)
    seeds = range(10)

    rows = []
    for iw in info_weights:
        for cw in cost_weights:
            for mv in min_values:
                results = [run(s, iw, cw, mv) for s in seeds]
                row = {
                    "info_weight": iw,
                    "cost_weight": cw,
                    "min_value": mv,
                    "mae": mean(r["mae"] for r in results),
                    "tail_mae": mean(r["tail_mae"] for r in results),
                    "effort": mean(r["effort"] for r in results),
                    "delay_accuracy": mean(r["delay_accuracy"] for r in results),
                    "probes": mean(r["probes"] for r in results),
                }
                # Screening score: control quality first, then identification,
                # then explicit penalties for effort and active probes.
                row["score"] = (
                    row["tail_mae"]
                    + 0.35 * row["mae"]
                    + 0.20 * (1.0 - row["delay_accuracy"])
                    + 2.0 * row["effort"]
                    + 0.0005 * row["probes"]
                )
                rows.append(row)

    rows.sort(key=lambda r: r["score"])
    print("rank iw cw minV score tailMAE MAE delayAcc effort probes")
    for rank, row in enumerate(rows[:12], 1):
        print(
            rank,
            row["info_weight"],
            row["cost_weight"],
            row["min_value"],
            f'{row["score"]:.6f}',
            f'{row["tail_mae"]:.6f}',
            f'{row["mae"]:.6f}',
            f'{row["delay_accuracy"]:.4f}',
            f'{row["effort"]:.6f}',
            f'{row["probes"]:.2f}',
        )


if __name__ == "__main__":
    main()
