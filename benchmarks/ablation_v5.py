"""Publication-oriented ablation for TRIVAX V5.

The experiment isolates the incremental contribution of temporal credit,
causal-delay confidence, and value-of-information probing on a delayed,
nonstationary scalar control problem. No parameters are tuned inside this file.
"""
from __future__ import annotations

import json
import math
import random
from collections import deque
from statistics import mean, median

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.baselines import PerturbAndObserve
from trivax.runtime_v2 import TrivaxRuntimeV2
from trivax.runtime_v3 import TrivaxRuntimeV3
from trivax.runtime_v5 import TrivaxRuntimeV5
from trivax.value_of_information import ValueOfInformationProbePolicy


def optimum(t: int) -> float:
    base = 0.50 + 0.09 * math.sin(2.0 * math.pi * t / 260.0)
    if t >= 900:
        base -= 0.07
    return max(0.12, min(0.88, base))


def delay_at(t: int) -> int:
    if t < 500:
        return 2
    if t < 1000:
        return 6
    return 1


def run(controller, seed: int, steps: int = 1500) -> dict[str, float]:
    rng = random.Random(seed)
    history = deque([float(controller.action)] * 24, maxlen=64)
    action = float(controller.action)
    previous_action = action
    errors: list[float] = []
    effort: list[float] = []
    delay_hits: list[float] = []
    probes = 0

    for t in range(steps):
        history.append(action)
        d = delay_at(t)
        delayed_action = list(history)[-(d + 1)]
        target = optimum(t)
        curvature = 3.0 if t < 900 else 5.5
        observation = 1.0 - curvature * (delayed_action - target) ** 2
        observation += rng.gauss(0.0, 0.006)
        if rng.random() < 0.006:
            observation += rng.choice((-1.0, 1.0)) * rng.uniform(0.10, 0.22)

        result = controller.step(observation)
        if isinstance(result, tuple):
            action = float(result[0])
            state = result[1]
            accepted = getattr(state, "accepted_delay", None)
            if accepted is not None:
                delay_hits.append(float(accepted == d))
            voi = getattr(state, "voi_state", None)
            if voi is not None:
                probes += int(voi.applied)
        else:
            action = float(result)

        errors.append(abs(action - target))
        effort.append(abs(action - previous_action))
        previous_action = action

    return {
        "mae": mean(errors),
        "median_ae": median(errors),
        "tail_mae": mean(errors[-400:]),
        "effort": mean(effort),
        "delay_accuracy": mean(delay_hits) if delay_hits else 0.0,
        "probes": float(probes),
    }


def make_controllers():
    return {
        "trivax_v5_full": TrivaxRuntimeV5(),
        "trivax_v5_no_probe": TrivaxRuntimeV5(
            probe_policy=ValueOfInformationProbePolicy(max_probes=0)
        ),
        "trivax_v3_causal_credit": TrivaxRuntimeV3(),
        "trivax_v2_temporal_credit": TrivaxRuntimeV2(),
        "adaptive_hill_climber": AdaptiveHillClimber(step_size=0.01),
        "perturb_and_observe": PerturbAndObserve(step_size=0.01),
    }


def benchmark(seeds=range(30)):
    rows = []
    for seed in seeds:
        for name, controller in make_controllers().items():
            row = {"seed": seed, "controller": name}
            row.update(run(controller, seed))
            rows.append(row)
    return rows


def summarize(rows):
    names = sorted({row["controller"] for row in rows})
    summary = []
    for name in names:
        group = [row for row in rows if row["controller"] == name]
        summary.append({
            "controller": name,
            "mae": mean(row["mae"] for row in group),
            "tail_mae": mean(row["tail_mae"] for row in group),
            "effort": mean(row["effort"] for row in group),
            "delay_accuracy": mean(row["delay_accuracy"] for row in group),
            "probes": mean(row["probes"] for row in group),
        })
    summary.sort(key=lambda row: row["tail_mae"])
    return summary


if __name__ == "__main__":
    rows = benchmark()
    print(json.dumps({"summary": summarize(rows), "runs": rows}, indent=2, sort_keys=True))
