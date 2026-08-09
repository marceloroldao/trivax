from __future__ import annotations

import math
import random
import time
from collections import deque
from statistics import mean

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.baselines import PerturbAndObserve
from trivax.core_rc import TrivaxCoreRC
from trivax.runtime_v2 import TrivaxRuntimeV2
from trivax.runtime_v5 import TrivaxRuntimeV5


def target(t: int) -> float:
    phase = (t % 600) / 600.0
    triangle = 4.0 * abs(phase - 0.5) - 1.0
    return 0.52 + 0.07 * triangle + 0.015 * math.sin(t / 41.0)


def true_delay(t: int) -> int:
    if t < 700:
        return 3
    if t < 1400:
        return 9
    if t < 2100:
        return 0
    return 6


def run_controller(factory, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    controller = factory()
    action = float(getattr(controller, "action", 0.5))
    history = deque([action] * 16, maxlen=16)
    errors = []
    effort = []
    validator_runs = 0
    probes = 0
    previous = action
    start = time.perf_counter()

    for t in range(2800):
        d = true_delay(t)
        delayed_action = list(history)[-(d + 1)]
        optimum = target(t)
        curvature = 0.8 if t < 900 else (2.2 if t < 1900 else 1.3)
        sigma = 0.0015 + 0.0025 * (0.5 + 0.5 * math.sin(t / 87.0))
        observation = 1.0 - curvature * (delayed_action - optimum) ** 2
        observation += rng.gauss(0.0, sigma)
        if rng.random() < 0.003:
            observation += rng.choice((-0.16, 0.16))

        result = controller.step(observation)
        if isinstance(result, tuple):
            action, state = result
            validator_runs += int(bool(getattr(state, "validator_ran", False)))
            probe_state = getattr(state, "probe_state", None)
            probes += int(bool(probe_state is not None and getattr(probe_state, "applied", False)))
        else:
            action = result

        action = float(action)
        history.append(action)
        errors.append(abs(action - optimum))
        effort.append(abs(action - previous))
        previous = action

    elapsed = time.perf_counter() - start
    return {
        "mae": mean(errors),
        "tail_mae": mean(errors[-700:]),
        "effort": mean(effort),
        "runtime_s": elapsed,
        "validator_fraction": validator_runs / 2800.0,
        "probes": float(probes),
    }


def main() -> None:
    factories = {
        "core_rc": lambda: TrivaxCoreRC(validator_interval=16, enable_probes=False),
        "runtime_v2": TrivaxRuntimeV2,
        "runtime_v5": TrivaxRuntimeV5,
        "adaptive_hill": AdaptiveHillClimber,
        "po": PerturbAndObserve,
    }
    seeds = range(40)

    print("controller,mae,tail_mae,effort,runtime_s,validator_fraction,probes")
    for name, factory in factories.items():
        rows = [run_controller(factory, seed) for seed in seeds]
        print(
            f"{name},"
            f"{mean(r['mae'] for r in rows):.8f},"
            f"{mean(r['tail_mae'] for r in rows):.8f},"
            f"{mean(r['effort'] for r in rows):.8f},"
            f"{mean(r['runtime_s'] for r in rows):.8f},"
            f"{mean(r['validator_fraction'] for r in rows):.6f},"
            f"{mean(r['probes'] for r in rows):.3f}"
        )


if __name__ == "__main__":
    main()
