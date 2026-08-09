from __future__ import annotations

import math
import random
from statistics import mean

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.regime_selector import RegimeMode, TrivaxRegimeSelector
from trivax.runtime_v2 import TrivaxRuntimeV2


def target(t: int) -> float:
    if t < 700:
        return 0.45 + 0.04 * math.sin(t / 70.0)
    if t < 1400:
        return 0.62 if (t // 90) % 2 == 0 else 0.38
    return 0.50 + 0.10 * math.sin(t / 28.0)


def delay(t: int) -> int:
    if t < 500:
        return 0
    if t < 1050:
        return 5
    if t < 1550:
        return 2
    return 8


def run_controller(kind: str, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    if kind == "selector":
        controller = TrivaxRegimeSelector()
    elif kind == "adaptive":
        controller = AdaptiveHillClimber()
    else:
        controller = TrivaxRuntimeV2()

    action = 0.5
    history = [action] * 16
    errors = []
    temporal_steps = 0
    switches = 0

    for t in range(2200):
        d = delay(t)
        delayed = history[-(d + 1)]
        optimum = target(t)
        noise_sigma = 0.001 if t < 1500 else 0.003
        observation = 1.0 - 1.3 * (delayed - optimum) ** 2 + rng.gauss(0.0, noise_sigma)
        if t in (820, 1210, 1760):
            observation += rng.choice((-0.08, 0.08))

        if kind == "selector":
            action, state = controller.step(observation)
            temporal_steps += int(state.mode is RegimeMode.TEMPORAL)
            switches += int(state.switched)
        elif kind == "adaptive":
            action, _ = controller.step(observation)
        else:
            action, _ = controller.step(observation)

        history.append(action)
        errors.append(abs(action - optimum))

    return {
        "mae": mean(errors),
        "tail_mae": mean(errors[-500:]),
        "temporal_fraction": temporal_steps / 2200.0,
        "switches": float(switches),
    }


def main() -> None:
    seeds = range(40)
    print("controller,mae,tail_mae,temporal_fraction,switches")
    for kind in ("selector", "adaptive", "temporal"):
        rows = [run_controller(kind, s) for s in seeds]
        print(
            f"{kind},{mean(r['mae'] for r in rows):.8f},"
            f"{mean(r['tail_mae'] for r in rows):.8f},"
            f"{mean(r['temporal_fraction'] for r in rows):.6f},"
            f"{mean(r['switches'] for r in rows):.3f}"
        )


if __name__ == "__main__":
    main()
