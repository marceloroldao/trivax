from __future__ import annotations

import math
import random
from collections import deque
from statistics import mean

from trivax.adaptive_baseline import AdaptiveHillClimber
from trivax.baselines import PerturbAndObserve
from trivax.runtime_v2 import TrivaxRuntimeV2
from trivax.runtime_v5 import TrivaxRuntimeV5
from trivax.runtime_v6 import RuntimeMode, TrivaxRuntimeV6


def target(t: int) -> float:
    phase = (t % 300) / 300.0
    triangle = 4.0 * abs(phase - 0.5) - 1.0
    return 0.52 + 0.07 * triangle + 0.02 * math.sin(t / 41.0)


def delay(t: int) -> int:
    if t < 700:
        return 3
    if t < 1400:
        return 9
    if t < 2100:
        return 0
    return 6


def run_controller(name: str, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    if name == "v6":
        controller = TrivaxRuntimeV6()
        action = controller.action
    elif name == "v5":
        controller = TrivaxRuntimeV5()
        action = controller.action
    elif name == "v2":
        controller = TrivaxRuntimeV2()
        action = controller.action
    elif name == "adaptive":
        controller = AdaptiveHillClimber(initial_action=0.5, initial_step=0.01)
        action = controller.action
    else:
        controller = PerturbAndObserve(initial_action=0.5, step_size=0.01)
        action = controller.action

    hist = deque([action] * 20, maxlen=20)
    errs=[]; efforts=[]; probes=0; verify=0; prev=action
    for t in range(2800):
        d=delay(t)
        delayed=list(hist)[-(d+1)]
        opt=target(t)
        curvature=0.7 if t < 900 else (2.1 if t < 1900 else 1.2)
        sigma=0.0015 + 0.0025 * abs(math.sin(t/137.0))
        obs=1.0-curvature*(delayed-opt)**2+rng.gauss(0.0,sigma)
        if rng.random() < 0.003:
            obs += rng.choice((-0.15,0.15))

        if name in ("v6","v5","v2"):
            action,state=controller.step(obs)
            if name == "v6":
                probes += int(state.voi_state.applied)
                verify += int(state.mode is RuntimeMode.VERIFY)
            elif name == "v5":
                probes += int(state.voi_state.applied)
        else:
            action=controller.step(obs)

        hist.append(action)
        errs.append(abs(action-opt))
        efforts.append(abs(action-prev))
        prev=action

    return {
        "mae":mean(errs),
        "tail_mae":mean(errs[-700:]),
        "effort":mean(efforts),
        "probes":float(probes),
        "verify_fraction":verify/2800.0,
    }


def main() -> None:
    names=("v6","v5","v2","adaptive","po")
    seeds=range(40)
    print("controller,mae,tail_mae,effort,probes,verify_fraction")
    for name in names:
        rows=[run_controller(name,s) for s in seeds]
        print(
            f"{name},{mean(r['mae'] for r in rows):.6f},"
            f"{mean(r['tail_mae'] for r in rows):.6f},"
            f"{mean(r['effort'] for r in rows):.6f},"
            f"{mean(r['probes'] for r in rows):.3f},"
            f"{mean(r['verify_fraction'] for r in rows):.4f}"
        )


if __name__ == "__main__":
    main()
