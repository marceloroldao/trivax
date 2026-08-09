from __future__ import annotations

import json
import random
from collections import deque
from math import pi, sin
from statistics import mean

from trivax.delay_aware import DelayAwareController
from trivax.observation_router import ObservationRouter
from trivax.probabilistic import ProbabilisticRegimeController


class SinusoidalPlant:
    def optimum_at(self, t):
        return 0.70 + 0.10 * sin(2.0 * pi * t / 200.0)

    def evaluate(self, action, t):
        target = self.optimum_at(t)
        return 1.0 - 4.0 * (action - target) ** 2


def run(mode, seed=0, steps=1000, delay=4):
    rng = random.Random(seed)
    plant = SinusoidalPlant()
    base = ProbabilisticRegimeController(step_size=0.01)
    router = ObservationRouter(window=5, outlier_z=4.0, innovation_clip=0.05)
    delayed = DelayAwareController(
        ProbabilisticRegimeController(step_size=0.01), sensor_delay=delay
    )
    delay_buffer = deque()

    controllers = {
        "raw": base,
        "routed": ProbabilisticRegimeController(step_size=0.01),
        "delay_aware": delayed,
    }
    actions = {name: controller.action for name, controller in controllers.items()}
    errors = {name: [] for name in controllers}

    for t in range(steps):
        for name, controller in controllers.items():
            action = actions[name]
            observation = plant.evaluate(action, t)

            if mode == "outlier" and rng.random() < 0.02:
                observation += rng.choice((-1.0, 1.0)) * 0.30

            if mode == "delay":
                if name == "delay_aware":
                    # Each controller needs its own causal delayed stream.
                    pass

            errors[name].append(abs(action - plant.optimum_at(t)))

            if name == "routed":
                observation = router.process(observation).output
                result = controller.step(observation)
            elif name == "delay_aware":
                result = controller.step(observation)
            else:
                result = controller.step(observation)
            actions[name] = result[0] if isinstance(result, tuple) else result

    return {name: mean(values) for name, values in errors.items()}


def run_delay(seed=0, steps=1000, delay=4):
    plant = SinusoidalPlant()
    raw = ProbabilisticRegimeController(step_size=0.01)
    wrapped = DelayAwareController(
        ProbabilisticRegimeController(step_size=0.01), sensor_delay=delay
    )
    controllers = {"raw": raw, "delay_aware": wrapped}
    actions = {name: controller.action for name, controller in controllers.items()}
    buffers = {name: deque() for name in controllers}
    errors = {name: [] for name in controllers}

    for t in range(steps):
        for name, controller in controllers.items():
            action = actions[name]
            actual_observation = plant.evaluate(action, t)
            buffer = buffers[name]
            buffer.append(actual_observation)
            if len(buffer) <= delay:
                observed = buffer[0]
            else:
                while len(buffer) > delay + 1:
                    buffer.popleft()
                observed = buffer[0]

            errors[name].append(abs(action - plant.optimum_at(t)))
            result = controller.step(observed)
            actions[name] = result[0] if isinstance(result, tuple) else result

    return {name: mean(values) for name, values in errors.items()}


def benchmark(seeds=range(20)):
    rows = []
    for seed in seeds:
        for mode in ("clean", "outlier"):
            result = run(mode=mode, seed=seed)
            rows.append({"seed": seed, "scenario": mode, **result})
        result = run_delay(seed=seed)
        rows.append({"seed": seed, "scenario": "delay", **result})
    return rows


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
