from __future__ import annotations

import json
import random
from collections import Counter

from trivax.delay_estimator import DelayEstimator


def simulate_fixed(delay, noise_sigma=0.005, steps=500, seed=0):
    rng = random.Random(seed)
    estimator = DelayEstimator(max_delay=10, window=200, min_samples=50, min_abs_correlation=0.2, min_margin=0.03)
    actions = []
    action = 0.5
    estimates = []

    for t in range(steps):
        action = max(0.0, min(1.0, action + rng.choice([-1.0, 1.0]) * rng.uniform(0.005, 0.03)))
        actions.append(action)
        source = actions[max(0, t - delay)]
        observation = 1.2 * source + rng.gauss(0.0, noise_sigma)
        result = estimator.update(action, observation)
        estimates.append(result)

    stable = [e for e in estimates if e.stable]
    accuracy = 0.0 if not stable else sum(e.delay == delay for e in stable) / len(stable)
    return {
        "true_delay": delay,
        "noise_sigma": noise_sigma,
        "stable_fraction": len(stable) / len(estimates),
        "stable_accuracy": accuracy,
        "final_delay": estimates[-1].delay,
        "final_score": estimates[-1].score,
    }


def simulate_variable(seed=0):
    rng = random.Random(seed)
    schedule = [(0, 200, 2), (200, 400, 5), (400, 600, 1)]
    estimator = DelayEstimator(max_delay=10, window=100, min_samples=40, min_abs_correlation=0.2, min_margin=0.03)
    actions = []
    action = 0.5
    records = []

    for t in range(600):
        delay = next(d for start, end, d in schedule if start <= t < end)
        action = max(0.0, min(1.0, action + rng.choice([-1.0, 1.0]) * rng.uniform(0.01, 0.04)))
        actions.append(action)
        source = actions[max(0, t - delay)]
        observation = 1.1 * source + rng.gauss(0.0, 0.005)
        estimate = estimator.update(action, observation)
        records.append((t, delay, estimate.delay, estimate.stable))

    segments = []
    for start, end, delay in schedule:
        region = [r for r in records[start:end] if r[3]]
        counts = Counter(r[2] for r in region)
        modal = counts.most_common(1)[0][0] if counts else None
        segments.append({"start": start, "end": end, "true_delay": delay, "modal_estimate": modal})
    return segments


if __name__ == "__main__":
    rows = []
    for delay in (0, 1, 2, 4, 7):
        for noise in (0.0, 0.005, 0.02):
            rows.append(simulate_fixed(delay, noise_sigma=noise, seed=2026))
    print(json.dumps({"fixed": rows, "variable": simulate_variable(seed=2026)}, indent=2, sort_keys=True))
