from __future__ import annotations

from statistics import mean, median
from time import perf_counter

from trivax import ScalarPlant, TrivaxController, run_closed_loop


def main() -> None:
    controller = TrivaxController(initial_action=0.25, step_size=0.03)
    plant = ScalarPlant(optimum=0.70, drift_amplitude=0.10, drift_period=200.0)

    start = perf_counter()
    records = run_closed_loop(controller, plant, steps=1000)
    elapsed = perf_counter() - start

    errors = [row["abs_error"] for row in records]
    effort = sum(
        abs(records[i]["action"] - records[i - 1]["action"])
        for i in range(1, len(records))
    )

    print("TRIVAX scalar tracking benchmark")
    print(f"steps={len(records)}")
    print(f"mean_abs_error={mean(errors):.8f}")
    print(f"median_abs_error={median(errors):.8f}")
    print(f"cumulative_abs_error={sum(errors):.8f}")
    print(f"control_effort={effort:.8f}")
    print(f"runtime_seconds={elapsed:.8f}")
    print(f"final_coherence={records[-1]['coherence']:.8f}")


if __name__ == "__main__":
    main()
