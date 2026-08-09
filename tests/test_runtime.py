from trivax.delay_estimator import DelayEstimator
from trivax.observation_router import ObservationRouter
from trivax.probabilistic import ProbabilisticRegimeController
from trivax.runtime import TrivaxRuntime


def test_runtime_stays_bounded():
    runtime = TrivaxRuntime(
        controller=ProbabilisticRegimeController(step_size=0.02),
        router=ObservationRouter(),
        delay_estimator=DelayEstimator(max_delay=4, min_samples=8, window=32),
    )
    for i in range(200):
        action, state = runtime.step(1.0 - 0.001 * (i % 7))
        assert 0.0 <= action <= 1.0
        assert state.hold_period >= 1


def test_runtime_requires_confirmed_delay_before_changing_hold_period():
    runtime = TrivaxRuntime(delay_confirmation=3)
    assert runtime.hold_period == 1
    assert runtime.estimated_delay is None


def test_runtime_rejects_invalid_confirmation():
    try:
        TrivaxRuntime(delay_confirmation=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
