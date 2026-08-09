from trivax.delay_estimator import DelayEstimator
from trivax.runtime_v2 import TrivaxRuntimeV2
from trivax.temporal_credit import HistoricalCreditController


def test_runtime_v2_emits_action_every_cycle():
    runtime = TrivaxRuntimeV2(
        controller=HistoricalCreditController(step_size=0.01, delay=0),
        delay_estimator=DelayEstimator(max_delay=4, window=32, min_samples=8),
    )
    actions = []
    for i in range(20):
        action, state = runtime.step(float(i) * 0.01)
        actions.append(action)
        assert state.action == action
    assert len(actions) == 20


def test_runtime_v2_applies_confirmed_delay():
    runtime = TrivaxRuntimeV2(delay_confirmation=1)

    class Estimate:
        stable = True
        delay = 4
        score = 0.9

    runtime._accept_delay(Estimate())
    assert runtime.estimated_delay == 4
    assert runtime.controller.delay == 4
