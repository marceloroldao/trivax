import pytest

from trivax.temporal_credit import HistoricalCreditController


def test_temporal_credit_stays_in_bounds():
    controller = HistoricalCreditController(step_size=0.1, delay=3)
    for observation in [0.0, 0.2, 0.1, 0.3, 0.25] * 20:
        action, state = controller.step(observation)
        assert 0.0 <= action <= 1.0
        assert state.delay == 3


def test_temporal_credit_uses_historical_action_difference():
    controller = HistoricalCreditController(initial_action=0.5, step_size=0.1, delay=2)
    controller.step(0.0)
    controller.step(0.0)
    controller.step(0.1)
    _, state = controller.step(0.2)

    assert state.credited_action is not None
    assert state.previous_credited_action is not None


def test_temporal_credit_can_update_delay_online():
    controller = HistoricalCreditController(delay=1)
    controller.set_delay(5)
    assert controller.delay == 5


def test_temporal_credit_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        HistoricalCreditController(step_size=0.0)
    with pytest.raises(ValueError):
        HistoricalCreditController(delay=-1)
    with pytest.raises(ValueError):
        HistoricalCreditController(min_action=1.0, max_action=0.0)
