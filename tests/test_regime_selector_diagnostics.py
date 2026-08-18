from trivax.regime_selector import RegimeMode, TrivaxRegimeSelector


def test_selector_reports_diagnostics_without_switching_early():
    selector = TrivaxRegimeSelector(min_dwell=8)
    for _ in range(5):
        action, state = selector.step(1.0)
    assert state.mode is RegimeMode.ADAPTIVE
    assert state.switch_count == 0
    assert state.temporal_duty_fraction == 0.0
    assert state.steps_in_mode >= 1
    assert state.switch_reason is None
    assert 0.0 <= action <= 1.0


def test_selector_duty_fraction_is_bounded():
    selector = TrivaxRegimeSelector(min_dwell=2, enter_threshold=0.2, exit_threshold=0.1)
    for i in range(80):
        _, state = selector.step(1.0 - 0.02 * ((i % 7) - 3) ** 2)
    assert 0.0 <= state.temporal_duty_fraction <= 1.0
    assert state.switch_count >= 0
