from trivax.regime_selector import RegimeMode, TrivaxRegimeSelector


def test_selector_starts_in_adaptive_mode():
    selector = TrivaxRegimeSelector()
    _, state = selector.step(1.0)
    assert state.mode is RegimeMode.ADAPTIVE


def test_selector_respects_action_bounds():
    selector = TrivaxRegimeSelector()
    for i in range(200):
        action, _ = selector.step(1.0 - (0.7 - 0.003 * i) ** 2)
        assert 0.0 <= action <= 1.0


def test_selector_can_enter_temporal_mode_under_delayed_signal():
    selector = TrivaxRegimeSelector(min_dwell=8, enter_threshold=0.5)
    history = [selector.action] * 8
    entered = False
    for t in range(240):
        delayed_action = history[-4]
        optimum = 0.45 + 0.0015 * t
        observation = 1.0 - (delayed_action - optimum) ** 2
        action, state = selector.step(observation)
        history.append(action)
        if state.mode is RegimeMode.TEMPORAL:
            entered = True
            break
    assert entered
