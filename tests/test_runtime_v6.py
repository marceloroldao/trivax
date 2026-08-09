from trivax.runtime_v6 import RuntimeMode, TrivaxRuntimeV6


def test_v6_starts_fast_and_stays_bounded():
    runtime = TrivaxRuntimeV6()
    assert runtime.mode is RuntimeMode.FAST
    for _ in range(120):
        action, state = runtime.step(1.0 - (runtime.action - 0.6) ** 2)
        assert 0.0 <= action <= 1.0
        assert state.mode in (RuntimeMode.FAST, RuntimeMode.VERIFY)


def test_v6_fast_mode_disables_probe():
    runtime = TrivaxRuntimeV6()
    _, state = runtime.step(0.9)
    assert state.mode is RuntimeMode.FAST
    assert state.voi_state.applied is False
