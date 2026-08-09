from trivax.core_rc import TrivaxCoreRC


def test_core_rc_runs_and_respects_bounds():
    runtime = TrivaxCoreRC()
    for i in range(200):
        action, state = runtime.step(1.0 - (runtime.action - 0.65) ** 2)
        assert 0.0 <= action <= 1.0
        assert 0.0 <= state.action <= 1.0


def test_validator_is_sampled_not_continuous():
    runtime = TrivaxCoreRC(validator_interval=8)
    flags = []
    for _ in range(20):
        _, state = runtime.step(1.0)
        flags.append(state.validator_ran)
    assert sum(flags) == 3
    assert flags[0] and flags[8] and flags[16]


def test_probes_disabled_by_default():
    runtime = TrivaxCoreRC(enable_probes=False, validator_interval=1)
    for _ in range(100):
        _, state = runtime.step(1.0)
        assert state.probe_state is None
