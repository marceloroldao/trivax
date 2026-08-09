from trivax.runtime_v5 import TrivaxRuntimeV5


def test_runtime_v5_stays_within_bounds_and_exposes_voi_state():
    runtime = TrivaxRuntimeV5()
    for i in range(100):
        observation = 1.0 - 4.0 * (runtime.action - 0.7) ** 2
        action, state = runtime.step(observation)
        assert 0.0 <= action <= 1.0
        assert state.voi_state.net_value == (
            state.voi_state.expected_information_gain - state.voi_state.perturbation_cost
        )


def test_runtime_v5_does_not_require_probe_every_step():
    runtime = TrivaxRuntimeV5()
    applied = 0
    for i in range(120):
        observation = 1.0 - 4.0 * (runtime.action - 0.65) ** 2
        _, state = runtime.step(observation)
        applied += int(state.voi_state.applied)
    assert applied < 120
