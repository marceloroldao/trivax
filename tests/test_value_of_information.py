import pytest

from trivax.value_of_information import ValueOfInformationProbePolicy


def test_probe_fires_when_uncertainty_is_high_and_excitation_is_low():
    policy = ValueOfInformationProbePolicy(min_interval=1, min_net_value=0.01)
    state = policy.step(confidence=0.1, excitation=0.0, update_blocked=False)
    assert state.applied
    assert state.offset != 0.0
    assert state.net_value > 0.0


def test_probe_is_suppressed_when_natural_excitation_is_sufficient():
    policy = ValueOfInformationProbePolicy(min_interval=1, min_net_value=0.01)
    state = policy.step(confidence=0.1, excitation=0.05, update_blocked=False)
    assert not state.applied
    assert state.expected_information_gain == 0.0


def test_probe_is_suppressed_when_confidence_is_high_and_cost_dominates():
    policy = ValueOfInformationProbePolicy(
        min_interval=1,
        min_net_value=0.05,
        cost_weight=20.0,
    )
    state = policy.step(confidence=0.99, excitation=0.0, update_blocked=False)
    assert not state.applied


def test_probe_is_blocked_during_invalid_observation_periods():
    policy = ValueOfInformationProbePolicy(min_interval=1, min_net_value=0.0)
    state = policy.step(confidence=0.0, excitation=0.0, update_blocked=True)
    assert not state.applied


def test_probe_budget_and_alternating_sign_are_enforced():
    policy = ValueOfInformationProbePolicy(
        min_interval=1,
        max_probes=2,
        min_net_value=0.0,
    )
    first = policy.step(confidence=0.0, excitation=0.0, update_blocked=False)
    policy.step(confidence=0.0, excitation=0.0, update_blocked=False)
    second = policy.step(confidence=0.0, excitation=0.0, update_blocked=False)
    policy.step(confidence=0.0, excitation=0.0, update_blocked=False)
    third = policy.step(confidence=0.0, excitation=0.0, update_blocked=False)

    assert first.applied
    assert second.applied
    assert first.offset == -second.offset
    assert not third.applied
    assert third.probe_count == 2


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        ValueOfInformationProbePolicy(amplitude=0.0)
    with pytest.raises(ValueError):
        ValueOfInformationProbePolicy(excitation_target=0.0)
