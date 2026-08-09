import pytest

from trivax.probing import IdentificationProbePolicy


def test_probe_waits_for_low_confidence_persistence():
    policy = IdentificationProbePolicy(amplitude=0.01, trigger_steps=3, min_interval=1)
    a = policy.step(confidence_ok=False, update_blocked=False)
    b = policy.step(confidence_ok=False, update_blocked=False)
    c = policy.step(confidence_ok=False, update_blocked=False)
    assert not a.applied
    assert not b.applied
    assert c.applied


def test_probe_alternates_sign():
    policy = IdentificationProbePolicy(amplitude=0.01, trigger_steps=1, min_interval=1)
    first = policy.step(confidence_ok=False, update_blocked=False)
    policy.step(confidence_ok=True, update_blocked=False)
    second = policy.step(confidence_ok=False, update_blocked=False)
    assert first.offset == pytest.approx(0.01)
    assert second.offset == pytest.approx(-0.01)


def test_probe_is_blocked_during_outlier_or_blocked_update():
    policy = IdentificationProbePolicy(trigger_steps=1, min_interval=1)
    state = policy.step(confidence_ok=False, update_blocked=True)
    assert not state.applied


def test_probe_budget_is_respected():
    policy = IdentificationProbePolicy(trigger_steps=1, min_interval=1, max_probes=2)
    applied = 0
    for _ in range(10):
        state = policy.step(confidence_ok=False, update_blocked=False)
        applied += int(state.applied)
    assert applied == 2


def test_invalid_probe_configuration():
    with pytest.raises(ValueError):
        IdentificationProbePolicy(amplitude=0.0)
    with pytest.raises(ValueError):
        IdentificationProbePolicy(trigger_steps=0)
