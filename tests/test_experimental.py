import math

import pytest

from trivax.baselines import PerturbAndObserve
from trivax.experimental import CoherenceAdaptiveController


def test_adaptive_controller_stays_within_bounds():
    controller = CoherenceAdaptiveController(initial_action=0.5, step_size=0.1)
    for observation in [0.0, 0.2, 0.1, 0.3, 0.25] * 20:
        action, state = controller.step(observation)
        assert 0.0 <= action <= 1.0
        assert 0.0 <= state.coherence <= 1.0
        assert state.effective_step > 0.0


def test_adaptive_step_contracts_after_degradation():
    controller = CoherenceAdaptiveController(step_size=0.05, coherence_alpha=0.5)
    _, first = controller.step(1.0)
    _, improved = controller.step(1.1)
    _, degraded = controller.step(0.9)

    assert improved.coherence > first.coherence
    assert degraded.coherence < improved.coherence
    assert degraded.effective_step < improved.effective_step


def test_po_reverses_after_degradation():
    controller = PerturbAndObserve(initial_action=0.5, step_size=0.05)
    controller.step(1.0)
    direction_before = controller.direction
    controller.step(0.9)
    assert controller.direction == -direction_before


def test_invalid_adaptive_configuration_is_rejected():
    with pytest.raises(ValueError):
        CoherenceAdaptiveController(step_size=0.0)
    with pytest.raises(ValueError):
        CoherenceAdaptiveController(coherence_alpha=0.0)
    with pytest.raises(ValueError):
        CoherenceAdaptiveController(min_gain=0.0)
