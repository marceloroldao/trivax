import pytest

from trivax.probabilistic import ProbabilisticRegimeController, ProbabilisticRegime


def test_probabilities_are_normalized_and_bounded():
    controller = ProbabilisticRegimeController(step_size=0.02)
    for observation in [0.1, 0.2, 0.15, 0.25, 0.22, 0.30]:
        action, state = controller.step(observation)
        assert 0.0 <= action <= 1.0
        assert 0.0 <= state.p_search <= 1.0
        assert 0.0 <= state.p_track <= 1.0
        assert 0.0 <= state.p_stabilize <= 1.0
        assert abs((state.p_search + state.p_track + state.p_stabilize) - 1.0) < 1e-12


def test_repeated_improvement_can_select_track():
    controller = ProbabilisticRegimeController(step_size=0.01)
    state = None
    for observation in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        _, state = controller.step(observation)
    assert state is not None
    assert state.regime in {ProbabilisticRegime.TRACK, ProbabilisticRegime.SEARCH}
    assert state.coherence > 0.5


def test_reversals_raise_reversal_rate():
    controller = ProbabilisticRegimeController(step_size=0.01, alpha=0.5)
    controller.step(1.0)
    _, state1 = controller.step(0.9)
    _, state2 = controller.step(0.8)
    assert state1.reversal_rate > 0.0
    assert state2.reversal_rate >= state1.reversal_rate


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        ProbabilisticRegimeController(step_size=0.0)
    with pytest.raises(ValueError):
        ProbabilisticRegimeController(alpha=0.0)
