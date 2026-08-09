from trivax.causal_delay import CausalDelayConfidence
from trivax.delay_estimator import DelayEstimate


class StubEstimator:
    def __init__(self, estimates):
        self.estimates = iter(estimates)

    def update(self, action, observation):
        return next(self.estimates)


def test_requires_persistent_confirmation():
    estimates = [DelayEstimate(2, 0.8, True, 64)] * 4
    layer = CausalDelayConfidence(
        estimator=StubEstimator(estimates),
        confirmation=3,
        change_confirmation=5,
        min_excitation=0.0,
    )

    state = None
    for i in range(3):
        state = layer.update(float(i), 0.0)
    assert state is not None
    assert state.accepted_delay == 2


def test_changing_existing_delay_requires_more_evidence():
    estimates = (
        [DelayEstimate(2, 0.8, True, 64)] * 3
        + [DelayEstimate(5, 0.9, True, 64)] * 5
    )
    layer = CausalDelayConfidence(
        estimator=StubEstimator(estimates),
        confirmation=3,
        change_confirmation=5,
        min_excitation=0.0,
    )

    for i in range(3):
        state = layer.update(float(i), 0.0)
    assert state.accepted_delay == 2

    for i in range(4):
        state = layer.update(float(i + 3), 0.0)
    assert state.accepted_delay == 2

    state = layer.update(8.0, 0.0)
    assert state.accepted_delay == 5


def test_outlier_block_prevents_delay_change():
    estimates = [DelayEstimate(4, 0.9, True, 64)] * 4
    layer = CausalDelayConfidence(
        estimator=StubEstimator(estimates),
        confirmation=2,
        change_confirmation=3,
        min_excitation=0.0,
    )

    state = layer.update(0.0, 0.0, block_update=True)
    assert state.accepted_delay is None
    state = layer.update(1.0, 0.0, block_update=False)
    assert state.accepted_delay is None
    state = layer.update(2.0, 0.0, block_update=False)
    assert state.accepted_delay == 4
