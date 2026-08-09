import random

from trivax.delay_estimator import DelayEstimator


def feed_linear_delay(estimator, delay, steps=300, noise=0.002, seed=123):
    rng = random.Random(seed)
    actions = []
    action = 0.5
    result = None
    for t in range(steps):
        action = max(0.0, min(1.0, action + rng.choice([-1.0, 1.0]) * rng.uniform(0.01, 0.04)))
        actions.append(action)
        source_index = max(0, t - delay)
        observation = 1.2 * actions[source_index] + rng.gauss(0.0, noise)
        result = estimator.update(action, observation)
    return result


def test_delay_estimator_recovers_known_delay():
    for delay in (0, 1, 2, 4, 7):
        estimator = DelayEstimator(max_delay=10, window=160, min_samples=40, min_abs_correlation=0.2, min_margin=0.03)
        result = feed_linear_delay(estimator, delay)
        assert result.stable
        assert result.delay == delay
        assert result.score > 0.7


def test_delay_estimator_waits_for_evidence():
    estimator = DelayEstimator(max_delay=6, min_samples=32)
    result = estimator.update(0.5, 1.0)
    assert not result.stable
    assert result.delay is None


def test_invalid_delay_estimator_configuration():
    import pytest

    with pytest.raises(ValueError):
        DelayEstimator(max_delay=-1)
    with pytest.raises(ValueError):
        DelayEstimator(window=4, min_samples=8)
    with pytest.raises(ValueError):
        DelayEstimator(min_samples=4)
