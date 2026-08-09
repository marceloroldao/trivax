import pytest

from trivax.robust_observation import RobustObservationLayer


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        RobustObservationLayer(median_window=2)
    with pytest.raises(ValueError):
        RobustObservationLayer(innovation_clip=0.0)
    with pytest.raises(ValueError):
        RobustObservationLayer(sensor_delay=-1)


def test_clips_large_innovation():
    layer = RobustObservationLayer(median_window=1, innovation_clip=0.1)
    first = layer.process(0.0)
    second = layer.process(1.0)
    assert first.filtered == 0.0
    assert second.filtered == pytest.approx(0.1)
    assert second.clipped_innovation == pytest.approx(0.1)


def test_short_median_rejects_single_impulse():
    layer = RobustObservationLayer(median_window=3, innovation_clip=10.0)
    layer.process(1.0)
    layer.process(1.0)
    state = layer.process(10.0)
    assert state.filtered == pytest.approx(1.0)


def test_delay_reference_is_exposed():
    layer = RobustObservationLayer(median_window=1, innovation_clip=10.0, sensor_delay=2)
    assert layer.process(1.0).delayed_reference is None
    assert layer.process(2.0).delayed_reference is None
    state = layer.process(3.0)
    assert state.delayed_reference == pytest.approx(1.0)
