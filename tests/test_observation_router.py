import pytest

from trivax.delay_aware import DelayAwareController
from trivax.observation_router import ObservationRoute, ObservationRouter
from trivax.probabilistic import ProbabilisticRegimeController


def test_router_defaults_to_raw_on_clean_signal():
    router = ObservationRouter(window=5)
    routes = []
    for value in [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]:
        state = router.process(value)
        routes.append(state.route)
    assert ObservationRoute.RAW in routes
    assert routes[-1] == ObservationRoute.RAW


def test_router_detects_impulsive_outlier():
    router = ObservationRouter(window=5, outlier_z=3.0)
    for value in [1.00, 1.01, 0.99, 1.00, 1.01]:
        router.process(value)
    state = router.process(2.0)
    assert state.route == ObservationRoute.ROBUST
    assert state.is_outlier
    assert abs(state.output - 1.0) < abs(state.raw - 1.0)


def test_router_marks_known_delay_without_smoothing():
    router = ObservationRouter(sensor_delay=4)
    state = router.process(1.23)
    assert state.route == ObservationRoute.DELAY_AWARE
    assert state.output == pytest.approx(1.23)


def test_delay_aware_wrapper_holds_actions_between_updates():
    inner = ProbabilisticRegimeController(step_size=0.01)
    wrapped = DelayAwareController(inner, sensor_delay=2)
    actions = []
    updates = []
    for observation in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        action, state = wrapped.step(observation)
        actions.append(action)
        updates.append(state.update_applied)
    assert updates == [True, False, False, True, False, False]
    assert actions[0] == actions[1] == actions[2]
    assert actions[3] == actions[4] == actions[5]


def test_invalid_router_and_delay_configuration():
    with pytest.raises(ValueError):
        ObservationRouter(window=4)
    with pytest.raises(ValueError):
        ObservationRouter(sensor_delay=-1)
    with pytest.raises(ValueError):
        DelayAwareController(ProbabilisticRegimeController(), sensor_delay=-1)
