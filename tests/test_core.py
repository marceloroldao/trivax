import pytest

from trivax import ScalarPlant, TrivaxController, run_closed_loop


def test_controller_rejects_invalid_step_size():
    with pytest.raises(ValueError):
        TrivaxController(step_size=0.0)


def test_controller_action_stays_within_bounds():
    controller = TrivaxController(initial_action=0.95, step_size=0.2)
    for observation in [0.1, 0.2, 0.3, 0.1, 0.4, 0.2]:
        action, _ = controller.step(observation)
        assert 0.0 <= action <= 1.0


def test_closed_loop_is_deterministic():
    plant_a = ScalarPlant()
    plant_b = ScalarPlant()
    controller_a = TrivaxController()
    controller_b = TrivaxController()

    result_a = run_closed_loop(controller_a, plant_a, steps=50)
    result_b = run_closed_loop(controller_b, plant_b, steps=50)

    assert result_a == result_b


def test_closed_loop_produces_expected_record_count():
    records = run_closed_loop(TrivaxController(), ScalarPlant(), steps=25)
    assert len(records) == 25
    assert all(0.0 <= row["action"] <= 1.0 for row in records)
    assert all(0.0 <= row["coherence"] <= 1.0 for row in records)
