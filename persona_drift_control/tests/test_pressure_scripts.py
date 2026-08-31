import pytest

from persona_drift.pressure_scripts import load_pressure_script


@pytest.mark.parametrize("category", ["character_traits", "language_constraints"])
def test_known_categories_return_a_long_enough_deterministic_script(category):
    script = load_pressure_script(category)
    assert len(script) >= 16
    assert script == load_pressure_script(category)  # deterministic, same object contents each call


def test_the_two_categories_have_different_scripts():
    assert load_pressure_script("character_traits") != load_pressure_script("language_constraints")


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="no escalating-pressure script"):
        load_pressure_script("not_a_real_category")
