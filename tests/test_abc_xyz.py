import pandas as pd

from src import abc_xyz


def test_abc_classes():  # Т-5
    cls = abc_xyz.abc_classification(pd.Series({"a": 80.0, "b": 15.0, "c": 5.0}))
    assert cls.to_dict() == {"a": "A", "b": "B", "c": "C"}


def test_abc_zero_total_all_c():
    assert set(abc_xyz.abc_classification(pd.Series({"a": 0.0, "b": 0.0}))) == {"C"}


def test_abc_zero_value_item_is_c():
    cls = abc_xyz.abc_classification(pd.Series({"a": 100.0, "b": 0.0}))
    assert cls["a"] == "A" and cls["b"] == "C"


def test_xyz_constant_demand_is_x():  # Т-6
    demand = pd.DataFrame({"I1": [10.0] * 14})
    result = abc_xyz.xyz_classification(demand)
    assert result.loc["I1", "xyz"] == "X"
    assert result.loc["I1", "cv"] == 0


def test_xyz_high_variation_is_z_and_unused_is_z():
    demand = pd.DataFrame({"I1": [0.0, 100.0] * 7, "I2": [0.0] * 14})
    result = abc_xyz.xyz_classification(demand)
    assert result.loc["I1", "xyz"] == "Z"
    assert result.loc["I2", "xyz"] == "Z"  # расхода нет, вариация не определена


def test_deseasonalize_removes_weekday_pattern():
    dates = pd.date_range("2026-09-07", periods=28)  # 4 полные недели, начиная с понедельника
    values = [10, 10, 10, 10, 20, 20, 20] * 4        # пятница-воскресенье вдвое выше, но ровно каждую неделю
    daily = pd.DataFrame({"date": dates, "store_id": "S1", "ingredient_id": "I1", "theory": values})
    raw = abc_xyz.demand_matrix(daily, dates.min(), dates.max(), deseasonalize=False)
    clean = abc_xyz.demand_matrix(daily, dates.min(), dates.max())
    assert abc_xyz.xyz_classification(raw).loc["I1", "xyz"] != "X"
    assert abc_xyz.xyz_classification(clean).loc["I1", "xyz"] == "X"


def test_classify_ingredients_combines_classes():
    ingredients = pd.DataFrame({"ingredient_id": ["I1", "I2"], "name": ["a", "b"], "unit": ["g", "g"]})
    cost = pd.Series({"I1": 95.0, "I2": 5.0})
    demand = pd.DataFrame({"I1": [5.0] * 10, "I2": [0.0, 20.0] * 5})
    result = abc_xyz.classify_ingredients(cost, demand, ingredients).set_index("ingredient_id")
    assert result.loc["I1", "abc_xyz"] == "AX"
    assert result.loc["I2", "abc_xyz"] == "CZ"
