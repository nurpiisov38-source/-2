import pandas as pd
import pytest

from src import consumption, demo_data, kpi


def test_food_cost_pct():  # Т-15
    assert kpi.food_cost_pct(30000, 100000) == pytest.approx(30.0)
    assert pd.isna(kpi.food_cost_pct(10, 0))


def test_dish_cost_from_recipe():
    dishes = pd.DataFrame({"dish_id": ["D1"], "name": ["Шаурма"], "price": [200.0]})
    recipe = pd.DataFrame({"dish_id": ["D1", "D1"], "ingredient_id": ["I1", "I2"], "qty_per_dish": [100.0, 2.0]})
    ingredients = pd.DataFrame({"ingredient_id": ["I1", "I2"], "price_per_unit": [0.5, 10.0]})
    row = kpi.dish_costs(dishes, recipe, ingredients).iloc[0]
    assert row["unit_cost"] == pytest.approx(70.0)
    assert row["food_cost_pct"] == pytest.approx(35.0)


def test_demo_data_is_deterministic():
    a = demo_data.generate_demo_data(weeks=4, seed=1)
    b = demo_data.generate_demo_data(weeks=4, seed=1)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_demo_data_losses_are_positive_and_realistic():
    t = demo_data.generate_demo_data(weeks=8, recount_errors=0)
    daily = consumption.daily_theory(t["sales"], t["recipe_items"])
    periods = consumption.calc_consumption(t["sales"], t["recipe_items"], t["deliveries"], t["inventory"],
                                           t["ingredients"], theory_daily=daily)
    k = kpi.compute_kpis(periods, t["sales"], t["dishes"])
    assert 0 < k["deviation_pct"] < 15
    assert k["revenue"] > 0 and 20 < k["food_cost_pct"] < 60


def test_demo_data_consistency_fact_equals_actual_usage():
    """Без ошибок пересчёта факт на конец периода должен быть неотрицательным и близким к теории."""
    t = demo_data.generate_demo_data(weeks=6, recount_errors=0)
    periods = consumption.calc_consumption(t["sales"], t["recipe_items"], t["deliveries"], t["inventory"], t["ingredients"])
    assert (periods["fact"] >= -1e-9).all()


def test_compute_kpis_empty_periods():
    t = demo_data.generate_demo_data(weeks=3)
    empty = consumption.calc_consumption(t["sales"], t["recipe_items"], t["deliveries"], t["inventory"], t["ingredients"]).iloc[:0]
    result = kpi.compute_kpis(empty, t["sales"], t["dishes"])
    assert result["loss_rub"] == 0 and pd.isna(result["deviation_pct"])
