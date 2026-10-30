import pandas as pd
import pytest

from src import consumption

D = pd.Timestamp


def make_inputs():
    """Мини-набор: 1 блюдо, 1 ингредиент (норма 100 г), инвентаризации пн 5.10 и чт 8.10, пн 12.10."""
    ingredients = pd.DataFrame({"ingredient_id": ["I1"], "name": ["Курица"], "unit": ["g"], "price_per_unit": [0.5]})
    recipe_items = pd.DataFrame({"dish_id": ["D1"], "ingredient_id": ["I1"], "qty_per_dish": [100.0]})
    sales = pd.DataFrame({
        "date": [D("2026-10-05"), D("2026-10-06"), D("2026-10-07"), D("2026-10-08"), D("2026-10-10")],
        "store_id": ["S1"] * 5, "dish_id": ["D1"] * 5, "qty": [7.0, 20.0, 15.0, 15.0, 10.0]})
    deliveries = pd.DataFrame({
        "date": [D("2026-10-05"), D("2026-10-06"), D("2026-10-10")], "store_id": ["S1"] * 3,
        "ingredient_id": ["I1"] * 3, "qty": [999.0, 3000.0, 4000.0]})
    inventory = pd.DataFrame({
        "date": [D("2026-10-05"), D("2026-10-08"), D("2026-10-12")], "store_id": ["S1"] * 3,
        "ingredient_id": ["I1"] * 3, "qty": [5000.0, 2000.0, 3000.0]})
    return sales, recipe_items, deliveries, inventory, ingredients


def test_theoretical_consumption():  # Т-1
    sales, recipe, *_ = make_inputs()
    daily = consumption.daily_theory(sales, recipe)
    assert daily.loc[daily["date"] == D("2026-10-06"), "theory"].iloc[0] == 2000.0


def test_periods_between_inventories():  # Т-4
    *_, inventory, _ = make_inputs()
    periods = consumption.build_periods(inventory)
    assert len(periods) == 2
    assert list(periods["start_date"]) == [D("2026-10-05"), D("2026-10-08")]


def test_fact_theory_and_deviation():  # Т-1, Т-2, Т-3
    result = consumption.calc_consumption(*make_inputs())
    first = result.iloc[0]
    assert first["days"] == 3
    assert first["theory"] == 5000.0          # 20+15+15 = 50 блюд × 100 г, день инвентаризации-старта (7 блюд) не входит
    assert first["deliveries"] == 3000.0      # поставка 5.10 относится к прошлому периоду
    assert first["fact"] == 6000.0            # 5000 + 3000 - 2000
    assert first["deviation"] == 1000.0
    assert first["deviation_pct"] == pytest.approx(20.0)
    assert first["deviation_cost"] == pytest.approx(500.0)


def test_second_period_has_four_days_and_includes_delivery():  # Т-4
    second = consumption.calc_consumption(*make_inputs()).iloc[1]
    assert second["days"] == 4
    assert second["deliveries"] == 4000.0
    assert second["theory"] == 1000.0
    assert second["fact"] == 2000.0 + 4000.0 - 3000.0


def test_no_theory_gives_nan_percent_not_error():
    sales, recipe, deliveries, inventory, ingredients = make_inputs()
    result = consumption.calc_consumption(sales.iloc[:0], recipe, deliveries, inventory, ingredients)
    assert result["deviation_pct"].isna().all()


def test_summarize_recomputes_percent():
    result = consumption.calc_consumption(*make_inputs())
    total = consumption.summarize(result, ["ingredient_id"]).iloc[0]
    assert total["theory"] == 6000.0 and total["fact"] == 9000.0   # 6000 + 3000
    assert total["deviation_pct"] == pytest.approx(3000 / 6000 * 100)


def test_flag_recount_marks_only_outlier():
    rows = [{"store_id": "S1", "ingredient_id": "I1", "deviation_pct": v} for v in [5, 6, 4, 5, 70]]
    flagged = consumption.flag_recount(pd.DataFrame(rows))
    assert list(flagged["check_recount"]) == [False, False, False, False, True]
