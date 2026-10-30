import pandas as pd
import pytest

from src import ordering

D = pd.Timestamp


def test_order_quantity_basic():  # Т-7: 10/день, поставка 2 дня, горизонт 2, страховой 1 день, остаток 20
    assert ordering.order_quantity(demand=40, safety=10, stock=20, pack_size=1) == 30


def test_order_quantity_rounds_up_to_pack():  # Т-8
    assert ordering.order_quantity(demand=40, safety=10, stock=20, pack_size=25) == 50


def test_order_quantity_exact_pack_not_rounded_up():
    assert ordering.order_quantity(demand=40, safety=10, stock=0, pack_size=25) == 50


def test_order_quantity_never_negative():  # Т-9
    assert ordering.order_quantity(demand=40, safety=10, stock=500, pack_size=25) == 0


def test_waste_risk():  # Т-10
    assert ordering.has_waste_risk(stock=100, daily_mean=10, shelf_life_days=5) is True
    assert ordering.has_waste_risk(stock=30, daily_mean=10, shelf_life_days=5) is False
    assert ordering.has_waste_risk(stock=0, daily_mean=0, shelf_life_days=5) is False


def make_daily(weekday_values):
    dates = pd.date_range("2026-08-31", "2026-10-15")  # понедельник ... четверг
    values = [weekday_values[d.dayofweek] for d in dates]
    return pd.DataFrame({"date": dates, "store_id": "S1", "ingredient_id": "I1", "theory": values})


def test_weekday_means_friday_higher_than_tuesday():  # Т-11
    daily = make_daily([10, 10, 10, 10, 20, 20, 10])
    means = ordering.weekday_means(daily, D("2026-10-15"), weeks=6).set_index("wd")["mean"]
    assert means[4] == 20 and means[1] == 10


def make_tables(stock=20):
    ingredients = pd.DataFrame({"ingredient_id": ["I1"], "name": ["Курица"], "unit": ["g"], "price_per_unit": [1.0],
                                "supplier": ["A"], "lead_time_days": [2], "shelf_life_days": [100], "pack_size": [1.0]})
    inventory = pd.DataFrame({"date": [D("2026-10-15")], "store_id": ["S1"], "ingredient_id": ["I1"], "qty": [float(stock)]})
    periods = pd.DataFrame({"store_id": ["S1"], "ingredient_id": ["I1"], "end_date": [D("2026-10-15")],
                            "theory": [100.0], "fact": [100.0]})
    return ingredients, inventory, periods


def test_build_order_list_constant_demand():
    ingredients, inventory, periods = make_tables(stock=20)
    daily = make_daily([10] * 7)
    result = ordering.build_order_list(ingredients, inventory, daily, periods, "2026-10-15", horizon_days=2)
    row = result.iloc[0]
    assert row["demand"] == pytest.approx(40) and row["safety"] == pytest.approx(10)
    assert row["target"] == pytest.approx(50) and row["to_order"] == 30


def test_build_order_list_uses_weekday_pattern():
    # Заказ в четверг, период покрытия 4 дня: пт, сб, вс, пн
    ingredients, inventory, periods = make_tables(stock=0)
    daily = make_daily([10, 10, 10, 10, 30, 30, 20])
    result = ordering.build_order_list(ingredients, inventory, daily, periods, "2026-10-15", horizon_days=2)
    assert result.iloc[0]["demand"] == pytest.approx(30 + 30 + 20 + 10)


def test_loss_factor_raises_forecast():
    ingredients, inventory, periods = make_tables(stock=0)
    periods["fact"] = 120.0  # потери 20%
    daily = make_daily([10] * 7)
    result = ordering.build_order_list(ingredients, inventory, daily, periods, "2026-10-15", horizon_days=2)
    assert result.iloc[0]["loss_factor"] == pytest.approx(1.2)
    assert result.iloc[0]["demand"] == pytest.approx(48)


def test_store_filter():
    ingredients, inventory, periods = make_tables()
    other = inventory.assign(store_id="S2")
    both = pd.concat([inventory, other])
    result = ordering.build_order_list(ingredients, both, make_daily([10] * 7), periods, "2026-10-15", store_id="S2")
    assert set(result["store_id"]) == {"S2"}
