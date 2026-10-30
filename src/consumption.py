"""Расход ингредиентов: теория (по продажам и техкартам) против факта (по остаткам и поставкам).

Инвентаризация проводится в конце дня, поэтому период между двумя инвентаризациями
(start, end] включает дни ПОСЛЕ первой даты и сам день второй инвентаризации.
"""
from __future__ import annotations

import pandas as pd

SUM_COLS = ["theory", "fact", "deviation", "theory_cost", "fact_cost", "deviation_cost"]


def daily_theory(sales: pd.DataFrame, recipe_items: pd.DataFrame) -> pd.DataFrame:
    """Теоретический расход по дням: продажи блюд × норма ингредиента в техкарте."""
    merged = sales.merge(recipe_items, on="dish_id")
    merged["theory"] = merged["qty"] * merged["qty_per_dish"]
    return merged.groupby(["date", "store_id", "ingredient_id"], as_index=False)["theory"].sum()


def daily_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Поставки по дням (несколько поставок в день складываются)."""
    out = deliveries.groupby(["date", "store_id", "ingredient_id"], as_index=False)["qty"].sum()
    return out.rename(columns={"qty": "deliveries"})


def build_periods(inventory: pd.DataFrame) -> pd.DataFrame:
    """Периоды между соседними инвентаризациями для каждой пары «точка + ингредиент»."""
    inv = inventory.sort_values(["store_id", "ingredient_id", "date"]).copy()
    grouped = inv.groupby(["store_id", "ingredient_id"])
    inv["start_date"] = grouped["date"].shift(1)
    inv["stock_start"] = grouped["qty"].shift(1)
    periods = inv.dropna(subset=["start_date"]).rename(columns={"date": "end_date", "qty": "stock_end"})
    return periods[["store_id", "ingredient_id", "start_date", "end_date",
                    "stock_start", "stock_end"]].reset_index(drop=True)


def _sum_into_periods(periods: pd.DataFrame, daily: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Суммирует дневные значения по периодам: день попадает в период, если start < день <= end."""
    left = daily.sort_values("date")
    right = periods.sort_values("end_date")[["store_id", "ingredient_id", "start_date", "end_date"]]
    merged = pd.merge_asof(left, right, left_on="date", right_on="end_date",
                           by=["store_id", "ingredient_id"], direction="forward")
    merged = merged[merged["date"] > merged["start_date"]]  # дни до первой инвентаризации отбрасываются
    return merged.groupby(["store_id", "ingredient_id", "end_date"], as_index=False)[value_col].sum()


def calc_consumption(sales, recipe_items, deliveries, inventory, ingredients,
                     theory_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Главная таблица: по каждому периоду теория, факт, отклонение в единицах и рублях.

    Факт = остаток на начало + поставки за период - остаток на конец.
    Отклонение = факт - теория (плюс = потери).
    """
    periods = build_periods(inventory)
    if theory_daily is None:
        theory_daily = daily_theory(sales, recipe_items)
    theory = _sum_into_periods(periods, theory_daily, "theory")
    delivered = _sum_into_periods(periods, daily_deliveries(deliveries), "deliveries")

    key = ["store_id", "ingredient_id", "end_date"]
    out = periods.merge(theory, on=key, how="left").merge(delivered, on=key, how="left")
    out[["theory", "deliveries"]] = out[["theory", "deliveries"]].fillna(0.0)
    out["days"] = (out["end_date"] - out["start_date"]).dt.days
    out["fact"] = out["stock_start"] + out["deliveries"] - out["stock_end"]
    out["deviation"] = out["fact"] - out["theory"]
    out["deviation_pct"] = out["deviation"] / out["theory"].where(out["theory"] > 0) * 100

    out = out.merge(ingredients[["ingredient_id", "name", "unit", "price_per_unit"]], on="ingredient_id", how="left")
    out["theory_cost"] = out["theory"] * out["price_per_unit"]
    out["fact_cost"] = out["fact"] * out["price_per_unit"]
    out["deviation_cost"] = out["deviation"] * out["price_per_unit"]
    return out


def summarize(periods: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Сводка по любым колонкам (ингредиент, точка и т.д.), отклонение в процентах пересчитывается."""
    summary = periods.groupby(by, as_index=False)[SUM_COLS].sum()
    summary["deviation_pct"] = summary["deviation"] / summary["theory"].where(summary["theory"] > 0) * 100
    return summary


def flag_recount(periods: pd.DataFrame, threshold_pp: float = 30.0, min_periods: int = 4) -> pd.DataFrame:
    """Помечает периоды, где отклонение резко выбивается из обычного для этой пары «точка + ингредиент».

    Это похоже на ошибку пересчёта при инвентаризации, а не на реальные потери.
    threshold_pp: на сколько процентных пунктов отклонение отличается от медианы.
    """
    out = periods.copy()
    grouped = out.groupby(["store_id", "ingredient_id"])["deviation_pct"]
    median = grouped.transform("median")
    count = grouped.transform("count")
    out["check_recount"] = (count >= min_periods) & ((out["deviation_pct"] - median).abs() > threshold_pp)
    return out
