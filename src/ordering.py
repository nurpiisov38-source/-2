"""Рекомендации по заказу: прогноз расхода, страховой запас 1 день, округление до упаковки, риск списания.

Формула (простая и объяснимая):
    цель = расход_в_день × (срок_поставки + горизонт_заказа + 1 день страхового запаса)
    заказ = max(0, цель - текущий_остаток), округлённый вверх до кратности упаковки.
Прогноз учитывает день недели: берётся средний расход за те же дни недели за последние недели
и умножается на коэффициент потерь (факт / теория за последний месяц, от 1,0 до 1,5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WD_COLS = [f"wd{i}" for i in range(7)]


def order_quantity(demand: float, safety: float, stock: float, pack_size: float = 1.0) -> float:
    """Сколько заказать: недостающее до цели, округлённое вверх до упаковки. Не бывает отрицательным."""
    raw = max(0.0, demand + safety - stock)
    if raw <= 0:
        return 0.0
    pack = pack_size if pack_size and pack_size > 0 else 1.0
    return float(np.ceil(raw / pack - 1e-9) * pack)


def has_waste_risk(stock: float, daily_mean: float, shelf_life_days: float) -> bool:
    """Риск списания: остатка больше, чем уйдёт за срок годности."""
    if stock <= 0:
        return False
    return stock > daily_mean * shelf_life_days


def weekday_means(daily: pd.DataFrame, order_date, weeks: int = 6) -> pd.DataFrame:
    """Средний теоретический расход по дням недели (0 = понедельник) за последние weeks недель."""
    order_date = pd.Timestamp(order_date)
    first = max(order_date - pd.Timedelta(days=7 * weeks - 1), daily["date"].min())
    days = pd.date_range(first, order_date)
    counts = pd.Series(days.dayofweek).value_counts()
    window = daily[(daily["date"] >= first) & (daily["date"] <= order_date)].copy()
    window["wd"] = window["date"].dt.dayofweek
    sums = window.groupby(["store_id", "ingredient_id", "wd"], as_index=False)["theory"].sum()
    sums["mean"] = sums["theory"] / sums["wd"].map(counts)
    return sums[["store_id", "ingredient_id", "wd", "mean"]]


def _weekday_wide(means: pd.DataFrame) -> pd.DataFrame:
    if means.empty:
        return pd.DataFrame(columns=["store_id", "ingredient_id"] + WD_COLS)
    wide = means.pivot_table(index=["store_id", "ingredient_id"], columns="wd", values="mean")
    wide = wide.reindex(columns=range(7)).fillna(0.0)
    wide.columns = WD_COLS
    return wide.reset_index()


def loss_factors(periods: pd.DataFrame, order_date, lookback_days: int = 28) -> pd.DataFrame:
    """Коэффициент потерь = факт / теория за последние lookback_days (не меньше 1,0 и не больше 1,5)."""
    order_date = pd.Timestamp(order_date)
    recent = periods[(periods["end_date"] <= order_date)
                     & (periods["end_date"] > order_date - pd.Timedelta(days=lookback_days))]
    grouped = recent.groupby(["store_id", "ingredient_id"])[["theory", "fact"]].sum()
    factor = (grouped["fact"] / grouped["theory"].where(grouped["theory"] > 0)).clip(lower=1.0, upper=1.5)
    return factor.fillna(1.0).rename("loss_factor").reset_index()


def build_order_list(ingredients: pd.DataFrame, inventory: pd.DataFrame, daily: pd.DataFrame,
                     periods: pd.DataFrame, order_date, horizon_days: int = 2,
                     history_weeks: int = 6, store_id: str | None = None) -> pd.DataFrame:
    """Таблица рекомендаций по каждой паре «точка + ингредиент» на дату инвентаризации order_date."""
    order_date = pd.Timestamp(order_date)
    stock = inventory.loc[inventory["date"] == order_date, ["store_id", "ingredient_id", "qty"]]
    stock = stock.rename(columns={"qty": "stock"})
    if store_id is not None:
        stock = stock[stock["store_id"] == store_id]

    base = stock.merge(ingredients, on="ingredient_id", how="left")
    base = base.merge(_weekday_wide(weekday_means(daily, order_date, history_weeks)),
                      on=["store_id", "ingredient_id"], how="left")
    base[WD_COLS] = base[WD_COLS].fillna(0.0)
    base = base.merge(loss_factors(periods, order_date), on=["store_id", "ingredient_id"], how="left")
    base["loss_factor"] = base["loss_factor"].fillna(1.0)

    week = base[WD_COLS].to_numpy(dtype=float) * base["loss_factor"].to_numpy(dtype=float)[:, None]
    days_ahead = (base["lead_time_days"].to_numpy(dtype=float) + horizon_days).astype(int)
    demand = np.zeros(len(base))
    for k in range(1, int(days_ahead.max()) + 1 if len(base) else 1):
        weekday = (order_date + pd.Timedelta(days=k)).dayofweek
        demand += np.where(days_ahead >= k, week[:, weekday], 0.0)
    daily_mean = week.mean(axis=1) if len(base) else np.array([])

    base["daily_forecast"] = daily_mean
    base["demand"] = demand
    base["safety"] = daily_mean  # страховой запас = расход за 1 день
    base["target"] = base["demand"] + base["safety"]
    base["to_order"] = [order_quantity(d, s, st, p) for d, s, st, p
                        in zip(base["demand"], base["safety"], base["stock"], base["pack_size"])]
    with np.errstate(divide="ignore", invalid="ignore"):
        base["coverage_days"] = np.where(daily_mean > 0, base["stock"] / np.where(daily_mean > 0, daily_mean, 1), np.nan)
    base["waste_risk"] = [has_waste_risk(st, m, sl) for st, m, sl
                          in zip(base["stock"], daily_mean, base["shelf_life_days"])]
    base["order_cost"] = base["to_order"] * base["price_per_unit"]

    cols = ["store_id", "ingredient_id", "name", "unit", "supplier", "stock", "daily_forecast", "coverage_days",
            "demand", "safety", "target", "to_order", "order_cost", "shelf_life_days", "waste_risk", "loss_factor"]
    return base[cols].sort_values(["supplier", "to_order"], ascending=[True, False]).reset_index(drop=True)
