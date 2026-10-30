"""KPI и себестоимость блюд."""
from __future__ import annotations

import numpy as np
import pandas as pd


def food_cost_pct(cost: float, revenue: float) -> float:
    """Food cost, %: стоимость израсходованных ингредиентов / выручка × 100."""
    return float(cost / revenue * 100) if revenue and revenue > 0 else float("nan")


def dish_costs(dishes: pd.DataFrame, recipe_items: pd.DataFrame, ingredients: pd.DataFrame) -> pd.DataFrame:
    """Себестоимость блюда по техкарте и текущим закупочным ценам."""
    merged = recipe_items.merge(ingredients[["ingredient_id", "price_per_unit"]], on="ingredient_id")
    merged["cost"] = merged["qty_per_dish"] * merged["price_per_unit"]
    cost = merged.groupby("dish_id", as_index=False)["cost"].sum().rename(columns={"cost": "unit_cost"})
    out = dishes.merge(cost, on="dish_id", how="left")
    out["unit_cost"] = out["unit_cost"].fillna(0.0)
    out["margin"] = out["price"] - out["unit_cost"]
    out["food_cost_pct"] = out["unit_cost"] / out["price"] * 100
    return out


def compute_kpis(periods: pd.DataFrame, sales: pd.DataFrame, dishes: pd.DataFrame) -> dict:
    """KPI по выбранным периодам (уже отфильтрованным по точке и датам)."""
    empty = {"theory_cost": 0.0, "fact_cost": 0.0, "deviation_cost": 0.0, "deviation_pct": float("nan"),
             "loss_rub": 0.0, "revenue": 0.0, "food_cost_pct": float("nan"), "stockouts": 0}
    if periods.empty:
        return empty

    theory_cost = float(periods["theory_cost"].sum())
    fact_cost = float(periods["fact_cost"].sum())
    deviation_cost = fact_cost - theory_cost
    # Потери: перерасход по каждой паре «точка + ингредиент», недорасход не вычитаем.
    per_pair = periods.groupby(["store_id", "ingredient_id"])["deviation_cost"].sum()

    stores = periods["store_id"].unique()
    window = sales[sales["store_id"].isin(stores)
                   & (sales["date"] > periods["start_date"].min())
                   & (sales["date"] <= periods["end_date"].max())]
    window = window.merge(dishes[["dish_id", "price"]], on="dish_id")
    revenue = float((window["qty"] * window["price"]).sum())

    return {
        "theory_cost": theory_cost,
        "fact_cost": fact_cost,
        "deviation_cost": deviation_cost,
        "deviation_pct": deviation_cost / theory_cost * 100 if theory_cost > 0 else float("nan"),
        "loss_rub": float(per_pair.clip(lower=0).sum()),
        "revenue": revenue,
        "food_cost_pct": food_cost_pct(fact_cost, revenue),
        # конец периода с нулевым остатком при наличии спроса = ингредиент заканчивался
        "stockouts": int(((periods["stock_end"] <= 0) & (periods["theory"] > 0)).sum()),
    }
