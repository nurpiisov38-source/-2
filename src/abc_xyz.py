"""ABC/XYZ-анализ ингредиентов.

ABC: насколько ингредиент важен по деньгам (стоимость расхода).
XYZ: насколько расход стабилен (коэффициент вариации дневного расхода = разброс / среднее).
Перед расчётом XYZ расход очищается от недельной сезонности (пятница всегда выше вторника),
иначе «предсказуемая» разница между днями недели выглядела бы как нестабильность.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGIES = {
    "AX": "Главный товар, расход стабилен: заказывать часто, держать минимальный запас.",
    "AY": "Главный товар, расход колеблется: заказ по прогнозу дня недели, контролировать каждый день.",
    "AZ": "Дорогой и непредсказуемый: небольшие частые заказы, высокий приоритет контроля.",
    "BX": "Средняя важность, стабильно: автоматический дозаказ по точке заказа.",
    "BY": "Средняя важность, колеблется: заказ по прогнозу, периодическая проверка.",
    "BZ": "Средняя важность, непредсказуемо: заказ под потребность, без больших запасов.",
    "CX": "Дешёвый и стабильный: заказывать с запасом большими партиями.",
    "CY": "Дешёвый, колеблется: простой запас, проверка раз в неделю.",
    "CZ": "Дешёвый и редкий: заказ по факту нехватки, не контролировать детально.",
}


def abc_classification(values: pd.Series, a: float = 0.80, b: float = 0.95) -> pd.Series:
    """Классы A/B/C. Ингредиент попадает в A, если до него накопленная доля стоимости ещё меньше a."""
    vals = values.fillna(0).clip(lower=0)
    total = vals.sum()
    if total <= 0:
        return pd.Series("C", index=values.index)
    ordered = vals.sort_values(ascending=False)
    share_before = (ordered.cumsum() - ordered) / total
    cls = pd.Series(np.where(share_before < a - 1e-12, "A", np.where(share_before < b - 1e-12, "B", "C")),
                    index=ordered.index)
    cls[ordered <= 0] = "C"
    return cls.reindex(values.index)


def xyz_classification(demand: pd.DataFrame, x: float = 0.10, y: float = 0.25) -> pd.DataFrame:
    """Классы X/Y/Z. demand: строки = дни, колонки = ингредиенты, значения = расход за день."""
    mean = demand.mean()
    std = demand.std(ddof=0)
    cv = std / mean.where(mean > 0)  # если среднего нет, cv не определён
    cls = pd.Series(np.where(cv <= x, "X", np.where(cv <= y, "Y", "Z")), index=cv.index)
    return pd.DataFrame({"cv": cv, "xyz": cls})


def demand_matrix(daily: pd.DataFrame, date_from, date_to, deseasonalize: bool = True) -> pd.DataFrame:
    """Таблица «день × ингредиент» с нулями в днях без расхода.

    deseasonalize=True: каждый день делится на средний расход этого дня недели, так остаётся только
    непредсказуемая часть колебаний.
    """
    pivot = daily.pivot_table(index="date", columns="ingredient_id", values="theory", aggfunc="sum")
    pivot = pivot.reindex(pd.date_range(date_from, date_to)).fillna(0.0)
    if deseasonalize:
        weekday_mean = pivot.groupby(pivot.index.dayofweek).transform("mean")
        pivot = (pivot / weekday_mean.where(weekday_mean > 0)).fillna(0.0)
    return pivot


def classify_ingredients(cost: pd.Series, demand: pd.DataFrame, ingredients: pd.DataFrame,
                         a=0.80, b=0.95, x=0.10, y=0.25) -> pd.DataFrame:
    """Сводная таблица ABC + XYZ. cost: индекс = ingredient_id, значения = стоимость расхода."""
    abc = abc_classification(cost, a, b)
    xyz = xyz_classification(demand.reindex(columns=cost.index).fillna(0.0), x, y)
    out = pd.DataFrame({"cost": cost, "abc": abc})
    out = out.join(xyz)
    total = out["cost"].sum()
    out["share"] = out["cost"] / total * 100 if total > 0 else 0.0
    out["abc_xyz"] = out["abc"] + out["xyz"]
    out["strategy"] = out["abc_xyz"].map(STRATEGIES)
    out.index.name = "ingredient_id"
    out = out.reset_index().merge(ingredients[["ingredient_id", "name", "unit"]], on="ingredient_id", how="left")
    return out.sort_values("cost", ascending=False).reset_index(drop=True)


def class_matrix(classified: pd.DataFrame, value: str = "count") -> pd.DataFrame:
    """Матрица 3×3: число ингредиентов (count) или доля стоимости, % (share)."""
    if value == "count":
        table = pd.crosstab(classified["abc"], classified["xyz"])
    else:
        table = classified.pivot_table(index="abc", columns="xyz", values="share", aggfunc="sum")
    return table.reindex(index=list("ABC"), columns=list("XYZ")).fillna(0)
