"""Генератор СИНТЕТИЧЕСКИХ демо-данных для сети шаурмы «Сочный вертел».

ВАЖНО: все цифры (цены, нормы, продажи, потери) выдуманы для демонстрации метода.
Это не данные реальной компании.

Как устроена симуляция (чтобы можно было объяснить на защите):
1. Каждый день для каждой точки генерируются продажи блюд (Пуассон, с поправкой на день недели и «размер» точки).
2. Продажи × техкарты = теоретический расход ингредиентов.
3. Реальный расход чуть больше теоретического (порча, переборы порций). У одних продуктов потери
   больше (соусы, овощи), у других меньше. Две «проблемные пары» точка+ингредиент сделаны специально.
4. Склад пополняется поставками по простому правилу; если товара не хватило, расход ограничен остатком.
5. Дважды в неделю (понедельник и четверг, в конце дня) фиксируется остаток, это и есть инвентаризация.
6. Несколько записей инвентаризации искажены (ошибка пересчёта), чтобы показать пометку «проверить пересчёт».
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import save_tables

ING_COLS = ["ingredient_id", "name", "unit", "price_per_unit", "supplier",
            "lead_time_days", "shelf_life_days", "pack_size"]

# id, название, ед., цена за единицу (₽), поставщик, срок поставки (дн.), срок годности (дн.), кратность упаковки
INGREDIENTS = [
    ("I01", "Курица (бедро)", "g", 0.32, "Мясокомбинат А", 1, 3, 5000),
    ("I02", "Говядина", "g", 0.62, "Мясокомбинат А", 1, 3, 5000),
    ("I03", "Баранина", "g", 0.78, "Мясокомбинат А", 1, 3, 5000),
    ("I04", "Индейка", "g", 0.45, "Мясокомбинат А", 1, 3, 5000),
    ("I05", "Лаваш", "pcs", 9.0, "Пекарня Б", 1, 4, 50),
    ("I06", "Пита", "pcs", 11.0, "Пекарня Б", 1, 4, 50),
    ("I07", "Огурцы свежие", "g", 0.09, "Овощебаза В", 1, 5, 5000),
    ("I08", "Помидоры", "g", 0.12, "Овощебаза В", 1, 5, 5000),
    ("I09", "Капуста белая", "g", 0.04, "Овощебаза В", 1, 10, 5000),
    ("I10", "Капуста красная", "g", 0.06, "Овощебаза В", 1, 10, 5000),
    ("I11", "Лук красный", "g", 0.03, "Овощебаза В", 1, 14, 5000),
    ("I12", "Морковь по-корейски", "g", 0.14, "Овощебаза В", 1, 5, 2000),
    ("I13", "Огурцы маринованные", "g", 0.11, "Овощебаза В", 1, 30, 3000),
    ("I14", "Картофель фри (заморозка)", "g", 0.15, "Заморозка Г", 2, 90, 5000),
    ("I15", "Зелень", "g", 0.40, "Овощебаза В", 1, 3, 500),
    ("I16", "Халапеньо", "g", 0.30, "Овощебаза В", 2, 30, 1000),
    ("I17", "Шампиньоны", "g", 0.25, "Овощебаза В", 1, 5, 2000),
    ("I18", "Сыр моцарелла", "g", 0.55, "Молочный Д", 2, 14, 2000),
    ("I19", "Сыр фета", "g", 0.65, "Молочный Д", 2, 14, 1000),
    ("I20", "Сметана", "g", 0.18, "Молочный Д", 2, 10, 5000),
    ("I21", "Соус чесночный", "ml", 0.20, "Соусы Е", 2, 21, 5000),
    ("I22", "Соус томатный", "ml", 0.12, "Соусы Е", 2, 30, 5000),
    ("I23", "Соус сырный", "ml", 0.25, "Соусы Е", 2, 21, 5000),
    ("I24", "Соус острый", "ml", 0.22, "Соусы Е", 2, 30, 3000),
    ("I25", "Кетчуп", "ml", 0.10, "Соусы Е", 3, 90, 5000),
    ("I26", "Майонез", "ml", 0.13, "Соусы Е", 3, 60, 5000),
    ("I27", "Масло подсолнечное", "ml", 0.12, "Бакалея Ж", 3, 180, 5000),
    ("I28", "Специи для шаурмы", "g", 0.90, "Бакалея Ж", 3, 180, 1000),
    ("I29", "Соль", "g", 0.03, "Бакалея Ж", 3, 365, 5000),
    ("I30", "Кола 0,33 л", "pcs", 45.0, "Напитки З", 2, 180, 24),
    ("I31", "Лимонад 0,33 л", "pcs", 42.0, "Напитки З", 2, 180, 24),
    ("I32", "Морс", "ml", 0.08, "Напитки З", 2, 30, 10000),
    ("I33", "Фольга для обёртки", "pcs", 2.2, "Упаковка И", 3, 999, 1000),
    ("I34", "Бумажный пакет", "pcs", 3.5, "Упаковка И", 3, 999, 500),
    ("I35", "Салфетка", "pcs", 0.4, "Упаковка И", 3, 999, 5000),
    ("I36", "Контейнер для фри", "pcs", 6.0, "Упаковка И", 3, 999, 500),
    ("I37", "Стакан 0,3 л", "pcs", 5.0, "Упаковка И", 3, 999, 1000),
]

DISHES = [
    ("D01", "Шаурма с курицей", 250.0),
    ("D02", "Шаурма с говядиной", 290.0),
    ("D03", "Шаурма сырная", 270.0),
    ("D04", "Шаурма острая", 270.0),
    ("D05", "Шаурма с бараниной", 320.0),
    ("D06", "Шаурма в пите (индейка)", 280.0),
    ("D07", "Картофель фри", 120.0),
    ("D08", "Кола 0,33 л", 110.0),
    ("D09", "Лимонад 0,33 л", 110.0),
    ("D10", "Морс 0,25 л", 90.0),
]

# Техкарты: блюдо -> {ингредиент: количество на 1 порцию}. Нормы синтетические.
RECIPES = {
    "D01": {"I05": 1, "I01": 130, "I07": 40, "I08": 40, "I09": 30, "I13": 20, "I21": 35, "I28": 2, "I33": 1, "I34": 1, "I35": 2},
    "D02": {"I05": 1, "I02": 130, "I07": 40, "I08": 40, "I11": 15, "I12": 20, "I22": 30, "I21": 20, "I28": 2, "I33": 1, "I34": 1, "I35": 2},
    "D03": {"I05": 1, "I01": 120, "I18": 40, "I08": 40, "I09": 30, "I23": 35, "I28": 2, "I33": 1, "I34": 1, "I35": 2},
    "D04": {"I05": 1, "I01": 120, "I16": 15, "I12": 30, "I07": 30, "I24": 30, "I21": 20, "I28": 3, "I33": 1, "I34": 1, "I35": 2},
    "D05": {"I05": 1, "I03": 130, "I08": 40, "I11": 20, "I10": 30, "I19": 25, "I20": 25, "I28": 2, "I33": 1, "I34": 1, "I35": 2},
    "D06": {"I06": 1, "I04": 120, "I07": 40, "I08": 40, "I15": 5, "I17": 30, "I21": 30, "I28": 2, "I33": 1, "I34": 1, "I35": 2},
    "D07": {"I14": 200, "I27": 15, "I29": 2, "I25": 25, "I36": 1, "I35": 1},
    "D08": {"I30": 1},
    "D09": {"I31": 1},
    "D10": {"I32": 250, "I37": 1},
}

POPULARITY = {"D01": 60, "D02": 40, "D03": 30, "D04": 25, "D05": 15,
              "D06": 20, "D07": 45, "D08": 40, "D09": 25, "D10": 18}  # продаж в день на «среднюю» точку
STORE_FACTORS = [1.2, 1.0, 0.9, 0.8, 1.1]                       # «размер» точек
WEEKDAY_FACTORS = np.array([0.8, 0.85, 0.9, 1.0, 1.3, 1.4, 1.2])  # пн...вс, выходные продают больше
ANOMALIES = {("S3", "I02"): 0.15, ("S4", "I21"): 0.10}          # дополнительные потери: точка + ингредиент


def _base_loss(ing_id: str) -> float:
    """Доля потерь сверх нормы техкарты."""
    if ing_id == "I01":
        return 0.07
    if ing_id in ("I02", "I03", "I04"):
        return 0.04
    if ing_id in {"I07", "I08", "I09", "I10", "I11", "I12", "I15", "I17"}:
        return 0.08
    if ing_id in {"I21", "I22", "I23", "I24", "I25", "I26"}:
        return 0.12
    if ing_id in {"I33", "I34", "I35", "I36", "I37"}:
        return 0.05
    if ing_id in {"I30", "I31"}:
        return 0.0
    return 0.02


def generate_demo_data(weeks: int = 26, end_date: str = "2026-10-25", seed: int = 42,
                       recount_errors: int = 8) -> dict[str, pd.DataFrame]:
    """Создаёт словарь таблиц (как после чтения CSV). Результат повторяется при том же seed."""
    rng = np.random.default_rng(seed)

    ingredients = pd.DataFrame(INGREDIENTS, columns=ING_COLS)
    stores = pd.DataFrame({"store_id": [f"S{i + 1}" for i in range(len(STORE_FACTORS))],
                           "name": [f"Точка {i + 1}" for i in range(len(STORE_FACTORS))]})
    dishes = pd.DataFrame(DISHES, columns=["dish_id", "name", "price"])
    recipe_items = pd.DataFrame([(d, i, q) for d, r in RECIPES.items() for i, q in r.items()],
                                columns=["dish_id", "ingredient_id", "qty_per_dish"])

    store_ids = list(stores["store_id"])
    ing_ids = list(ingredients["ingredient_id"])
    dish_ids = list(dishes["dish_id"])
    n_st, n_ing = len(store_ids), len(ing_ids)

    recipe = np.zeros((len(dish_ids), n_ing))  # матрица «блюдо × ингредиент»
    for d, i, q in recipe_items.itertuples(index=False):
        recipe[dish_ids.index(d), ing_ids.index(i)] = q

    base = np.array([POPULARITY[d] for d in dish_ids], dtype=float)
    store_f = np.array(STORE_FACTORS)
    loss = np.array([[_base_loss(i) + ANOMALIES.get((s, i), 0.0) for i in ing_ids] for s in store_ids])
    lead = ingredients["lead_time_days"].to_numpy()
    interval = np.where(lead <= 1, 1, np.where(lead == 2, 2, 3))  # как часто приходят поставки
    pack = ingredients["pack_size"].to_numpy(dtype=float)

    mean_cons = ((store_f[:, None] * base[None, :]) @ recipe) * WEEKDAY_FACTORS.mean() * (1 + loss)
    stock = np.round(mean_cons * (interval + 1))

    dates = pd.date_range(end=end_date, periods=weeks * 7)
    sales_rows, deliv_rows, inv_rows = [], [], []
    for k, day in enumerate(dates):
        lam = store_f[:, None] * base[None, :] * WEEKDAY_FACTORS[day.dayofweek]
        sold = rng.poisson(lam)
        theory = sold @ recipe
        desired = np.round(theory * (1 + loss) * rng.normal(1.0, 0.02, theory.shape))

        due = (k % interval) == 0
        target = mean_cons * (interval + 1) * rng.uniform(0.55, 1.3, mean_cons.shape)
        need = np.where(due[None, :], np.maximum(0.0, target - stock), 0.0)
        delivered = np.maximum(np.ceil(need / pack[None, :] - 1e-9) * pack[None, :], 0.0)

        stock = stock + delivered
        used = np.minimum(desired, stock)  # если товара не хватило, расход ограничен остатком
        stock = stock - used

        for s, d in zip(*np.nonzero(sold)):
            sales_rows.append((day, store_ids[s], dish_ids[d], int(sold[s, d])))
        for s, i in zip(*np.nonzero(delivered)):
            deliv_rows.append((day, store_ids[s], ing_ids[i], int(delivered[s, i])))
        if day.dayofweek in (0, 3):  # инвентаризация в понедельник и четверг, в конце дня
            for s in range(n_st):
                for i in range(n_ing):
                    inv_rows.append((day, store_ids[s], ing_ids[i], int(stock[s, i])))

    sales = pd.DataFrame(sales_rows, columns=["date", "store_id", "dish_id", "qty"])
    deliveries = pd.DataFrame(deliv_rows, columns=["date", "store_id", "ingredient_id", "qty"])
    inventory = pd.DataFrame(inv_rows, columns=["date", "store_id", "ingredient_id", "qty"])

    if recount_errors > 0:  # имитируем ошибки пересчёта
        idx = rng.choice(len(inventory), size=min(recount_errors, len(inventory)), replace=False)
        factors = rng.choice([0.6, 1.5], size=len(idx))
        inventory.loc[idx, "qty"] = (inventory.loc[idx, "qty"] * factors).round().astype(int)

    return {"stores": stores, "ingredients": ingredients, "dishes": dishes,
            "recipe_items": recipe_items, "sales": sales, "deliveries": deliveries, "inventory": inventory}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сгенерировать синтетические демо-CSV")
    default_out = Path(__file__).resolve().parents[1] / "data" / "demo"
    parser.add_argument("--out", default=str(default_out), help="папка для CSV")
    parser.add_argument("--weeks", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    save_tables(generate_demo_data(weeks=args.weeks, seed=args.seed), args.out)
    print(f"Демо-данные (синтетика) сохранены в {args.out}")
