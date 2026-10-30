"""Проверка входных CSV-файлов.

Перед расчётами убеждаемся, что файлы корректны, и если нет, возвращаем понятные
сообщения с номерами строк (а не падаем с ошибкой pandas).
Номер строки считается как в Excel: заголовок = строка 1, первые данные = строка 2.
"""
from __future__ import annotations

import pandas as pd

# Описание каждого файла: обязательные колонки, числовые, даты, ключ уникальности.
SCHEMAS: dict[str, dict] = {
    "stores": {"columns": ["store_id", "name"], "numeric": [], "dates": [], "key": ["store_id"]},
    "ingredients": {
        "columns": ["ingredient_id", "name", "unit", "price_per_unit", "supplier",
                    "lead_time_days", "shelf_life_days", "pack_size"],
        "numeric": ["price_per_unit", "lead_time_days", "shelf_life_days", "pack_size"],
        "dates": [], "key": ["ingredient_id"],
    },
    "dishes": {"columns": ["dish_id", "name", "price"], "numeric": ["price"], "dates": [], "key": ["dish_id"]},
    "recipe_items": {
        "columns": ["dish_id", "ingredient_id", "qty_per_dish"], "numeric": ["qty_per_dish"],
        "dates": [], "key": ["dish_id", "ingredient_id"],
    },
    "sales": {
        "columns": ["date", "store_id", "dish_id", "qty"], "numeric": ["qty"],
        "dates": ["date"], "key": ["date", "store_id", "dish_id"],
    },
    # Поставок в один день может быть несколько, поэтому ключа уникальности нет.
    "deliveries": {
        "columns": ["date", "store_id", "ingredient_id", "qty"], "numeric": ["qty"],
        "dates": ["date"], "key": [],
    },
    "inventory": {
        "columns": ["date", "store_id", "ingredient_id", "qty"], "numeric": ["qty"],
        "dates": ["date"], "key": ["date", "store_id", "ingredient_id"],
    },
}

# (таблица, колонка, справочник, колонка справочника): значение должно существовать в справочнике.
REFERENCES = [
    ("recipe_items", "dish_id", "dishes", "dish_id"),
    ("recipe_items", "ingredient_id", "ingredients", "ingredient_id"),
    ("sales", "store_id", "stores", "store_id"),
    ("sales", "dish_id", "dishes", "dish_id"),
    ("deliveries", "store_id", "stores", "store_id"),
    ("deliveries", "ingredient_id", "ingredients", "ingredient_id"),
    ("inventory", "store_id", "stores", "store_id"),
    ("inventory", "ingredient_id", "ingredients", "ingredient_id"),
]

MAX_ROWS_SHOWN = 5


def _rows_text(mask: pd.Series) -> str:
    """Текст «строки 3, 5 и ещё 2» по булевой маске."""
    bad = mask[mask].index
    shown = ", ".join(str(int(i) + 2) for i in bad[:MAX_ROWS_SHOWN])
    extra = len(bad) - MAX_ROWS_SHOWN
    return f"строки {shown}" + (f" и ещё {extra}" if extra > 0 else "")


def _check_table(name: str, df: pd.DataFrame, schema: dict) -> list[str]:
    errors: list[str] = []
    f = f"{name}.csv"
    cols = schema["columns"]

    empty = df[cols].isna().any(axis=1)
    if empty.any():
        errors.append(f"{f}: пустые значения ({_rows_text(empty)})")

    for col in schema["numeric"]:
        num = pd.to_numeric(df[col], errors="coerce")
        not_number = num.isna() & df[col].notna()
        if not_number.any():
            errors.append(f"{f}: в колонке {col} не числа ({_rows_text(not_number)})")
        negative = num < 0
        if negative.any():
            errors.append(f"{f}: в колонке {col} отрицательные значения ({_rows_text(negative)})")
        if name == "ingredients" and col == "pack_size":
            zero = num <= 0
            if zero.any():
                errors.append(f"{f}: pack_size должен быть больше нуля ({_rows_text(zero)})")

    for col in schema["dates"]:
        parsed = pd.to_datetime(df[col], format="ISO8601", errors="coerce")
        bad_date = parsed.isna() & df[col].notna()
        if bad_date.any():
            errors.append(f"{f}: в колонке {col} неверная дата, нужен формат ГГГГ-ММ-ДД ({_rows_text(bad_date)})")

    if schema["key"]:
        dup = df.duplicated(subset=schema["key"], keep="first")
        if dup.any():
            errors.append(f"{f}: повторяющиеся записи по ({', '.join(schema['key'])}) ({_rows_text(dup)})")
    return errors


def _check_references(tables: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    for table, col, ref_table, ref_col in REFERENCES:
        known = set(tables[ref_table][ref_col].astype(str))
        unknown = sorted(set(tables[table][col].astype(str)) - known)
        if unknown:
            errors.append(f"{table}.csv: значения {col} нет в {ref_table}.csv: {', '.join(unknown[:5])}")
    return errors


def validate_tables(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Возвращает список ошибок. Пустой список = данные корректны."""
    errors: list[str] = []
    for name, schema in SCHEMAS.items():
        df = tables.get(name)
        f = f"{name}.csv"
        if df is None:
            errors.append(f"Нет файла {f}")
            continue
        missing = [c for c in schema["columns"] if c not in df.columns]
        if missing:
            errors.append(f"{f}: нет колонок {', '.join(missing)}")
            continue
        if df.empty:
            errors.append(f"{f}: файл пустой")
            continue
        errors += _check_table(name, df, schema)
    if errors:
        return errors
    errors += _check_references(tables)
    dates = pd.to_datetime(tables["inventory"]["date"], format="ISO8601")
    if dates.nunique() < 2:
        errors.append("inventory.csv: нужно минимум две даты инвентаризации, иначе нечего сравнивать")
    return errors
