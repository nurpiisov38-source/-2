"""Чтение CSV, проверка и приведение типов."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validators import SCHEMAS, validate_tables


def sources_from_folder(folder) -> dict:
    """Находит файлы <имя>.csv в папке."""
    folder = Path(folder)
    return {name: folder / f"{name}.csv" for name in SCHEMAS if (folder / f"{name}.csv").exists()}


def normalize_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Приводит типы: идентификаторы в строки, числа во float, даты в datetime."""
    out = {}
    for name, schema in SCHEMAS.items():
        df = tables[name][schema["columns"]].copy()
        for col in df.columns:
            if col.endswith("_id"):
                df[col] = df[col].astype(str).str.strip()
        for col in schema["numeric"]:
            df[col] = pd.to_numeric(df[col]).astype(float)
        for col in schema["dates"]:
            df[col] = pd.to_datetime(df[col], format="ISO8601").dt.normalize().astype("datetime64[ns]")
        out[name] = df.reset_index(drop=True)
    return out


def load_tables(sources: dict) -> tuple[dict | None, list[str]]:
    """Читает и проверяет файлы. sources: имя таблицы -> путь или файловый объект.

    Возвращает (таблицы, []) при успехе или (None, список ошибок).
    """
    raw: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for name in SCHEMAS:
        source = sources.get(name)
        if source is None:
            errors.append(f"Нет файла {name}.csv")
            continue
        try:
            raw[name] = pd.read_csv(source, encoding="utf-8-sig")
        except Exception as exc:  # битый файл, пустой файл, не тот формат
            errors.append(f"{name}.csv: не удалось прочитать файл ({exc})")
    if errors:
        return None, errors
    errors = validate_tables(raw)
    if errors:
        return None, errors
    return normalize_tables(raw), []


def save_tables(tables: dict[str, pd.DataFrame], folder) -> None:
    """Сохраняет таблицы в CSV (используется для демо-данных)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8")
