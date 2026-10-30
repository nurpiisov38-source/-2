import pandas as pd
import pytest

from src import data_loader, demo_data, validators


@pytest.fixture(scope="module")
def tables():
    return demo_data.generate_demo_data(weeks=4)


def copy_of(tables):
    return {k: v.copy() for k, v in tables.items()}


def test_valid_demo_data_has_no_errors(tables):
    assert validators.validate_tables(tables) == []


def test_missing_column(tables):  # Т-12
    t = copy_of(tables)
    t["sales"] = t["sales"].drop(columns=["qty"])
    errors = validators.validate_tables(t)
    assert any("sales.csv" in e and "qty" in e for e in errors)


def test_negative_value_reports_row_number(tables):  # Т-13
    t = copy_of(tables)
    t["sales"].loc[2, "qty"] = -5
    errors = validators.validate_tables(t)
    assert any("отрицательные" in e and "строки 4" in e for e in errors)


def test_text_in_numeric_column(tables):
    t = copy_of(tables)
    t["ingredients"]["price_per_unit"] = t["ingredients"]["price_per_unit"].astype(object)
    t["ingredients"].loc[0, "price_per_unit"] = "дорого"
    assert any("не числа" in e for e in validators.validate_tables(t))


def test_unknown_ingredient_in_recipe(tables):  # Т-14
    t = copy_of(tables)
    t["recipe_items"].loc[0, "ingredient_id"] = "I99"
    errors = validators.validate_tables(t)
    assert any("I99" in e and "ingredients.csv" in e for e in errors)


def test_bad_date(tables):
    t = copy_of(tables)
    t["sales"]["date"] = t["sales"]["date"].astype(str)
    t["sales"].loc[0, "date"] = "31/12/2026"
    assert any("неверная дата" in e for e in validators.validate_tables(t))


def test_duplicate_inventory_record(tables):
    t = copy_of(tables)
    t["inventory"] = pd.concat([t["inventory"], t["inventory"].iloc[[0]]], ignore_index=True)
    assert any("повторяющиеся" in e for e in validators.validate_tables(t))


def test_zero_pack_size(tables):
    t = copy_of(tables)
    t["ingredients"].loc[0, "pack_size"] = 0
    assert any("pack_size" in e for e in validators.validate_tables(t))


def test_single_inventory_date_is_error(tables):
    t = copy_of(tables)
    first_date = t["inventory"]["date"].min()
    t["inventory"] = t["inventory"][t["inventory"]["date"] == first_date]
    assert any("две даты" in e for e in validators.validate_tables(t))


def test_roundtrip_through_csv(tables, tmp_path):
    data_loader.save_tables(tables, tmp_path)
    loaded, errors = data_loader.load_tables(data_loader.sources_from_folder(tmp_path))
    assert errors == []
    assert len(loaded["sales"]) == len(tables["sales"])
    assert str(loaded["sales"]["date"].dtype).startswith("datetime64")


def test_missing_file_is_reported(tmp_path):
    loaded, errors = data_loader.load_tables({})
    assert loaded is None and any("Нет файла" in e for e in errors)
