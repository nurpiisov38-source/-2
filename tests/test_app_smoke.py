"""Дымовой тест интерфейса: приложение стартует, грузит демо-данные и не падает."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_starts_without_data():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("демо-данные" in i.value for i in at.info)


def test_app_loads_demo_and_switches_store():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Данные", "Техкарты", "Расход", "ABC/XYZ", "Заказ", "Дашборд"]
    at.sidebar.selectbox[0].set_value("S3").run()
    assert not at.exception and not at.error
