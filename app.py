"""Stock Control для «Сочного вертела»: запуск командой `streamlit run app.py`.

Учебный проект. Все данные по умолчанию синтетические.
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src import consumption, data_loader, demo_data, ordering, ui_pages
from src.validators import SCHEMAS

st.set_page_config(page_title="Stock Control — Сочный вертел", page_icon="🌯", layout="wide")


def set_tables(tables: dict, synthetic: bool) -> None:
    """Считает периоды расхода один раз и кладёт всё в память сессии."""
    daily = consumption.daily_theory(tables["sales"], tables["recipe_items"])
    periods = consumption.calc_consumption(tables["sales"], tables["recipe_items"], tables["deliveries"],
                                           tables["inventory"], tables["ingredients"], theory_daily=daily)
    periods = consumption.flag_recount(periods)
    st.session_state.update(tables=tables, daily=daily, periods=periods, synthetic=synthetic, errors=[])


def load_demo() -> None:
    set_tables(demo_data.generate_demo_data(), synthetic=True)


def load_uploads(files: dict) -> None:
    missing = [f"Нет файла {name}.csv" for name, f in files.items() if f is None]
    if missing:
        st.session_state["errors"] = missing
        return
    sources = {name: io.BytesIO(f.getvalue()) for name, f in files.items()}
    tables, errors = data_loader.load_tables(sources)
    if errors:
        st.session_state["errors"] = errors
        return
    try:
        set_tables(tables, synthetic=False)
    except Exception as exc:  # неожиданная проблема в данных: показываем, а не роняем приложение
        st.session_state["errors"] = [f"Не удалось выполнить расчёт: {exc}"]


# ------------------------------------------------------------------ боковая панель: данные
with st.sidebar:
    st.header("Данные")
    if st.button("Загрузить демо-данные (синтетика)"):
        load_demo()
    with st.expander("Загрузить свои CSV"):
        uploads = {name: st.file_uploader(f"{name}.csv", type="csv", key=f"up_{name}") for name in SCHEMAS}
        if st.button("Проверить и загрузить"):
            load_uploads(uploads)

st.title("🌯 Stock Control — «Сочный вертел»")

errors = st.session_state.get("errors", [])
if errors:
    st.error("Файлы не прошли проверку:\n\n" + "\n".join(f"- {e}" for e in errors))

if "tables" not in st.session_state:
    st.info("Нажмите «Загрузить демо-данные» слева или загрузите свои CSV. Шаблоны файлов можно скачать на вкладке «Данные».")
    st.stop()

tables = st.session_state["tables"]
daily = st.session_state["daily"]
all_periods = st.session_state["periods"]
if all_periods.empty:
    st.error("Не удалось построить периоды между инвентаризациями. Проверьте файл inventory.csv.")
    st.stop()

# ------------------------------------------------------------------ боковая панель: фильтры и параметры
store_names = dict(zip(tables["stores"]["store_id"], tables["stores"]["name"]))
with st.sidebar:
    st.header("Фильтры")
    store_id = st.selectbox("Точка", [None] + list(store_names), format_func=lambda s: "Вся сеть" if s is None else store_names[s])
    min_d, max_d = all_periods["end_date"].min().date(), all_periods["end_date"].max().date()
    date_range = st.date_input("Период (по дате инвентаризации)", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    st.header("Параметры")
    threshold = st.number_input("Порог отклонения, %", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    abc_a = st.slider("Граница A, % накопленной стоимости", 50, 95, 80)
    abc_b = st.slider("Граница B, %", 85, 99, 95)
    xyz_x = st.slider("Граница X (вариация), %", 1, 30, 10)
    xyz_y = st.slider("Граница Y (вариация), %", 10, 80, 25)
    horizon = st.number_input("Горизонт заказа, дней", min_value=1, max_value=7, value=2)
    inv_dates = sorted(tables["inventory"]["date"].unique(), reverse=True)
    order_date = st.selectbox("Дата заказа (инвентаризация)", inv_dates, format_func=lambda d: f"{pd.Timestamp(d):%d.%m.%Y}")

if len(date_range) != 2:
    st.info("Выберите начальную и конечную дату периода.")
    st.stop()
if abc_b <= abc_a or xyz_y <= xyz_x:
    st.error("Границы заданы неверно: B должна быть больше A, а Y больше X.")
    st.stop()

date_from, date_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
mask = (all_periods["end_date"] >= date_from) & (all_periods["end_date"] <= date_to)
daily_mask = (daily["date"] >= date_from) & (daily["date"] <= date_to)
if store_id is not None:
    mask &= all_periods["store_id"] == store_id
    daily_mask &= daily["store_id"] == store_id
periods, daily_f = all_periods[mask], daily[daily_mask]

order = ordering.build_order_list(tables["ingredients"], tables["inventory"], daily, all_periods,
                                  order_date, horizon_days=int(horizon), store_id=store_id)

# ------------------------------------------------------------------ вкладки
tabs = st.tabs(["Данные", "Техкарты", "Расход", "ABC/XYZ", "Заказ", "Дашборд"])
with tabs[0]:
    ui_pages.render_data_page(tables, st.session_state["synthetic"])
with tabs[1]:
    ui_pages.render_recipes_page(tables)
with tabs[2]:
    ui_pages.render_consumption_page(periods, tables, threshold, store_id is None)
with tabs[3]:
    ui_pages.render_abc_page(periods, daily_f, tables, date_from, date_to, abc_a, abc_b, xyz_x, xyz_y)
with tabs[4]:
    ui_pages.render_order_page(order, tables, order_date, int(horizon))
with tabs[5]:
    ui_pages.render_dashboard_page(periods, tables, order, store_id is None)
