"""Страницы (вкладки) интерфейса Streamlit. Здесь только показ: все расчёты в других модулях."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import abc_xyz, consumption, kpi
from .validators import SCHEMAS

RU = {
    "ingredient_id": "Код", "name": "Ингредиент", "unit": "Ед.", "theory": "Теория", "fact": "Факт",
    "deviation": "Отклонение", "deviation_pct": "Отклонение, %", "deviation_cost": "Отклонение, ₽",
    "theory_cost": "Теория, ₽", "fact_cost": "Факт, ₽", "store": "Точка", "supplier": "Поставщик",
    "stock": "Остаток", "daily_forecast": "Прогноз/день", "coverage_days": "Покрытие, дн.",
    "demand": "Нужно на период", "safety": "Страховой запас", "target": "Цель", "to_order": "К заказу",
    "order_cost": "Сумма, ₽", "shelf_life_days": "Срок годности, дн.", "waste_risk": "Риск списания",
    "abc": "ABC", "xyz": "XYZ", "abc_xyz": "Класс", "cv": "Вариация", "share": "Доля стоимости, %",
    "cost": "Стоимость, ₽", "strategy": "Рекомендация",
}


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig, чтобы Excel открыл кириллицу


def _store_names(tables: dict) -> dict:
    return dict(zip(tables["stores"]["store_id"], tables["stores"]["name"]))


def _highlight(row: pd.Series, column: str, threshold: float) -> list[str]:
    value = row[column]
    bad = pd.notna(value) and value > threshold
    return ["background-color: #fbe3e3; color: #7a1d1d" if bad else "" for _ in row]


# ---------------------------------------------------------------- Данные
def render_data_page(tables: dict, synthetic: bool) -> None:
    if synthetic:
        st.warning("Данные синтетические: цифры выдуманы для демонстрации метода, это не данные компании.")
    cols = st.columns(4)
    cols[0].metric("Точек", len(tables["stores"]))
    cols[1].metric("Ингредиентов", len(tables["ingredients"]))
    cols[2].metric("Блюд", len(tables["dishes"]))
    cols[3].metric("Дат инвентаризации", tables["inventory"]["date"].nunique())
    st.caption(f"Период продаж: {tables['sales']['date'].min():%d.%m.%Y} — {tables['sales']['date'].max():%d.%m.%Y}")
    for name, df in tables.items():
        with st.expander(f"{name}.csv ({len(df)} строк)"):
            st.dataframe(df.head(200), hide_index=True)
            template = ",".join(SCHEMAS[name]["columns"]) + "\n"
            st.download_button(f"Скачать шаблон {name}.csv", template.encode("utf-8-sig"),
                               file_name=f"{name}.csv", key=f"tpl_{name}")


# ---------------------------------------------------------------- Техкарты
def render_recipes_page(tables: dict) -> None:
    costs = kpi.dish_costs(tables["dishes"], tables["recipe_items"], tables["ingredients"])
    st.subheader("Себестоимость блюд")
    show = costs.rename(columns={"dish_id": "Код", "name": "Блюдо", "price": "Цена, ₽", "unit_cost": "Себестоимость, ₽",
                                 "margin": "Маржа, ₽", "food_cost_pct": "Food cost, %"})
    st.dataframe(show.style.format({"Цена, ₽": "{:.0f}", "Себестоимость, ₽": "{:.1f}", "Маржа, ₽": "{:.1f}",
                                    "Food cost, %": "{:.1f}"}), hide_index=True)
    st.subheader("Состав блюда")
    dish = st.selectbox("Блюдо", costs["dish_id"], format_func=lambda d: costs.set_index("dish_id").loc[d, "name"])
    items = (tables["recipe_items"][tables["recipe_items"]["dish_id"] == dish]
             .merge(tables["ingredients"], on="ingredient_id"))
    items["cost"] = items["qty_per_dish"] * items["price_per_unit"]
    st.dataframe(items[["name", "qty_per_dish", "unit", "price_per_unit", "cost"]].rename(columns={
        "name": "Ингредиент", "qty_per_dish": "Норма на порцию", "unit": "Ед.", "price_per_unit": "Цена за ед., ₽",
        "cost": "Стоимость, ₽"}), hide_index=True)


# ---------------------------------------------------------------- Расход
def render_consumption_page(periods: pd.DataFrame, tables: dict, threshold: float, all_network: bool) -> None:
    if periods.empty:
        st.info("За выбранный период нет данных.")
        return
    st.caption("Отклонение = факт − теория. Плюс означает перерасход (потери: порча, переборы порций, возможные хищения). "
               "Красным выделены ингредиенты с отклонением выше порога.")
    by_ing = consumption.summarize(periods, ["ingredient_id", "name", "unit"]).sort_values("deviation_cost", ascending=False)
    table = by_ing[["name", "unit", "theory", "fact", "deviation", "deviation_pct", "deviation_cost"]].rename(columns=RU)
    styled = (table.style.apply(_highlight, column="Отклонение, %", threshold=threshold, axis=1)
              .format({"Теория": "{:,.0f}", "Факт": "{:,.0f}", "Отклонение": "{:+,.0f}",
                       "Отклонение, %": "{:+.1f}", "Отклонение, ₽": "{:+,.0f}"}, na_rep="—"))
    st.dataframe(styled, hide_index=True)

    top = by_ing[by_ing["deviation_cost"] > 0].head(10)
    fig = px.bar(top, x="deviation_cost", y="name", orientation="h", title="Топ-10 ингредиентов по потерям, ₽",
                 labels={"deviation_cost": "Потери, ₽", "name": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
    st.plotly_chart(fig)

    if all_network:
        by_store = consumption.summarize(periods, ["store_id"])
        by_store["store"] = by_store["store_id"].map(_store_names(tables))
        fig2 = px.bar(by_store, x="store", y="deviation_cost", title="Отклонение по точкам, ₽",
                      labels={"deviation_cost": "Отклонение, ₽", "store": ""})
        st.plotly_chart(fig2)

    flagged = periods[periods["check_recount"]]
    with st.expander(f"Проверить пересчёт: {len(flagged)} подозрительных периодов"):
        st.caption("Отклонение резко отличается от обычного для этой точки и ингредиента. "
                   "Скорее всего, ошибка при пересчёте остатков, а не реальные потери: стоит пересчитать.")
        if not flagged.empty:
            view = flagged.assign(store=flagged["store_id"].map(_store_names(tables)))
            st.dataframe(view[["store", "name", "start_date", "end_date", "theory", "fact", "deviation_pct"]].rename(
                columns={**RU, "start_date": "С", "end_date": "По"}), hide_index=True)


# ---------------------------------------------------------------- ABC/XYZ
def render_abc_page(periods: pd.DataFrame, daily: pd.DataFrame, tables: dict, date_from, date_to,
                    a: float, b: float, x: float, y: float) -> None:
    if periods.empty:
        st.info("За выбранный период нет данных.")
        return
    cost = periods.groupby("ingredient_id")["fact_cost"].sum()
    demand = abc_xyz.demand_matrix(daily, date_from, date_to)
    result = abc_xyz.classify_ingredients(cost, demand, tables["ingredients"], a / 100, b / 100, x / 100, y / 100)

    left, right = st.columns(2)
    left.markdown("**Число ингредиентов**")
    left.dataframe(abc_xyz.class_matrix(result, "count").astype(int))
    right.markdown("**Доля стоимости расхода, %**")
    right.dataframe(abc_xyz.class_matrix(result, "share").round(1))

    pareto = result.assign(cum=result["share"].cumsum())
    fig = go.Figure()
    fig.add_bar(x=pareto["name"], y=pareto["cost"], name="Стоимость, ₽")
    fig.add_scatter(x=pareto["name"], y=pareto["cum"], name="Накопленная доля, %", yaxis="y2")
    fig.update_layout(title="Диаграмма Парето", yaxis2={"overlaying": "y", "side": "right", "range": [0, 105]},
                      legend={"orientation": "h"}, height=420)
    st.plotly_chart(fig)

    table = result[["name", "unit", "cost", "share", "abc", "cv", "xyz", "abc_xyz", "strategy"]].rename(columns=RU)
    st.dataframe(table.style.format({"Стоимость, ₽": "{:,.0f}", "Доля стоимости, %": "{:.1f}", "Вариация": "{:.2f}"},
                                    na_rep="—"), hide_index=True)
    st.caption("ABC: по стоимости расхода. XYZ: по вариации расхода после поправки на день недели. "
               "Если ингредиент не используется в техкартах, вариация не определена и он получает класс Z.")


# ---------------------------------------------------------------- Заказ
def render_order_page(order: pd.DataFrame, tables: dict, order_date, horizon: int) -> None:
    if order.empty:
        st.info("Нет данных для выбранной даты.")
        return
    order = order.assign(store=order["store_id"].map(_store_names(tables)))
    to_buy = order[order["to_order"] > 0]
    cols = st.columns(3)
    cols[0].metric("Позиций к заказу", len(to_buy))
    cols[1].metric("Сумма заказа, ₽", f"{to_buy['order_cost'].sum():,.0f}".replace(",", " "))
    cols[2].metric("Позиций с риском списания", int(order["waste_risk"].sum()))
    st.caption(f"Остатки на конец дня {pd.Timestamp(order_date):%d.%m.%Y}. Цель = расход в день × (срок поставки + {horizon} дн. "
               "горизонта + 1 день страхового запаса). Количество округлено до упаковки.")

    st.subheader("Заказать сейчас")
    view = to_buy[["store", "supplier", "name", "unit", "stock", "daily_forecast", "coverage_days", "target", "to_order",
                   "order_cost"]].rename(columns=RU)
    st.dataframe(view.style.format({"Остаток": "{:,.0f}", "Прогноз/день": "{:,.0f}", "Покрытие, дн.": "{:.1f}",
                                    "Цель": "{:,.0f}", "К заказу": "{:,.0f}", "Сумма, ₽": "{:,.0f}"}, na_rep="—"),
                 hide_index=True)
    st.download_button("Скачать заказ (CSV)", _csv_bytes(view), file_name=f"order_{pd.Timestamp(order_date):%Y-%m-%d}.csv")

    by_supplier = to_buy.groupby("supplier", as_index=False).agg(positions=("name", "count"), total=("order_cost", "sum"))
    st.dataframe(by_supplier.rename(columns={"supplier": "Поставщик", "positions": "Позиций", "total": "Сумма, ₽"})
                 .style.format({"Сумма, ₽": "{:,.0f}"}), hide_index=True)

    risk = order[order["waste_risk"]]
    with st.expander(f"Риск списания: {len(risk)} позиций"):
        st.caption("Остаток больше, чем уйдёт за срок годности. Не заказывать, перераспределить между точками или запустить в акцию.")
        st.dataframe(risk[["store", "name", "stock", "daily_forecast", "shelf_life_days"]].rename(columns=RU), hide_index=True)


# ---------------------------------------------------------------- Дашборд
def render_dashboard_page(periods: pd.DataFrame, tables: dict, order: pd.DataFrame, all_network: bool) -> None:
    if periods.empty:
        st.info("За выбранный период нет данных.")
        return
    k = kpi.compute_kpis(periods, tables["sales"], tables["dishes"])
    row1 = st.columns(4)
    row1[0].metric("Отклонение расхода", f"{k['deviation_pct']:+.1f} %")
    row1[1].metric("Потери, ₽", f"{k['loss_rub']:,.0f}".replace(",", " "))
    row1[2].metric("Food cost", f"{k['food_cost_pct']:.1f} %")
    row1[3].metric("Выручка, ₽", f"{k['revenue']:,.0f}".replace(",", " "))
    row2 = st.columns(3)
    row2[0].metric("Ингредиент заканчивался (раз)", k["stockouts"], help="Число периодов, где остаток на конец был нулевым при наличии спроса")
    row2[1].metric("Позиций с риском списания", int(order["waste_risk"].sum()) if not order.empty else 0)
    covered = order["coverage_days"].dropna() if not order.empty else pd.Series(dtype=float)
    row2[2].metric("Среднее покрытие запаса, дн.", f"{covered.mean():.1f}" if len(covered) else "—")

    if all_network:
        by_store = consumption.summarize(periods, ["store_id"])
        by_store["store"] = by_store["store_id"].map(_store_names(tables))
        st.plotly_chart(px.bar(by_store, x="store", y="deviation_pct", title="Отклонение расхода по точкам, %",
                               labels={"deviation_pct": "Отклонение, %", "store": ""}))

    names = tables["ingredients"].set_index("ingredient_id")["name"]
    choice = st.selectbox("Ингредиент для графика", names.index, format_func=lambda i: names[i])
    series = periods[periods["ingredient_id"] == choice].groupby("end_date", as_index=False)[["theory", "fact"]].sum()
    long = series.melt("end_date", var_name="Показатель", value_name="Расход").replace({"theory": "Теория", "fact": "Факт"})
    st.plotly_chart(px.line(long, x="end_date", y="Расход", color="Показатель", markers=True,
                            title=f"Теория и факт: {names[choice]}", labels={"end_date": "Конец периода"}))
    st.caption("Food cost считается по стоимости фактического расхода (включая потери и упаковку) к выручке за тот же период. "
               "Доля списаний и время подготовки заказа в приложении не рассчитываются: списания не ведутся отдельно, время замеряется вручную.")
