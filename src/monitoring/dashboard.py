"""
MATS Dashboard — мониторинг спреда ТАТок (TATN / TATNP)
Данные: MOEX ISS API (бесплатно, без токена)
Запуск: streamlit run src/monitoring/dashboard.py
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
REFRESH_SEC = 10       # секунд между обновлениями
HISTORY_DAYS = 1       # дней истории по умолчанию
BOARD = "TQBR"         # секция MOEX

SPREAD_LEVELS = [
    (0.0,  "red",    "solid", "0 — пол"),
    (2.5,  "orange", "dash",  "2.5 — алерт"),
    (5.0,  "green",  "dash",  "5.0 — скальп"),
    (7.0,  "blue",   "dash",  "7.0 — хороший вход"),
    (10.0, "purple", "dot",   "10.0 — среднесрочка"),
]
RATIO_LEVELS = [
    (1.000, "red",    "1.000 — паритет"),
    (1.003, "orange", "1.003 — алерт"),
    (1.008, "green",  "1.008 — скальп"),
    (1.011, "blue",   "1.011 — хороший вход"),
    (1.016, "purple", "1.016 — среднесрочка"),
]


# ---------------------------------------------------------------------------
# MOEX ISS API
# ---------------------------------------------------------------------------
def get_current_price(ticker: str) -> float | None:
    """Текущая цена через MOEX ISS (последняя сделка или цена закрытия)."""
    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares"
        f"/boards/{BOARD}/securities/{ticker}.json"
    )
    params = {
        "iss.meta": "off",
        "iss.only": "marketdata",
        "marketdata.columns": "SECID,LAST,LCLOSEPRICE",
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        rows = r.json()["marketdata"]["data"]
        if rows:
            last = rows[0][1]
            close = rows[0][2]
            return float(last) if last else (float(close) if close else None)
    except Exception:
        return None


def get_candles(ticker: str, days: int = 1) -> pd.DataFrame:
    """Минутные свечи за последние N дней через MOEX ISS."""
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares"
        f"/boards/{BOARD}/securities/{ticker}/candles.json"
    )
    params = {"interval": 1, "from": date_from, "iss.meta": "off"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()["candles"]
        if not data["data"]:
            return pd.DataFrame()
        df = pd.DataFrame(data["data"], columns=data["columns"])
        df["begin"] = pd.to_datetime(df["begin"])
        return df[["begin", "close"]].copy()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Логика сигнала
# ---------------------------------------------------------------------------
def calc_signal(spread: float, ratio: float) -> tuple[str, str, str]:
    """Возвращает (код сигнала, описание, тип для st)."""
    if ratio < 1.000:
        return "STRONG_LONG", "TATNP дороже обычки — сильный лонг спреда", "error"
    if spread >= 7.0 and ratio >= 1.011:
        return "LONG_GOOD", "Хороший вход. Ставить лимитки на обе ноги", "success"
    if spread >= 5.0 and ratio >= 1.008:
        return "LONG_SCALP", "Вход скальп. Лимитный ордер, ждать схождения", "warning"
    if spread >= 2.5:
        return "ALERT", "Наблюдать. Готовить ордера, не входить пока", "info"
    if spread <= 0.5:
        return "EXIT", "Линии сошлись. Закрывать обе ноги", "error"
    return "NO_SIGNAL", "Нет сигнала. Ждать расхождения ≥ 2.5 руб", "info"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MATS — ТАТки",
    page_icon="📈",
    layout="wide",
)
st.title("📈 MATS Dashboard — ТАТки (TATN / TATNP)")

# Боковая панель параметров
with st.sidebar:
    st.header("⚙️ Параметры")
    days_history = st.selectbox("История", [1, 3, 5, 10], index=0, format_func=lambda x: f"{x} дн.")
    refresh_sec = st.slider("Обновление (сек)", 5, 60, REFRESH_SEC, 5)
    st.divider()
    st.caption("Данные: MOEX ISS API")
    st.caption("Все уровни из bot_spec/01_STRATEGY_RULES.md")
    update_time = st.empty()

# Заглушки для живого обновления
ph_metrics = st.empty()
ph_signal  = st.empty()
ph_charts  = st.empty()

# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------
while True:
    update_time.caption(f"Обновлено: {datetime.now().strftime('%H:%M:%S')}")

    # --- Текущие цены ---
    price_tatn  = get_current_price("TATN")
    price_tatnp = get_current_price("TATNP")

    if not price_tatn or not price_tatnp:
        ph_signal.error(
            "⛔ Нет данных от MOEX. Биржа закрыта (10:00–18:50 и 19:05–23:50 МСК) "
            "или нет интернета."
        )
        time.sleep(refresh_sec)
        st.rerun()

    spread_rub = price_tatn  - price_tatnp
    ratio      = price_tatn  / price_tatnp
    spread_pct = (ratio - 1) * 100

    # --- Метрики ---
    with ph_metrics.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("TATN",        f"{price_tatn:.2f} ₽")
        c2.metric("TATNP",       f"{price_tatnp:.2f} ₽")
        c3.metric("Спред",       f"{spread_rub:.2f} ₽")
        c4.metric("Ratio",       f"{ratio:.4f}")
        c5.metric("Откл. от нормы", f"{spread_pct:.2f}%")

    # --- Сигнал ---
    code, desc, stype = calc_signal(spread_rub, ratio)
    with ph_signal.container():
        label = f"**{code}** — {desc}"
        if stype == "success":
            st.success(label)
        elif stype == "warning":
            st.warning(label)
        elif stype == "error":
            st.error(label)
        else:
            st.info(label)

    # --- История свечей ---
    df_tatn  = get_candles("TATN",  days=days_history)
    df_tatnp = get_candles("TATNP", days=days_history)

    with ph_charts.container():
        if df_tatn.empty or df_tatnp.empty:
            st.warning("История свечей недоступна (биржа закрыта или нет данных).")
        else:
            df = (
                pd.merge(df_tatn, df_tatnp, on="begin", suffixes=("_tatn", "_tatnp"))
                .sort_values("begin")
                .reset_index(drop=True)
            )

            # Нормализация для наложения
            df["tatn_pct"]  = (df["close_tatn"]  / df["close_tatn"].iloc[0]  - 1) * 100
            df["tatnp_pct"] = (df["close_tatnp"] / df["close_tatnp"].iloc[0] - 1) * 100
            df["spread"]    = df["close_tatn"] - df["close_tatnp"]
            df["ratio_h"]   = df["close_tatn"] / df["close_tatnp"]

            col1, col2, col3 = st.columns(3)

            # --- График 1: Наложение % ---
            with col1:
                st.subheader("1. Наложение %")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=df["begin"], y=df["tatn_pct"],
                    name="TATN (синий)", line=dict(color="#2196F3", width=2),
                ))
                fig1.add_trace(go.Scatter(
                    x=df["begin"], y=df["tatnp_pct"],
                    name="TATNP (оранж.)", line=dict(color="#FF9800", width=2),
                ))
                fig1.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
                fig1.update_layout(
                    height=380,
                    margin=dict(l=0, r=10, t=10, b=0),
                    legend=dict(orientation="h", y=1.12),
                    yaxis_title="%",
                    hovermode="x unified",
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.caption(
                    "Синий НИЖЕ оранж. → ШОРТ TATN + ЛОНГ TATNP  |  "
                    "Оранж. НИЖЕ синего → ЛОНГ TATN + ШОРТ TATNP"
                )

            # --- График 2: Разница (руб) ---
            with col2:
                st.subheader("2. Разница (₽)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df["begin"], y=df["spread"],
                    name="Спред", line=dict(color="#4CAF50", width=2),
                    fill="tozeroy", fillcolor="rgba(76,175,80,0.08)",
                ))
                for level, color, dash, label in SPREAD_LEVELS:
                    fig2.add_hline(
                        y=level, line_dash=dash, line_color=color, line_width=1,
                        annotation_text=label, annotation_position="right",
                        annotation_font_size=10,
                    )
                fig2.update_layout(
                    height=380,
                    margin=dict(l=0, r=80, t=10, b=0),
                    yaxis_title="руб",
                    hovermode="x unified",
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Основной рабочий график. Вход при спреде ≥ 5 руб.")

            # --- График 3: Ratio ---
            with col3:
                st.subheader("3. Ratio (TATN/TATNP)")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=df["begin"], y=df["ratio_h"],
                    name="Ratio", line=dict(color="#9C27B0", width=2),
                ))
                for level, color, label in RATIO_LEVELS:
                    fig3.add_hline(
                        y=level, line_dash="dash", line_color=color, line_width=1,
                        annotation_text=label, annotation_position="right",
                        annotation_font_size=10,
                    )
                fig3.update_layout(
                    height=380,
                    margin=dict(l=0, r=110, t=10, b=0),
                    yaxis_title="ratio",
                    hovermode="x unified",
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Нормализованный сигнал. Вход при ratio ≥ 1.008.")

    time.sleep(refresh_sec)
    st.rerun()
