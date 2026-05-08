"""
MATS Dashboard — мониторинг спреда ТАТок (TATN / TATNP)
Данные: T-Invest REST API (токен из .env)
Запуск: streamlit run src/monitoring/dashboard.py
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
TINKOFF_TOKEN = os.getenv("TINKOFF_TOKEN", "")
FIGI = {
    "TATN":  "BBG004RVFFC0",
    "TATNP": "BBG004S68829",
}
_TINVEST_BASE = "https://invest-public-api.tinkoff.ru/rest"

SPREAD_LEVELS = [
    (0.0,  "red",    "solid", "0 — пол"),
    (2.5,  "orange", "dash",  "2.5 — алерт"),
    (5.0,  "green",  "dash",  "5.0 — скальп"),
    (7.0,  "blue",   "dash",  "7.0 — хороший вход"),
    (10.0, "purple", "dot",   "10.0 — среднесрочка"),
    (20.0, "red",    "dot",   "20.0 — выход"),
]
RATIO_LEVELS = [
    (1.000, "red",    "1.000 — паритет"),
    (1.003, "orange", "1.003 — алерт"),
    (1.008, "green",  "1.008 — скальп"),
    (1.011, "blue",   "1.011 — хороший вход"),
    (1.016, "purple", "1.016 — среднесрочка"),
    (1.032, "red",    "1.032 — выход"),
]


# ---------------------------------------------------------------------------
# T-Invest REST API
# ---------------------------------------------------------------------------
_TF_MAP = {
    1:  "CANDLE_INTERVAL_1_MIN",
    5:  "CANDLE_INTERVAL_5_MIN",
    10: "CANDLE_INTERVAL_10_MIN",
    15: "CANDLE_INTERVAL_15_MIN",
    60: "CANDLE_INTERVAL_HOUR",
}


def _headers() -> dict:
    return {"Authorization": f"Bearer {TINKOFF_TOKEN}", "Content-Type": "application/json"}


def _q2f(q: dict) -> float:
    return float(q.get("units", 0)) + q.get("nano", 0) / 1_000_000_000


def get_current_price(ticker: str) -> float | None:
    figi = FIGI.get(ticker)
    if not figi or not TINKOFF_TOKEN:
        return None
    try:
        r = requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.MarketDataService/GetLastPrices",
            json={"figi": [figi]},
            headers=_headers(),
            timeout=5,
        )
        price = r.json()["lastPrices"][0]["price"]
        value = _q2f(price)
        return value if value > 0 else None
    except Exception:
        return None


def get_candles(ticker: str, days: int = 1, interval: int = 1) -> pd.DataFrame:
    figi = FIGI.get(ticker)
    if not figi or not TINKOFF_TOKEN:
        return pd.DataFrame()
    ti_interval = _TF_MAP.get(interval, "CANDLE_INTERVAL_1_MIN")
    to_dt   = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days)
    try:
        r = requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles",
            json={
                "figi":     figi,
                "from":     from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to":       to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "interval": ti_interval,
            },
            headers=_headers(),
            timeout=10,
        )
        candles = r.json().get("candles", [])
        if not candles:
            return pd.DataFrame()
        rows = []
        for c in candles:
            close = _q2f(c["close"])
            if close == 0:
                continue
            # Конвертируем UTC → МСК (+3ч), убираем tzinfo для Plotly
            ts = pd.to_datetime(c["time"]) + timedelta(hours=3)
            rows.append({"begin": ts.replace(tzinfo=None), "close": close})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Логика сигнала
# ---------------------------------------------------------------------------
def calc_signal(spread: float, ratio: float) -> tuple[str, str, str]:
    if spread >= 20.0:
        return "TAKE_PROFIT", "Спред > 20 руб — зона выхода из среднесрочки", "error"
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

# Боковая панель — рендерится один раз, не мигает
with st.sidebar:
    st.header("⚙️ Параметры")
    days_history = st.selectbox(
        "История", [1, 3, 5, 10], index=0,
        format_func=lambda x: f"{x} дн.",
    )
    timeframe = st.selectbox(
        "Таймфрейм свечей", [1, 5, 10, 15, 60], index=0,
        format_func=lambda x: f"{x} мин",
    )
    refresh_sec = st.slider("Обновление (сек)", 2, 60, 5, 1)
    st.divider()
    st.caption("Данные: T-Invest API")
    st.caption("Уровни: bot_spec/01_STRATEGY_RULES.md")


# ---------------------------------------------------------------------------
# Живой фрагмент — обновляется без перезагрузки страницы
# ---------------------------------------------------------------------------
@st.fragment(run_every=refresh_sec)
def live_dashboard(days: int, tf: int) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    tf_label = f"{tf} мин · {days} дн."

    # --- Текущие цены ---
    price_tatn  = get_current_price("TATN")
    price_tatnp = get_current_price("TATNP")

    if not TINKOFF_TOKEN:
        st.error("⛔ TINKOFF_TOKEN не найден. Создайте файл .env с токеном.")
        return
    if not price_tatn or not price_tatnp:
        st.warning(
            f"⏸ [{now}] Нет котировок от T-Invest. "
            "Биржа закрыта (10:00–18:50 и 19:05–23:50 МСК) или нет соединения."
        )
        return

    spread_rub = price_tatn  - price_tatnp
    ratio      = price_tatn  / price_tatnp
    spread_pct = (ratio - 1) * 100

    # --- Метрики ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("TATN",          f"{price_tatn:.2f} ₽")
    c2.metric("TATNP",         f"{price_tatnp:.2f} ₽")
    c3.metric("Спред",         f"{spread_rub:.2f} ₽")
    c4.metric("Ratio",         f"{ratio:.4f}")
    c5.metric("Откл. от нормы", f"{spread_pct:.2f}%")
    c6.metric("Обновлено",     now)

    # --- Сигнал ---
    code, desc, stype = calc_signal(spread_rub, ratio)
    label = f"**{code}** — {desc}"
    if stype == "success":
        st.success(label)
    elif stype == "warning":
        st.warning(label)
    elif stype == "error":
        st.error(label)
    else:
        st.info(label)

    # --- Исторические свечи ---
    df_tatn  = get_candles("TATN",  days=days, interval=tf)
    df_tatnp = get_candles("TATNP", days=days, interval=tf)

    if df_tatn.empty or df_tatnp.empty:
        st.warning("История свечей недоступна (биржа закрыта или нет данных).")
        return

    df = (
        pd.merge(df_tatn, df_tatnp, on="begin", suffixes=("_tatn", "_tatnp"))
        .sort_values("begin")
        .reset_index(drop=True)
    )
    df["spread"]  = df["close_tatn"] - df["close_tatnp"]   # TATN − TATNP ₽
    df["ratio_h"] = df["close_tatn"] / df["close_tatnp"]   # TATN / TATNP

    # Нормализация к % от первой свечи (как TradingView compare mode)
    tatn_base      = df["close_tatn"].iloc[0]
    tatnp_base     = df["close_tatnp"].iloc[0]
    df["tatn_pct"]    = (df["close_tatn"]  / tatn_base  - 1) * 100
    df["tatnp_pct"]   = (df["close_tatnp"] / tatnp_base - 1) * 100
    df["spread_pct"]  = df["tatn_pct"] - df["tatnp_pct"]   # относительный спред в %

    # --- Три графика: один под другим, синхронная ось X ---
    # Row 1 использует две Y-оси (TATN справа, TATNP слева)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.33, 0.33, 0.34],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # Заголовки — маленький шрифт, как подпись
    TITLE_FONT = dict(size=9, color="rgba(180,180,180,0.9)")
    for title, y in [
        (f"Цены, ₽  [{tf_label}]",                 1.005),
        (f"TATN% − TATNP% (относит. спред)  [{tf_label}]",  0.650),
        (f"TATN / TATNP ratio  [{tf_label}]",         0.300),
    ]:
        fig.add_annotation(
            text=title, xref="paper", yref="paper",
            x=0.01, y=y, showarrow=False,
            font=TITLE_FONT, xanchor="left",
        )

    # График 1 — двойная ось: TATN справа (оранж.), TATNP слева (синий)
    # Каждая линия на своей шкале → видны микродвижения обеих
    fig.add_trace(go.Scatter(
        x=df["begin"], y=df["close_tatn"],
        name="TATN", line=dict(color="#FF9800", width=1.5),
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["begin"], y=df["close_tatnp"],
        name="TATNP", line=dict(color="#2196F3", width=1.5),
    ), row=1, col=1, secondary_y=True)

    # График 2 — относительный спред в % (TATN% − TATNP%), осциллятор вокруг 0
    fig.add_trace(go.Scatter(
        x=df["begin"], y=df["spread_pct"],
        name="Спред%", line=dict(color="#4CAF50", width=1.5),
        showlegend=True,
    ), row=2, col=1)
    fig.add_hline(
        y=0, line_dash="solid", line_color="rgba(255,255,255,0.8)",
        line_width=2, row=2, col=1,
    )

    # График 3 — TATN / TATNP ratio, зум в реальный диапазон
    fig.add_trace(go.Scatter(
        x=df["begin"], y=df["ratio_h"],
        name="Ratio", line=dict(color="#9C27B0", width=1.5),
    ), row=3, col=1)
    ratio_open = df["ratio_h"].iloc[0]
    fig.add_hline(
        y=ratio_open, line_dash="dash", line_color="rgba(255,255,255,0.5)",
        line_width=1, row=3, col=1,
    )

    # Вертикальные линии — каждый час (или каждые 4ч при истории > 2 дней)
    freq = "4h" if days > 2 else "1h"
    hourly = pd.date_range(
        start=df["begin"].min().floor("h"),
        end=df["begin"].max(),
        freq=freq,
    )
    for t in hourly:
        fig.add_vline(
            x=t,
            line_width=1,
            line_color="rgba(160,160,160,0.2)",
            line_dash="solid",
        )

    # Y-оси справа (как в TradingView)
    # Верхний: две независимые Y-оси — каждая линия заполняет всю высоту
    _tatn_pad = max(df["close_tatn"].max() - df["close_tatn"].min(), 1.0) * 0.5
    fig.update_yaxes(
        secondary_y=False,
        side="right", tickfont=dict(size=10), tickformat=".2f",
        range=[df["close_tatn"].min() - _tatn_pad, df["close_tatn"].max() + _tatn_pad],
        row=1, col=1,
    )
    _tatnp_pad = max(df["close_tatnp"].max() - df["close_tatnp"].min(), 1.0) * 0.5
    fig.update_yaxes(
        secondary_y=True,
        side="left", tickfont=dict(size=10), tickformat=".2f",
        range=[df["close_tatnp"].min() - _tatnp_pad, df["close_tatnp"].max() + _tatnp_pad],
        row=1, col=1,
    )
    # Средний: относительный спред %, 0 посередине симметрично
    _s_abs = max(abs(df["spread_pct"].min()), abs(df["spread_pct"].max())) * 1.15 + 0.05
    fig.update_yaxes(
        side="right", tickfont=dict(size=10), tickformat=".2f",
        ticksuffix="%", range=[-_s_abs, _s_abs],
        row=2, col=1,
    )
    # Нижний: ratio зумирован в реальный диапазон (видны микродвижения)
    _r_lo = df["ratio_h"].min()
    _r_hi = df["ratio_h"].max()
    _r_pad = max((_r_hi - _r_lo) * 0.5, 0.0005)
    fig.update_yaxes(
        side="right", tickfont=dict(size=10), tickformat=".4f",
        range=[_r_lo - _r_pad, _r_hi + _r_pad],
        row=3, col=1,
    )

    # Вертикальная линия = текущее время (ISS запаздывает, но линия точная)
    now_t = datetime.now().replace(second=0, microsecond=0)
    fig.add_vline(
        x=now_t,
        line_width=1,
        line_color="rgba(180,180,180,0.55)",
        line_dash="solid",
    )

    # Отступ справа 5% от диапазона данных
    x_start = df["begin"].min()
    x_pad   = (now_t - x_start) * 0.05

    # Временная шкала + диапазон на всех трёх графиках
    for row in (1, 2, 3):
        fig.update_xaxes(
            showticklabels=True,
            tickformat="%H:%M",
            tickfont=dict(size=9),
            range=[x_start, now_t + x_pad],
            row=row, col=1,
        )

    # Кнопки зума — только на верхнем графике, управляют всеми тремя (shared_xaxes)
    fig.update_xaxes(
        rangeselector=dict(
            bgcolor="rgba(30,30,30,0.85)",
            activecolor="rgba(80,80,80,0.9)",
            font=dict(size=10, color="white"),
            buttons=[
                dict(count=1,  label="1ч",  step="hour", stepmode="backward"),
                dict(count=2,  label="2ч",  step="hour", stepmode="backward"),
                dict(count=4,  label="4ч",  step="hour", stepmode="backward"),
                dict(step="all", label="День"),
            ],
        ),
        row=1, col=1,
    )

    fig.update_layout(
        height=900,
        margin=dict(l=60, r=80, t=20, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Верхний: цены ₽ (TATN оранж. выше TATNP синего)  |  "
        "Средний: относит. спред% — осциллятор  |  "
        "Нижний: ratio TATN/TATNP  |  Zoom/pan синхронный"
    )


# Запускаем живой фрагмент
live_dashboard(days=days_history, tf=timeframe)
