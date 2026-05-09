"""
MATS Backtester — тестирование стратегии ТАТок на истории T-Invest
Запуск: streamlit run src/monitoring/backtest.py
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
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
_TF_MAP = {
    1:  "CANDLE_INTERVAL_1_MIN",
    5:  "CANDLE_INTERVAL_5_MIN",
    10: "CANDLE_INTERVAL_10_MIN",
    15: "CANDLE_INTERVAL_15_MIN",
    60: "CANDLE_INTERVAL_HOUR",
}

# ---------------------------------------------------------------------------
# T-Invest API
# ---------------------------------------------------------------------------
def _headers() -> dict:
    return {"Authorization": f"Bearer {TINKOFF_TOKEN}", "Content-Type": "application/json"}

def _q2f(q: dict) -> float:
    return float(q.get("units", 0)) + q.get("nano", 0) / 1_000_000_000

def get_candles(ticker: str, days: int, interval: int) -> pd.DataFrame:
    figi = FIGI.get(ticker)
    if not figi or not TINKOFF_TOKEN:
        return pd.DataFrame()
    to_dt   = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days)
    try:
        r = requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles",
            json={
                "figi":     figi,
                "from":     from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to":       to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "interval": _TF_MAP.get(interval, "CANDLE_INTERVAL_5_MIN"),
            },
            headers=_headers(),
            timeout=20,
        )
        candles = r.json().get("candles", [])
        if not candles:
            return pd.DataFrame()
        rows = []
        for c in candles:
            close = _q2f(c["close"])
            if close == 0:
                continue
            ts = pd.to_datetime(c["time"]) + timedelta(hours=3)
            rows.append({"begin": ts.replace(tzinfo=None), "close": close})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# Движок бэктеста
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, p: dict) -> dict:
    """
    Простой state machine: IDLE → IN_POSITION → IDLE.
    Вход: сигнал LONG_SCALP или LONG_GOOD по свечам.
    Выход: тейк (спред ≤ target) или стоп (спред ≥ entry + stop_add).
    Слипадж смоделирован как фиксированный параметр.
    """
    trades      = []
    equity      = [p["deposit"]]
    eq_times    = [df["begin"].iloc[0]]
    position    = None
    cooldown_ts = None
    capital     = float(p["deposit"])

    for _, row in df.iterrows():
        spread = row["spread"]
        ratio  = row["ratio_h"]
        ts     = row["begin"]

        if position is None:
            if cooldown_ts is not None and ts < cooldown_ts:
                continue

            # Сигнал входа
            signal = None
            if spread >= p["entry_good"] and ratio >= p["ratio_good"]:
                signal = "LONG_GOOD"
            elif spread >= p["entry_scalp"] and ratio >= p["ratio_scalp"]:
                signal = "LONG_SCALP"

            if signal:
                entry_spread = spread + p["slippage"]  # слипадж ухудшает вход
                position = {
                    "entry_time":   ts,
                    "entry_spread": entry_spread,
                    "signal":       signal,
                    "lots":         p["lots"],
                }
        else:
            entry_spread = position["entry_spread"]
            stop_spread  = entry_spread + p["stop_add"]

            result_type = None
            exit_spread = None

            if spread <= p["target"]:
                result_type = "TP"
                exit_spread = p["target"]
            elif spread >= stop_spread:
                result_type = "SL"
                exit_spread = stop_spread

            if result_type:
                commission = position["lots"] * 4 * p["commission_per_lot"]
                pnl_gross  = (entry_spread - exit_spread) * position["lots"]
                pnl_net    = pnl_gross - commission
                capital   += pnl_net

                trades.append({
                    "Вход":       position["entry_time"].strftime("%d.%m %H:%M"),
                    "Выход":      ts.strftime("%d.%m %H:%M"),
                    "entry_ts":   position["entry_time"],
                    "exit_ts":    ts,
                    "Сигнал":     position["signal"],
                    "Спред вх.":  round(entry_spread, 2),
                    "Спред вых.": round(exit_spread,  2),
                    "Лотов":      position["lots"],
                    "P&L ₽":      round(pnl_net, 2),
                    "Результат":  "✅ TP" if result_type == "TP" else "🛑 SL",
                    "Капитал":    round(capital, 2),
                })
                equity.append(capital)
                eq_times.append(ts)

                position    = None
                cooldown_ts = ts + timedelta(seconds=p["cooldown_s"])

    return {"trades": trades, "equity": equity, "eq_times": eq_times}


def calc_stats(trades: list, deposit: float) -> dict:
    if not trades:
        return {}
    pnls   = [t["P&L ₽"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Максимальная просадка
    cap, peak, max_dd = deposit, deposit, 0.0
    for p in pnls:
        cap  += p
        peak  = max(peak, cap)
        dd    = (peak - cap) / peak * 100
        max_dd = max(max_dd, dd)

    arr    = np.array(pnls)
    sharpe = (arr.mean() / arr.std() * len(pnls) ** 0.5) if len(pnls) > 1 and arr.std() > 0 else 0.0

    return {
        "total":     len(trades),
        "wins":      len(wins),
        "losses":    len(losses),
        "winrate":   len(wins) / len(trades) * 100,
        "total_pnl": sum(pnls),
        "avg_pnl":   sum(pnls) / len(trades),
        "avg_win":   sum(wins)   / len(wins)   if wins   else 0.0,
        "avg_loss":  sum(losses) / len(losses) if losses else 0.0,
        "max_dd":    max_dd,
        "sharpe":    sharpe,
        "roi":       sum(pnls) / deposit * 100,
    }

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MATS Backtester", page_icon="📊", layout="wide")
st.markdown(
    '<p style="font-size:1.1rem;font-weight:600;margin:0 0 4px 0">'
    '📊 MATS Backtester — ТАТки (TATN / TATNP)</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Параметры")

    st.subheader("Данные")
    days = st.selectbox("История", [3, 5, 10, 30, 60], index=1,
                        format_func=lambda x: f"{x} дн.")
    tf   = st.selectbox("Таймфрейм", [1, 5, 10, 15, 60], index=1,
                        format_func=lambda x: f"{x} мин")
    st.divider()

    st.subheader("Сигнал входа")
    entry_scalp = st.slider("Спред скальп ≥, ₽",  2.0, 10.0, 5.0, 0.5)
    ratio_scalp = st.slider("Ratio скальп ≥",     1.000, 1.020, 1.008, 0.001, format="%.3f")
    entry_good  = st.slider("Спред хороший ≥, ₽", 3.0, 15.0, 7.0, 0.5)
    ratio_good  = st.slider("Ratio хороший ≥",    1.000, 1.025, 1.011, 0.001, format="%.3f")
    st.divider()

    st.subheader("Риск-менеджмент")
    deposit  = st.number_input("Депозит ₽",            value=10_000, step=1_000)
    lots     = st.number_input("Лотов на сделку",      value=10,     step=1, min_value=1)
    stop_add = st.slider("Стоп: доп. к спреду ₽",      0.5, 10.0, 3.0, 0.5)
    target   = st.slider("Тейк: спред схождения ₽",    0.0, 2.0,  0.5, 0.1)
    slippage = st.slider("Слипадж (ухудш. входа) ₽",   0.0, 2.0,  0.2, 0.1)
    cooldown = st.slider("Cooldown после сделки, мин", 0, 60, 2, 1)
    st.divider()

    run_btn = st.button("▶  Запустить бэктест", type="primary", use_container_width=True)

if not run_btn:
    st.info("Настройте параметры в панели слева и нажмите **▶ Запустить бэктест**.")
    st.stop()

# --- Загрузка данных ---
with st.spinner(f"Загружаю свечи TATN + TATNP за {days} дн. ({tf} мин)..."):
    df_tatn  = get_candles("TATN",  days=days, interval=tf)
    df_tatnp = get_candles("TATNP", days=days, interval=tf)

if df_tatn.empty or df_tatnp.empty:
    st.error("Нет данных. Проверьте TINKOFF_TOKEN в .env или биржа закрыта.")
    st.stop()

df = (
    pd.merge(df_tatn, df_tatnp, on="begin", suffixes=("_tatn", "_tatnp"))
    .sort_values("begin")
    .reset_index(drop=True)
)
df["spread"]  = df["close_tatn"] - df["close_tatnp"]
df["ratio_h"] = df["close_tatn"] / df["close_tatnp"]

st.caption(
    f"Свечей загружено: {len(df)} · "
    f"Период: {df['begin'].iloc[0].strftime('%d.%m.%Y %H:%M')} — "
    f"{df['begin'].iloc[-1].strftime('%d.%m.%Y %H:%M')} · "
    f"Спред: {df['spread'].min():.2f}–{df['spread'].max():.2f} ₽"
)

params = {
    "deposit":            float(deposit),
    "lots":               int(lots),
    "entry_scalp":        entry_scalp,
    "ratio_scalp":        ratio_scalp,
    "entry_good":         entry_good,
    "ratio_good":         ratio_good,
    "stop_add":           stop_add,
    "target":             target,
    "slippage":           slippage,
    "cooldown_s":         cooldown * 60,
    "commission_per_lot": 0.05,
}

result = run_backtest(df, params)
trades = result["trades"]
stats  = calc_stats(trades, float(deposit))

if not trades:
    st.warning("За выбранный период сигналов нет. Попробуйте другие пороги или более длинный период.")
    st.stop()

# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
st.markdown("### Результаты")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Сделок",         stats["total"])
c2.metric("Winrate",        f"{stats['winrate']:.0f}%")
c3.metric("P&L итого",      f"{stats['total_pnl']:+.0f} ₽",
          delta=f"ROI {stats['roi']:+.1f}%")
c4.metric("Средний P&L",    f"{stats['avg_pnl']:+.1f} ₽")
c5.metric("Макс. просадка", f"{stats['max_dd']:.1f}%")
c6.metric("Sharpe",         f"{stats['sharpe']:.2f}")

c7, c8, c9 = st.columns(3)
c7.metric("Прибыльных", f"{stats['wins']}  ({stats['winrate']:.0f}%)")
c8.metric("Ср. прибыль", f"{stats['avg_win']:+.1f} ₽")
c9.metric("Ср. убыток",  f"{stats['avg_loss']:+.1f} ₽")

# ---------------------------------------------------------------------------
# Equity + Drawdown
# ---------------------------------------------------------------------------
eq_df = pd.DataFrame({"time": result["eq_times"], "equity": result["equity"]})
eq_df["peak"]     = eq_df["equity"].cummax()
eq_df["dd_pct"]   = (eq_df["peak"] - eq_df["equity"]) / eq_df["peak"] * 100

fig_eq = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.65, 0.35], vertical_spacing=0.06,
    subplot_titles=("Кривая капитала ₽", "Просадка %"),
)
fig_eq.add_trace(go.Scatter(
    x=eq_df["time"], y=eq_df["equity"],
    name="Капитал", line=dict(color="#4CAF50", width=2),
    fill="tozeroy", fillcolor="rgba(76,175,80,0.08)",
), row=1, col=1)
fig_eq.add_hline(y=float(deposit), line_dash="dash",
                 line_color="rgba(255,255,255,0.35)", row=1, col=1)
fig_eq.add_trace(go.Scatter(
    x=eq_df["time"], y=-eq_df["dd_pct"],
    name="Просадка", line=dict(color="#f85149", width=1.5),
    fill="tozeroy", fillcolor="rgba(248,81,73,0.12)",
), row=2, col=1)
fig_eq.update_yaxes(tickformat=".0f", ticksuffix=" ₽", side="right", row=1, col=1)
fig_eq.update_yaxes(tickformat=".1f", ticksuffix="%",  side="right", row=2, col=1)
fig_eq.update_layout(
    height=420, margin=dict(l=40, r=70, t=30, b=20),
    hovermode="x unified", showlegend=False,
)
st.plotly_chart(fig_eq, use_container_width=True)

# ---------------------------------------------------------------------------
# Спред с точками входа/выхода
# ---------------------------------------------------------------------------
fig_sp = go.Figure()
fig_sp.add_trace(go.Scatter(
    x=df["begin"], y=df["spread"],
    name="Спред ₽", line=dict(color="#9C27B0", width=1),
))

for level, color, dash, label in [
    (target,      "lime",     "dash", f"Тейк {target} ₽"),
    (entry_scalp, "#FF9800",  "dot",  f"Скальп {entry_scalp} ₽"),
    (entry_good,  "#2196F3",  "dot",  f"Хороший {entry_good} ₽"),
]:
    fig_sp.add_hline(y=level, line_dash=dash, line_color=color,
                     annotation_text=label, annotation_position="right")

tp_trades = [t for t in trades if "TP" in t["Результат"]]
sl_trades = [t for t in trades if "SL" in t["Результат"]]

if tp_trades:
    fig_sp.add_trace(go.Scatter(
        x=[t["entry_ts"] for t in tp_trades], y=[t["Спред вх."] for t in tp_trades],
        mode="markers", name="Вход TP",
        marker=dict(color="lime", size=9, symbol="triangle-up"),
    ))
    fig_sp.add_trace(go.Scatter(
        x=[t["exit_ts"] for t in tp_trades], y=[t["Спред вых."] for t in tp_trades],
        mode="markers", name="Выход TP",
        marker=dict(color="lime", size=9, symbol="triangle-down"),
    ))
if sl_trades:
    fig_sp.add_trace(go.Scatter(
        x=[t["entry_ts"] for t in sl_trades], y=[t["Спред вх."] for t in sl_trades],
        mode="markers", name="Вход SL",
        marker=dict(color="#f85149", size=9, symbol="triangle-up"),
    ))
    fig_sp.add_trace(go.Scatter(
        x=[t["exit_ts"] for t in sl_trades], y=[t["Спред вых."] for t in sl_trades],
        mode="markers", name="Выход SL",
        marker=dict(color="#f85149", size=9, symbol="triangle-down"),
    ))

fig_sp.update_layout(
    height=350, margin=dict(l=40, r=70, t=20, b=20),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.08, font=dict(size=11)),
)
fig_sp.update_yaxes(title_text="Спред ₽", side="right")
st.plotly_chart(fig_sp, use_container_width=True)

# ---------------------------------------------------------------------------
# Таблица сделок
# ---------------------------------------------------------------------------
st.divider()
st.markdown("**📋 История сделок**")
display_cols = ["Вход", "Выход", "Сигнал", "Спред вх.", "Спред вых.", "Лотов", "P&L ₽", "Результат", "Капитал"]
df_trades = pd.DataFrame(trades)[display_cols].iloc[::-1].reset_index(drop=True)
st.dataframe(df_trades, use_container_width=True, hide_index=True)
