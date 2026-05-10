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


def run_grid_search(df: pd.DataFrame, param_grid: dict, fixed: dict) -> pd.DataFrame:
    """Grid search: запускает бэктест по всем комбинациям param_grid."""
    import itertools
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    rows   = []
    for combo in combos:
        p = {**fixed, **dict(zip(keys, combo))}
        res   = run_backtest(df, p)
        stats = calc_stats(res["trades"], fixed["deposit"])
        if stats:
            rows.append({**dict(zip(keys, combo)), **stats, "сделок": stats["total"]})
    return pd.DataFrame(rows)


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
# Сигналы и параметры входа (общие для дашборда и тренажёра)
# ---------------------------------------------------------------------------
def calc_signal(spread: float, ratio: float) -> tuple[str, str, str]:
    if spread >= 20.0: return "TAKE_PROFIT", "Спред > 20 ₽ — зона выхода",       "error"
    if ratio  <  1.000: return "STRONG_LONG", "TATNP дороже обычки — сильный лонг", "error"
    if spread >= 7.0 and ratio >= 1.011: return "LONG_GOOD",  "Хороший вход",  "success"
    if spread >= 5.0 and ratio >= 1.008: return "LONG_SCALP", "Вход скальп",   "warning"
    if spread >= 2.5: return "ALERT",     "Наблюдать, готовить",               "info"
    if spread <= 0.5: return "EXIT",      "Линии сошлись. Закрывать!",         "error"
    return "NO_SIGNAL", "Нет сигнала", "info"


def calc_entry_params(spread: float, deposit: float, stop_add: float, target: float) -> dict:
    risk_per_trade = deposit * 0.20 / 5
    lots           = max(1, int(risk_per_trade / stop_add))
    commission     = lots * 4 * 0.05
    profit_net     = (spread - target) * lots - commission
    risk           = stop_add * lots
    rr             = profit_net / risk if risk > 0 else 0.0
    return {"lots": lots, "commission": commission, "profit_net": profit_net, "risk": risk, "rr": rr}


def run_sim_to_end(df: pd.DataFrame, deposit: float, stop_add: float, target: float) -> None:
    """Прокрутить все оставшиеся свечи в режиме полного автомата."""
    ss     = st.session_state
    cursor = ss["sim_cursor"]
    n      = len(df)

    while cursor < n:
        row    = df.iloc[cursor]
        spread = row["spread"]
        ratio  = row["ratio_h"]
        ts     = row["begin"]

        pos = ss["sim_position"]
        if pos is not None:
            stop_spread = pos["entry_spread"] + stop_add
            rt = ex = None
            if spread <= target:
                rt, ex = "TP", target
            elif spread >= stop_spread:
                rt, ex = "SL", stop_spread
            if rt:
                commission = pos["lots"] * 4 * 0.05
                pnl = (pos["entry_spread"] - ex) * pos["lots"] - commission
                ss["sim_capital"] += pnl
                ss["sim_trades"].append({
                    "Вход":       pos["entry_time"].strftime("%d.%m %H:%M"),
                    "entry_ts":   pos["entry_time"],
                    "exit_ts":    ts,
                    "Выход":      ts.strftime("%d.%m %H:%M"),
                    "Сигнал":     pos["signal"],
                    "Спред вх.":  round(pos["entry_spread"], 2),
                    "Спред вых.": round(ex, 2),
                    "Лотов":      pos["lots"],
                    "P&L ₽":      round(pnl, 2),
                    "Результат":  "✅ TP" if rt == "TP" else "🛑 SL",
                    "Капитал":    round(ss["sim_capital"], 2),
                    "hour":       ts.hour,
                })
                ss["sim_position"]     = None
                ss["sim_cooldown_end"] = cursor + 5

        if ss["sim_position"] is None and cursor >= ss["sim_cooldown_end"]:
            sig, _, _ = calc_signal(spread, ratio)
            if sig in {"LONG_SCALP", "LONG_GOOD"}:
                ep = calc_entry_params(spread, deposit, stop_add, target)
                if ep["rr"] >= 2.0:
                    ss["sim_position"] = {
                        "entry_time":   ts,
                        "entry_spread": spread,
                        "signal":       sig,
                        "lots":         ep["lots"],
                    }

        cursor += 1

    ss["sim_cursor"]         = n
    ss["sim_last_processed"] = n

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MATS Backtester", page_icon="📊", layout="wide")
st.markdown(
    '<p style="font-size:1.1rem;font-weight:600;margin:0 0 4px 0">'
    '📊 MATS Backtester — ТАТки (TATN / TATNP)</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — общие параметры и загрузка данных
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Параметры")

    st.subheader("Данные")
    days = st.selectbox("История", [3, 5, 10, 30, 60], index=1,
                        format_func=lambda x: f"{x} дн.")
    tf   = st.selectbox("Таймфрейм", [1, 5, 10, 15, 60], index=1,
                        format_func=lambda x: f"{x} мин")
    st.divider()

    st.subheader("Общие")
    deposit  = st.number_input("Депозит ₽",          value=10_000, step=1_000)
    lots     = st.number_input("Лотов на сделку",    value=10, step=1, min_value=1)
    slippage = st.slider("Слипадж ₽",                0.0, 2.0, 0.2, 0.1)
    cooldown = st.slider("Cooldown, мин",            0, 60, 2, 1)
    st.divider()

    load_btn = st.button("📥 Загрузить данные", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Загрузка данных (в session_state)
# ---------------------------------------------------------------------------
if load_btn:
    with st.spinner(f"Загружаю свечи за {days} дн. ({tf} мин)..."):
        df_t  = get_candles("TATN",  days=days, interval=tf)
        df_tp = get_candles("TATNP", days=days, interval=tf)
    if df_t.empty or df_tp.empty:
        st.error("Нет данных. Проверьте TINKOFF_TOKEN в .env.")
        st.stop()
    df_merged = (
        pd.merge(df_t, df_tp, on="begin", suffixes=("_tatn", "_tatnp"))
        .sort_values("begin").reset_index(drop=True)
    )
    df_merged["spread"]  = df_merged["close_tatn"] - df_merged["close_tatnp"]
    df_merged["ratio_h"] = df_merged["close_tatn"]  / df_merged["close_tatnp"]
    st.session_state["df"]   = df_merged
    st.session_state["days"] = days
    st.session_state["tf"]   = tf

if "df" not in st.session_state:
    st.info("Нажмите **📥 Загрузить данные** в боковой панели.")
    st.stop()

df = st.session_state["df"]
st.caption(
    f"Свечей: {len(df)} · "
    f"{df['begin'].iloc[0].strftime('%d.%m.%Y %H:%M')} — "
    f"{df['begin'].iloc[-1].strftime('%d.%m.%Y %H:%M')} · "
    f"Спред: {df['spread'].min():.2f}–{df['spread'].max():.2f} ₽"
)

# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------
tab_bt, tab_opt, tab_sim = st.tabs(["📊 Бэктест", "🔍 Оптимизатор", "🎮 Тренажёр"])

# ===========================================================================
# ВКЛАДКА 1: БЭКТЕСТ
# ===========================================================================
with tab_bt:
    col1, col2 = st.columns(2)
    with col1:
        entry_scalp = st.slider("Спред скальп ≥, ₽",  2.0, 10.0, 5.0, 0.5)
        ratio_scalp = st.slider("Ratio скальп ≥",     1.000, 1.020, 1.008, 0.001, format="%.3f")
        stop_add    = st.slider("Стоп: доп. к спреду ₽", 0.5, 10.0, 3.0, 0.5)
    with col2:
        entry_good  = st.slider("Спред хороший ≥, ₽", 3.0, 15.0, 7.0, 0.5)
        ratio_good  = st.slider("Ratio хороший ≥",    1.000, 1.025, 1.011, 0.001, format="%.3f")
        target      = st.slider("Тейк: спред схождения ₽", 0.0, 2.0, 0.5, 0.1)

    run_bt = st.button("▶ Запустить бэктест", type="primary")

    if run_bt:
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
            st.warning("Сигналов нет. Попробуйте другие пороги или период.")
        else:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Сделок",         stats["total"])
            c2.metric("Winrate",        f"{stats['winrate']:.0f}%")
            c3.metric("P&L итого",      f"{stats['total_pnl']:+.0f} ₽",
                      delta=f"ROI {stats['roi']:+.1f}%")
            c4.metric("Средний P&L",    f"{stats['avg_pnl']:+.1f} ₽")
            c5.metric("Макс. просадка", f"{stats['max_dd']:.1f}%")
            c6.metric("Sharpe",         f"{stats['sharpe']:.2f}")

            # Equity + Drawdown
            eq_df = pd.DataFrame({"time": result["eq_times"], "equity": result["equity"]})
            eq_df["peak"]   = eq_df["equity"].cummax()
            eq_df["dd_pct"] = (eq_df["peak"] - eq_df["equity"]) / eq_df["peak"] * 100

            fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.65, 0.35], vertical_spacing=0.06,
                                   subplot_titles=("Кривая капитала ₽", "Просадка %"))
            fig_eq.add_trace(go.Scatter(x=eq_df["time"], y=eq_df["equity"],
                name="Капитал", line=dict(color="#4CAF50", width=2),
                fill="tozeroy", fillcolor="rgba(76,175,80,0.08)"), row=1, col=1)
            fig_eq.add_hline(y=float(deposit), line_dash="dash",
                             line_color="rgba(255,255,255,0.35)", row=1, col=1)
            fig_eq.add_trace(go.Scatter(x=eq_df["time"], y=-eq_df["dd_pct"],
                name="Просадка", line=dict(color="#f85149", width=1.5),
                fill="tozeroy", fillcolor="rgba(248,81,73,0.12)"), row=2, col=1)
            fig_eq.update_yaxes(tickformat=".0f", ticksuffix=" ₽", side="right", row=1, col=1)
            fig_eq.update_yaxes(tickformat=".1f", ticksuffix="%",  side="right", row=2, col=1)
            fig_eq.update_layout(height=400, margin=dict(l=40, r=70, t=30, b=20),
                                 hovermode="x unified", showlegend=False)
            st.plotly_chart(fig_eq, use_container_width=True)

            # Спред с точками входа/выхода
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(x=df["begin"], y=df["spread"],
                name="Спред ₽", line=dict(color="#9C27B0", width=1)))
            for level, color, dash, label in [
                (target,      "lime",    "dash", f"Тейк {target} ₽"),
                (entry_scalp, "#FF9800", "dot",  f"Скальп {entry_scalp} ₽"),
                (entry_good,  "#2196F3", "dot",  f"Хороший {entry_good} ₽"),
            ]:
                fig_sp.add_hline(y=level, line_dash=dash, line_color=color,
                                 annotation_text=label, annotation_position="right")
            tp_t = [t for t in trades if "TP" in t["Результат"]]
            sl_t = [t for t in trades if "SL" in t["Результат"]]
            if tp_t:
                fig_sp.add_trace(go.Scatter(x=[t["entry_ts"] for t in tp_t],
                    y=[t["Спред вх."] for t in tp_t], mode="markers", name="Вход TP",
                    marker=dict(color="lime", size=9, symbol="triangle-up")))
                fig_sp.add_trace(go.Scatter(x=[t["exit_ts"] for t in tp_t],
                    y=[t["Спред вых."] for t in tp_t], mode="markers", name="Выход TP",
                    marker=dict(color="lime", size=9, symbol="triangle-down")))
            if sl_t:
                fig_sp.add_trace(go.Scatter(x=[t["entry_ts"] for t in sl_t],
                    y=[t["Спред вх."] for t in sl_t], mode="markers", name="Вход SL",
                    marker=dict(color="#f85149", size=9, symbol="triangle-up")))
                fig_sp.add_trace(go.Scatter(x=[t["exit_ts"] for t in sl_t],
                    y=[t["Спред вых."] for t in sl_t], mode="markers", name="Выход SL",
                    marker=dict(color="#f85149", size=9, symbol="triangle-down")))
            fig_sp.update_layout(height=330, margin=dict(l=40, r=70, t=20, b=20),
                                 hovermode="x unified",
                                 legend=dict(orientation="h", y=1.08, font=dict(size=11)))
            fig_sp.update_yaxes(title_text="Спред ₽", side="right")
            st.plotly_chart(fig_sp, use_container_width=True)

            st.divider()
            st.markdown("**📋 История сделок**")
            cols_show = ["Вход", "Выход", "Сигнал", "Спред вх.", "Спред вых.",
                         "Лотов", "P&L ₽", "Результат", "Капитал"]
            df_tr = pd.DataFrame(trades)[cols_show].iloc[::-1].reset_index(drop=True)
            st.dataframe(df_tr, use_container_width=True, hide_index=True)

            # --- Анализ по сессиям MOEX ---
            st.divider()
            st.markdown("### 📅 Анализ по сессиям MOEX")
            st.caption("⚠️ Урок 11б: спред расходится чаще на **вечерней сессии** (19:05–23:50 МСК).")

            df_tr_full = pd.DataFrame(trades).copy()
            df_tr_full["hour"] = pd.to_datetime(df_tr_full["entry_ts"]).dt.hour

            def _session(h: int) -> str:
                if 10 <= h <= 18: return "Утренняя (10–18)"
                if 19 <= h <= 23: return "Вечерняя (19–23)"
                return "Другое"

            df_tr_full["Сессия"] = df_tr_full["hour"].apply(_session)

            sess_rows = []
            for sess in ["Утренняя (10–18)", "Вечерняя (19–23)"]:
                sub = df_tr_full[df_tr_full["Сессия"] == sess]
                if sub.empty:
                    continue
                pnls = sub["P&L ₽"].tolist()
                wins = [p for p in pnls if p > 0]
                arr  = np.array(pnls)
                sh   = (arr.mean() / arr.std() * len(pnls) ** 0.5) if len(pnls) > 1 and arr.std() > 0 else 0.0
                sess_rows.append({
                    "Сессия":  sess,
                    "Сделок":  len(sub),
                    "Winrate": f"{len(wins)/len(sub)*100:.0f}%",
                    "P&L ₽":   f"{sum(pnls):+.0f}",
                    "ROI %":   f"{sum(pnls)/float(deposit)*100:+.1f}%",
                    "Sharpe":  f"{sh:.2f}",
                })
            if sess_rows:
                st.dataframe(pd.DataFrame(sess_rows), hide_index=True, use_container_width=True)

            # Bar: P&L по часам
            hourly_pnl = df_tr_full.groupby("hour")["P&L ₽"].sum().reset_index()
            fig_hr = go.Figure(go.Bar(
                x=hourly_pnl["hour"],
                y=hourly_pnl["P&L ₽"],
                marker_color=["#4CAF50" if v >= 0 else "#f85149" for v in hourly_pnl["P&L ₽"]],
                text=[f"{v:+.0f}" for v in hourly_pnl["P&L ₽"]],
                textposition="outside",
            ))
            fig_hr.update_layout(
                title="P&L по часам входа (МСК)", height=280,
                xaxis=dict(title="Час", tickmode="linear", dtick=1),
                yaxis=dict(title="P&L ₽", side="right"),
                margin=dict(l=20, r=60, t=40, b=30),
            )
            st.plotly_chart(fig_hr, use_container_width=True)

            # Scatter: каждая сделка
            fig_sc = go.Figure()
            for res, color, sym in [("TP", "#4CAF50", "circle"), ("SL", "#f85149", "x")]:
                sub_sc = df_tr_full[df_tr_full["Результат"].str.contains(res)]
                if not sub_sc.empty:
                    fig_sc.add_trace(go.Scatter(
                        x=sub_sc["hour"],
                        y=sub_sc["P&L ₽"],
                        mode="markers",
                        name=res,
                        marker=dict(color=color, size=11, symbol=sym),
                        text=sub_sc["Вход"],
                        hovertemplate="%{text}<br>P&L: %{y:.0f} ₽",
                    ))
            fig_sc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)")
            fig_sc.update_layout(
                title="Каждая сделка: час входа vs P&L", height=280,
                xaxis=dict(title="Час входа (МСК)", tickmode="linear", dtick=1),
                yaxis=dict(title="P&L ₽", side="right"),
                margin=dict(l=20, r=60, t=40, b=30),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

# ===========================================================================
# ВКЛАДКА 2: ОПТИМИЗАТОР
# ===========================================================================
with tab_opt:
    st.markdown("Запускает бэктест по сетке параметров и строит тепловую карту результатов.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Спред входа (скальп)**")
        es_min  = st.number_input("entry_scalp min", value=3.0, step=0.5)
        es_max  = st.number_input("entry_scalp max", value=7.0, step=0.5)
        es_step = st.number_input("entry_scalp шаг", value=1.0, step=0.5, min_value=0.5)
    with col_b:
        st.markdown("**Стоп (доп. к спреду)**")
        sa_min  = st.number_input("stop_add min", value=2.0, step=0.5)
        sa_max  = st.number_input("stop_add max", value=5.0, step=0.5)
        sa_step = st.number_input("stop_add шаг", value=1.0, step=0.5, min_value=0.5)
    with col_c:
        st.markdown("**Тейк (схождение)**")
        target_vals = st.multiselect("target ₽ (варианты)",
                                     [0.3, 0.5, 0.7, 1.0], default=[0.3, 0.5, 0.7])
        st.markdown("**Метрика**")
        metric_key = st.selectbox("Оптимизировать по",
                                  ["sharpe", "roi", "winrate", "total_pnl"],
                                  format_func=lambda x: {
                                      "sharpe": "Sharpe",
                                      "roi": "ROI %",
                                      "winrate": "Winrate %",
                                      "total_pnl": "P&L ₽",
                                  }[x])

    run_opt = st.button("🔍 Запустить оптимизацию", type="primary")

    if run_opt:
        import numpy as _np

        def _arange(mn, mx, step):
            vals = []
            v = mn
            while v <= mx + 1e-9:
                vals.append(round(v, 2))
                v += step
            return vals

        es_vals = _arange(es_min, es_max, es_step)
        sa_vals = _arange(sa_min, sa_max, sa_step)
        if not target_vals:
            st.warning("Выберите хотя бы один target.")
            st.stop()

        n_combos = len(es_vals) * len(sa_vals) * len(target_vals)
        st.info(f"Прогонов: {n_combos} ({len(es_vals)} entry × {len(sa_vals)} stop × {len(target_vals)} target)")

        fixed = {
            "deposit":            float(deposit),
            "lots":               int(lots),
            "ratio_scalp":        1.008,
            "entry_good":         999.0,   # отключаем сигнал LONG_GOOD
            "ratio_good":         1.025,
            "slippage":           slippage,
            "cooldown_s":         cooldown * 60,
            "commission_per_lot": 0.05,
        }
        param_grid = {
            "entry_scalp": es_vals,
            "stop_add":    sa_vals,
            "target":      target_vals,
        }

        with st.spinner("Оптимизация..."):
            progress = st.progress(0)
            # Запускаем через run_grid_search с псевдо-прогрессом
            import itertools
            keys   = list(param_grid.keys())
            combos = list(itertools.product(*param_grid.values()))
            rows   = []
            for i, combo in enumerate(combos):
                p = {**fixed, **dict(zip(keys, combo))}
                res   = run_backtest(df, p)
                stats = calc_stats(res["trades"], fixed["deposit"])
                if stats:
                    rows.append({**dict(zip(keys, combo)), **stats})
                progress.progress((i + 1) / len(combos))

        if not rows:
            st.warning("Ни одна комбинация не дала сделок. Попробуйте другие диапазоны.")
            st.stop()

        df_opt = pd.DataFrame(rows)

        # --- Топ-10 ---
        st.markdown(f"### Топ-10 по {metric_key}")
        top10_cols = ["entry_scalp", "stop_add", "target",
                      "total", "winrate", "roi", "sharpe", "max_dd", "total_pnl"]
        top10_cols = [c for c in top10_cols if c in df_opt.columns]
        df_top = (df_opt.sort_values(metric_key, ascending=False)
                  .head(10)[top10_cols].reset_index(drop=True))
        st.dataframe(df_top.style.format({
            "winrate":   "{:.0f}%",
            "roi":       "{:+.1f}%",
            "sharpe":    "{:.2f}",
            "max_dd":    "{:.1f}%",
            "total_pnl": "{:+.0f} ₽",
        }), use_container_width=True, hide_index=True)

        # --- Heatmap для каждого target ---
        st.markdown(f"### Тепловые карты ({metric_key})")
        for tgt in sorted(df_opt["target"].unique()):
            sub = df_opt[df_opt["target"] == tgt]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="stop_add", columns="entry_scalp",
                                    values=metric_key, aggfunc="mean")
            z = pivot.values.astype(float)
            fig_hm = go.Figure(go.Heatmap(
                z=z,
                x=[f"{v} ₽" for v in pivot.columns],
                y=[f"{v} ₽" for v in pivot.index],
                colorscale="RdYlGn",
                text=_np.round(z, 2),
                texttemplate="%{text}",
                colorbar=dict(title=metric_key),
            ))
            fig_hm.update_layout(
                title=f"target = {tgt} ₽ | X: entry_scalp | Y: stop_add | Цвет: {metric_key}",
                height=350, margin=dict(l=60, r=40, t=50, b=40),
                xaxis_title="entry_scalp ₽", yaxis_title="stop_add ₽",
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # --- Лучшая комбинация ---
        best = df_opt.loc[df_opt[metric_key].idxmax()]
        st.success(
            f"**Лучшая комбинация по {metric_key}:** "
            f"entry_scalp={best['entry_scalp']} ₽, "
            f"stop_add={best['stop_add']} ₽, "
            f"target={best['target']} ₽ → "
            f"Sharpe={best['sharpe']:.2f}, ROI={best['roi']:+.1f}%, "
            f"WR={best['winrate']:.0f}%, Сделок={int(best['total'])}"
        )

# ===========================================================================
# ВКЛАДКА 3: ТРЕНАЖЁР
# ===========================================================================
with tab_sim:
    st.markdown(
        "Пошаговое воспроизведение торговли на реальной истории. "
        "**График показывает только прошлое** — будущего не видно."
    )

    # --- Параметры тренажёра ---
    col_sm, col_sp = st.columns([1, 3])
    with col_sm:
        is_auto = (
            st.radio("Режим", ["🖐 Полуавтомат", "🤖 Полный автомат"], key="sim_mode_radio")
            == "🤖 Полный автомат"
        )
    with col_sp:
        spc1, spc2, spc3 = st.columns(3)
        sim_dep  = spc1.number_input("Депозит ₽", value=10_000, step=1_000, key="sim_dep_inp")
        sim_stop = spc2.slider("Стоп ₽",  0.5, 10.0, 3.0, 0.5, key="sim_stop_sl")
        sim_tgt  = spc3.slider("Тейк ₽",  0.0,  2.0, 0.5, 0.1, key="sim_tgt_sl")

    # --- Инициализация / сброс session_state ---
    _sim_df_key = len(df)
    _need_init  = (
        "sim_cursor" not in st.session_state
        or st.session_state.get("sim_df_key") != _sim_df_key
    )
    if _need_init:
        st.session_state.update({
            "sim_cursor":         0,
            "sim_position":       None,
            "sim_trades":         [],
            "sim_capital":        float(sim_dep),
            "sim_cooldown_end":   0,
            "sim_last_processed": -1,
            "sim_df_key":         _sim_df_key,
        })

    if st.button("🔄 Сбросить", key="sim_rst"):
        st.session_state.update({
            "sim_cursor":         0,
            "sim_position":       None,
            "sim_trades":         [],
            "sim_capital":        float(sim_dep),
            "sim_cooldown_end":   0,
            "sim_last_processed": -1,
        })
        st.rerun()

    cursor  = st.session_state["sim_cursor"]
    n_total = len(df)

    # --- Обработка текущей свечи (один раз за позицию курсора) ---
    if st.session_state["sim_last_processed"] != cursor and cursor < n_total:
        _row = df.iloc[cursor]
        _spr = _row["spread"]
        _rat = _row["ratio_h"]
        _ts  = _row["begin"]
        _pos = st.session_state["sim_position"]

        # Проверка TP / SL
        if _pos is not None:
            _stop_lv = _pos["entry_spread"] + sim_stop
            _rt = _ex = None
            if _spr <= sim_tgt:
                _rt, _ex = "TP", sim_tgt
            elif _spr >= _stop_lv:
                _rt, _ex = "SL", _stop_lv
            if _rt:
                _comm = _pos["lots"] * 4 * 0.05
                _pnl  = (_pos["entry_spread"] - _ex) * _pos["lots"] - _comm
                st.session_state["sim_capital"] += _pnl
                st.session_state["sim_trades"].append({
                    "Вход":       _pos["entry_time"].strftime("%d.%m %H:%M"),
                    "entry_ts":   _pos["entry_time"],
                    "exit_ts":    _ts,
                    "Выход":      _ts.strftime("%d.%m %H:%M"),
                    "Сигнал":     _pos["signal"],
                    "Спред вх.":  round(_pos["entry_spread"], 2),
                    "Спред вых.": round(_ex, 2),
                    "Лотов":      _pos["lots"],
                    "P&L ₽":      round(_pnl, 2),
                    "Результат":  "✅ TP" if _rt == "TP" else "🛑 SL",
                    "Капитал":    round(st.session_state["sim_capital"], 2),
                    "hour":       _ts.hour,
                })
                st.session_state["sim_position"] = None
                st.session_state["sim_cooldown_end"] = cursor + 5
                _pos = None

        # Авто-вход (только полный автомат)
        if is_auto and _pos is None and cursor >= st.session_state["sim_cooldown_end"]:
            _sig, _, _ = calc_signal(_spr, _rat)
            if _sig in {"LONG_SCALP", "LONG_GOOD"}:
                _ep = calc_entry_params(_spr, float(sim_dep), sim_stop, sim_tgt)
                if _ep["rr"] >= 2.0:
                    st.session_state["sim_position"] = {
                        "entry_time":   _ts,
                        "entry_spread": _spr,
                        "signal":       _sig,
                        "lots":         _ep["lots"],
                    }

        st.session_state["sim_last_processed"] = cursor

    # --- Читаем актуальное состояние для отображения ---
    cursor         = st.session_state["sim_cursor"]
    position       = st.session_state["sim_position"]
    capital        = st.session_state["sim_capital"]
    sim_trades_lst = st.session_state["sim_trades"]

    if cursor >= n_total:
        st.success(f"✅ Все {n_total} свечей просмотрены!")
        cursor = n_total - 1

    _row   = df.iloc[cursor]
    spread = _row["spread"]
    ratio  = _row["ratio_h"]
    ts     = _row["begin"]
    signal, signal_desc, sig_type = calc_signal(spread, ratio)
    ep     = calc_entry_params(spread, float(sim_dep), sim_stop, sim_tgt)

    in_cooldown = cursor < st.session_state["sim_cooldown_end"]
    can_enter   = (
        position is None
        and not in_cooldown
        and signal in {"LONG_SCALP", "LONG_GOOD"}
        and ep["rr"] >= 2.0
    )

    # --- Информационные метрики ---
    mi1, mi2, mi3, mi4, mi5 = st.columns(5)
    mi1.metric("Свеча",   f"{cursor + 1} / {n_total}")
    mi2.metric("Время",   ts.strftime("%d.%m %H:%M"))
    mi3.metric("Спред",   f"{spread:.2f} ₽")
    mi4.metric("Ratio",   f"{ratio:.4f}")
    mi5.metric("Капитал", f"{capital:.0f} ₽",
               delta=f"{capital - float(sim_dep):+.0f} ₽")

    # Сигнал
    _icons = {"success": "🟢", "warning": "🟡", "error": "🔴", "info": "⚪"}
    st.markdown(
        f"**Сигнал:** {_icons.get(sig_type, '⚪')} **{signal}** — {signal_desc}"
        + (" *(cooldown)*" if in_cooldown else "")
    )

    # Параметры входа
    if signal in {"LONG_SCALP", "LONG_GOOD"}:
        pe1, pe2, pe3, pe4, pe5 = st.columns(5)
        pe1.metric("Лотов",     ep["lots"])
        pe2.metric("Комиссия",  f"{ep['commission']:.2f} ₽")
        pe3.metric("Потенциал", f"{ep['profit_net']:+.2f} ₽")
        pe4.metric("Риск",      f"{ep['risk']:.2f} ₽")
        pe5.metric("R/R",       f"{ep['rr']:.2f}",
                   delta="✅ OK" if ep["rr"] >= 2.0 else "❌ мало")

    # Открытая позиция
    if position is not None:
        _stop_lv  = position["entry_spread"] + sim_stop
        _cur_pnl  = (position["entry_spread"] - spread) * position["lots"] - position["lots"] * 4 * 0.05
        st.info(
            f"📊 **В позиции** | Вход: {position['entry_time'].strftime('%H:%M')} | "
            f"Спред входа: {position['entry_spread']:.2f} ₽ | "
            f"Тейк: ≤ {sim_tgt:.1f} ₽ | Стоп: ≥ {_stop_lv:.2f} ₽ | "
            f"P&L сейчас: {_cur_pnl:+.2f} ₽"
        )

    # --- Кнопки навигации ---
    nb1, nb2, nb3, _ = st.columns([1, 1, 1, 3])
    with nb1:
        if st.button("▶ Следующая", key="sim_next", disabled=cursor >= n_total - 1):
            st.session_state["sim_cursor"] += 1
            st.rerun()

    if not is_auto:
        # Полуавтомат: кнопки входа при сигнале
        if can_enter:
            with nb2:
                if st.button("✅ Войти", key="sim_enter", type="primary"):
                    st.session_state["sim_position"] = {
                        "entry_time":   ts,
                        "entry_spread": spread,
                        "signal":       signal,
                        "lots":         ep["lots"],
                    }
                    st.session_state["sim_last_processed"] = cursor
                    st.session_state["sim_cursor"] += 1
                    st.rerun()
            with nb3:
                if st.button("⏭ Пропустить", key="sim_skip"):
                    st.session_state["sim_cursor"] += 1
                    st.rerun()
    else:
        # Полный автомат: прокрутить всё до конца
        with nb2:
            if st.button("⏩ До конца", key="sim_run_all"):
                run_sim_to_end(df, float(sim_dep), sim_stop, sim_tgt)
                st.rerun()

    # --- График (история до текущей свечи, будущего нет) ---
    hist = df.iloc[: cursor + 1]
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=hist["begin"], y=hist["spread"],
        name="Спред", line=dict(color="#9C27B0", width=1.5),
    ))

    for _lvl, _clr, _lbl in [
        (sim_tgt, "lime",    f"Тейк {sim_tgt} ₽"),
        (5.0,     "#FF9800", "Скальп 5.0 ₽"),
        (7.0,     "#2196F3", "Хороший 7.0 ₽"),
    ]:
        fig_sim.add_hline(y=_lvl, line_dash="dash", line_color=_clr,
                          annotation_text=_lbl, annotation_position="right")

    for _tr in sim_trades_lst:
        _clr_t = "lime" if "TP" in _tr["Результат"] else "#f85149"
        fig_sim.add_trace(go.Scatter(
            x=[_tr["entry_ts"]], y=[_tr["Спред вх."]],
            mode="markers", marker=dict(color=_clr_t, size=10, symbol="triangle-up"),
            showlegend=False,
        ))
        fig_sim.add_trace(go.Scatter(
            x=[_tr["exit_ts"]], y=[_tr["Спред вых."]],
            mode="markers", marker=dict(color=_clr_t, size=10, symbol="triangle-down"),
            showlegend=False,
        ))

    if position is not None:
        fig_sim.add_trace(go.Scatter(
            x=[position["entry_time"]], y=[position["entry_spread"]],
            mode="markers", name="Позиция",
            marker=dict(color="yellow", size=13, symbol="triangle-up"),
        ))

    fig_sim.update_layout(height=320, margin=dict(l=40, r=80, t=20, b=20),
                          hovermode="x unified", showlegend=False)
    fig_sim.update_yaxes(title_text="Спред ₽", side="right")
    st.plotly_chart(fig_sim, use_container_width=True)

    # --- Лог сделок и итоговые метрики ---
    if sim_trades_lst:
        st.divider()
        st.markdown("**📋 Лог сделок тренажёра**")
        _cols_sim = ["Вход", "Выход", "Сигнал", "Спред вх.", "Спред вых.",
                     "Лотов", "P&L ₽", "Результат", "Капитал"]
        df_sim_log = (
            pd.DataFrame(sim_trades_lst)[_cols_sim]
            .iloc[::-1].reset_index(drop=True)
        )
        st.dataframe(df_sim_log, use_container_width=True, hide_index=True)

        st.markdown("#### 📊 Итог тренажёра")
        sim_stats = calc_stats(sim_trades_lst, float(sim_dep))
        if sim_stats:
            ms1, ms2, ms3, ms4, ms5 = st.columns(5)
            ms1.metric("Сделок",  sim_stats["total"])
            ms2.metric("Winrate", f"{sim_stats['winrate']:.0f}%")
            ms3.metric("P&L",     f"{sim_stats['total_pnl']:+.0f} ₽")
            ms4.metric("ROI",     f"{sim_stats['roi']:+.1f}%")
            ms5.metric("Sharpe",  f"{sim_stats['sharpe']:.2f}")
