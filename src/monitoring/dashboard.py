"""
MATS Dashboard — мониторинг спреда ТАТок (TATN / TATNP)
Данные: T-Invest REST API (токен из .env)
Запуск: streamlit run src/monitoring/dashboard.py
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import csv
import socket
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
TINKOFF_TOKEN = os.getenv("TINKOFF_TOKEN", "")
DRY_RUN       = os.getenv("DRY_RUN", "true").lower() != "false"  # по умолчанию ВСЕГДА true
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

# Риск-менеджмент (Виктор р9-09: депо 20% / 5 частей)
DEPOSIT_RUB      = 10_000   # депо ₽ — менять под реальный
MAX_DRAWDOWN_PCT = 0.20     # макс. просадка 20%
RISK_PARTS       = 5        # частей
STOP_SPREAD_ADD  = 3.0      # стоп = спред_входа + 3 руб
TARGET_SPREAD    = 0.5      # цель схождения ₽
MIN_RR           = 2.0      # не входить если R/R < 2
ALERT_TIMEOUT_S  = 30       # сек до автоотмены сигнала
ALERT_COOLDOWN_S = 120      # сек тишины после закрытия алерта


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


def _f2q(price: float) -> dict:
    units = int(price)
    nano  = round((price - units) * 1_000_000_000)
    return {"units": units, "nano": nano}


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
# T-Invest API — ордера и стакан
# ---------------------------------------------------------------------------

def get_orderbook(ticker: str, depth: int = 5) -> dict | None:
    figi = FIGI.get(ticker)
    if not figi or not TINKOFF_TOKEN:
        return None
    try:
        r = requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.MarketDataService/GetOrderBook",
            json={"figi": figi, "depth": depth},
            headers=_headers(),
            timeout=5,
        )
        data = r.json()
        asks = [(_q2f(i["price"]), int(i["quantity"])) for i in data.get("asks", [])]
        bids = [(_q2f(i["price"]), int(i["quantity"])) for i in data.get("bids", [])]
        if not asks or not bids:
            return None
        return {"asks": asks, "bids": bids}
    except Exception:
        return None


def get_account_id() -> str | None:
    if not TINKOFF_TOKEN:
        return None
    try:
        r = requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts",
            json={},
            headers=_headers(),
            timeout=5,
        )
        accounts = r.json().get("accounts", [])
        return accounts[0]["id"] if accounts else None
    except Exception:
        return None


def _post_order(figi: str, direction: str, qty: int, price: float, account_id: str) -> str | None:
    """direction: 'BUY' или 'SELL'. Возвращает orderId или None."""
    try:
        r = requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.OrdersService/PostOrder",
            json={
                "figi":      figi,
                "quantity":  qty,
                "price":     _f2q(price),
                "direction": f"ORDER_DIRECTION_{direction}",
                "accountId": account_id,
                "orderType": "ORDER_TYPE_LIMIT",
                "orderId":   str(uuid.uuid4()),
            },
            headers=_headers(),
            timeout=10,
        )
        return r.json().get("orderId")
    except Exception:
        return None


def _get_order_state(account_id: str, order_id: str) -> dict | None:
    try:
        r = requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.OrdersService/GetOrderState",
            json={"accountId": account_id, "orderId": order_id},
            headers=_headers(),
            timeout=5,
        )
        return r.json()
    except Exception:
        return None


def _cancel_order(account_id: str, order_id: str) -> None:
    try:
        requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.OrdersService/CancelOrder",
            json={"accountId": account_id, "orderId": order_id},
            headers=_headers(),
            timeout=5,
        )
    except Exception:
        pass


def _post_stop_order(figi: str, direction: str, qty: int, stop_price: float, account_id: str) -> None:
    """Выставить стоп-лосс на сервер T-Invest (работает без интернета на стороне брокера)."""
    try:
        requests.post(
            f"{_TINVEST_BASE}/tinkoff.public.invest.api.contract.v1.StopOrdersService/PostStopOrder",
            json={
                "figi":           figi,
                "quantity":       qty,
                "stopPrice":      _f2q(stop_price),
                "direction":      f"STOP_ORDER_DIRECTION_{direction}",
                "accountId":      account_id,
                "stopOrderType":  "STOP_ORDER_TYPE_STOP_LOSS",
                "expirationType": "STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL",
            },
            headers=_headers(),
            timeout=10,
        )
    except Exception:
        pass


def _log_trade(data: dict) -> None:
    log_dir  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
    log_path = os.path.join(log_dir, "trades.csv")
    os.makedirs(log_dir, exist_ok=True)
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


# ---------------------------------------------------------------------------
# Расчёт параметров входа
# ---------------------------------------------------------------------------
def calc_entry_params(spread_rub: float, price_tatn: float, price_tatnp: float) -> dict:
    risk_per_trade = DEPOSIT_RUB * MAX_DRAWDOWN_PCT / RISK_PARTS   # 400 ₽
    lots           = max(1, int(risk_per_trade / STOP_SPREAD_ADD))
    commission     = lots * 4 * 0.05                                # 4 сделки ~0.05₽/лот
    profit_gross   = (spread_rub - TARGET_SPREAD) * lots
    profit_net     = profit_gross - commission
    risk           = STOP_SPREAD_ADD * lots
    rr             = profit_net / risk if risk > 0 else 0
    return {
        "lots":         lots,
        "buy_price":    price_tatnp,
        "sell_price":   price_tatn,
        "spread_entry": spread_rub,
        "profit_net":   profit_net,
        "risk":         risk,
        "commission":   commission,
        "rr":           rr,
    }


LEG_TIMEOUT_S = 15  # сек ожидания заполнения каждой ноги


def execute_entry(alert_data: dict, account_id: str) -> dict:
    """
    Обе ноги одновременно через ThreadPoolExecutor (аналог asyncio.gather).
    Агрессивные лимитки: TATNP buy=ask[0], TATN sell=bid[0].
    Таймаут 15 сек — если нога не заполнилась, отменить обе + аварийно закрыть заполненную.
    После входа — стоп-ордера на сервер T-Invest (защита без интернета).
    """
    ob_tatnp = get_orderbook("TATNP")
    ob_tatn  = get_orderbook("TATN")
    if not ob_tatnp or not ob_tatn:
        return {"status": "FAILED", "reason": "Нет стакана"}

    buy_price  = ob_tatnp["asks"][0][0]   # TATNP: покупаем по ask[0]
    sell_price = ob_tatn["bids"][0][0]    # TATN:  продаём по bid[0]
    lots       = alert_data["lots"]

    # Обе ноги одновременно (правило р9-09)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_buy  = ex.submit(_post_order, FIGI["TATNP"], "BUY",  lots, buy_price,  account_id)
        f_sell = ex.submit(_post_order, FIGI["TATN"],  "SELL", lots, sell_price, account_id)
    oid_buy  = f_buy.result()
    oid_sell = f_sell.result()

    if not oid_buy or not oid_sell:
        if oid_buy:  _cancel_order(account_id, oid_buy)
        if oid_sell: _cancel_order(account_id, oid_sell)
        return {"status": "FAILED", "reason": "Ошибка PostOrder"}

    # Ждём заполнения обеих ног (таймаут 15 сек)
    filled_buy = filled_sell = False
    price_buy  = buy_price
    price_sell = sell_price
    deadline   = time.time() + LEG_TIMEOUT_S

    while time.time() < deadline:
        if not filled_buy:
            s = _get_order_state(account_id, oid_buy)
            if s and s.get("executionReportStatus") == "EXECUTION_REPORT_STATUS_FILL":
                filled_buy = True
                price_buy  = _q2f(s.get("averagePositionPrice", _f2q(buy_price)))
        if not filled_sell:
            s = _get_order_state(account_id, oid_sell)
            if s and s.get("executionReportStatus") == "EXECUTION_REPORT_STATUS_FILL":
                filled_sell = True
                price_sell  = _q2f(s.get("averagePositionPrice", _f2q(sell_price)))
        if filled_buy and filled_sell:
            break
        time.sleep(1)

    # Таймаут — нога-риск: отменить незаполненные, аварийно закрыть заполненные
    if not (filled_buy and filled_sell):
        if not filled_buy:  _cancel_order(account_id, oid_buy)
        if not filled_sell: _cancel_order(account_id, oid_sell)
        ob2_tatnp = get_orderbook("TATNP")
        ob2_tatn  = get_orderbook("TATN")
        if filled_buy  and ob2_tatnp:
            _post_order(FIGI["TATNP"], "SELL", lots, ob2_tatnp["bids"][0][0], account_id)
        if filled_sell and ob2_tatn:
            _post_order(FIGI["TATN"],  "BUY",  lots, ob2_tatn["asks"][0][0],  account_id)
        return {"status": "FAILED", "reason": "Таймаут 15 сек — нога не заполнилась"}

    # Обе ноги заполнены — выставить стоп-ордера на сервер T-Invest
    entry_spread = price_sell - price_buy
    stop_half    = STOP_SPREAD_ADD / 2   # каждая нога защищается на половину стопа
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(_post_stop_order, FIGI["TATNP"], "SELL", lots, price_buy  - stop_half, account_id)
        ex.submit(_post_stop_order, FIGI["TATN"],  "BUY",  lots, price_sell + stop_half, account_id)

    _log_trade({
        "time":         datetime.now().isoformat(),
        "type":         "ENTRY",
        "ticker_buy":   "TATNP",
        "ticker_sell":  "TATN",
        "buy_price":    round(price_buy,   2),
        "sell_price":   round(price_sell,  2),
        "entry_spread": round(entry_spread, 2),
        "stop_spread":  round(entry_spread + STOP_SPREAD_ADD, 2),
        "lots":         lots,
        "oid_buy":      oid_buy,
        "oid_sell":     oid_sell,
    })

    return {
        "status":       "OK",
        "buy_price":    price_buy,
        "sell_price":   price_sell,
        "entry_spread": entry_spread,
        "oid_buy":      oid_buy,
        "oid_sell":     oid_sell,
    }


def close_position_real(pos: dict, account_id: str) -> dict:
    """Закрыть реальную позицию: SELL TATNP + BUY TATN одновременно."""
    ob_tatnp = get_orderbook("TATNP")
    ob_tatn  = get_orderbook("TATN")
    if not ob_tatnp or not ob_tatn:
        return {"status": "FAILED", "reason": "Нет стакана при закрытии"}

    lots       = pos["lots"]
    sell_tatnp = ob_tatnp["bids"][0][0]   # TATNP: продаём по bid[0]
    buy_tatn   = ob_tatn["asks"][0][0]    # TATN:  покупаем по ask[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_sell = ex.submit(_post_order, FIGI["TATNP"], "SELL", lots, sell_tatnp, account_id)
        f_buy  = ex.submit(_post_order, FIGI["TATN"],  "BUY",  lots, buy_tatn,   account_id)
    oid_sell = f_sell.result()
    oid_buy  = f_buy.result()

    if not oid_sell or not oid_buy:
        if oid_sell: _cancel_order(account_id, oid_sell)
        if oid_buy:  _cancel_order(account_id, oid_buy)
        return {"status": "FAILED", "reason": "Ошибка PostOrder при закрытии"}

    filled_sell = filled_buy_close = False
    deadline = time.time() + LEG_TIMEOUT_S

    while time.time() < deadline:
        if not filled_sell:
            s = _get_order_state(account_id, oid_sell)
            if s and s.get("executionReportStatus") == "EXECUTION_REPORT_STATUS_FILL":
                filled_sell = True
                sell_tatnp  = _q2f(s.get("averagePositionPrice", _f2q(sell_tatnp)))
        if not filled_buy_close:
            s = _get_order_state(account_id, oid_buy)
            if s and s.get("executionReportStatus") == "EXECUTION_REPORT_STATUS_FILL":
                filled_buy_close = True
                buy_tatn         = _q2f(s.get("averagePositionPrice", _f2q(buy_tatn)))
        if filled_sell and filled_buy_close:
            break
        time.sleep(1)

    if not (filled_sell and filled_buy_close):
        if not filled_sell:      _cancel_order(account_id, oid_sell)
        if not filled_buy_close: _cancel_order(account_id, oid_buy)
        return {"status": "PARTIAL", "reason": "Таймаут при закрытии — проверьте позиции вручную"}

    close_spread = buy_tatn - sell_tatnp
    pnl = (pos["entry_spread"] - close_spread) * lots

    _log_trade({
        "time":         datetime.now().isoformat(),
        "type":         "EXIT",
        "close_tatnp":  round(sell_tatnp,  2),
        "close_tatn":   round(buy_tatn,    2),
        "entry_spread": round(pos["entry_spread"], 2),
        "close_spread": round(close_spread, 2),
        "pnl_gross":    round(pnl,          2),
        "lots":         lots,
    })

    return {"status": "OK", "close_spread": close_spread, "pnl": pnl}


def _render_alert(signal_code: str) -> None:
    p         = st.session_state.alert_data
    elapsed   = int(time.time() - st.session_state.alert_time)
    remaining = max(0, ALERT_TIMEOUT_S - elapsed)
    label     = "🟡 LONG SCALP" if signal_code == "LONG_SCALP" else "🟢 LONG GOOD"
    rr_color  = "#3fb950" if p["rr"] >= MIN_RR else "#f78166"

    st.markdown(f"""
<style>
@keyframes mats-glow {{
    0%,100% {{ box-shadow: 0 0 8px #f78166;  border-color: #f78166; }}
    50%     {{ box-shadow: 0 0 28px #ff6644; border-color: #ff6644; }}
}}
.mats-alert-box {{
    border: 2px solid #f78166; border-radius: 10px;
    padding: 12px 18px; background: #1a0808; margin-bottom: 10px;
    animation: mats-glow 1s ease-in-out infinite;
}}
</style>
<div class="mats-alert-box">
  <span style="color:#f78166;font-size:15px;font-weight:700">🚨 СИГНАЛ: {label}</span>
  &nbsp;&nbsp;<span style="color:#7d8590;font-size:12px">Подтвердите вход или пропустите</span>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Купить TATNP",  f"{p['buy_price']:.2f} ₽",  f"{p['lots']} лотов")
    c2.metric("Продать TATN",  f"{p['sell_price']:.2f} ₽", f"{p['lots']} лотов")
    c3.metric("Спред входа",   f"{p['spread_entry']:.2f} ₽", f"→ цель {TARGET_SPREAD} ₽")
    c4.metric("⏱ Осталось",   f"{remaining} сек")

    c5, c6, c7 = st.columns(3)
    c5.metric("Прогноз (чистый)", f"+{p['profit_net']:.0f} ₽")
    c6.metric("Риск (стоп −3₽)", f"−{p['risk']:.0f} ₽")
    c7.metric("R / R",            f"1 : {p['rr']:.1f}")

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅  РАЗРЕШИТЬ", type="primary",
                     use_container_width=True, key="btn_approve"):
            st.session_state.alert_active       = False
            st.session_state.alert_action       = "APPROVED"
            st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S
    with col_no:
        if st.button("❌  ПРОПУСТИТЬ",
                     use_container_width=True, key="btn_skip"):
            st.session_state.alert_active       = False
            st.session_state.alert_action       = "SKIPPED"
            st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S


# ---------------------------------------------------------------------------
# Рендер открытой бумажной позиции
# ---------------------------------------------------------------------------
def _render_position(price_tatn: float, price_tatnp: float) -> None:
    pos            = st.session_state.paper_position
    current_spread = price_tatn - price_tatnp
    entry_spread   = pos["entry_spread"]
    lots           = pos["lots"]
    commission_close = lots * 4 * 0.05

    pnl_gross = (entry_spread - current_spread) * lots
    pnl_net   = pnl_gross - pos["commission"] - commission_close
    elapsed   = datetime.now() - pos["entry_time"]
    minutes   = int(elapsed.total_seconds() // 60)
    seconds   = int(elapsed.total_seconds() % 60)

    pnl_color  = "#3fb950" if pnl_net >= 0 else "#f85149"
    pnl_sign   = "+" if pnl_net >= 0 else ""
    stop_loss  = -STOP_SPREAD_ADD * lots
    stop_hit   = pnl_net <= stop_loss

    st.markdown(f"""
<style>
@keyframes pos-pulse {{
    0%,100% {{ border-color: {pnl_color}; box-shadow: 0 0 8px {pnl_color}44; }}
    50%     {{ border-color: {pnl_color}; box-shadow: 0 0 20px {pnl_color}88; }}
}}
.pos-box {{
    border: 2px solid {pnl_color}; border-radius: 10px;
    padding: 12px 18px; background: #0d1117; margin-bottom: 10px;
    animation: pos-pulse 2s ease-in-out infinite;
}}
</style>
<div class="pos-box">
  <span style="color:{pnl_color};font-size:15px;font-weight:700">
    📊 ПОЗИЦИЯ ОТКРЫТА &nbsp;·&nbsp;
    <span style="font-size:20px">{pnl_sign}{pnl_net:.0f} ₽</span>
  </span>
  &nbsp;&nbsp;<span style="color:#7d8590;font-size:12px">⏱ {minutes:02d}:{seconds:02d} в позиции</span>
  {"&nbsp;&nbsp;<span style='color:#f85149;font-weight:700'>⚠️ СТОП-ЛОСС!</span>" if stop_hit else ""}
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Спред входа",   f"{entry_spread:.2f} ₽")
    c2.metric("Спред сейчас",  f"{current_spread:.2f} ₽",
              delta=f"{current_spread - entry_spread:+.2f} ₽",
              delta_color="inverse")
    c3.metric("Лотов",         f"{lots}")
    c4.metric("P&L (чистый)",  f"{pnl_sign}{pnl_net:.0f} ₽")
    c5.metric("Стоп-лосс",     f"{stop_loss:.0f} ₽")

    if st.button("⛔  ЗАКРЫТЬ СДЕЛКУ", type="primary",
                 use_container_width=True, key="btn_close"):
        st.session_state.trade_history.append({
            "Время входа":   pos["entry_time"].strftime("%H:%M:%S"),
            "Спред входа":   f"{entry_spread:.2f}",
            "Спред выхода":  f"{current_spread:.2f}",
            "Лотов":         lots,
            "P&L ₽":         f"{pnl_sign}{pnl_net:.0f}",
            "Результат":     "✅ Прибыль" if pnl_net > 0 else "❌ Убыток",
        })
        st.session_state.paper_position   = None
        st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S


# ---------------------------------------------------------------------------
# Рендер реальной позиции
# ---------------------------------------------------------------------------
def _render_real_position(price_tatn: float, price_tatnp: float, account_id: str) -> None:
    pos            = st.session_state.real_position
    current_spread = price_tatn - price_tatnp
    entry_spread   = pos["entry_spread"]
    lots           = pos["lots"]
    commission_est = lots * 4 * 0.05

    pnl_gross = (entry_spread - current_spread) * lots
    pnl_net   = pnl_gross - commission_est
    elapsed   = datetime.now() - pos["entry_time"]
    minutes   = int(elapsed.total_seconds() // 60)
    seconds   = int(elapsed.total_seconds() % 60)

    stop_loss = -STOP_SPREAD_ADD * lots
    pnl_color = "#3fb950" if pnl_net >= 0 else "#f85149"
    pnl_sign  = "+" if pnl_net >= 0 else ""

    # --- Авто-стоп: срабатывает без кнопки ---
    if pnl_net <= stop_loss:
        st.error(f"🚨 АВТО-СТОП: P&L {pnl_sign}{pnl_net:.0f} ₽ достиг лимита {stop_loss:.0f} ₽")
        with st.spinner("Аварийное закрытие позиции..."):
            result = close_position_real(pos, account_id)
        if result["status"] == "OK":
            st.session_state.trade_history.append({
                "Время входа":  pos["entry_time"].strftime("%H:%M:%S"),
                "Спред входа":  f"{entry_spread:.2f}",
                "Спред выхода": f"{result['close_spread']:.2f}",
                "Лотов":        lots,
                "P&L ₽":        f"{result['pnl']:+.0f}",
                "Результат":    "🛑 Стоп-лосс",
                "Режим":        "РЕАЛЬНЫЙ",
            })
            st.session_state.real_position        = None
            st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S
        else:
            st.error(f"⚠️ Авто-стоп не сработал: {result['reason']} — ЗАКРОЙТЕ ВРУЧНУЮ в CScalp!")
        return

    # --- Тейк-профит: спред сошёлся ---
    if current_spread <= TARGET_SPREAD:
        st.success(f"🎯 ТЕЙК-ПРОФИТ — спред {current_spread:.2f} ₽ ≤ {TARGET_SPREAD} ₽. Закрывайте!")

    # --- Основной блок ---
    st.markdown(f"""
<style>
@keyframes real-pulse {{
    0%,100% {{ border-color: {pnl_color}; box-shadow: 0 0 10px {pnl_color}55; }}
    50%     {{ border-color: {pnl_color}; box-shadow: 0 0 28px {pnl_color}99; }}
}}
.real-pos-box {{
    border: 3px solid {pnl_color}; border-radius: 10px;
    padding: 12px 18px; background: #0d1117; margin-bottom: 10px;
    animation: real-pulse 1.5s ease-in-out infinite;
}}
</style>
<div class="real-pos-box">
  <span style="color:#f85149;font-size:11px;font-weight:700;letter-spacing:1px">
    ⚠️ РЕАЛЬНАЯ ПОЗИЦИЯ
  </span><br>
  <span style="color:{pnl_color};font-size:15px;font-weight:700">
    📊 P&L (ориентир.): <span style="font-size:20px">{pnl_sign}{pnl_net:.0f} ₽</span>
  </span>
  &nbsp;&nbsp;<span style="color:#7d8590;font-size:12px">⏱ {minutes:02d}:{seconds:02d} в позиции</span>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Спред входа",     f"{entry_spread:.2f} ₽")
    c2.metric("Спред сейчас",    f"{current_spread:.2f} ₽",
              delta=f"{current_spread - entry_spread:+.2f} ₽", delta_color="inverse")
    c3.metric("Лотов",           f"{lots}")
    c4.metric("P&L (ориентир.)", f"{pnl_sign}{pnl_net:.0f} ₽")
    c5.metric("Стоп-лосс",       f"{stop_loss:.0f} ₽")

    if st.button("⛔  ЗАКРЫТЬ РЕАЛЬНУЮ ПОЗИЦИЮ", type="primary",
                 use_container_width=True, key="btn_close_real"):
        with st.spinner("Закрываю позицию на бирже..."):
            result = close_position_real(pos, account_id)
        if result["status"] == "OK":
            st.session_state.trade_history.append({
                "Время входа":  pos["entry_time"].strftime("%H:%M:%S"),
                "Спред входа":  f"{entry_spread:.2f}",
                "Спред выхода": f"{result['close_spread']:.2f}",
                "Лотов":        lots,
                "P&L ₽":        f"{result['pnl']:+.0f}",
                "Результат":    "✅ Прибыль" if result["pnl"] > 0 else "❌ Убыток",
                "Режим":        "РЕАЛЬНЫЙ",
            })
            st.session_state.real_position        = None
            st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S
            st.success(f"✅ Позиция закрыта. P&L: {result['pnl']:+.0f} ₽")
        else:
            st.error(f"⚠️ {result['reason']} — проверьте позиции в CScalp!")


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
st.markdown(
    '<p style="font-size:1.1rem;font-weight:600;margin:0 0 4px 0">'
    '📈 MATS Dashboard — ТАТки (TATN / TATNP)</p>',
    unsafe_allow_html=True,
)
st.markdown("""
<style>
[data-testid="stMetricLabel"] p { font-size: 0.72rem !important; }
[data-testid="stMetricValue"]   { font-size: 1.15rem !important; line-height: 1.2; }
</style>
""", unsafe_allow_html=True)

# Боковая панель — рендерится один раз, не мигает
with st.sidebar:
    if DRY_RUN:
        st.markdown(
            '<div style="background:#3d2e00;border:1px solid #e3b341;border-radius:6px;'
            'padding:8px 12px;margin-bottom:12px;font-size:12px;color:#e3b341">'
            '🛡️ <b>DRY_RUN = ON</b><br>'
            '<span style="color:#7d8590">Ордера не выставляются</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#3d0f0e;border:2px solid #f85149;border-radius:6px;'
            'padding:8px 12px;margin-bottom:12px;font-size:12px;color:#f85149">'
            '⚠️ <b>DRY_RUN = OFF</b><br>'
            '<span style="color:#f85149">РЕАЛЬНЫЕ ОРДЕРА АКТИВНЫ</span></div>',
            unsafe_allow_html=True,
        )
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
    st.divider()
    try:
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _local_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        _local_ip = "localhost"
    st.caption(f"📱 Мобильный: http://{_local_ip}:8501")
    st.divider()
    paper_mode = st.toggle("📋 Тренажёр", value=True,
                           help="Виртуальные сделки на реальных котировках")
    st.divider()
    auto_mode = st.toggle(
        "🤖 Полный автомат", value=False,
        help="Авто-вход и авто-выход без нажатия кнопок. В тренажёре — для тестирования логики.",
    )
    if auto_mode and not paper_mode and not DRY_RUN:
        auto_daily_limit = st.number_input(
            "Дневной лимит убытка ₽", value=400, step=100, min_value=100,
            help="Авто-режим выключится если дневной убыток достигнет лимита",
        )
    else:
        auto_daily_limit = 99_999


# ---------------------------------------------------------------------------
# Инициализация состояния алерта
# ---------------------------------------------------------------------------
for _k, _v in [
    ("alert_active",         False),
    ("alert_data",           None),
    ("alert_time",           0.0),
    ("alert_action",         None),
    ("alert_cooldown_until", 0.0),
    ("paper_position",       None),
    ("real_position",        None),
    ("trade_history",        []),
    ("account_id",           None),
    ("auto_log",             []),
    ("daily_pnl",            0.0),
    ("daily_date",           ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Живой фрагмент — обновляется без перезагрузки страницы
# ---------------------------------------------------------------------------
@st.fragment(run_every=refresh_sec)
def live_dashboard(days: int, tf: int, paper: bool = True) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    tf_label = f"{tf} мин · {days} дн."

    # Кэшируем account_id один раз на сессию (только для реальной торговли)
    if st.session_state.account_id is None and not DRY_RUN:
        st.session_state.account_id = get_account_id()
    account_id = st.session_state.account_id

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
    ENTRY_SIGNALS = {"LONG_SCALP", "LONG_GOOD"}

    # Сбросить дневной P&L при смене даты
    today_str = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.daily_date != today_str:
        st.session_state.daily_pnl  = 0.0
        st.session_state.daily_date = today_str

    daily_limit_hit = st.session_state.daily_pnl <= -auto_daily_limit

    # =========================================================
    # ПОЛНЫЙ АВТОМАТ
    # =========================================================
    if auto_mode:
        # Баннер режима
        if daily_limit_hit:
            st.error(f"🛑 АВТО-РЕЖИМ: дневной лимит убытка достигнут "
                     f"({st.session_state.daily_pnl:+.0f} ₽). Торговля остановлена.")
        else:
            mode_label = "🤖 ПОЛНЫЙ АВТОМАТ" + (" · ТРЕНАЖЁР" if paper_mode else " · РЕАЛЬНЫЙ")
            st.markdown(
                f'<div style="background:#0d2137;border:1px solid #2196F3;border-radius:6px;'
                f'padding:6px 14px;margin-bottom:8px;font-size:13px;color:#2196F3">'
                f'<b>{mode_label}</b> &nbsp;·&nbsp; '
                f'Дневной P&L: <b>{st.session_state.daily_pnl:+.0f} ₽</b> '
                f'/ лимит −{auto_daily_limit:.0f} ₽</div>',
                unsafe_allow_html=True,
            )

        # --- Авто-тейк: позиция открыта + спред сошёлся ---
        if not daily_limit_hit:
            if (st.session_state.real_position is not None
                    and spread_rub <= TARGET_SPREAD
                    and not paper_mode and not DRY_RUN):
                pos = st.session_state.real_position
                with st.spinner("🤖 Авто-тейк..."):
                    result = close_position_real(pos, account_id)
                if result["status"] == "OK":
                    pnl = result["pnl"]
                    st.session_state.daily_pnl += pnl
                    st.session_state.trade_history.append({
                        "Время входа":  pos["entry_time"].strftime("%H:%M:%S"),
                        "Спред входа":  f"{pos['entry_spread']:.2f}",
                        "Спред выхода": f"{result['close_spread']:.2f}",
                        "Лотов":        pos["lots"],
                        "P&L ₽":        f"{pnl:+.0f}",
                        "Результат":    "✅ Авто-тейк",
                        "Режим":        "РЕАЛЬНЫЙ",
                    })
                    st.session_state.real_position        = None
                    st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S
                    st.session_state.auto_log.insert(0,
                        f"{now}  ✅ АВТО-ТЕЙК: P&L {pnl:+.0f} ₽, спред {result['close_spread']:.2f} ₽"
                    )

            # Авто-тейк в тренажёре
            if (st.session_state.paper_position is not None
                    and spread_rub <= TARGET_SPREAD
                    and paper_mode):
                pos = st.session_state.paper_position
                commission_close = pos["lots"] * 4 * 0.05
                pnl_gross = (pos["entry_spread"] - spread_rub) * pos["lots"]
                pnl_net   = pnl_gross - pos["commission"] - commission_close
                st.session_state.trade_history.append({
                    "Время входа":  pos["entry_time"].strftime("%H:%M:%S"),
                    "Спред входа":  f"{pos['entry_spread']:.2f}",
                    "Спред выхода": f"{spread_rub:.2f}",
                    "Лотов":        pos["lots"],
                    "P&L ₽":        f"{pnl_net:+.0f}",
                    "Результат":    "✅ Авто-тейк",
                    "Режим":        "ТРЕНАЖЁР",
                })
                st.session_state.paper_position        = None
                st.session_state.alert_cooldown_until  = time.time() + ALERT_COOLDOWN_S
                st.session_state.auto_log.insert(0,
                    f"{now}  ✅ АВТО-ТЕЙК (тренажёр): P&L {pnl_net:+.0f} ₽"
                )

            # --- Авто-вход: нет позиции + сигнал + cooldown прошёл ---
            no_position = (st.session_state.real_position is None
                           and st.session_state.paper_position is None)
            if (no_position
                    and code in ENTRY_SIGNALS
                    and time.time() > st.session_state.alert_cooldown_until):
                p = calc_entry_params(spread_rub, price_tatn, price_tatnp)
                if p["rr"] >= MIN_RR:
                    if paper_mode or DRY_RUN:
                        # Авто-вход в тренажёре
                        st.session_state.paper_position = {
                            "entry_time":   datetime.now(),
                            "entry_spread": spread_rub,
                            "entry_tatn":   price_tatn,
                            "entry_tatnp":  price_tatnp,
                            "lots":         p["lots"],
                            "commission":   p["commission"],
                        }
                        st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S
                        st.session_state.auto_log.insert(0,
                            f"{now}  🤖 АВТО-ВХОД (тренажёр): {code}, "
                            f"спред {spread_rub:.2f} ₽, {p['lots']} лот., R/R {p['rr']:.1f}"
                        )
                    else:
                        # Авто-вход реальный
                        if not account_id:
                            st.session_state.auto_log.insert(0, f"{now}  ❌ Нет account_id")
                        else:
                            with st.spinner(f"🤖 Авто-вход {code}..."):
                                result = execute_entry(p, account_id)
                            if result["status"] == "OK":
                                st.session_state.real_position = {
                                    "entry_time":   datetime.now(),
                                    "entry_spread": result["entry_spread"],
                                    "entry_tatn":   result["sell_price"],
                                    "entry_tatnp":  result["buy_price"],
                                    "lots":         p["lots"],
                                    "oid_buy":      result["oid_buy"],
                                    "oid_sell":     result["oid_sell"],
                                }
                                st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S
                                st.session_state.auto_log.insert(0,
                                    f"{now}  🤖 АВТО-ВХОД: {code}, "
                                    f"спред {result['entry_spread']:.2f} ₽, "
                                    f"{p['lots']} лот., R/R {p['rr']:.1f}"
                                )
                            else:
                                st.session_state.auto_log.insert(0,
                                    f"{now}  ❌ Авто-вход неудача: {result['reason']}"
                                )
                else:
                    st.session_state.auto_log.insert(0,
                        f"{now}  ⏭ Пропуск {code}: R/R {p['rr']:.1f} < {MIN_RR}"
                    )

        # Обрезаем лог до 30 записей
        st.session_state.auto_log = st.session_state.auto_log[:30]

    # =========================================================
    # ПОЛУАВТОМАТ (ручное подтверждение)
    # =========================================================
    else:
        # Активировать алерт при новом сигнале входа
        if (code in ENTRY_SIGNALS
                and not st.session_state.alert_active
                and time.time() > st.session_state.alert_cooldown_until):
            st.session_state.alert_active = True
            st.session_state.alert_time   = time.time()
            st.session_state.alert_data   = calc_entry_params(spread_rub, price_tatn, price_tatnp)
            st.session_state.alert_action = None

        # Таймаут алерта
        if st.session_state.alert_active:
            if time.time() - st.session_state.alert_time > ALERT_TIMEOUT_S:
                st.session_state.alert_active         = False
                st.session_state.alert_action         = "TIMEOUT"
                st.session_state.alert_cooldown_until = time.time() + ALERT_COOLDOWN_S

        # Открыть позицию при нажатии РАЗРЕШИТЬ
        if (st.session_state.alert_action == "APPROVED"
                and st.session_state.paper_position is None
                and st.session_state.real_position is None
                and st.session_state.alert_data is not None):
            p = st.session_state.alert_data
            if paper_mode or DRY_RUN:
                st.session_state.paper_position = {
                    "entry_time":   datetime.now(),
                    "entry_spread": spread_rub,
                    "entry_tatn":   price_tatn,
                    "entry_tatnp":  price_tatnp,
                    "lots":         p["lots"],
                    "commission":   p["commission"],
                }
            else:
                if not account_id:
                    st.error("❌ Не удалось получить account_id — проверьте TINKOFF_TOKEN")
                else:
                    with st.spinner("⚙️ Выставляю ордера на биржу..."):
                        result = execute_entry(p, account_id)
                    if result["status"] == "OK":
                        st.session_state.real_position = {
                            "entry_time":   datetime.now(),
                            "entry_spread": result["entry_spread"],
                            "entry_tatn":   result["sell_price"],
                            "entry_tatnp":  result["buy_price"],
                            "lots":         p["lots"],
                            "oid_buy":      result["oid_buy"],
                            "oid_sell":     result["oid_sell"],
                        }
                    else:
                        st.error(f"❌ Вход не удался: {result['reason']}")
            st.session_state.alert_action = None

    # --- Блок позиции (если открыта) ---
    if st.session_state.real_position is not None:
        _render_real_position(price_tatn, price_tatnp, account_id)
    elif st.session_state.paper_position is not None:
        _render_position(price_tatn, price_tatnp)
    elif not auto_mode and st.session_state.alert_active:
        _render_alert(code)
    else:
        # Строка сигнала
        label = f"**{code}** — {desc}"
        if stype == "success":   st.success(label)
        elif stype == "warning": st.warning(label)
        elif stype == "error":   st.error(label)
        else:                    st.info(label)

        if not auto_mode:
            action = st.session_state.alert_action
            if action == "SKIPPED":
                st.info("⏭ Сигнал пропущен. Новый алерт через 2 мин.")
            elif action == "TIMEOUT":
                st.warning("⏱ Время истекло — сигнал отменён. Новый алерт через 2 мин.")

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

    # --- История сделок тренажёра ---
    history = st.session_state.trade_history
    if history:
        st.divider()
        st.markdown("**📋 История сделок тренажёра**")
        df_hist = pd.DataFrame(history[::-1])   # новые сверху
        total   = sum(float(r["P&L ₽"]) for r in history)
        wins    = sum(1 for r in history if r["Результат"].startswith("✅"))
        sign    = "+" if total >= 0 else ""
        st.caption(
            f"Сделок: {len(history)}  |  "
            f"Прибыльных: {wins}  |  "
            f"Итого P&L: **{sign}{total:.0f} ₽**"
        )
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        if st.button("🗑 Очистить историю", key="btn_clear_history"):
            st.session_state.trade_history = []

    # --- Лог авто-режима ---
    if auto_mode and st.session_state.auto_log:
        st.divider()
        with st.expander("🤖 Лог автомата", expanded=True):
            for entry in st.session_state.auto_log:
                st.caption(entry)
            if st.button("🗑 Очистить лог", key="btn_clear_log"):
                st.session_state.auto_log = []


# Запускаем живой фрагмент
live_dashboard(days=days_history, tf=timeframe, paper=paper_mode)
