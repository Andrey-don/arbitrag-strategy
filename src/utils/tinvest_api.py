"""T-Invest REST API — низкоуровневый враппер."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


_TF_MAP = {
    1:  "CANDLE_INTERVAL_1_MIN",
    5:  "CANDLE_INTERVAL_5_MIN",
    15: "CANDLE_INTERVAL_15_MIN",
    60: "CANDLE_INTERVAL_HOUR",
}


class TInvestAPI:
    def __init__(self, token: str, base_url: str = "https://invest-public-api.tinkoff.ru/rest"):
        self._token = token
        self._base = base_url

    # ------------------------------------------------------------------
    # Низкоуровневые хелперы
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    @staticmethod
    def _q2f(q: dict) -> float:
        return float(q.get("units", 0)) + q.get("nano", 0) / 1_000_000_000

    @staticmethod
    def _f2q(price: float) -> dict:
        units = int(price)
        nano = round((price - units) * 1_000_000_000)
        return {"units": units, "nano": nano}

    def _post(self, endpoint: str, body: dict, timeout: int = 10) -> dict:
        try:
            r = requests.post(
                f"{self._base}/{endpoint}",
                json=body,
                headers=self._headers(),
                timeout=timeout,
            )
            return r.json()
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Рыночные данные
    # ------------------------------------------------------------------

    def get_last_price(self, figi: str) -> float | None:
        data = self._post(
            "tinkoff.public.invest.api.contract.v1.MarketDataService/GetLastPrices",
            {"figi": [figi]},
            timeout=5,
        )
        try:
            price = data["lastPrices"][0]["price"]
            v = self._q2f(price)
            return v if v > 0 else None
        except (KeyError, IndexError):
            return None

    def get_candles(self, figi: str, days: int, interval: int) -> pd.DataFrame:
        ti_interval = _TF_MAP.get(interval, "CANDLE_INTERVAL_15_MIN")
        to_dt = datetime.now(timezone.utc)
        from_dt = to_dt - timedelta(days=days)
        data = self._post(
            "tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles",
            {
                "figi":     figi,
                "from":     from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to":       to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "interval": ti_interval,
            },
        )
        candles = data.get("candles", [])
        if not candles:
            return pd.DataFrame()
        rows = []
        for c in candles:
            close = self._q2f(c["close"])
            if close == 0:
                continue
            ts = pd.to_datetime(c["time"]) + timedelta(hours=3)  # UTC → МСК
            rows.append({"begin": ts.replace(tzinfo=None), "close": close})
        return pd.DataFrame(rows)

    def get_orderbook(self, figi: str, depth: int = 5) -> dict | None:
        data = self._post(
            "tinkoff.public.invest.api.contract.v1.MarketDataService/GetOrderBook",
            {"figi": figi, "depth": depth},
            timeout=5,
        )
        asks = [self._q2f(i["price"]) for i in data.get("asks", [])]
        bids = [self._q2f(i["price"]) for i in data.get("bids", [])]
        if not asks or not bids:
            return None
        return {"best_ask": asks[0], "best_bid": bids[0]}

    # ------------------------------------------------------------------
    # Счёт
    # ------------------------------------------------------------------

    def get_account_id(self) -> str | None:
        data = self._post(
            "tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts",
            {},
            timeout=5,
        )
        accounts = data.get("accounts", [])
        return accounts[0]["id"] if accounts else None

    # ------------------------------------------------------------------
    # Ордера
    # ------------------------------------------------------------------

    def post_order(self, figi: str, direction: str, qty: int, price: float, account_id: str) -> str | None:
        """direction: 'BUY' или 'SELL'. Возвращает orderId или None."""
        data = self._post(
            "tinkoff.public.invest.api.contract.v1.OrdersService/PostOrder",
            {
                "figi":      figi,
                "quantity":  qty,
                "price":     self._f2q(price),
                "direction": f"ORDER_DIRECTION_{direction}",
                "accountId": account_id,
                "orderType": "ORDER_TYPE_LIMIT",
                "orderId":   str(uuid.uuid4()),
            },
        )
        return data.get("orderId")

    def get_order_state(self, account_id: str, order_id: str) -> dict:
        return self._post(
            "tinkoff.public.invest.api.contract.v1.OrdersService/GetOrderState",
            {"accountId": account_id, "orderId": order_id},
            timeout=5,
        )

    def cancel_order(self, account_id: str, order_id: str) -> None:
        self._post(
            "tinkoff.public.invest.api.contract.v1.OrdersService/CancelOrder",
            {"accountId": account_id, "orderId": order_id},
            timeout=5,
        )

    def post_stop_order(
        self, figi: str, direction: str, qty: int, stop_price: float, account_id: str
    ) -> None:
        """Серверный стоп-лосс — исполняется брокером даже при отключении интернета."""
        self._post(
            "tinkoff.public.invest.api.contract.v1.StopOrdersService/PostStopOrder",
            {
                "figi":           figi,
                "quantity":       qty,
                "stopPrice":      self._f2q(stop_price),
                "direction":      f"STOP_ORDER_DIRECTION_{direction}",
                "accountId":      account_id,
                "stopOrderType":  "STOP_ORDER_TYPE_STOP_LOSS",
                "expirationType": "STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL",
            },
        )
