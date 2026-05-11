"""Получение рыночных данных: онлайн (T-Invest API) + офлайн (data/history/)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.config import BotConfig
from src.utils.tinvest_api import TInvestAPI

_HIST_DIR = Path(__file__).resolve().parents[2] / "data" / "history"


class DataFetcher:
    def __init__(self, config: BotConfig, api: TInvestAPI):
        self.cfg = config
        self.api = api

    def get_prices(self) -> tuple[float, float] | None:
        """Текущие last prices пары. None если API недоступен."""
        pa = self.api.get_last_price(self.cfg.figi_a)
        pb = self.api.get_last_price(self.cfg.figi_b)
        if pa is None or pb is None:
            return None
        return pa, pb

    def get_spread_history_online(self, days: int = 25) -> list[tuple[datetime, float, float]]:
        """История спреда из T-Invest API: (timestamp, spread, ratio)."""
        iv = self.cfg.candle_interval_min
        df_a = self.api.get_candles(self.cfg.figi_a, days, iv)
        df_b = self.api.get_candles(self.cfg.figi_b, days, iv)
        if df_a.empty or df_b.empty:
            return []
        merged = pd.merge(df_a, df_b, on="begin", suffixes=("_a", "_b")).sort_values("begin")
        result = []
        for _, row in merged.iterrows():
            spread = row["close_a"] - row["close_b"]
            ratio = row["close_a"] / row["close_b"] if row["close_b"] > 0 else 1.0
            result.append((row["begin"].to_pydatetime(), spread, ratio))
        return result

    def get_spread_history_offline(self, days: int | None = None) -> list[tuple[datetime, float, float]]:
        """История спреда из CSV-файлов data/history/. Используется для инициализации фильтра."""
        iv = self.cfg.candle_interval_min
        path_a = _HIST_DIR / f"{self.cfg.ticker_a}_{iv}min.csv"
        path_b = _HIST_DIR / f"{self.cfg.ticker_b}_{iv}min.csv"
        if not path_a.exists() or not path_b.exists():
            return []
        df_a = pd.read_csv(path_a, parse_dates=["begin"])
        df_b = pd.read_csv(path_b, parse_dates=["begin"])
        merged = pd.merge(df_a, df_b, on="begin", suffixes=("_a", "_b")).sort_values("begin")
        if days is not None:
            cutoff = merged["begin"].max() - pd.Timedelta(days=days)
            merged = merged[merged["begin"] >= cutoff]
        result = []
        for _, row in merged.iterrows():
            spread = row["close_a"] - row["close_b"]
            ratio = row["close_a"] / row["close_b"] if row["close_b"] > 0 else 1.0
            ts = row["begin"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            result.append((ts, spread, ratio))
        return result

    def get_spread_history(self, days: int = 25) -> list[tuple[datetime, float, float]]:
        """Приоритет: офлайн CSV → онлайн API."""
        offline = self.get_spread_history_offline(days)
        if offline:
            return offline
        return self.get_spread_history_online(days)
