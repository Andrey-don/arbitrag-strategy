"""Конфигурация бота SBER/SBERP — все параметры из walk-forward оптимизации."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class BotConfig:
    # Пара
    ticker_a: str = "SBER"
    ticker_b: str = "SBERP"
    figi_a: str = "BBG004730N88"
    figi_b: str = "BBG0047315Y7"
    hedge: int = 1

    # Пороги стратегии (walk-forward оптимизация на 2024)
    entry_scalp: float = 0.30       # ₽ — скальп вход
    ratio_scalp: float = 1.0005
    entry_good: float = 0.40        # ₽ — хороший вход
    ratio_good: float = 1.0008
    target: float = 0.08            # ₽ — цель схождения (take profit)
    stop_add: float = 0.50          # ₽ — стоп = entry_spread + stop_add
    max_entry_spread: float = 0.50  # ₽ — не входить если спред выше
    slippage: float = 0.05          # ₽ — оценка проскальзывания
    commission_per_lot: float = 0.05  # ₽ за лот

    # Режимный фильтр (ключевое открытие walk-forward)
    # |rolling_mean(спред, 20 дней)| > 0.40₽ → не входить
    regime_window_days: int = 20
    regime_threshold: float = 0.40

    # Исполнение
    lots: int = 10
    leg_timeout_s: int = 15         # сек ожидания заполнения каждой ноги
    candle_interval_min: int = 15   # таймфрейм (начать на 15, потом 5min)
    poll_interval_s: int = 60       # интервал опроса API

    # Риск
    deposit_rub: float = 10_000.0
    max_drawdown_pct: float = 0.20  # 20% просадки → стоп торговли

    # T-Invest API
    base_url: str = "https://invest-public-api.tinkoff.ru/rest"

    @property
    def token(self) -> str:
        return os.getenv("TINKOFF_TOKEN", "")

    @property
    def dry_run(self) -> bool:
        return os.getenv("DRY_RUN", "true").lower() != "false"
