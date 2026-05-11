"""Управление рисками: просадка, лимиты."""
from __future__ import annotations

import logging

from src.utils.config import BotConfig

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: BotConfig):
        self.cfg = config
        self._capital = config.deposit_rub
        self._peak = config.deposit_rub
        self._trade_count = 0
        self._wins = 0
        self._losses = 0

    def update_pnl(self, pnl: float) -> None:
        self._capital += pnl
        self._peak = max(self._peak, self._capital)
        self._trade_count += 1
        if pnl >= 0:
            self._wins += 1
        else:
            self._losses += 1

    @property
    def drawdown(self) -> float:
        if self._peak == 0:
            return 0.0
        return (self._peak - self._capital) / self._peak

    @property
    def win_rate(self) -> float:
        if self._trade_count == 0:
            return 0.0
        return self._wins / self._trade_count

    def can_enter(self) -> bool:
        if self.drawdown >= self.cfg.max_drawdown_pct:
            log.warning(
                "СТОП: просадка %.1f%% ≥ лимита %.1f%% — торговля остановлена",
                self.drawdown * 100,
                self.cfg.max_drawdown_pct * 100,
            )
            return False
        return True

    def status(self) -> str:
        return (
            f"Капитал: {self._capital:.0f}₽  "
            f"Просадка: {self.drawdown * 100:.1f}%  "
            f"Сделок: {self._trade_count}  "
            f"WR: {self.win_rate * 100:.0f}%"
        )
