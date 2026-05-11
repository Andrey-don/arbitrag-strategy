"""Стратегия парного арбитража SBER/SBERP.

spread = SBER_close - SBERP_close
Вход: spread >= entry_scalp И ratio >= ratio_scalp → SELL SBER, BUY SBERP
Выход TP: spread <= target
Выход SL: spread >= entry_spread + stop_add
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.utils.config import BotConfig
from src.strategies.regime_filter import RegimeFilter


class SignalType(Enum):
    NONE = "NONE"
    ENTRY_SCALP = "ENTRY_SCALP"
    ENTRY_GOOD = "ENTRY_GOOD"
    EXIT_TP = "EXIT_TP"
    EXIT_SL = "EXIT_SL"


@dataclass
class Signal:
    type: SignalType
    spread: float
    ratio: float
    timestamp: datetime
    entry_spread: float = 0.0   # заполняется только для EXIT сигналов


class PairArbitrageStrategy:
    def __init__(self, config: BotConfig):
        self.cfg = config
        self.regime = RegimeFilter(config.regime_window_days, config.regime_threshold)
        self._in_position = False
        self._entry_spread: float = 0.0
        self._stop_spread: float = 0.0

    @property
    def in_position(self) -> bool:
        return self._in_position

    def force_close(self) -> None:
        """Принудительно сбросить позицию (при ручном вмешательстве)."""
        self._in_position = False

    def on_candle(self, timestamp: datetime, spread: float, ratio: float) -> Signal:
        """Вызывается на каждом закрытии свечи. Возвращает Signal."""
        self.regime.update(timestamp, spread)

        if self._in_position:
            return self._check_exit(timestamp, spread, ratio)
        return self._check_entry(timestamp, spread, ratio)

    # ------------------------------------------------------------------

    def _check_entry(self, timestamp: datetime, spread: float, ratio: float) -> Signal:
        if self.regime.is_blocked():
            return Signal(SignalType.NONE, spread, ratio, timestamp)

        if spread > self.cfg.max_entry_spread:
            return Signal(SignalType.NONE, spread, ratio, timestamp)

        if spread >= self.cfg.entry_good and ratio >= self.cfg.ratio_good:
            self._open(spread)
            return Signal(SignalType.ENTRY_GOOD, spread, ratio, timestamp)

        if spread >= self.cfg.entry_scalp and ratio >= self.cfg.ratio_scalp:
            self._open(spread)
            return Signal(SignalType.ENTRY_SCALP, spread, ratio, timestamp)

        return Signal(SignalType.NONE, spread, ratio, timestamp)

    def _check_exit(self, timestamp: datetime, spread: float, ratio: float) -> Signal:
        if spread <= self.cfg.target:
            self._in_position = False
            return Signal(SignalType.EXIT_TP, spread, ratio, timestamp,
                          entry_spread=self._entry_spread)

        if spread >= self._stop_spread:
            self._in_position = False
            return Signal(SignalType.EXIT_SL, spread, ratio, timestamp,
                          entry_spread=self._entry_spread)

        return Signal(SignalType.NONE, spread, ratio, timestamp)

    def _open(self, spread: float) -> None:
        self._in_position = True
        self._entry_spread = spread + self.cfg.slippage
        self._stop_spread = self._entry_spread + self.cfg.stop_add
