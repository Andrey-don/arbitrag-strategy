"""Режимный фильтр: |rolling_mean(спред, 20 дней)| > 0.40₽ → не входить.

Ключевое открытие walk-forward теста:
- 2025: mean спреда = +2₽ весь год (структурный/дивидендный сдвиг)
- Mean-reversion не работает при трендовом спреде
- Фильтр превращает -42₽ убыток в -3₽
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta


class RegimeFilter:
    """Каузальный rolling mean по времени (только прошлые данные)."""

    def __init__(self, window_days: int = 20, threshold: float = 0.40):
        self.window_days = window_days
        self.threshold = threshold
        self._data: deque[tuple[datetime, float]] = deque()

    def update(self, timestamp: datetime, spread: float) -> None:
        self._data.append((timestamp, spread))
        cutoff = timestamp - timedelta(days=self.window_days)
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def rolling_mean(self) -> float:
        if not self._data:
            return 0.0
        return sum(s for _, s in self._data) / len(self._data)

    def is_blocked(self) -> bool:
        """True → рынок трендовый, вход запрещён."""
        if len(self._data) < 10:
            return False
        return abs(self.rolling_mean()) > self.threshold

    def status(self) -> str:
        mean = self.rolling_mean()
        return (
            f"regime_mean={mean:+.3f}₽  n={len(self._data)}  "
            f"{'🔴 BLOCKED' if self.is_blocked() else '🟢 OK'}"
        )
