"""Order Executor — параллельное исполнение обеих ног парного арбитража.

Правила:
- Обе ноги одновременно через ThreadPoolExecutor
- После входа — немедленно PostStopOrder на сервер T-Invest (защита при обрыве инета)
- Таймаут 15 сек на заполнение, иначе отмена обеих ног
- DRY_RUN=true: всё логируется, реальных ордеров нет
"""
from __future__ import annotations

import concurrent.futures
import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.config import BotConfig
from src.utils.tinvest_api import TInvestAPI
from src.strategies.pair_strategy import Signal, SignalType

log = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


@dataclass
class Position:
    open_time: datetime
    entry_spread: float
    lots: int
    stop_spread: float
    sell_price: float = 0.0   # цена продажи SBER
    buy_price: float = 0.0    # цена покупки SBERP
    order_ids: dict = field(default_factory=dict)


class OrderExecutor:
    def __init__(self, config: BotConfig, api: TInvestAPI):
        self.cfg = config
        self.api = api
        self.position: Position | None = None

    # ------------------------------------------------------------------
    # Вход
    # ------------------------------------------------------------------

    def enter(self, signal: Signal, account_id: str | None) -> bool:
        """SELL SBER + BUY SBERP. Возвращает True если вход выполнен."""
        cfg = self.cfg

        if cfg.dry_run:
            log.info(
                "[DRY_RUN] ВХОД %s: SELL %s %d лот, BUY %s %d лот  spread=%.4f₽  ratio=%.6f",
                signal.type.value, cfg.ticker_a, cfg.lots, cfg.ticker_b, cfg.lots,
                signal.spread, signal.ratio,
            )
            self.position = Position(
                open_time=signal.timestamp,
                entry_spread=signal.spread + cfg.slippage,
                lots=cfg.lots,
                stop_spread=signal.spread + cfg.slippage + cfg.stop_add,
            )
            self._log_trade({
                "event": "ENTRY", "type": signal.type.value,
                "timestamp": signal.timestamp, "spread": signal.spread,
                "entry_spread": self.position.entry_spread,
                "lots": cfg.lots, "dry_run": True,
            })
            return True

        # Стакан для агрессивных лимиток
        ob_a = self.api.get_orderbook(cfg.figi_a)
        ob_b = self.api.get_orderbook(cfg.figi_b)
        if not ob_a or not ob_b:
            log.error("ВХОД: нет стакана")
            return False

        sell_price = ob_a["best_bid"]  # SBER продаём по лучшему bid
        buy_price  = ob_b["best_ask"]  # SBERP покупаем по лучшему ask

        # Обе ноги одновременно
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_sell = ex.submit(self.api.post_order, cfg.figi_a, "SELL", cfg.lots, sell_price, account_id)
            f_buy  = ex.submit(self.api.post_order, cfg.figi_b, "BUY",  cfg.lots, buy_price,  account_id)
        oid_sell = f_sell.result()
        oid_buy  = f_buy.result()

        if not oid_sell or not oid_buy:
            if oid_sell: self.api.cancel_order(account_id, oid_sell)
            if oid_buy:  self.api.cancel_order(account_id, oid_buy)
            log.error("ВХОД: ошибка PostOrder")
            return False

        if not self._wait_fill([oid_sell, oid_buy], account_id):
            self.api.cancel_order(account_id, oid_sell)
            self.api.cancel_order(account_id, oid_buy)
            log.error("ВХОД: таймаут заполнения")
            return False

        entry_spread = sell_price - buy_price
        stop_spread  = entry_spread + cfg.stop_add

        # Серверные стоп-лоссы (защита при обрыве интернета)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            ex.submit(self.api.post_stop_order,
                      cfg.figi_a, "BUY",  cfg.lots, sell_price + cfg.stop_add, account_id)
            ex.submit(self.api.post_stop_order,
                      cfg.figi_b, "SELL", cfg.lots, buy_price  - cfg.stop_add, account_id)

        self.position = Position(
            open_time=signal.timestamp,
            entry_spread=entry_spread,
            lots=cfg.lots,
            stop_spread=stop_spread,
            sell_price=sell_price,
            buy_price=buy_price,
            order_ids={"sell": oid_sell, "buy": oid_buy},
        )
        log.info("ВХОД OK %s: entry=%.4f₽ stop=%.4f₽", signal.type.value, entry_spread, stop_spread)
        self._log_trade({
            "event": "ENTRY", "type": signal.type.value,
            "timestamp": signal.timestamp, "spread": signal.spread,
            "entry_spread": entry_spread, "lots": cfg.lots, "dry_run": False,
        })
        return True

    # ------------------------------------------------------------------
    # Выход
    # ------------------------------------------------------------------

    def exit(self, signal: Signal, account_id: str | None) -> float | None:
        """BUY SBER + SELL SBERP. Возвращает P&L или None при ошибке."""
        if not self.position:
            return None
        pos = self.position
        cfg = self.cfg
        commission = pos.lots * 4 * cfg.commission_per_lot

        if cfg.dry_run:
            pnl = (pos.entry_spread - signal.spread) * pos.lots - commission
            log.info(
                "[DRY_RUN] ВЫХОД %s: spread=%.4f₽  P&L=%.2f₽",
                signal.type.value, signal.spread, pnl,
            )
            self._log_trade({
                "event": "EXIT", "type": signal.type.value,
                "timestamp": signal.timestamp, "spread": signal.spread,
                "entry_spread": pos.entry_spread, "pnl": pnl,
                "lots": pos.lots, "dry_run": True,
            })
            self.position = None
            return pnl

        ob_a = self.api.get_orderbook(cfg.figi_a)
        ob_b = self.api.get_orderbook(cfg.figi_b)
        if not ob_a or not ob_b:
            log.error("ВЫХОД: нет стакана")
            return None

        buy_price  = ob_a["best_ask"]  # SBER откупаем по ask
        sell_price = ob_b["best_bid"]  # SBERP продаём по bid

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_buy  = ex.submit(self.api.post_order, cfg.figi_a, "BUY",  pos.lots, buy_price,  account_id)
            f_sell = ex.submit(self.api.post_order, cfg.figi_b, "SELL", pos.lots, sell_price, account_id)
        oid_buy  = f_buy.result()
        oid_sell = f_sell.result()

        if not oid_buy or not oid_sell:
            if oid_buy:  self.api.cancel_order(account_id, oid_buy)
            if oid_sell: self.api.cancel_order(account_id, oid_sell)
            log.error("ВЫХОД: ошибка PostOrder")
            return None

        self._wait_fill([oid_buy, oid_sell], account_id)
        pnl = (pos.sell_price - buy_price) * pos.lots + (sell_price - pos.buy_price) * pos.lots - commission
        log.info("ВЫХОД OK %s: P&L=%.2f₽", signal.type.value, pnl)
        self._log_trade({
            "event": "EXIT", "type": signal.type.value,
            "timestamp": signal.timestamp, "spread": signal.spread,
            "entry_spread": pos.entry_spread, "pnl": pnl,
            "lots": pos.lots, "dry_run": False,
        })
        self.position = None
        return pnl

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------

    def _wait_fill(self, order_ids: list[str], account_id: str) -> bool:
        deadline = time.time() + self.cfg.leg_timeout_s
        pending = set(order_ids)
        while time.time() < deadline and pending:
            time.sleep(1)
            for oid in list(pending):
                state = self.api.get_order_state(account_id, oid)
                status = state.get("executionReportStatus", "")
                if "FILL" in status:
                    pending.discard(oid)
        return len(pending) == 0

    @staticmethod
    def _log_trade(data: dict) -> None:
        _LOG_DIR.mkdir(exist_ok=True)
        path = _LOG_DIR / "trades.csv"
        file_exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
