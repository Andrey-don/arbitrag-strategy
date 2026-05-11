"""
MATS Bot — SBER/SBERP парный арбитраж.

Запуск:  python -m src.bot
Стоп:    Ctrl+C

Переменные среды (.env):
  TINKOFF_TOKEN=...   — токен T-Invest API
  DRY_RUN=true        — симуляция (не отправлять реальные ордера)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from dotenv import load_dotenv

from src.utils.config import BotConfig
from src.utils.tinvest_api import TInvestAPI
from src.data.fetcher import DataFetcher
from src.strategies.pair_strategy import PairArbitrageStrategy, SignalType
from src.risk.risk_manager import RiskManager
from src.execution.order_executor import OrderExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    cfg      = BotConfig()
    api      = TInvestAPI(token=cfg.token, base_url=cfg.base_url)
    fetcher  = DataFetcher(cfg, api)
    strategy = PairArbitrageStrategy(cfg)
    risk     = RiskManager(cfg)
    executor = OrderExecutor(cfg, api)

    log.info(
        "=== MATS Bot === пара=%s/%s  TF=%dmin  DRY_RUN=%s",
        cfg.ticker_a, cfg.ticker_b, cfg.candle_interval_min, cfg.dry_run,
    )

    # Авторизация (только для реальной торговли)
    account_id: str | None = None
    if not cfg.dry_run:
        account_id = api.get_account_id()
        if not account_id:
            log.error("Не удалось получить account_id. Проверьте TINKOFF_TOKEN.")
            return
        log.info("Счёт: %s", account_id)

    # Инициализация режимного фильтра историческими данными
    log.info("Загрузка истории для режимного фильтра (последние %d дней)...", cfg.regime_window_days + 5)
    history = fetcher.get_spread_history(days=cfg.regime_window_days + 5)
    if history:
        for ts, spread, ratio in history:
            strategy.regime.update(ts, spread)
        log.info("Загружено %d свечей. %s", len(history), strategy.regime.status())
    else:
        log.warning("История недоступна — режимный фильтр не инициализирован (нужно %d свечей).", 10)

    # Основной цикл
    last_processed_minute: int = -1

    log.info("Запуск цикла (опрос каждые %ds, свечи %dmin)...", cfg.poll_interval_s, cfg.candle_interval_min)

    while True:
        try:
            now = datetime.now()
            # Синхронизация: обрабатывать только один раз за интервал свечи
            candle_minute = (now.hour * 60 + now.minute) // cfg.candle_interval_min
            if candle_minute == last_processed_minute:
                time.sleep(cfg.poll_interval_s)
                continue
            last_processed_minute = candle_minute

            prices = fetcher.get_prices()
            if prices is None:
                log.warning("Нет котировок, пропуск тика...")
                time.sleep(cfg.poll_interval_s)
                continue

            price_a, price_b = prices
            spread = price_a - price_b
            ratio  = price_a / price_b if price_b > 0 else 1.0

            signal = strategy.on_candle(now, spread, ratio)

            log.info(
                "Тик: %s=%.2f  %s=%.2f  spread=%.4f₽  ratio=%.6f  %s  %s",
                cfg.ticker_a, price_a, cfg.ticker_b, price_b,
                spread, ratio,
                strategy.regime.status(),
                "IN_POS" if strategy.in_position else "FLAT",
            )

            # Обработка сигналов
            if signal.type in (SignalType.ENTRY_SCALP, SignalType.ENTRY_GOOD):
                if risk.can_enter():
                    log.info(">>> СИГНАЛ ВХОД %s <<<", signal.type.value)
                    ok = executor.enter(signal, account_id)
                    if not ok:
                        strategy.force_close()  # откат состояния стратегии

            elif signal.type in (SignalType.EXIT_TP, SignalType.EXIT_SL):
                log.info(">>> СИГНАЛ ВЫХОД %s <<<", signal.type.value)
                pnl = executor.exit(signal, account_id)
                if pnl is not None:
                    risk.update_pnl(pnl)
                    log.info("P&L сделки: %.2f₽  |  %s", pnl, risk.status())
                else:
                    strategy.force_close()  # откат если ордер провалился

        except KeyboardInterrupt:
            log.info("Остановка по Ctrl+C")
            break
        except Exception as exc:
            log.error("Ошибка в основном цикле: %s", exc, exc_info=True)

        time.sleep(cfg.poll_interval_s)


if __name__ == "__main__":
    main()
