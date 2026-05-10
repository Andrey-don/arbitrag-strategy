"""
MATS History Downloader — загрузка исторических данных с T-Invest API.
Сохраняет CSV-файлы в data/history/ для офлайн-тестирования в бэктестере.

Запуск:  python src/monitoring/download_history.py
Или:     двойной клик на download_history.bat
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ─── Пути ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent.parent
HISTORY_DIR = ROOT_DIR / "data" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT_DIR / ".env")
TINKOFF_TOKEN = os.getenv("TINKOFF_TOKEN", "")
_BASE = "https://invest-public-api.tinkoff.ru/rest"

# ─── Инструменты ─────────────────────────────────────────────────────────────
# Чтобы добавить новый инструмент — найди FIGI через T-Invest API (FindInstrument)
# или на сайте https://www.tinkoff.ru/invest/ в URL тикера.
INSTRUMENTS: dict[str, str] = {
    "TATN":  "BBG004RVFFC0",   # Татнефть обыкновенная
    "TATNP": "BBG004S68829",   # Татнефть привилегированная
    # ── Добавляй новые пары сюда ──────────────────────────────────────────
    # "SBER":  "BBG004730N88",   # Сбербанк обыкновенная
    # "SBERP": "BBG0047315Y7",   # Сбербанк привилегированная
    # "GAZP":  "BBG004730RP0",   # Газпром
    # "LKOH":  "BBG004731032",   # Лукойл
    # "GOLD":  "<FIGI>",         # Золото — уточни FIGI через FindInstrument
}

# ─── Параметры загрузки ───────────────────────────────────────────────────────
MONTHS_BACK = 3        # глубина истории в месяцах
TIMEFRAMES  = [5, 60]  # таймфреймы в минутах: 5-мин и 1-час

# ─── T-Invest API ─────────────────────────────────────────────────────────────
_TF_MAP: dict[int, str] = {
    1:  "CANDLE_INTERVAL_1_MIN",
    5:  "CANDLE_INTERVAL_5_MIN",
    10: "CANDLE_INTERVAL_10_MIN",
    15: "CANDLE_INTERVAL_15_MIN",
    60: "CANDLE_INTERVAL_HOUR",
}
# Максимальный размер чанка на один запрос (ограничение T-Invest API)
_TF_CHUNK: dict[int, int] = {1: 1, 5: 1, 10: 1, 15: 1, 60: 7}


def _headers() -> dict:
    return {"Authorization": f"Bearer {TINKOFF_TOKEN}", "Content-Type": "application/json"}


def _q2f(q: dict) -> float:
    return float(q.get("units", 0)) + q.get("nano", 0) / 1_000_000_000


def fetch_candles(figi: str, from_dt: datetime, to_dt: datetime, interval: int) -> pd.DataFrame:
    chunk = timedelta(days=_TF_CHUNK.get(interval, 1))
    rows: list[dict] = []
    cur = from_dt.replace(tzinfo=timezone.utc) if from_dt.tzinfo is None else from_dt
    end = to_dt.replace(tzinfo=timezone.utc)   if to_dt.tzinfo   is None else to_dt

    while cur < end:
        cur_to = min(cur + chunk, end)
        try:
            r = requests.post(
                f"{_BASE}/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles",
                json={
                    "figi":     figi,
                    "from":     cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to":       cur_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "interval": _TF_MAP.get(interval, "CANDLE_INTERVAL_5_MIN"),
                },
                headers=_headers(),
                timeout=20,
            )
            for c in r.json().get("candles", []):
                close = _q2f(c["close"])
                if close:
                    ts = pd.to_datetime(c["time"]) + timedelta(hours=3)
                    rows.append({"begin": ts.replace(tzinfo=None), "close": close})
        except Exception as e:
            print(f"    ⚠ Ошибка запроса ({cur.date()}): {e}")
        cur = cur_to
        time.sleep(0.05)  # небольшая пауза, чтобы не перегружать API

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def tf_label(tf: int) -> str:
    """Метка таймфрейма для имени файла: 5 → '5min', 60 → '1h'."""
    return f"{tf}min" if tf < 60 else "1h"


def download_all() -> None:
    if not TINKOFF_TOKEN:
        print("❌ TINKOFF_TOKEN не найден в .env")
        print("   Добавь строку: TINKOFF_TOKEN=<твой_токен>")
        input("Нажми Enter для выхода...")
        sys.exit(1)

    to_dt   = datetime.now()
    from_dt = to_dt - timedelta(days=30 * MONTHS_BACK)

    print("=" * 60)
    print("  MATS History Downloader")
    print("=" * 60)
    print(f"  Период:       {from_dt.date()} — {to_dt.date()} ({MONTHS_BACK} мес.)")
    print(f"  Инструменты:  {', '.join(INSTRUMENTS)}")
    print(f"  Таймфреймы:   {', '.join(str(t) + ' мин' for t in TIMEFRAMES)}")
    print(f"  Сохранение:   {HISTORY_DIR}")
    print("=" * 60)

    total = len(INSTRUMENTS) * len(TIMEFRAMES)
    done  = 0
    errors = 0

    for ticker, figi in INSTRUMENTS.items():
        for tf in TIMEFRAMES:
            done += 1
            label  = tf_label(tf)
            out    = HISTORY_DIR / f"{ticker}_{label}.csv"
            chunks = (MONTHS_BACK * 30) // _TF_CHUNK.get(tf, 1)
            print(f"\n  [{done}/{total}] {ticker} {label} (~{chunks} запросов)...", end=" ", flush=True)
            df = fetch_candles(figi, from_dt, to_dt, tf)
            if df.empty:
                print("⚠ нет данных")
                errors += 1
                continue
            df = df.drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
            df.to_csv(out, index=False)
            size_kb = out.stat().st_size // 1024
            print(f"✅ {len(df):,} свечей → {out.name} ({size_kb} КБ)")

    print("\n" + "=" * 60)
    if errors == 0:
        print("  ✅ Все файлы загружены успешно!")
    else:
        print(f"  ⚠ Завершено с {errors} ошибками.")
    print(f"  Папка: {HISTORY_DIR}")
    print("=" * 60)
    input("\nНажми Enter для выхода...")


if __name__ == "__main__":
    download_all()
