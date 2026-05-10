"""
MATS History Downloader — загрузка исторических данных с T-Invest API.

Скачивает: 1-мин и 5-мин (напрямую с API)
Генерирует: 15-мин и 1-час (пересемплирование из 5-мин, без лишних запросов)

Запуск: python src/monitoring/download_history.py
Или:    двойной клик на download_history.bat
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
# Добавить новый инструмент: найди FIGI через T-Invest API (FindInstrument)
# или на https://www.tinkoff.ru/invest/ → страница инструмента → URL.
INSTRUMENTS: dict[str, str] = {
    "TATN":  "BBG004RVFFC0",   # Татнефть обыкновенная
    "TATNP": "BBG004S68829",   # Татнефть привилегированная
    # ── Золото ───────────────────────────────────────────────────────────
    # Пара TGLD/AKGD — аналог ТАТок: два ETF на физ. золото.
    # Спред между ними = арбитражная возможность.
    "TGLD":  "TCS80A101X50",   # Т-Капитал Золото (ETF, ~13 руб./лот)
    "AKGD":  "BBG014M8NBM4",   # Альфа-Капитал Золото (ETF)
    # Альтернативы:
    # "SBGD":  "BBG019HZM0H0",   # Первая — Доступное Золото (ETF)
    # "GLDRUB_TOM": "BBG000VJ5YR4",  # Биржевое золото спот (пара с GD-фьючерсом)
    # ── Другие акционные пары ────────────────────────────────────────────
    # "SBER":  "BBG004730N88",   # Сбербанк обыкновенная
    # "SBERP": "BBG0047315Y7",   # Сбербанк привилегированная
    # "GAZP":  "BBG004730RP0",   # Газпром
    # "LKOH":  "BBG004731032",   # Лукойл
}

# ─── Параметры загрузки ───────────────────────────────────────────────────────
MONTHS_BACK = 3        # глубина истории в месяцах

# Таймфреймы, скачиваемые с API.
# 15-мин и 1-час НЕ качаем — они генерируются из 5-мин автоматически.
# Нельзя получить 1-мин из 5-мин: только скачать отдельно (≈90 запросов за 3 мес.)
DOWNLOAD_TF = [1, 5]

# Таймфреймы, получаемые пересемплированием из 5-мин данных (бесплатно)
RESAMPLE_FROM_5MIN = {
    "15min": 15,   # каждые 3 свечи 5-мин = 1 свеча 15-мин
    "1h":    60,   # каждые 12 свечей 5-мин = 1 свеча 1-час
}

# ─── T-Invest API ─────────────────────────────────────────────────────────────
_TF_MAP: dict[int, str] = {
    1:  "CANDLE_INTERVAL_1_MIN",
    5:  "CANDLE_INTERVAL_5_MIN",
    10: "CANDLE_INTERVAL_10_MIN",
    15: "CANDLE_INTERVAL_15_MIN",
    60: "CANDLE_INTERVAL_HOUR",
}
# Максимальный размер одного запроса к API (иначе пустой ответ)
_TF_CHUNK: dict[int, int] = {1: 1, 5: 1, 10: 1, 15: 1, 60: 7}


def _headers() -> dict:
    return {"Authorization": f"Bearer {TINKOFF_TOKEN}", "Content-Type": "application/json"}


def _q2f(q: dict) -> float:
    return float(q.get("units", 0)) + q.get("nano", 0) / 1_000_000_000


def fetch_candles(figi: str, from_dt: datetime, to_dt: datetime, interval: int) -> pd.DataFrame:
    """Загрузить свечи с T-Invest API с автоматическим чанкованием."""
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
            print(f"    ⚠ Ошибка ({cur.date()}): {e}")
        cur = cur_to
        time.sleep(0.05)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def resample_candles(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Пересемплировать 5-мин свечи в старший таймфрейм по цене закрытия."""
    df = df.copy()
    df["begin"] = pd.to_datetime(df["begin"])
    resampled = (
        df.set_index("begin")["close"]
        .resample(f"{minutes}min")
        .last()
        .dropna()
        .reset_index()
    )
    return resampled


def tf_label(tf: int) -> str:
    return f"{tf}min" if tf < 60 else "1h"


def download_all() -> None:
    if not TINKOFF_TOKEN:
        print("❌ TINKOFF_TOKEN не найден в .env")
        print("   Добавь строку: TINKOFF_TOKEN=t.твой_токен")
        input("\nНажми Enter для выхода...")
        sys.exit(1)

    to_dt   = datetime.now()
    from_dt = to_dt - timedelta(days=30 * MONTHS_BACK)

    print("=" * 65)
    print("  MATS History Downloader")
    print("=" * 65)
    print(f"  Период:      {from_dt.date()} — {to_dt.date()} ({MONTHS_BACK} мес.)")
    print(f"  Инструменты: {', '.join(INSTRUMENTS)}")
    print(f"  Скачиваем:   {', '.join(tf_label(t) for t in DOWNLOAD_TF)}")
    print(f"  Генерируем:  {', '.join(RESAMPLE_FROM_5MIN)} (из 5-мин)")
    print(f"  Папка:       {HISTORY_DIR}")
    print("=" * 65)

    total  = len(INSTRUMENTS) * len(DOWNLOAD_TF)
    done   = 0
    errors = 0
    # Буфер 5-мин данных для генерации производных таймфреймов
    buf_5min: dict[str, pd.DataFrame] = {}

    # ── Шаг 1: скачать с API ─────────────────────────────────────────────────
    print("\n📡 Шаг 1: загрузка с T-Invest API")
    for ticker, figi in INSTRUMENTS.items():
        for tf in DOWNLOAD_TF:
            done += 1
            label  = tf_label(tf)
            out    = HISTORY_DIR / f"{ticker}_{label}.csv"
            n_req  = (MONTHS_BACK * 30) // _TF_CHUNK.get(tf, 1)
            print(f"  [{done}/{total}] {ticker} {label} (~{n_req} запросов)...", end=" ", flush=True)

            df = fetch_candles(figi, from_dt, to_dt, tf)
            if df.empty:
                print("⚠ нет данных")
                errors += 1
                continue

            df = df.drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
            df.to_csv(out, index=False)
            kb = out.stat().st_size // 1024
            print(f"✅ {len(df):,} свечей → {out.name} ({kb} КБ)")

            if tf == 5:
                buf_5min[ticker] = df  # сохраняем для пересемплирования

    # ── Шаг 2: генерация производных таймфреймов из 5-мин ───────────────────
    if buf_5min:
        print("\n🔄 Шаг 2: генерация из 5-мин данных")
        for label, minutes in RESAMPLE_FROM_5MIN.items():
            for ticker in INSTRUMENTS:
                if ticker not in buf_5min:
                    # 5-мин не загрузили — читаем из файла если есть
                    f5 = HISTORY_DIR / f"{ticker}_5min.csv"
                    if f5.exists():
                        buf_5min[ticker] = pd.read_csv(f5, parse_dates=["begin"])
                    else:
                        print(f"  ⚠ {ticker}_{label}: нет 5-мин данных, пропуск")
                        continue

                out = HISTORY_DIR / f"{ticker}_{label}.csv"
                df_res = resample_candles(buf_5min[ticker], minutes)
                df_res.to_csv(out, index=False)
                kb = out.stat().st_size // 1024
                print(f"  ✅ {ticker} {label}: {len(df_res):,} свечей → {out.name} ({kb} КБ)")

    # ── Итог ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    all_files = sorted(HISTORY_DIR.glob("*.csv"))
    print(f"  Файлов в data/history/: {len(all_files)}")
    for f in all_files:
        print(f"    {f.name}  ({f.stat().st_size // 1024} КБ)")
    print()
    if errors == 0:
        print("  ✅ Все файлы загружены и сгенерированы!")
    else:
        print(f"  ⚠ Завершено с {errors} ошибками (проверь TINKOFF_TOKEN).")
    print(f"\n  Путь: {HISTORY_DIR}")
    print("=" * 65)
    input("\nНажми Enter для выхода...")


if __name__ == "__main__":
    download_all()
