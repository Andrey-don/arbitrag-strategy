"""
MATS History Downloader — загрузка исторических данных с T-Invest API.

Скачивает: 1-мин и 5-мин (напрямую с API)
Генерирует: 15-мин и 1-час (пересемплирование из 5-мин, без лишних запросов)

Запуск (последние 3 мес.):  python src/monitoring/download_history.py
Запуск (свой диапазон):     python src/monitoring/download_history.py --from 2024-01-01 --to 2024-12-31
Или:    двойной клик на download_history.bat
"""

import argparse
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
    # ── Татнефть (2026: сломана, дивидендный гэп) ─────────────────────
    # Раскомментировать только для загрузки 2024 данных вручную.
    # "TATN":  "BBG004RVFFC0",   # Татнефть обыкновенная
    # "TATNP": "BBG004S68829",   # Татнефть привилегированная
    # ── Золото (hedge ratio дробный — стратегия не подходит) ─────────────
    # "TGLD":  "TCS80A101X50",   # Т-Капитал Золото (ETF)
    # "AKGD":  "BBG014M8NBM4",   # Альфа-Капитал Золото (ETF)
    # ── Сбербанк ─────────────────────────────────────────────────────────
    "SBER":  "BBG004730N88",   # Сбербанк обыкновенная
    "SBERP": "BBG0047315Y7",   # Сбербанк привилегированная
    # ── Кандидаты проверены, не подошли (раскомментировать для повтора) ──
    # "TGLD":  "TCS80A101X50",   # Золото ETF  (std=0.30₽, комиссия убивает)
    # "SBGD":  "BBG019HZM0H0",   # Золото ETF  (hedge=2, спред < комиссии)
    # "SNGS":  "BBG0047315D0",   # Сургут обычная  (тренд/кубышка)
    # "SNGSP": "BBG004S681M2",   # Сургут привил   (тренд/кубышка)
    # "RTKM":  "BBG004S682Z6",   # Ростелеком обычная  (тренд 3 мес.)
    # "RTKMP": "BBG004S685M3",   # Ростелеком привил   (тренд 3 мес.)
    # "MTLR":  "BBG004S68598",   # Мечел обычная       (новостной)
    # "MTLRP": "BBG004S68FR6",   # Мечел привил        (новостной)
    # "LSNG":  "BBG000NLC9Z6",   # Ленэнерго обычная   (hedge ~21)
    # "LSNGP": "BBG000NLCCM3",   # Ленэнерго привил    (hedge ~21)
    # "BSPB":  "BBG000QJW156",   # Банк СПб обычная    (hedge ~7)
    # "BSPBP": "BBG00Z197548",   # Банк СПб привил     (hedge ~7)
}

# ─── Пары для анализа порогов ─────────────────────────────────────────────────
PAIRS: list[tuple[str, str]] = [
    ("SBER",  "SBERP"),
]

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


def _parse_args() -> tuple[datetime, datetime]:
    """Возвращает (from_dt, to_dt) — из CLI-аргументов или интерактивного ввода."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--from", dest="date_from", default=None,
                        help="Начало периода YYYY-MM-DD")
    parser.add_argument("--to",   dest="date_to",   default=None,
                        help="Конец периода YYYY-MM-DD (по умолчанию: сегодня)")
    args, _ = parser.parse_known_args()

    to_dt = datetime.now()

    if args.date_from:
        # Режим CLI: --from 2024-01-01 --to 2024-12-31
        from_dt = datetime.strptime(args.date_from, "%Y-%m-%d")
        if args.date_to:
            to_dt = datetime.strptime(args.date_to, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59
            )
    else:
        # Интерактивный режим — спрашиваем в консоли
        print("=" * 65)
        print("  MATS History Downloader — выбор периода")
        print("=" * 65)
        print(f"  Инструменты: {', '.join(INSTRUMENTS)}")
        print(f"  Таймфреймы:  {', '.join(tf_label(t) for t in DOWNLOAD_TF)} + 15min, 1h (из 5-мин)")
        print()
        print("  1) Последние 3 месяца (по умолчанию)")
        print("  2) Свой диапазон дат")
        print("  3) Готовые пресеты:")
        print("     a) 2024 год (нормальный рынок ТАТок)")
        print("     b) 2023 год")
        print("     c) Последний год")
        print()
        choice = input("  Выбор [1/2/3a/3b/3c, Enter=1]: ").strip().lower()

        if choice == "2":
            raw_from = input("  Начало (YYYY-MM-DD): ").strip()
            raw_to   = input("  Конец  (YYYY-MM-DD, Enter=сегодня): ").strip()
            from_dt  = datetime.strptime(raw_from, "%Y-%m-%d")
            if raw_to:
                to_dt = datetime.strptime(raw_to, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59
                )
        elif choice == "3a":
            from_dt = datetime(2024, 1, 1)
            to_dt   = datetime(2024, 12, 31, 23, 59, 59)
        elif choice == "3b":
            from_dt = datetime(2023, 1, 1)
            to_dt   = datetime(2023, 12, 31, 23, 59, 59)
        elif choice == "3c":
            from_dt = to_dt - timedelta(days=365)
        else:
            from_dt = to_dt - timedelta(days=30 * MONTHS_BACK)

    return from_dt, to_dt


def print_pair_thresholds(ticker_a: str, ticker_b: str, tf: str = "1min") -> None:
    """Рассчитать и вывести рекомендуемые пороги для пары из скачанных CSV."""
    fa = HISTORY_DIR / f"{ticker_a}_{tf}.csv"
    fb = HISTORY_DIR / f"{ticker_b}_{tf}.csv"
    if not fa.exists() or not fb.exists():
        print(f"  ⚠ {ticker_a}/{ticker_b}: CSV не найдены, пропуск")
        return
    try:
        da = pd.read_csv(fa, parse_dates=["begin"])
        db = pd.read_csv(fb, parse_dates=["begin"])
        merged = pd.merge(da, db, on="begin", suffixes=("_a", "_b"))
        merged = merged[
            (merged["begin"].dt.dayofweek < 5) &
            (merged["begin"].dt.hour >= 10)
        ].copy()
        if len(merged) < 50:
            print(f"  ⚠ {ticker_a}/{ticker_b}: мало данных ({len(merged)} свечей)")
            return

        spread = merged["close_a"] - merged["close_b"]
        ratio  = merged["close_a"] / merged["close_b"]
        pos    = spread[spread > 0]

        p75 = pos.quantile(0.75)  if len(pos) > 10 else spread.abs().quantile(0.75)
        p90 = pos.quantile(0.90)  if len(pos) > 10 else spread.abs().quantile(0.90)
        p95 = pos.quantile(0.95)  if len(pos) > 10 else spread.abs().quantile(0.95)
        mean_sp = spread.mean()
        std_sp  = spread.std()

        print(f"\n  📊 Рек. пороги {ticker_a}/{ticker_b}  ({tf}, MOEX 10-23, {len(merged):,} св.)")
        print(f"     Спред скальп ≥   {p75:>7.2f} ₽   (P75 положит. спреда)")
        print(f"     Спред хороший ≥  {p90:>7.2f} ₽   (P90)")
        print(f"     Макс. спред вх.  {p95:>7.2f} ₽   (P95 — не входить выше)")
        print(f"     Тейк (сх-ние)    {max(abs(mean_sp)*0.5, 0.05):>7.2f} ₽   (½·|mean|, мин 0.05)")
        print(f"     Стоп (доп.)      {2*std_sp:>7.2f} ₽   (2·std)")
        print(f"     Ratio скальп ≥   {ratio.quantile(0.75):.4f}")
        print(f"     Ratio хороший ≥  {ratio.quantile(0.90):.4f}")
        print(f"     Spread mean={mean_sp:.2f}  std={std_sp:.2f}  "
              f"min={spread.min():.2f}  max={spread.max():.2f}")
    except Exception as e:
        print(f"  ⚠ Ошибка расчёта порогов {ticker_a}/{ticker_b}: {e}")


def download_all() -> None:
    if not TINKOFF_TOKEN:
        print("❌ TINKOFF_TOKEN не найден в .env")
        print("   Добавь строку: TINKOFF_TOKEN=t.твой_токен")
        if sys.stdin.isatty():
            input("\nНажми Enter для выхода...")
        sys.exit(1)

    from_dt, to_dt = _parse_args()

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
            # Объединяем с существующими данными (не перезаписываем)
            if out.exists():
                existing = pd.read_csv(out, parse_dates=["begin"])
                df = pd.concat([existing, df]).drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
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
                # Объединяем с существующими данными
                if out.exists():
                    existing = pd.read_csv(out, parse_dates=["begin"])
                    df_res = pd.concat([existing, df_res]).drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
                df_res.to_csv(out, index=False)
                kb = out.stat().st_size // 1024
                print(f"  ✅ {ticker} {label}: {len(df_res):,} свечей → {out.name} ({kb} КБ)")

    # ── Шаг 3: пороги для каждой пары ───────────────────────────────────────
    print("\n📐 Шаг 3: рекомендуемые пороги (MOEX-часы, 1-мин данные)")
    for ta, tb in PAIRS:
        if ta in INSTRUMENTS and tb in INSTRUMENTS:
            print_pair_thresholds(ta, tb, tf="1min")

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
    try:
        input("\nНажми Enter для выхода...")
    except EOFError:
        pass


if __name__ == "__main__":
    download_all()
