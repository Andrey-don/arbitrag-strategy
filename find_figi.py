"""
Поиск FIGI инструментов по тикеру через T-Invest API.
Запуск: python find_figi.py
"""

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")
TOKEN = os.getenv("TINKOFF_TOKEN", "")
BASE  = "https://invest-public-api.tinkoff.ru/rest"

TICKERS_TO_FIND = [
    "TGLD",       # Т-Капитал Золото (ETF)
    "AKGD",       # Альфа-Капитал Золото (ETF)
    "SBGD",       # Первая — Доступное Золото (ETF)
    "GLDRUB_TOM", # Биржевое золото спот
    "TATN",       # Проверка — должен быть знакомый FIGI
    "TATNP",      # Проверка
]

def find_instrument(query: str) -> list[dict]:
    try:
        r = requests.post(
            f"{BASE}/tinkoff.public.invest.api.contract.v1.InstrumentsService/FindInstrument",
            json={"query": query, "apiTradeAvailableFlag": True},
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
            timeout=10,
        )
        return r.json().get("instruments", [])
    except Exception as e:
        print(f"  ⚠ Ошибка: {e}")
        return []

def main():
    if not TOKEN:
        print("❌ TINKOFF_TOKEN не найден в .env")
        input("Нажми Enter...")
        return

    print("=" * 70)
    print("  Поиск FIGI инструментов — T-Invest API")
    print("=" * 70)

    for ticker in TICKERS_TO_FIND:
        print(f"\n🔍 {ticker}:")
        instruments = find_instrument(ticker)
        if not instruments:
            print("  — не найдено")
            continue
        # Фильтруем: ищем точное совпадение тикера (если есть) или выводим топ-3
        exact = [i for i in instruments if i.get("ticker", "").upper() == ticker.upper()]
        show  = exact if exact else instruments[:3]
        for inst in show:
            print(
                f"  ticker={inst.get('ticker','?'):12s} "
                f"figi={inst.get('figi','?'):16s} "
                f"type={inst.get('instrumentType','?'):15s} "
                f"name={inst.get('name','?')}"
            )

    print("\n" + "=" * 70)
    print("  Скопируй нужные FIGI в download_history.py → INSTRUMENTS")
    print("=" * 70)
    input("\nНажми Enter для выхода...")

if __name__ == "__main__":
    main()
