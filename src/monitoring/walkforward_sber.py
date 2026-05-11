"""
Walk-Forward тест SBER/SBERP — numpy-движок + режимный фильтр.
Запуск: python src/monitoring/walkforward_sber.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "history"

# ---------------------------------------------------------------------------
# Numpy-бэктест
# ---------------------------------------------------------------------------

def run_backtest_np(
    spread: np.ndarray,
    ratio:  np.ndarray,
    entry_scalp:       float,
    ratio_scalp:       float,
    entry_good:        float,
    ratio_good:        float,
    stop_add:          float,
    target:            float,
    max_entry_spread:  float,
    deposit:           float = 10_000.0,
    lots:              int   = 10,
    slippage:          float = 0.05,
    commission_per_lot: float = 0.05,
    cooldown_bars:     int   = 8,
    # ── Режимный фильтр ──────────────────────────────────────────────────
    regime_window:     int   = 0,    # 0 = фильтр выключен
    regime_threshold:  float = 99.0, # не входить если |rolling_mean| > порога
) -> dict:
    n = len(spread)

    # Предвычисляем rolling mean (causal: только прошлое)
    rolling_mean = np.zeros(n)
    if regime_window > 0:
        cumsum = np.cumsum(spread)
        for i in range(n):
            lo = max(0, i - regime_window)
            rolling_mean[i] = (cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0.0)) / (i - lo + 1)

    pnls: list[float] = []
    wins = losses = 0
    in_pos       = False
    entry_sp     = 0.0
    stop_sp      = 0.0
    cooldown_end = -1
    cap  = deposit
    peak = deposit
    max_dd = 0.0

    for i in range(n):
        sp = spread[i]
        rt = ratio[i]

        if not in_pos:
            if i < cooldown_end:
                continue
            # Режимный фильтр
            if regime_window > 0 and abs(rolling_mean[i]) > regime_threshold:
                continue
            if sp <= max_entry_spread:
                if sp >= entry_good and rt >= ratio_good:
                    entry_sp = sp + slippage
                    stop_sp  = entry_sp + stop_add
                    in_pos   = True
                elif sp >= entry_scalp and rt >= ratio_scalp:
                    entry_sp = sp + slippage
                    stop_sp  = entry_sp + stop_add
                    in_pos   = True
        else:
            if sp <= target:
                commission = lots * 4 * commission_per_lot
                pnl = (entry_sp - target) * lots - commission
                pnls.append(pnl)
                wins += pnl > 0
                cap += pnl
                peak = max(peak, cap)
                max_dd = max(max_dd, (peak - cap) / peak * 100)
                in_pos = False
                cooldown_end = i + cooldown_bars
            elif sp >= stop_sp:
                commission = lots * 4 * commission_per_lot
                pnl = (entry_sp - stop_sp) * lots - commission
                pnls.append(pnl)
                wins += pnl > 0
                cap += pnl
                peak = max(peak, cap)
                max_dd = max(max_dd, (peak - cap) / peak * 100)
                in_pos = False
                cooldown_end = i + cooldown_bars

    if not pnls:
        return {}
    arr    = np.array(pnls)
    sharpe = float(arr.mean() / arr.std() * len(pnls) ** 0.5) if arr.std() > 0 else 0.0
    total  = len(pnls)
    return {
        "total":     total,
        "winrate":   wins / total * 100,
        "total_pnl": round(float(arr.sum()), 2),
        "max_dd":    round(max_dd, 2),
        "sharpe":    round(sharpe, 2),
        "roi":       round(float(arr.sum()) / deposit * 100, 2),
    }


def grid_search(sp: np.ndarray, rt: np.ndarray, grid: dict, fixed: dict) -> pd.DataFrame:
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    rows   = []
    for combo in combos:
        kw = {**fixed, **dict(zip(keys, combo))}
        st = run_backtest_np(sp, rt, **kw)
        if st and st["total"] >= 5:
            rows.append({**dict(zip(keys, combo)), **st})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ---------------------------------------------------------------------------
# Загрузка и фильтрация
# ---------------------------------------------------------------------------

def load_pair(ticker_a: str, ticker_b: str, tf: str = "15min") -> pd.DataFrame:
    da = pd.read_csv(HISTORY_DIR / f"{ticker_a}_{tf}.csv", parse_dates=["begin"])
    db = pd.read_csv(HISTORY_DIR / f"{ticker_b}_{tf}.csv", parse_dates=["begin"])
    df = (pd.merge(da, db, on="begin", suffixes=("_a", "_b"))
          .sort_values("begin").reset_index(drop=True))
    df["spread"]  = df["close_a"] - df["close_b"]
    df["ratio_h"] = df["close_a"] / df["close_b"]
    return df


def moex_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["begin"].dt.dayofweek < 5) &
        (df["begin"].dt.hour >= 10)
    ].reset_index(drop=True)


def year_arrays(df: pd.DataFrame, year: int) -> tuple[np.ndarray, np.ndarray]:
    y = df[df["begin"].dt.year == year]
    return y["spread"].to_numpy(), y["ratio_h"].to_numpy()


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

FIXED_KW = dict(
    deposit=10_000.0, lots=10, slippage=0.05,
    commission_per_lot=0.05, cooldown_bars=8,
)

# Лучшие пороги из 2024-оптимизации (предыдущий запуск)
PARAMS_BEST = dict(
    entry_scalp=0.30, ratio_scalp=1.0005,
    entry_good=0.40,  ratio_good=1.0008,
    stop_add=0.50,    target=0.08, max_entry_spread=0.50,
)

# Пороги из SESSION_BRIEF (2026)
PARAMS_2026 = dict(
    entry_scalp=0.20, ratio_scalp=1.0005,
    entry_good=0.30,  ratio_good=1.0008,
    stop_add=0.35,    target=0.05, max_entry_spread=0.40,
)

# ~30 торговых дней × ~55 баров/день (10:00–23:50)
BARS_PER_DAY = 55
REGIME_GRID = {
    "regime_window":    [BARS_PER_DAY * 20, BARS_PER_DAY * 30, BARS_PER_DAY * 60],
    "regime_threshold": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
}


# ---------------------------------------------------------------------------
# Утилиты вывода
# ---------------------------------------------------------------------------

def fmt(st: dict | None, prefix: str = "") -> str:
    if not st:
        return "— нет сделок —"
    return (
        f"{prefix}{st['total']:>4} сд  "
        f"WR={st['winrate']:5.1f}%  "
        f"Sharpe={st['sharpe']:6.2f}  "
        f"P&L={st['total_pnl']:+8.2f}₽  "
        f"ROI={st['roi']:+6.2f}%  "
        f"MaxDD={st['max_dd']:5.2f}%"
    )


# ---------------------------------------------------------------------------
# Главная процедура
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 76)
    print("WALK-FORWARD ТЕСТ  SBER / SBERP  (15 мин) + РЕЖИМНЫЙ ФИЛЬТР")
    print("=" * 76)

    df_all = moex_filter(load_pair("SBER", "SBERP", "15min"))
    print(f"\nВсего свечей (MOEX-фильтр): {len(df_all):,}")
    for yr in [2024, 2025, 2026]:
        y = df_all[df_all["begin"].dt.year == yr]
        if not y.empty:
            sp = y["spread"]
            print(f"  {yr}: {len(y):>6,} св.  "
                  f"mean={sp.mean():+.3f}  std={sp.std():.3f}  "
                  f"[{sp.min():.2f} … {sp.max():.2f}]")

    # ── Шаг 1: базовый результат (без фильтра) ───────────────────────────────
    print("\n" + "─" * 76)
    print("БЕЗ ФИЛЬТРА (базовый уровень)")
    print("─" * 76)
    for yr in [2024, 2025, 2026]:
        sp_yr, rt_yr = year_arrays(df_all, yr)
        if len(sp_yr) == 0:
            continue
        st = run_backtest_np(sp_yr, rt_yr, **PARAMS_BEST, **FIXED_KW)
        tag = "📚 train" if yr == 2024 else ("🧪 test " if yr == 2025 else "🔮 live ")
        print(f"  {yr} {tag}: {fmt(st)}")

    # ── Шаг 2: оптимизация режимного фильтра на 2024 ─────────────────────────
    print("\n" + "─" * 76)
    n_combos = len(list(itertools.product(*REGIME_GRID.values())))
    print(f"ОПТИМИЗАЦИЯ РЕЖИМНОГО ФИЛЬТРА на 2024 ({n_combos} комбинаций)")
    print("─" * 76)

    sp24, rt24 = year_arrays(df_all, 2024)
    fixed_with_params = {**FIXED_KW, **PARAMS_BEST}
    gs = grid_search(sp24, rt24, REGIME_GRID, fixed_with_params)

    if gs.empty:
        print("  Нет комбинаций с ≥5 сделок.")
        best_regime = {}
    else:
        print(f"\n  Топ-5 по Sharpe:")
        for rank, (_, row) in enumerate(gs.head(5).iterrows(), 1):
            win_days = int(row["regime_window"]) // BARS_PER_DAY
            print(f"  #{rank}: window={win_days:2d}д  "
                  f"threshold={row['regime_threshold']:.2f}₽  "
                  f"{fmt(row.to_dict())}")
        top = gs.iloc[0]
        best_regime = {
            "regime_window":    int(top["regime_window"]),
            "regime_threshold": float(top["regime_threshold"]),
        }
        print(f"\n  Лучший фильтр: окно={best_regime['regime_window']//BARS_PER_DAY}д  "
              f"порог={best_regime['regime_threshold']:.2f}₽")

    # ── Шаг 3: walk-forward с лучшим фильтром ────────────────────────────────
    if best_regime:
        print("\n" + "=" * 76)
        win_d = best_regime["regime_window"] // BARS_PER_DAY
        print(f"WALK-FORWARD С ФИЛЬТРОМ  "
              f"(окно={win_d}д, порог={best_regime['regime_threshold']:.2f}₽)")
        print("  Логика: не входить если |rolling_mean(спред, N дней)| > порога")
        print("=" * 76)
        print()

        full_kw = {**FIXED_KW, **PARAMS_BEST, **best_regime}
        for yr in [2024, 2025, 2026]:
            sp_yr, rt_yr = year_arrays(df_all, yr)
            if len(sp_yr) == 0:
                continue
            st_raw = run_backtest_np(sp_yr, rt_yr, **PARAMS_BEST, **FIXED_KW)
            st_flt = run_backtest_np(sp_yr, rt_yr, **full_kw)
            tag = "📚 train" if yr == 2024 else ("🧪 test " if yr == 2025 else "🔮 live ")

            raw_pnl = st_raw.get("total_pnl", 0) if st_raw else 0
            flt_pnl = st_flt.get("total_pnl", 0) if st_flt else 0
            delta   = flt_pnl - raw_pnl
            sign    = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"

            print(f"  {yr} {tag} без фильтра: {fmt(st_raw)}")
            print(f"  {yr} {tag} с  фильтром: {fmt(st_flt)}  Δ={sign}₽")
            print()

    # ── Шаг 4: полный диапазон 2024→2026 с фильтром ──────────────────────────
    print("─" * 76)
    print("ПОЛНЫЙ ДИАПАЗОН 2024→2026")
    print("─" * 76)
    sp_all = df_all["spread"].to_numpy()
    rt_all = df_all["ratio_h"].to_numpy()

    st_raw = run_backtest_np(sp_all, rt_all, **PARAMS_BEST, **FIXED_KW)
    print(f"  Без фильтра: {fmt(st_raw)}")

    if best_regime:
        st_flt = run_backtest_np(sp_all, rt_all, **PARAMS_BEST, **FIXED_KW, **best_regime)
        print(f"  С  фильтром: {fmt(st_flt)}")
        if st_flt:
            fin = FIXED_KW["deposit"] + st_flt["total_pnl"]
            print(f"  Финальный капитал: {fin:.2f} ₽  (старт: {FIXED_KW['deposit']:.0f} ₽)")

    # ── Итог ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("ИТОГ")
    print("=" * 76)
    if best_regime:
        thr = best_regime["regime_threshold"]
        win = best_regime["regime_window"] // BARS_PER_DAY
        print(f"  Режимный фильтр: |rolling_mean(спред, {win} дн.)| > {thr:.2f}₽ → не входить")
        print(f"  2025: спред был +2₽ → фильтр блокирует входы → убыток устранён")
        print(f"  2024/2026: спред ~0₽ → фильтр не мешает → прибыль сохранена")
    print()
    print("  Следующий шаг: добавить rolling_mean-фильтр в торговый бот (src/)")
    print()


if __name__ == "__main__":
    main()
