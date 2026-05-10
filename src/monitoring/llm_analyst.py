"""
LLM Analyst — анализ результатов бэктеста через AI-модели.
Провайдеры: Claude (Anthropic), GPT-4o (OpenAI), Ollama (локально).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Системный промпт — контекст стратегии ТАТок
# ---------------------------------------------------------------------------
_SYSTEM = """\
Ты — профессиональный квантовый аналитик торговых стратегий.
Специализация: парный арбитраж на Московской бирже (MOEX).
Текущий проект — система MATS (MOEX Arbitrage Trading System).

Стратегия: пара TATN (Татнефть обыкновенная) / TATNP (Татнефть привилегированная).
Логика: продаём спред при расхождении, закрываем при схождении.
Депозит небольшой (~10 000 ₽), торговля через T-Invest API.

Ключевые параметры стратегии:
- entry_scalp: минимальный спред для скальп-входа (обычно 5 ₽)
- entry_good: минимальный спред для хорошего входа (обычно 7 ₽)
- ratio_scalp: минимальное отношение TATN/TATNP для скальпа (1.008)
- ratio_good: минимальное отношение для хорошего входа (1.011)
- target: спред при тейк-профите (обычно 0.5 ₽ — схождение линий)
- stop_add: добавка к спреду входа для стоп-лосса (обычно 3 ₽)
- cooldown_s: пауза после закрытия позиции (120 сек)
- commission_per_lot: ~0.05 ₽/лот × 4 ноги

Факты из уроков:
- Вечерняя сессия (19:05–23:50 МСК) прибыльнее утренней
- Обе ноги выставляются одновременно (лимитные ордера)
- Стоп выставляется как PostStopOrder на сервер
- R/R должен быть ≥ 2.0 для входа

Отвечай кратко, по существу, на русском языке.
Не объясняй базовые понятия — пользователь их знает.
Давай конкретные числовые рекомендации, если данных достаточно.
"""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _fmt_stats(stats: dict) -> str:
    return (
        f"Сделок: {stats.get('total', 0)}, "
        f"Winrate: {stats.get('winrate', 0):.0f}%, "
        f"ROI: {stats.get('roi', 0):+.1f}%, "
        f"P&L: {stats.get('total_pnl', 0):+.0f} ₽, "
        f"Sharpe: {stats.get('sharpe', 0):.2f}, "
        f"Max DD: {stats.get('max_dd', 0):.1f}%, "
        f"Avg win: {stats.get('avg_win', 0):+.1f} ₽, "
        f"Avg loss: {stats.get('avg_loss', 0):+.1f} ₽"
    )


def _fmt_params(params: dict) -> str:
    lines = []
    for k, v in params.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.3f}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _worst_trades(trades: list, n: int = 5) -> str:
    if not trades:
        return "нет сделок"
    worst = sorted(trades, key=lambda t: t["P&L ₽"])[:n]
    lines = []
    for t in worst:
        lines.append(
            f"  {t['Вход']} → {t['Выход']} | {t['Результат']} | "
            f"спред вх: {t['Спред вх.']:.2f}→{t['Спред вых.']:.2f} | P&L: {t['P&L ₽']:+.0f} ₽"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class LLMAnalyst:
    """Аналитик результатов бэктеста на основе LLM."""

    PROVIDERS = ["OpenRouter", "Claude (Anthropic)", "GPT-4o (OpenAI)", "Ollama (локально)"]

    MODELS = {
        "OpenRouter": [
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5",
            "google/gemini-2.5-flash",
            "google/gemini-2.0-flash-001",
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-small-3.2-24b-instruct",
            "deepseek/deepseek-chat-v3-0324",
        ],
        "Claude (Anthropic)": [
            "claude-sonnet-4-6",
            "claude-opus-4-7",
            "claude-haiku-4-5-20251001",
        ],
        "GPT-4o (OpenAI)": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ],
        "Ollama (локально)": [],  # заполняется динамически
    }

    def __init__(self, provider: str, model: str, api_key: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Публичные методы анализа
    # ------------------------------------------------------------------

    def analyze_backtest(self, stats: dict, params: dict, trades: list) -> str:
        """Анализ результатов прогона бэктеста."""
        msg = f"""\
Результаты бэктеста стратегии ТАТки:

Параметры запуска:
{_fmt_params(params)}

Метрики:
{_fmt_stats(stats)}

5 худших сделок:
{_worst_trades(trades, 5)}

Проанализируй результаты:
1. Насколько стратегия жизнеспособна? Что говорят Sharpe и просадка?
2. Есть ли проблемы (переобучение, мало сделок, высокая просадка)?
3. Что конкретно изменить в параметрах — дай числа.
4. На что обратить внимание в худших сделках?
"""
        return self._call(msg)

    def analyze_optimizer(self, top10_df: pd.DataFrame, best_row: dict,
                          metric_key: str) -> str:
        """Интерпретация топ-10 результатов оптимизатора."""
        top10_str = top10_df.to_string(index=False, max_rows=10)
        best_str = ", ".join(f"{k}={v}" for k, v in best_row.items()
                             if k in ("entry_scalp", "stop_add", "target",
                                      "winrate", "roi", "sharpe", "max_dd", "total"))
        msg = f"""\
Оптимизатор прогнал grid search по параметрам стратегии ТАТки.
Метрика оптимизации: {metric_key}

Топ-10 комбинаций:
{top10_str}

Лучшая комбинация: {best_str}

Интерпретируй результаты:
1. Насколько стабилен «лучший» результат — или это случайный пик?
2. Какой диапазон параметров реально работает (не одна точка)?
3. Есть ли риск переподгонки под историю?
4. Какие параметры поставить в бот для реальной торговли?
"""
        return self._call(msg)

    def advise_trainer(self, spread: float, ratio: float, signal: str,
                       position: Optional[dict], recent_trades: list,
                       sim_dep: float, stop_add: float, target: float) -> str:
        """Совет тренажёра по текущей рыночной ситуации."""
        pos_str = "нет открытой позиции"
        if position:
            pnl_now = (position["entry_spread"] - spread) * position["lots"]
            pos_str = (
                f"В позиции с {position['entry_time'].strftime('%H:%M')}, "
                f"спред входа: {position['entry_spread']:.2f} ₽, "
                f"текущий спред: {spread:.2f} ₽, "
                f"лотов: {position['lots']}, "
                f"P&L сейчас: {pnl_now:+.1f} ₽"
            )

        recent_str = "нет недавних сделок"
        if recent_trades:
            last3 = recent_trades[-3:]
            recent_str = "; ".join(
                f"{t['Результат']} {t['P&L ₽']:+.0f} ₽ "
                f"(вх: {t['Спред вх.']:.1f}→{t['Спред вых.']:.1f})"
                for t in last3
            )

        msg = f"""\
Текущая ситуация в тренажёре:
- Спред TATN-TATNP: {spread:.2f} ₽
- Ratio TATN/TATNP: {ratio:.4f}
- Сигнал системы: {signal}
- Позиция: {pos_str}
- Последние 3 сделки: {recent_str}
- Параметры: стоп +{stop_add} ₽, тейк ≤{target} ₽, депозит {sim_dep:.0f} ₽

Дай краткий совет трейдеру (2–4 предложения):
- Что делать прямо сейчас?
- На что обратить внимание?
"""
        return self._call(msg)

    # ------------------------------------------------------------------
    # Вызовы LLM
    # ------------------------------------------------------------------

    def _call(self, user_msg: str) -> str:
        try:
            if self.provider == "OpenRouter":
                return self._call_openrouter(user_msg)
            if self.provider == "Claude (Anthropic)":
                return self._call_claude(user_msg)
            if self.provider == "GPT-4o (OpenAI)":
                return self._call_openai(user_msg)
            if self.provider == "Ollama (локально)":
                return self._call_ollama(user_msg)
            return "Неизвестный провайдер."
        except Exception as e:
            return f"❌ Ошибка LLM ({self.provider}): {e}"

    def _call_openrouter(self, user_msg: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    def _call_claude(self, user_msg: str) -> str:
        import anthropic  # импорт только при вызове

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text

    def _call_openai(self, user_msg: str) -> str:
        from openai import OpenAI  # импорт только при вызове

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    def _call_ollama(self, user_msg: str) -> str:
        payload = {
            "model":    self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            "stream": False,
        }
        r = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    # ------------------------------------------------------------------
    # Утилита: список моделей Ollama
    # ------------------------------------------------------------------

    @staticmethod
    def get_ollama_models() -> list[str]:
        """Возвращает список моделей, доступных в локальном Ollama."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []
