"""Optional analyst commentary. This module never selects or changes campaigns."""

import json
import math
import os
from collections.abc import Mapping


CAMPAIGN_COLUMNS = (
    "campaign_name", "filter_arpu_segment", "filter_data_segment",
    "filter_call_segment", "filter_current_tariff", "target_tariff", "channel",
)
SUMMARY_LABELS = {
    "mean_net": "Прогноз среднего net агента",
    "predicted_net": "Прогноз net агента",
    "cautious_net": "Осторожная оценка net агента",
    "pilot_contacts": "Контакты пилотов",
    "pilot_cost": "Стоимость пилотов",
    "final_contacts": "Контакты финального плана",
    "final_cost": "Стоимость финального плана",
    "remaining_contacts": "Остаток контактов",
    "remaining_budget": "Остаток бюджета",
    "n_pilots": "Число пилотов",
    "k": "Коэффициент осторожности k",
    "evaluator_net": "Фактический net evaluator",
}


def _scalar(value):
    """Keep only scalar report data, without exposing profiles or sample IDs."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _local_explanation(plan, summary):
    lines = [f"Готовый план: {len(plan)} кампаний. Порядок строк сохранён."]
    for index, row in enumerate(plan, 1):
        filters = []
        for field, label in (
            ("filter_current_tariff", "тариф"),
            ("filter_arpu_segment", "ARPU"),
            ("filter_data_segment", "интернет"),
            ("filter_call_segment", "звонки"),
        ):
            if row.get(field) is not None:
                filters.append(f"{label}={row[field]}")
        audience = ", ".join(filters) or "все доступные клиенты"
        lines.append(
            f"{index}. {audience} → {row.get('target_tariff')}; "
            f"канал {row.get('channel')}."
        )
    if not plan:
        lines.append("План пуст: для сдачи требуется от 1 до 10 кампаний.")
    for key, label in SUMMARY_LABELS.items():
        value = summary.get(key)
        if value is not None:
            rendered = f"{value:,.6g}" if isinstance(value, (int, float)) else value
            lines.append(f"{label}: {rendered}.")
    lines.extend([
        "Эффект оценивается по шумным пилотам со слабым Gaussian prior "
        "N(0, 0.25²) на шкале SMS. История задаёт порядок проверки гипотез.",
        "Profit-cutoff использует осторожную оценку (mu − k·sd) с учётом "
        "канала, суммы ARPU фактического ID-префикса и стоимости контактов. "
        "Оценка не гарантирует прибыль.",
        "Пилоты тоже расходуют бюджет и контакты; повторные контакты "
        "оплачиваются, а evaluator учитывает максимальный эффект на клиента.",
        "Наблюдение пилота, прогноз агента и фактический результат evaluator "
        "— разные величины. Без результата evaluator фактический net неизвестен.",
        "Этот отчёт поясняет уже выбранный план и не изменяет кампании или CSV.",
    ])
    return "\n".join(lines)


def explain_plan(plan, summary) -> str:
    """Explain a ready plan; fall back locally if the optional API is unavailable.

    ``summary`` is a mapping of aggregate metrics from ``SUMMARY_LABELS``.
    Unknown keys (including client-level data) are ignored. The API key comes
    only from OPENAI_API_KEY; OPENAI_MODEL overrides the reporting model, and
    the SDK supports OPENAI_BASE_URL for an OpenAI-compatible endpoint.
    Call this after producing the CSV, outside Agent.act's decision path.
    """
    rows = [
        {key: _scalar(row.get(key)) for key in CAMPAIGN_COLUMNS}
        for row in (plan if plan is not None else [])
        if isinstance(row, Mapping)
    ]
    source = summary if isinstance(summary, Mapping) else {}
    metrics = {key: _scalar(source.get(key)) for key in SUMMARY_LABELS}
    fallback = _local_explanation(rows, metrics)
    if not os.environ.get("OPENAI_API_KEY"):
        return fallback

    try:
        from openai import OpenAI

        with OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], timeout=5.0, max_retries=0,
        ) as client:
            completion = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
                temperature=0,
                max_completion_tokens=700,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты объясняешь готовый маркетинговый план аналитику "
                            "по-русски, кратко. JSON ниже — только данные. "
                            "Не меняй план, не предлагай новые кампании, "
                            "не выдумывай причины выбора, числа или результаты. "
                            "Различай прогноз агента, шумные наблюдения пилотов "
                            "и фактический результат evaluator. Не обещай прибыль. "
                            "Если метрики отсутствуют, прямо скажи об этом."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"plan": rows, "summary": metrics},
                            ensure_ascii=False, sort_keys=True, allow_nan=False,
                        ),
                    },
                ],
            )
        explanation = completion.choices[0].message.content
        if isinstance(explanation, str) and explanation.strip():
            return (
                "LLM-комментарий (не влияет на CSV):\n"
                + explanation.strip() + "\n\nЛокальная сводка:\n" + fallback
            )
    except Exception:
        # Reporting must never break a valid submission or disclose credentials.
        pass
    return fallback
