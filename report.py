"""
report.py — демографический портрет популяции агентов.

Генерирует отчёт для заданного снимка (DataFrame):
  - Общая численность и распределение по районам
  - Возрастные бины (пирамида)
  - Образование
  - Занятость
  - Семейный статус
  - Средние зарплаты по районам
  - Динамика переездов (из tick_stats)
"""

import pandas as pd
import numpy as np
from typing import Optional


AGE_BINS    = [0, 18, 30, 45, 60, 75, 150]
AGE_LABELS  = ["0–17", "18–29", "30–44", "45–59", "60–74", "75+"]

DISTRICT_SHORT = {
    "District of Trnava":             "Trnava",
    "District of Dunajská\xa0Streda": "D. Streda",
    "District of Galanta":            "Galanta",
    "District of Hlohovec":           "Hlohovec",
    "District of Piešťany":           "Piešťany",
    "District of Senica":             "Senica",
    "District of Skalica":            "Skalica",
}


def _pct(part, total) -> str:
    if total == 0:
        return " 0.0%"
    return f"{part/total*100:5.1f}%"


def _bar(value, max_value, width=20) -> str:
    if max_value == 0:
        return " " * width
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


def demographic_portrait(
    df: pd.DataFrame,
    label: str = "Снимок",
    tick_num: Optional[int] = None,
) -> str:
    """Возвращает текстовый демографический портрет."""
    lines = []
    total = len(df)

    header = f"{'='*65}"
    title = f"  ДЕМОГРАФИЧЕСКИЙ ПОРТРЕТ — {label}"
    if tick_num is not None:
        year = tick_num // 12
        month = tick_num % 12 or 12
        title += f"  [тик {tick_num} / год {year} мес {month}]"
    lines += [header, title, header]
    lines.append(f"\n  Агентов всего: {total:,}\n")

    # ── Районы ───────────────────────────────────────────────────────────────
    lines.append("  РАСПРЕДЕЛЕНИЕ ПО РАЙОНАМ")
    lines.append(f"  {'Район':<14} {'Агентов':>8}  {'Доля':>6}  {'':20}")
    lines.append("  " + "-" * 52)
    by_district = df.groupby("district")["id"].count().sort_values(ascending=False)
    max_d = by_district.max()
    for district, count in by_district.items():
        short = DISTRICT_SHORT.get(district, district)
        bar = _bar(count, max_d)
        lines.append(f"  {short:<14} {count:>8,}  {_pct(count, total)}  {bar}")

    # ── Возраст ───────────────────────────────────────────────────────────────
    lines.append("\n  ВОЗРАСТНАЯ СТРУКТУРА")
    df_copy = df.copy()
    df_copy["age_bin"] = pd.cut(df_copy["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    age_dist = df_copy["age_bin"].value_counts().reindex(AGE_LABELS, fill_value=0)
    max_a = age_dist.max()
    lines.append(f"  {'Группа':>6} {'Агентов':>8}  {'Доля':>6}  {'':20}")
    lines.append("  " + "-" * 45)
    for label_a, count in age_dist.items():
        bar = _bar(count, max_a)
        lines.append(f"  {label_a:>6} {count:>8,}  {_pct(count, total)}  {bar}")
    lines.append(f"\n  Средний возраст: {df['age'].mean():.1f} лет  |  Медиана: {df['age'].median():.1f}")

    # ── Пол ───────────────────────────────────────────────────────────────────
    lines.append("\n  ПОЛ")
    sex_dist = df["sex"].value_counts()
    for sex, count in sex_dist.items():
        lines.append(f"  {sex:<8} {count:>8,}  {_pct(count, total)}")

    # ── Образование ───────────────────────────────────────────────────────────
    lines.append("\n  ОБРАЗОВАНИЕ")
    edu_order = [
        "Without education", "Basic",
        "Secondary without school-leavi~",
        "Secondary with school-leaving ~",
        "University", "Unknown"
    ]
    edu_dist = df["education"].value_counts()
    max_e = edu_dist.max()
    for edu in edu_order:
        count = edu_dist.get(edu, 0)
        short = edu[:32]
        bar = _bar(count, max_e)
        lines.append(f"  {short:<32} {count:>7,}  {_pct(count, total)}  {bar}")

    # ── Занятость ─────────────────────────────────────────────────────────────
    lines.append("\n  ЗАНЯТОСТЬ (топ-6)")
    occ_dist = df["occupation"].value_counts().head(6)
    max_o = occ_dist.max()
    for occ, count in occ_dist.items():
        short = occ[:32]
        lines.append(f"  {short:<32} {count:>7,}  {_pct(count, total)}")

    # ── Семейный статус ───────────────────────────────────────────────────────
    lines.append("\n  СЕМЕЙНЫЙ СТАТУС")
    mar_dist = df["marital"].value_counts()
    for mar, count in mar_dist.items():
        lines.append(f"  {mar:<22} {count:>7,}  {_pct(count, total)}")

    # ── Зарплата ──────────────────────────────────────────────────────────────
    lines.append("\n  ЗАРПЛАТА ПО РАЙОНАМ (средняя, EUR/мес)")
    wage_by_d = df.groupby("district")["wage"].mean().sort_values(ascending=False)
    for district, wage in wage_by_d.items():
        short = DISTRICT_SHORT.get(district, district)
        lines.append(f"  {short:<14} {wage:>8,.0f}€")

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)


def migration_summary(tick_stats: list) -> str:
    """Краткая сводка по миграции за всю симуляцию."""
    if not tick_stats:
        return "Нет данных о тиках."

    total_moves = sum(s["moves"] for s in tick_stats)
    max_tick = max(tick_stats, key=lambda s: s["moves"])
    min_tick = min(tick_stats, key=lambda s: s["moves"])
    avg_moves = total_moves / len(tick_stats)

    lines = [
        "\n" + "=" * 65,
        "  СВОДКА МИГРАЦИИ",
        "=" * 65,
        f"  Всего переездов за симуляцию: {total_moves:,}",
        f"  Среднее переездов/тик:        {avg_moves:.1f}",
        f"  Максимум: тик {max_tick['tick']} → {max_tick['moves']} переездов",
        f"  Минимум:  тик {min_tick['tick']} → {min_tick['moves']} переездов",
        "",
        "  Динамика переездов по годам:",
    ]

    # Группируем по годам
    yearly = {}
    for s in tick_stats:
        year = (s["tick"] - 1) // 12 + 1
        yearly.setdefault(year, []).append(s["moves"])

    max_annual = max(sum(v) for v in yearly.values())
    for year, monthly in sorted(yearly.items()):
        annual = sum(monthly)
        bar = _bar(annual, max_annual, width=25)
        lines.append(f"  Год {year:2d}  {annual:>5,} переездов  {bar}")

    lines.append("=" * 65)
    return "\n".join(lines)


def compare_snapshots(snapshots: dict, tick_stats: list) -> str:
    """Сравнивает начальный, средний и финальный снимки."""
    ticks = sorted(snapshots.keys())
    lines = [
        "\n" + "=" * 65,
        "  СРАВНЕНИЕ СНИМКОВ — динамика по районам",
        "=" * 65,
        f"  {'Район':<14}", 
    ]

    # Заголовки столбцов
    header = f"  {'Район':<14}"
    for t in ticks:
        label = f"Тик {t}" if t > 0 else "Старт"
        header += f"  {label:>8}"
    header += f"  {'Δ нач→кон':>10}"
    lines[3] = header
    lines.append("  " + "-" * (14 + len(ticks) * 10 + 14))

    all_districts = sorted(list(snapshots[ticks[0]]["district"].unique()))
    for district in all_districts:
        short = DISTRICT_SHORT.get(district, district)
        row = f"  {short:<14}"
        counts = []
        for t in ticks:
            count = (snapshots[t]["district"] == district).sum()
            counts.append(count)
            row += f"  {count:>8,}"
        # Изменение начало → конец
        delta = counts[-1] - counts[0]
        sign = "+" if delta >= 0 else ""
        row += f"  {sign}{delta:>8,}"
        lines.append(row)

    lines.append("=" * 65)
    lines.append(migration_summary(tick_stats))
    return "\n".join(lines)
