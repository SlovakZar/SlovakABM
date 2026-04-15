"""
report.py v2 — демографический портрет для всей Словакии.

Адаптирован под новую структуру агента:
  - Четыре домена satisfaction
  - TPB intention_state
  - agent_type
  - Региональная агрегация (8 краёв)
"""

import pandas as pd
import numpy as np
from typing import Optional

AGE_BINS   = [0, 18, 30, 45, 60, 75, 150]
AGE_LABELS = ["0–17", "18–29", "30–44", "45–59", "60–74", "75+"]

REGION_NAMES = {
    "BA": "Bratislavský",
    "TT": "Trnavský",
    "TN": "Trenčínský",
    "NR": "Nitrianský",
    "ZA": "Žilinský",
    "BB": "Banskobystrický",
    "PO": "Prešovský",
    "KE": "Košický",
}


def _pct(part, total) -> str:
    if total == 0: return "  0.0%"
    return f"{part/total*100:5.1f}%"


def _bar(value, max_value, width=18) -> str:
    if max_value == 0: return " " * width
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


def demographic_portrait(
    df: pd.DataFrame,
    label: str = "Снимок",
    tick_num: Optional[int] = None,
) -> str:
    lines = []
    total = len(df)

    header = "=" * 70
    title  = f"  ДЕМОГРАФИЧЕСКИЙ ПОРТРЕТ — {label}"
    if tick_num is not None:
        yr = tick_num // 12
        mo = tick_num % 12 or 12
        title += f"  [тик {tick_num} / год {yr} мес {mo}]"
    lines += [header, title, header]
    lines.append(f"\n  Агентов всего: {total:,}\n")

    # ── Регионы ───────────────────────────────────────────────────────────────
    lines.append("  РАСПРЕДЕЛЕНИЕ ПО РЕГИОНАМ (краям)")
    lines.append(f"  {'Регион':<22} {'Агентов':>8}  {'Доля':>6}  {'':18}")
    lines.append("  " + "-" * 58)
    by_region = df.groupby("region")["id"].count().sort_values(ascending=False)
    max_r = by_region.max()
    for code, count in by_region.items():
        name = REGION_NAMES.get(code, code)
        lines.append(f"  {name:<22} {count:>8,}  {_pct(count, total)}  {_bar(count, max_r)}")

    # ── Топ-10 районов по агентам ─────────────────────────────────────────────
    lines.append("\n  ТОП-10 РАЙОНОВ")
    by_d = df.groupby("district")["id"].count().sort_values(ascending=False).head(10)
    for dist, count in by_d.items():
        name = dist.replace("District of ", "")
        lines.append(f"  {name:<30} {count:>8,}  {_pct(count, total)}")

    # ── Типы агентов ──────────────────────────────────────────────────────────
    lines.append("\n  ТИПЫ АГЕНТОВ")
    type_dist = df["agent_type"].value_counts()
    max_t = type_dist.max()
    for t, count in type_dist.items():
        lines.append(f"  {t:<16} {count:>8,}  {_pct(count, total)}  {_bar(count, max_t)}")

    # ── TPB состояния ─────────────────────────────────────────────────────────
    lines.append("\n  TPB INTENTION STATE")
    intent_dist = df["intention_state"].value_counts()
    for state, count in intent_dist.items():
        lines.append(f"  {state:<14} {count:>8,}  {_pct(count, total)}")

    # ── Возраст ───────────────────────────────────────────────────────────────
    lines.append("\n  ВОЗРАСТНАЯ СТРУКТУРА")
    df2 = df.copy()
    df2["age_bin"] = pd.cut(df2["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    age_dist = df2["age_bin"].value_counts().reindex(AGE_LABELS, fill_value=0)
    max_a = age_dist.max()
    for lbl, count in age_dist.items():
        lines.append(f"  {lbl:>6} {count:>8,}  {_pct(count, total)}  {_bar(count, max_a)}")
    lines.append(f"\n  Средний возраст: {df['age'].mean():.1f} | Медиана: {df['age'].median():.1f}")

    # ── Домены satisfaction ───────────────────────────────────────────────────
    lines.append("\n  SATISFACTION ПО ДОМЕНАМ (средние)")
    for col, label_d in [
        ("sat_economic", "Economic"),
        ("sat_social",   "Social  "),
        ("sat_family",   "Family  "),
        ("sat_place",    "Place   "),
    ]:
        if col in df.columns:
            m = df[col].mean()
            bar = _bar(m, 1.0, width=20)
            lines.append(f"  {label_d}  {m:.4f}  {bar}")

    # ── Inertia и dissatisfaction ─────────────────────────────────────────────
    lines.append("\n  ПОВЕДЕНЧЕСКИЕ ПАРАМЕТРЫ")
    lines.append(f"  Ср. inertia:           {df['inertia'].mean():.3f}")
    lines.append(f"  Ср. perceived_control: {df['perceived_control'].mean():.3f}")
    lines.append(f"  Ср. info_quality:      {df['info_quality'].mean():.3f}")
    lines.append(f"  network_location:      {df['network_location'].mean():.1%}")
    lines.append(f"  network_signal +ve:    {(df['network_signal']=='positive').mean():.1%}")

    # ── Зарплата ──────────────────────────────────────────────────────────────
    employed = df[df["wage"] > 0]
    lines.append(f"\n  ЗАРПЛАТА (занятые, n={len(employed):,})")
    lines.append(f"  Средняя: {employed['wage'].mean():,.0f}€  |  Медиана: {employed['wage'].median():,.0f}€")
    lines.append(f"  P25: {employed['wage'].quantile(.25):,.0f}€  |  P75: {employed['wage'].quantile(.75):,.0f}€")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def migration_summary(tick_stats: list) -> str:
    if not tick_stats:
        return "Нет данных."
    total_moves    = sum(s["moves"] for s in tick_stats)
    total_commutes = sum(s.get("commutes", 0) for s in tick_stats)
    total_adapts   = sum(s.get("adapts", 0) for s in tick_stats)
    avg_moves = total_moves / len(tick_stats)

    lines = [
        "\n" + "=" * 70,
        "  СВОДКА МИГРАЦИИ",
        "=" * 70,
        f"  Переездов за симуляцию:  {total_moves:,}",
        f"  Маятниковых решений:      {total_commutes:,}",
        f"  Адаптаций на месте:       {total_adapts:,}",
        f"  Среднее переездов/тик:    {avg_moves:.1f}",
        "",
        "  Динамика переездов по годам:",
    ]

    yearly = {}
    for s in tick_stats:
        year = (s["tick"] - 1) // 12 + 1
        yearly.setdefault(year, []).append(s["moves"])

    max_annual = max(sum(v) for v in yearly.values()) if yearly else 1
    for year, monthly in sorted(yearly.items()):
        annual = sum(monthly)
        bar = _bar(annual, max_annual, width=22)
        lines.append(f"  Год {year:2d}  {annual:>6,} переездов  {bar}")

    lines.append("=" * 70)
    return "\n".join(lines)


def compare_snapshots(snapshots: dict, tick_stats: list) -> str:
    ticks = sorted(snapshots.keys())
    lines = [
        "\n" + "=" * 70,
        "  ДИНАМИКА ПО РЕГИОНАМ (агенты)",
        "=" * 70,
    ]

    header = f"  {'Регион':<22}"
    for t in ticks:
        lbl = "Старт" if t == 0 else f"Тик {t}"
        header += f"  {lbl:>8}"
    header += f"  {'Δ':>8}"
    lines.append(header)
    lines.append("  " + "-" * (22 + len(ticks) * 10 + 12))

    all_regions = sorted(snapshots[ticks[0]]["region"].unique())
    for region in all_regions:
        name = REGION_NAMES.get(region, region)
        row = f"  {name:<22}"
        counts = []
        for t in ticks:
            count = (snapshots[t]["region"] == region).sum()
            counts.append(count)
            row += f"  {count:>8,}"
        delta = counts[-1] - counts[0]
        sign  = "+" if delta >= 0 else ""
        row  += f"  {sign}{delta:>7,}"
        lines.append(row)

    lines.append("=" * 70)
    lines.append(migration_summary(tick_stats))
    return "\n".join(lines)
