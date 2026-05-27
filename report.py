"""
report.py v3 — демографический портрет, матрицы потоков и агентский аудит.

Адаптирован под архитектуру FFT:
  - Разделение на residence_district и workplace_district.
  - Матрица маятниковой миграции (Commute-срез).
  - Поведенческий аудит 30 случайных агентов, принявших решения.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict

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

    header = "=" * 75
    title  = f"  ДЕМОГРАФИЧЕСКИЙ ПОРТРЕТ — {label}"
    if tick_num is not None:
        yr = tick_num // 12
        mo = tick_num % 12 or 12
        title += f"  [тик {tick_num} / год {yr} мес {mo}]"
    lines += [header, title, header]
    lines.append(f"\n  Агентов всего: {total:,}\n")

    # ── Регионы проживания (residence_district -> region) ─────────────────────
    lines.append("  РАСПРЕДЕЛЕНИЕ ПО РЕГИОНАМ ПРОЖИВАНИЯ")
    lines.append(f"  {'Регион':<22} {'Агентов':>8}  {'Доля':>6}  {'':18}")
    lines.append("  " + "-" * 58)
    by_region = df.groupby("region")["id"].count().sort_values(ascending=False)
    max_r = by_region.max() if not by_region.empty else 1
    for code, count in by_region.items():
        name = REGION_NAMES.get(code, code)
        lines.append(f"  {name:<22} {count:>8,}  {_pct(count, total)}  {_bar(count, max_r)}")

    # ── Топ-10 маятниковых маршрутов (Где живут -> Где работают) ──────────────
    lines.append("\n  ТОП-10 НАПРАВЛЕНИЙ МАЯТНИКОВОЙ МИГРАЦИИ (COMMUTING)")
    lines.append(f"  {'Живут в районе':<22} →  {'Работают в районе':<22} | {'Агентов':>6}")
    lines.append("  " + "-" * 60)
    
    # Фильтруем тех, у кого район работы отличается от района проживания
    commuters = df[df["residence_district"] != df["workplace_district"]]
    if not commuters.empty:
        top_commutes = commuters.groupby(["residence_district", "workplace_district"]).size().sort_values(ascending=False).head(10)
        for (res, work), count in top_commutes.items():
            r_name = str(res).replace("District of ", "")[:20]
            w_name = str(work).replace("District of ", "")[:20]
            lines.append(f"  {r_name:<22} →  {w_name:<22} | {count:>6,}")
    else:
        lines.append("  [Маятниковые связи между районами не обнаружены или все работают дома]")

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

    lines.append("\n" + "=" * 75)
    return "\n".join(lines)


def agent_behavior_audit(agents_list: List[dict], sample_size: int = 30) -> str:
    """
    Генерирует подробный поведенческий срез по фиксированной группе агентов,
    сделавших выбор на текущем шаге. Разделяет их по стратегиям адаптации.
    """
    lines = [
        "\n" + "═" * 85,
        f"  ПОВЕДЕНЧЕСКИЙ АУДИТ АГЕНТОВ (Срез {sample_size} случайных решений)",
        "═" * 85
    ]
    
    if not agents_list:
        lines.append("  [Нет зарегистрированных поведенческих актов за данный период]")
        lines.append("═" * 85)
        return "\n".join(lines)

    # Выбираем случайную выборку, если агентов больше лимита
    np.random.seed(42)  # Фиксация для воспроизводимости отчета
    sample_indices = np.random.choice(len(agents_list), min(sample_size, len(agents_list)), replace=False)
    sampled_agents = [agents_list[i] for i in sample_indices]

    lines.append(f"  {'ID':<6} | {'Тип':<10} | {'Решение':<10} | {'Откуда → Куда (Жилье)':<25} | {'Зарплата':<8} | {'Ctrl':<4} | {'Gap':<4}")
    lines.append("  " + "-" * 81)

    for idx, ag in enumerate(sampled_agents):
        ag_id = ag.get('id', 'N/A')
        ag_type = ag.get('agent_type', 'norm')
        
        # Определяем тип принятого решения (из лога изменений агента)
        decision = ag.get('last_decision', 'stay') # move, commute, adapt, none
        
        # Форматируем локации
        prev_res = str(ag.get('prev_residence', ag.get('residence_district'))).replace("District of ", "")[:10]
        curr_res = str(ag.get('residence_district')).replace("District of ", "")[:10]
        curr_work = str(ag.get('workplace_district')).replace("District of ", "")[:10]
        
        loc_flow = f"{prev_res}→{curr_res}" if decision == "move" else f"{curr_res} [W:{curr_work}]"
        
        wage = f"{ag.get('wage', 0):,.0f}€"
        ctrl = f"{ag.get('econ_perceived_control', 0.5):.2f}"
        gap = f"{ag.get('domain_economic_gap', 0.0):.2f}"
        
        lines.append(f"  {ag_id:<6} | {ag_type:<10} | {decision:<10} | {loc_flow:<25} | {wage:>8} | {ctrl:<4} | {gap:<4}")
    
    # Сводная аналитика по выбранному срезу
    lines.append("  " + "-" * 81)
    decisions_series = pd.Series([a.get('last_decision', 'stay') for a in sampled_agents])
    counts = decisions_series.value_counts()
    
    summary_str = "Распределение в срезе: " + ", ".join([f"{k}: {v}" for k, v in counts.items()])
    lines.append(f"  {summary_str}")
    lines.append("═" * 85)
    return "\n".join(lines)


def migration_summary(tick_stats: list) -> str:
    if not tick_stats:
        return "Нет данных."
    total_moves    = sum(s["moves"] for s in tick_stats)
    total_commutes = sum(s.get("commutes", 0) for s in tick_stats)
    total_adapts   = sum(s.get("adapts", 0) for s in tick_stats)
    avg_moves = total_moves / len(tick_stats)

    lines = [
        "\n" + "=" * 75,
        "  СВОДКА ДИНАМИКИ И ИЗМЕНЕНИЙ СТРАТЕГИЙ",
        "=" * 75,
        f"  Физических переездов (Move):     {total_moves:,}",
        f"  Маятниковых решений (Commute):   {total_commutes:,}",
        f"  Вынужденных адаптаций (Adapt):   {total_adapts:,}",
        f"  Средняя интенсивность миграции: {avg_moves:.1f} переездов/тик",
        "",
        "  Динамика изменения стратегий по годам (Переезды):",
    ]

    yearly = {}
    for s in tick_stats:
        year = (s["tick"] - 1) // 12 + 1
        yearly.setdefault(year, []).append(s["moves"])

    max_annual = max(sum(v) for v in yearly.values()) if yearly else 1
    for year, monthly in sorted(yearly.items()):
        annual = sum(monthly)
        bar = _bar(annual, max_annual, width=22)
        lines.append(f"  Год {year:2d}  {annual:>6,} актов переезда  {bar}")

    lines.append("=" * 75)
    return "\n".join(lines)


def compare_snapshots(snapshots: dict, tick_stats: list, active_agents_log: Optional[List[dict]] = None) -> str:
    ticks = sorted(snapshots.keys())
    lines = [
        "\n" + "=" * 75,
        "  МЕЖРЕГИОНАЛЬНЫЙ БАЛАНС НАСЕЛЕНИЯ (Краи Словакии)",
        "=" * 75,
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

    lines.append("=" * 75)
    
    # Добавляем общую сводку
    lines.append(migration_summary(tick_stats))
    
    # Если передан список активных агентов за тик — выводим глубокий аудит
    if active_agents_log is not None:
        lines.append(agent_behavior_audit(active_agents_log, sample_size=30))
        
    return "\n".join(lines)
