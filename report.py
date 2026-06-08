"""
report.py v4 — демографический портрет, матрицы потоков и агентский аудит.

v4 (адаптация под архитектуру v6):
  - Демографический портрет: статусы занятости, топ-5 отраслей, средняя зарплата.
  - Поведенческий аудит: работает на основе action_log из FFT-pipeline.
  - Сводка динамики: econ/place-активации, спутниковые переезды.
  - Устойчивость к старым данным: все поля проверяются через .get().
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

STATUS_LABELS = {
    "stay":       "Живут и работают дома",
    "commute":    "Маятники",
    "unemployed": "Безработные",
    "student":    "Студенты",
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

    header = "=" * 78
    title  = f"  ДЕМОГРАФИЧЕСКИЙ ПОРТРЕТ — {label}"
    if tick_num is not None:
        yr = tick_num // 12
        mo = tick_num % 12 or 12
        title += f"  [тик {tick_num} / год {yr} мес {mo}]"
    lines += [header, title, header]
    lines.append(f"\n  Агентов всего: {total:,}\n")

    # ── Статусы занятости ────────────────────────────────────────────────────
    lines.append("  СТАТУСЫ ЗАНЯТОСТИ")
    lines.append(f"  {'Статус':<28} {'Агентов':>8}  {'Доля':>6}  {'':18}")
    lines.append("  " + "-" * 64)
    if "status" in df.columns:
        by_status = df["status"].value_counts()
        max_s = by_status.max() if not by_status.empty else 1
        for code in ["stay", "commute", "unemployed", "student"]:
            count = by_status.get(code, 0)
            name = STATUS_LABELS.get(code, code)
            lines.append(f"  {name:<28} {count:>8,}  {_pct(count, total)}  {_bar(count, max_s)}")
    else:
        lines.append("  [Столбец status отсутствует]")

    # ── Регионы проживания ──────────────────────────────────────────────────
    lines.append("\n  РАСПРЕДЕЛЕНИЕ ПО РЕГИОНАМ ПРОЖИВАНИЯ")
    lines.append(f"  {'Регион':<22} {'Агентов':>8}  {'Доля':>6}  {'':18}")
    lines.append("  " + "-" * 58)
    if "region" in df.columns:
        by_region = df.groupby("region")["id"].count().sort_values(ascending=False)
        max_r = by_region.max() if not by_region.empty else 1
        for code, count in by_region.items():
            name = REGION_NAMES.get(code, code)
            lines.append(f"  {name:<22} {count:>8,}  {_pct(count, total)}  {_bar(count, max_r)}")
    else:
        lines.append("  [Столбец region отсутствует]")

    # ── Топ-5 отраслей среди занятых ────────────────────────────────────────
    employed = df[df["is_employed"] == True] if "is_employed" in df.columns else df[df["status"].isin(["stay", "commute"])]
    if not employed.empty and "industry" in employed.columns:
        lines.append("\n  ТОП-5 ОТРАСЛЕЙ СРЕДИ ЗАНЯТЫХ")
        lines.append(f"  {'Отрасль':<30} {'Агентов':>8}  {'Ср.зарплата':>12}")
        lines.append("  " + "-" * 54)
        top_ind = employed["industry"].value_counts().head(5)
        for ind, count in top_ind.items():
            avg_w = employed[employed["industry"] == ind]["wage"].mean()
            lines.append(f"  {str(ind)[:30]:<30} {count:>8,}  {avg_w:>10,.0f} €")
    else:
        lines.append("\n  [Нет данных по отраслям занятых]")

    # ── Топ-5 отраслей среди безработных ────────────────────────────────────
    unemployed = df[df["status"] == "unemployed"] if "status" in df.columns else pd.DataFrame()
    if not unemployed.empty and "industry" in unemployed.columns:
        lines.append("\n  ТОП-5 ОТРАСЛЕЙ СРЕДИ БЕЗРАБОТНЫХ")
        lines.append(f"  {'Отрасль':<30} {'Агентов':>8}")
        lines.append("  " + "-" * 42)
        top_unemp = unemployed["industry"].value_counts().head(5)
        for ind, count in top_unemp.items():
            lines.append(f"  {str(ind)[:30]:<30} {count:>8,}")
    else:
        lines.append("\n  [Нет данных по отраслям безработных]")

    # ── Топ-10 маятниковых маршрутов ────────────────────────────────────────
    lines.append("\n  ТОП-10 НАПРАВЛЕНИЙ МАЯТНИКОВОЙ МИГРАЦИИ (COMMUTING)")
    lines.append(f"  {'Живут в районе':<22} →  {'Работают в районе':<22} | {'Агентов':>6}")
    lines.append("  " + "-" * 60)
    if "residence_district" in df.columns and "workplace_district" in df.columns:
        commuters = df[df["residence_district"] != df["workplace_district"]]
        if not commuters.empty:
            top_commutes = commuters.groupby(["residence_district", "workplace_district"]).size().sort_values(ascending=False).head(10)
            for (res, work), count in top_commutes.items():
                r_name = str(res).replace("District of ", "")[:20]
                w_name = str(work).replace("District of ", "")[:20]
                lines.append(f"  {r_name:<22} →  {w_name:<22} | {count:>6,}")
        else:
            lines.append("  [Маятниковые связи между районами не обнаружены]")
    else:
        lines.append("  [Нет данных residence_district / workplace_district]")

    # ── Домены satisfaction ─────────────────────────────────────────────────
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

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


def agent_behavior_audit(action_log: Optional[List[dict]], sample_size: int = 30) -> str:
    """
    Поведенческий аудит на основе action_log из FFT-pipeline.

    Формат строки:
      ID | Тип агента | Домен активации | Решение |
      Пред. жильё → Новое жильё | Пред. работа → Нов. работа |
      Зарплата | Отрасль | Желаемая прибавка

    Устойчив к отсутствию лога и старым данным.
    """
    lines = [
        "\n" + "═" * 100,
        f"  ПОВЕДЕНЧЕСКИЙ АУДИТ АГЕНТОВ (Срез до {sample_size} случайных решений из лога)",
        "═" * 100,
    ]

    if not action_log:
        lines.append("  [Нет данных поведенческого аудита — action_log пуст]")
        lines.append("═" * 100)
        return "\n".join(lines)

    # Выбираем случайную выборку
    rng = np.random.default_rng(42)
    n_sample = min(sample_size, len(action_log))
    indices = rng.choice(len(action_log), n_sample, replace=False)
    sampled = [action_log[i] for i in indices]

    # Заголовок
    lines.append(
        f"  {'ID':<6} | {'Тип':<12} | {'Домен':<10} | {'Решение':<9} | "
        f"{'Жильё до→после':<28} | {'Работа до→после':<28} | "
        f"{'Зарплата':>8} | {'Отрасль':<18} | {'Надбавка':>8}"
    )
    lines.append("  " + "─" * 97)

    for ag in sampled:
        ag_id       = str(ag.get("id", "?"))[:6]
        ag_type     = str(ag.get("agent_type", "?"))[:12]
        act_domain  = str(ag.get("activation_domain", "?"))[:10]
        decision    = str(ag.get("decision", "?"))[:9]

        prev_res    = str(ag.get("prev_residence", "?")).replace("District of ", "")[:12]
        new_res     = str(ag.get("new_residence", "?")).replace("District of ", "")[:12]
        prev_wp     = str(ag.get("prev_workplace", "?")).replace("District of ", "")[:12]
        new_wp      = str(ag.get("new_workplace", "?")).replace("District of ", "")[:12]

        res_flow    = f"{prev_res}→{new_res}"
        work_flow   = f"{prev_wp}→{new_wp}"

        wage_val    = ag.get("wage", 0)
        wage_str    = f"{wage_val:,.0f}€" if wage_val else "—"
        industry    = str(ag.get("industry", "—"))[:18]
        desired     = ag.get("desired_raise", 0)
        desired_str = f"{desired:.1%}" if desired else "—"

        lines.append(
            f"  {ag_id:<6} | {ag_type:<12} | {act_domain:<10} | {decision:<9} | "
            f"{res_flow:<28} | {work_flow:<28} | "
            f"{wage_str:>8} | {industry:<18} | {desired_str:>8}"
        )

    # ── Сводка по выборке ───────────────────────────────────────────────────
    lines.append("  " + "─" * 97)

    # Распределение по доменам активации
    domains = [str(a.get("activation_domain", "?")) for a in sampled]
    dom_counts = pd.Series(domains).value_counts()
    dom_str = ", ".join([f"{k}: {v}" for k, v in dom_counts.items()])
    lines.append(f"  Активации по доменам: {dom_str}")

    # Распределение решений
    decisions = [str(a.get("decision", "?")) for a in sampled]
    dec_counts = pd.Series(decisions).value_counts()
    dec_str = ", ".join([f"{k}: {v}" for k, v in dec_counts.items()])
    lines.append(f"  Решения: {dec_str}")

    # Средняя желаемая прибавка (только для econ-driven)
    raises = [a.get("desired_raise", 0) for a in sampled
              if a.get("desired_raise", 0) > 0]
    if raises:
        avg_raise = sum(raises) / len(raises)
        lines.append(f"  Средняя желаемая прибавка (среди ищущих работу): {avg_raise:.1%}")
    else:
        lines.append(f"  Средняя желаемая прибавка: — (нет данных)")

    lines.append("═" * 100)
    return "\n".join(lines)


def migration_summary(tick_stats: list) -> str:
    if not tick_stats:
        return "Нет данных."

    total_moves       = sum(s.get("moves", 0) for s in tick_stats)
    total_commutes    = sum(s.get("commutes", 0) for s in tick_stats)
    total_adapts      = sum(s.get("adapts", 0) for s in tick_stats)
    total_satellite   = sum(s.get("satellite_moves", 0) for s in tick_stats)
    total_econ_moves  = sum(s.get("econ_driven_moves", 0) for s in tick_stats)
    total_place_moves = sum(s.get("place_driven_moves", 0) for s in tick_stats)
    total_econ_act    = sum(s.get("econ_activated", 0) for s in tick_stats)
    total_place_act   = sum(s.get("place_activated", 0) for s in tick_stats)
    avg_moves = total_moves / len(tick_stats) if tick_stats else 0

    lines = [
        "\n" + "=" * 78,
        "  СВОДКА ДИНАМИКИ И ИЗМЕНЕНИЙ СТРАТЕГИЙ",
        "=" * 78,
        f"  Активаций — экономика:     {total_econ_act:>8,}",
        f"  Активаций — место:         {total_place_act:>8,}",
        f"  Физических переездов (Move):     {total_moves:>8,}",
        f"    из-за экономики:               {total_econ_moves:>8,}",
        f"    из-за места:                   {total_place_moves:>8,}",
        f"    в спутники:                    {total_satellite:>8,}",
        f"  Маятниковых решений (Commute):   {total_commutes:>8,}",
        f"  Вынужденных адаптаций (Adapt):   {total_adapts:>8,}",
        f"  Средняя интенсивность миграции: {avg_moves:.1f} переездов/тик",
        "",
        "  Динамика изменения стратегий по годам (Переезды):",
    ]

    # Годовые переезды: общие + по типам если есть данные
    yearly_total = {}
    yearly_econ  = {}
    yearly_place = {}
    for s in tick_stats:
        year = (s["tick"] - 1) // 12 + 1
        yearly_total.setdefault(year, []).append(s.get("moves", 0))
        yearly_econ.setdefault(year, []).append(s.get("econ_driven_moves", 0))
        yearly_place.setdefault(year, []).append(s.get("place_driven_moves", 0))

    has_econ  = any(sum(v) > 0 for v in yearly_econ.values())
    has_place = any(sum(v) > 0 for v in yearly_place.values())

    max_annual = max(sum(v) for v in yearly_total.values()) if yearly_total else 1

    if has_econ or has_place:
        lines.append(f"  {'Год':<6} {'Всего':>8}  {'Econ':>8}  {'Place':>8}  {'':22}")
        lines.append("  " + "-" * 58)
        for year in sorted(yearly_total.keys()):
            t = sum(yearly_total[year])
            e = sum(yearly_econ[year])
            p = sum(yearly_place[year])
            bar = _bar(t, max_annual, width=22)
            lines.append(f"  {year:<6} {t:>8,}  {e:>8,}  {p:>8,}  {bar}")
    else:
        for year, monthly in sorted(yearly_total.items()):
            annual = sum(monthly)
            bar = _bar(annual, max_annual, width=22)
            lines.append(f"  Год {year:2d}  {annual:>6,} актов переезда  {bar}")

    lines.append("=" * 78)
    return "\n".join(lines)


def compare_snapshots(
    snapshots: dict,
    tick_stats: list,
    all_action_log: Optional[List[dict]] = None,
) -> str:
    ticks = sorted(snapshots.keys())
    lines = [
        "\n" + "=" * 78,
        "  МЕЖРЕГИОНАЛЬНЫЙ БАЛАНС НАСЕЛЕНИЯ (Краи Словакии)",
        "=" * 78,
    ]

    header = f"  {'Регион':<22}"
    for t in ticks:
        lbl = "Старт" if t == 0 else f"Тик {t}"
        header += f"  {lbl:>8}"
    header += f"  {'Δ':>8}"
    lines.append(header)
    lines.append("  " + "-" * (22 + len(ticks) * 10 + 12))

    first_tick = ticks[0]
    if "region" in snapshots[first_tick].columns:
        all_regions = sorted(snapshots[first_tick]["region"].unique())
    else:
        all_regions = []

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

    lines.append("=" * 78)

    # Добавляем общую сводку динамики
    lines.append(migration_summary(tick_stats))

    # Поведенческий аудит из агрегированного лога решений
    lines.append(agent_behavior_audit(all_action_log, sample_size=30))

    return "\n".join(lines)
