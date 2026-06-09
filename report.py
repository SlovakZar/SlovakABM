"""
report.py v5 — Приоритезированный консольный отчёт с критическими метриками.

v5 (переработка под требования приоритезации):
  - demographic_portrait: exclude_students, критические метрики (безработица, зарплата),
    регионы с экономикой (зарплата, безработица), gaps по доменам и статусам.
  - migration_summary: тренды экономических индикаторов по годам (бары).
  - compare_snapshots: региональный баланс с Δ безработицы и зарплаты.
  - agent_behavior_audit: только в полном режиме (detail=True).
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

# ── Вспомогательные ───────────────────────────────────────────────────────────

def _pct(part, total) -> str:
    if total == 0: return "  0.0%"
    return f"{part/total*100:5.1f}%"


def _bar(value, max_value, width=18) -> str:
    if max_value == 0: return " " * width
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


def _hline(width=78, char="─") -> str:
    return char * width


def _section(title: str, width=78) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEMOGRAPHIC PORTRAIT — основной портрет с приоритезацией метрик
# ══════════════════════════════════════════════════════════════════════════════

def demographic_portrait(
    df: pd.DataFrame,
    label: str = "Снимок",
    tick_num: Optional[int] = None,
    exclude_students: bool = True,
    detail: bool = False,
) -> str:
    """
    Демографический портрет с приоритезацией метрик.

    Критические (всегда):
      - Количество агентов (без студентов), доля безработных
      - Средняя зарплата (общая, медиана)
      - Таблица регионов: население, средняя зарплата, уровень безработицы

    Важные (всегда):
      - Satisfaction по 4 доменам
      - Gaps по доменам с разбивкой по статусам занятости
      - Топ-5 отраслей среди занятых и безработных

    Дополнительные (detail=True):
      - Распределение типов агентов
      - Психологические параметры по типам
      - Топ-10 маятниковых маршрутов
    """
    lines = []
    total_all = len(df)
    n_students = int((df["status"] == "student").sum()) if "status" in df.columns else 0

    # Фильтр студентов
    if exclude_students and "status" in df.columns:
        work_df = df[df["status"] != "student"].copy()
    else:
        work_df = df.copy()
    total = len(work_df)

    # ── Шапка ────────────────────────────────────────────────────────────────
    header = "=" * 78
    title  = f"  ДЕМОГРАФИЧЕСКИЙ ПОРТРЕТ — {label}"
    if tick_num is not None:
        yr = tick_num // 12
        mo = tick_num % 12 or 12
        title += f"  [тик {tick_num} / год {yr} мес {mo}]"
    lines += [header, title, header]

    lines.append(f"\n  Агентов всего: {total_all:,}")
    if exclude_students and n_students > 0:
        lines.append(f"    из них студентов: {n_students:,} (исключены из анализа)")
    lines.append(f"  Анализируемая популяция (без студентов): {total:,}\n")

    # ═══ КРИТИЧЕСКИЕ МЕТРИКИ ════════════════════════════════════════════════
    lines.append(_section("КРИТИЧЕСКИЕ МЕТРИКИ"))

    # Безработица
    if "status" in work_df.columns:
        n_unemp = int((work_df["status"] == "unemployed").sum())
        unemp_rate = n_unemp / total if total > 0 else 0
        n_commuters = int((work_df["status"] == "commute").sum())
        n_stay = int((work_df["status"] == "stay").sum())
        lines.append(f"  Безработных:        {n_unemp:>8,}  ({unemp_rate:.1%})")
        lines.append(f"  Занятых на месте:   {n_stay:>8,}  ({n_stay/total*100 if total else 0:.1f}%)")
        lines.append(f"  Маятников:          {n_commuters:>8,}  ({n_commuters/total*100 if total else 0:.1f}%)")
        bar_max = max(n_stay, n_commuters, n_unemp, 1)
        lines.append(f"  {'Статусы:':<20}  {_bar(n_stay, bar_max)} stay")
        lines.append(f"  {'':20}  {_bar(n_commuters, bar_max)} commute")
        lines.append(f"  {'':20}  {_bar(n_unemp, bar_max)} unemployed")

    # Зарплата
    if "wage" in work_df.columns:
        employed_w = work_df[work_df["wage"] > 0]
        if not employed_w.empty:
            avg_w = employed_w["wage"].mean()
            med_w = employed_w["wage"].median()
            q25 = employed_w["wage"].quantile(0.25)
            q75 = employed_w["wage"].quantile(0.75)
            lines.append(f"\n  Средняя зарплата (занятые):  {avg_w:>10,.0f} €")
            lines.append(f"  Медианная зарплата:         {med_w:>10,.0f} €")
            lines.append(f"  Q25–Q75:                    {q25:>10,.0f} – {q75:,.0f} €")
        else:
            lines.append("\n  [Нет данных по зарплатам занятых]")

    # ═══ РЕГИОНАЛЬНАЯ ТАБЛИЦА ═══════════════════════════════════════════════
    lines.append(_section("РЕГИОНЫ: НАСЕЛЕНИЕ, ЗАРПЛАТА, БЕЗРАБОТИЦА"))

    if "region" in work_df.columns:
        lines.append(f"  {'Регион':<22} {'Население':>8}  {'Доля':>6}  "
                     f"{'Ср.зарплата':>11}  {'Безраб.':>8}  {'Кол-во':>6}")
        lines.append("  " + _hline(72))

        region_stats = []
        for code, name in REGION_NAMES.items():
            reg = work_df[work_df["region"] == code]
            pop = len(reg)
            if pop == 0:
                continue
            share = pop / total if total else 0
            avg_w = reg[reg["wage"] > 0]["wage"].mean() if "wage" in reg.columns else 0
            n_unemp_r = int((reg["status"] == "unemployed").sum()) if "status" in reg.columns else 0
            unemp_r = n_unemp_r / pop if pop else 0
            region_stats.append((name, pop, share, avg_w, unemp_r, n_unemp_r))

        # Сортируем по населению
        region_stats.sort(key=lambda x: x[1], reverse=True)

        max_pop = max(r[1] for r in region_stats) if region_stats else 1
        for name, pop, share, avg_w, unemp_r, n_unemp_r in region_stats:
            lines.append(f"  {name:<22} {pop:>8,}  {share:>5.1%}  "
                         f"{avg_w:>9,.0f} €  {unemp_r:>7.1%}  {n_unemp_r:>6,}")
    else:
        lines.append("  [Столбец region отсутствует]")

    # ═══ ВАЖНЫЕ МЕТРИКИ ══════════════════════════════════════════════════════
    lines.append(_section("ВАЖНЫЕ МЕТРИКИ"))

    # Satisfaction по доменам
    lines.append("  SATISFACTION ПО ДОМЕНАМ (0–1, средние)")
    sat_items = [
        ("sat_economic", "Economic"),
        ("sat_social",   "Social  "),
        ("sat_family",   "Family  "),
        ("sat_place",    "Place   "),
    ]
    sat_max = 0
    sat_vals = {}
    for col, label_d in sat_items:
        if col in work_df.columns:
            m = work_df[col].mean()
            sat_vals[col] = m
            if m > sat_max:
                sat_max = m
    for col, label_d in sat_items:
        if col in sat_vals:
            m = sat_vals[col]
            bar = _bar(m, max(sat_max, 1.0), width=22)
            lines.append(f"  {label_d}  {m:.4f}  {bar}")

    # Gaps по доменам с разбивкой по статусам
    lines.append(_section("ДЕФИЦИТЫ (GAPS) ПО ДОМЕНАМ × СТАТУСАМ ЗАНЯТОСТИ"))
    domains = [
        ("sat_economic", "thr_economic", "Economic"),
        ("sat_social",   "thr_social",   "Social  "),
        ("sat_family",   "thr_family",   "Family  "),
        ("sat_place",    "thr_place",    "Place   "),
    ]

    statuses = ["stay", "commute", "unemployed"]
    status_labels = {"stay": "Stay    ", "commute": "Commute ", "unemployed": "Unemp   "}

    for sat_col, thr_col, dom_name in domains:
        if sat_col not in work_df.columns or thr_col not in work_df.columns:
            continue
        lines.append(f"\n  ── {dom_name} ──")
        lines.append(f"  {'Статус':<10} {'Gap средний':>12}  {'Gap медиана':>12}  {'Доля gap>0':>10}")

        for st in statuses:
            subset = work_df[work_df["status"] == st] if "status" in work_df.columns else pd.DataFrame()
            if subset.empty:
                continue
            thr = subset[thr_col].clip(lower=0.01)
            sat = subset[sat_col]
            gaps = (thr - sat) / thr
            gap_mean = gaps.mean()
            gap_med  = gaps.median()
            gap_pos  = (gaps > 0).mean()
            lines.append(f"  {status_labels.get(st, st):<10} {gap_mean:>12.4f}  {gap_med:>12.4f}  {gap_pos:>9.1%}")

    # Топ-5 отраслей
    employed = work_df[work_df["is_employed"] == True] if "is_employed" in work_df.columns else work_df[work_df["status"].isin(["stay", "commute"])]
    if not employed.empty and "industry" in employed.columns:
        lines.append(_section("ТОП-5 ОТРАСЛЕЙ СРЕДИ ЗАНЯТЫХ"))
        lines.append(f"  {'Отрасль':<30} {'Агентов':>8}  {'Ср.зарплата':>12}")
        lines.append("  " + _hline(54))
        top_ind = employed["industry"].value_counts().head(5)
        for ind, count in top_ind.items():
            avg_w = employed[employed["industry"] == ind]["wage"].mean()
            lines.append(f"  {str(ind)[:30]:<30} {count:>8,}  {avg_w:>10,.0f} €")

    unemployed = work_df[work_df["status"] == "unemployed"] if "status" in work_df.columns else pd.DataFrame()
    if not unemployed.empty and "industry" in unemployed.columns:
        lines.append(_section("ТОП-5 ОТРАСЛЕЙ СРЕДИ БЕЗРАБОТНЫХ"))
        lines.append(f"  {'Отрасль':<30} {'Агентов':>8}")
        lines.append("  " + _hline(42))
        top_unemp = unemployed["industry"].value_counts().head(5)
        for ind, count in top_unemp.items():
            lines.append(f"  {str(ind)[:30]:<30} {count:>8,}")

    # ═══ ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ (detail=True) ═══════════════════════════════
    if detail:
        lines.append(_section("ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ (detail=True)"))

        # Распределение типов агентов
        if "agent_type" in work_df.columns:
            lines.append("\n  ТИПЫ АГЕНТОВ")
            type_counts = work_df["agent_type"].value_counts()
            max_t = type_counts.max() if not type_counts.empty else 1
            for t, c in type_counts.items():
                lines.append(f"  {str(t):<20} {c:>8,}  {_pct(c, total)}  {_bar(c, max_t)}")

        # Психологические параметры по типам
        psych_params = [
            ("inertia",                 "Inertia"),
            ("perceived_control",       "Perceived Control"),
            ("econ_perceived_control",  "Econ PC"),
            ("job_flexibility",         "Job Flexibility"),
            ("shock_sensitivity",       "Shock Sensitivity"),
            ("info_quality",            "Info Quality"),
            ("commuter_threshold",      "Commuter Thr."),
            ("internal_mig_thr",        "Internal Mig Thr."),
        ]
        available_psych = [(c, l) for c, l in psych_params if c in work_df.columns]
        if available_psych and "agent_type" in work_df.columns:
            lines.append(_section("ПСИХОЛОГИЧЕСКИЕ ПАРАМЕТРЫ ПО ТИПАМ АГЕНТОВ"))
            agent_types = sorted(work_df["agent_type"].unique())
            header = f"  {'Параметр':<22}"
            for at in agent_types:
                header += f"  {str(at):>12}"
            lines.append(header)
            lines.append("  " + _hline(22 + len(agent_types) * 14))
            for col, label_p in available_psych:
                row = f"  {label_p:<22}"
                for at in agent_types:
                    val = work_df[work_df["agent_type"] == at][col].mean()
                    row += f"  {val:>12.4f}"
                lines.append(row)
        elif available_psych:
            lines.append(_section("ПСИХОЛОГИЧЕСКИЕ ПАРАМЕТРЫ (ОБЩИЕ)"))
            for col, label_p in available_psych:
                m = work_df[col].mean()
                lines.append(f"  {label_p:<24} {m:.4f}")

        # Топ-10 маятниковых маршрутов
        if "residence_district" in work_df.columns and "workplace_district" in work_df.columns:
            lines.append(_section("ТОП-10 НАПРАВЛЕНИЙ МАЯТНИКОВОЙ МИГРАЦИИ"))
            lines.append(f"  {'Живут в районе':<22} →  {'Работают в районе':<22} | {'Агентов':>6}")
            lines.append("  " + _hline(60))
            commuters_df = work_df[work_df["residence_district"] != work_df["workplace_district"]]
            if not commuters_df.empty:
                top_commutes = (commuters_df.groupby(["residence_district", "workplace_district"])
                                .size().sort_values(ascending=False).head(10))
                for (res, work), count in top_commutes.items():
                    r_name = str(res).replace("District of ", "")[:20]
                    w_name = str(work).replace("District of ", "")[:20]
                    lines.append(f"  {r_name:<22} →  {w_name:<22} | {count:>6,}")
            else:
                lines.append("  [Маятниковые связи между районами не обнаружены]")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 2. AGENT BEHAVIOR AUDIT — глубинная диагностика (только detail=True)
# ══════════════════════════════════════════════════════════════════════════════

def agent_behavior_audit(action_log: Optional[List[dict]], sample_size: int = 30) -> str:
    """
    Поведенческий аудит на основе action_log из FFT-pipeline.

    Выводится только в полном отчёте (detail=True / mode='full').
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

    rng = np.random.default_rng(42)
    n_sample = min(sample_size, len(action_log))
    indices = rng.choice(len(action_log), n_sample, replace=False)
    sampled = [action_log[i] for i in indices]

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

    lines.append("  " + "─" * 97)

    # Сводка по выборке
    domains = [str(a.get("activation_domain", "?")) for a in sampled]
    dom_counts = pd.Series(domains).value_counts()
    dom_str = ", ".join([f"{k}: {v}" for k, v in dom_counts.items()])
    lines.append(f"  Активации по доменам: {dom_str}")

    decisions = [str(a.get("decision", "?")) for a in sampled]
    dec_counts = pd.Series(decisions).value_counts()
    dec_str = ", ".join([f"{k}: {v}" for k, v in dec_counts.items()])
    lines.append(f"  Решения: {dec_str}")

    raises = [a.get("desired_raise", 0) for a in sampled
              if a.get("desired_raise", 0) > 0]
    if raises:
        avg_raise = sum(raises) / len(raises)
        lines.append(f"  Средняя желаемая прибавка (среди ищущих работу): {avg_raise:.1%}")
    else:
        lines.append("  Средняя желаемая прибавка: — (нет данных)")

    lines.append("═" * 100)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MIGRATION SUMMARY — сводка динамики с экономическими трендами
# ══════════════════════════════════════════════════════════════════════════════

def migration_summary(tick_stats: list) -> str:
    """
    Сводка динамики: переезды, активации + экономические тренды по годам.

    Годовые бары для:
      - avg_wage (средняя зарплата)
      - avg_dissat (средняя неудовлетворённость)
      - n_unemployed (число безработных)
      - jobs_pressure_max (макс. давление на рынок труда)
      - moves (переезды)
    """
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
        "  СВОДКА ДИНАМИКИ И ЭКОНОМИЧЕСКИЕ ТРЕНДЫ",
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
    ]

    # ── Годовые тренды: агрегация ────────────────────────────────────────────
    yearly_total    = {}
    yearly_econ     = {}
    yearly_place    = {}
    yearly_wage     = {}
    yearly_dissat   = {}
    yearly_unemp    = {}
    yearly_pressure = {}

    for s in tick_stats:
        year = (s["tick"] - 1) // 12 + 1
        yearly_total.setdefault(year, []).append(s.get("moves", 0))
        yearly_econ.setdefault(year, []).append(s.get("econ_driven_moves", 0))
        yearly_place.setdefault(year, []).append(s.get("place_driven_moves", 0))
        yearly_wage.setdefault(year, []).append(s.get("avg_wage", 0))
        yearly_dissat.setdefault(year, []).append(s.get("avg_dissat", 0))
        yearly_unemp.setdefault(year, []).append(s.get("n_unemployed", 0))
        yearly_pressure.setdefault(year, []).append(s.get("jobs_pressure_max", 0))

    years = sorted(yearly_total.keys())

    # ── Тренд переездов (с разбивкой econ/place) ────────────────────────────
    lines.append(_section("ТРЕНД ПЕРЕЕЗДОВ ПО ГОДАМ"))
    has_econ  = any(sum(v) > 0 for v in yearly_econ.values())
    has_place = any(sum(v) > 0 for v in yearly_place.values())
    max_moves = max(sum(v) for v in yearly_total.values()) if yearly_total else 1

    if has_econ or has_place:
        lines.append(f"  {'Год':<6} {'Всего':>8}  {'Econ':>8}  {'Place':>8}  {'':22}")
        lines.append("  " + _hline(58))
        for year in years:
            t = sum(yearly_total[year])
            e = sum(yearly_econ[year])
            p = sum(yearly_place[year])
            bar = _bar(t, max_moves, width=22)
            lines.append(f"  {year:<6} {t:>8,}  {e:>8,}  {p:>8,}  {bar}")
    else:
        for year in years:
            t = sum(yearly_total[year])
            bar = _bar(t, max_moves, width=22)
            lines.append(f"  Год {year:2d}  {t:>6,} актов переезда  {bar}")

    # ── Тренд зарплат ───────────────────────────────────────────────────────
    lines.append(_section("ТРЕНД СРЕДНЕЙ ЗАРПЛАТЫ ПО ГОДАМ"))
    wage_vals = [sum(yearly_wage[y]) / len(yearly_wage[y]) if yearly_wage[y] else 0 for y in years]
    wage_min = min(wage_vals) if wage_vals else 0
    wage_max = max(wage_vals) if wage_vals else 1
    wage_range = wage_max - wage_min if wage_max > wage_min else 1
    lines.append(f"  {'Год':<6} {'Ср.зарплата':>12}  {'Норм':>6}  {'':30}")
    lines.append("  " + _hline(60))
    for year, w_avg in zip(years, wage_vals):
        norm = (w_avg - wage_min) / wage_range if wage_range > 0 else 0.5
        bar = _bar(norm, 1.0, width=30)
        lines.append(f"  {year:<6} {w_avg:>10,.0f} €  {norm:>5.2f}  {bar}")

    # ── Тренд безработицы ───────────────────────────────────────────────────
    lines.append(_section("ТРЕНД БЕЗРАБОТИЦЫ ПО ГОДАМ"))
    unemp_vals = [sum(yearly_unemp[y]) / len(yearly_unemp[y]) if yearly_unemp[y] else 0 for y in years]
    unemp_max = max(unemp_vals) if unemp_vals else 1
    lines.append(f"  {'Год':<6} {'Безработных':>12}  {'':30}")
    lines.append("  " + _hline(52))
    for year, u_avg in zip(years, unemp_vals):
        bar = _bar(u_avg, unemp_max, width=30) if unemp_max > 0 else " " * 30
        lines.append(f"  {year:<6} {u_avg:>10,.0f}  {bar}")

    # ── Тренд dissatisfaction ───────────────────────────────────────────────
    lines.append(_section("ТРЕНД НЕУДОВЛЕТВОРЁННОСТИ (DISSAT) ПО ГОДАМ"))
    dissat_vals = [sum(yearly_dissat[y]) / len(yearly_dissat[y]) if yearly_dissat[y] else 0 for y in years]
    dissat_max = max(dissat_vals) if dissat_vals else 1
    lines.append(f"  {'Год':<6} {'Avg Dissat':>12}  {'':30}")
    lines.append("  " + _hline(52))
    for year, d_avg in zip(years, dissat_vals):
        bar = _bar(d_avg, dissat_max, width=30) if dissat_max > 0 else " " * 30
        lines.append(f"  {year:<6} {d_avg:>12.4f}  {bar}")

    # ── Тренд jobs_pressure_max ─────────────────────────────────────────────
    lines.append(_section("ТРЕНД МАКС. ДАВЛЕНИЯ НА РЫНОК ТРУДА (JOBS PRESSURE)"))
    press_vals = [sum(yearly_pressure[y]) / len(yearly_pressure[y]) if yearly_pressure[y] else 0 for y in years]
    press_max = max(press_vals) if press_vals else 1
    lines.append(f"  {'Год':<6} {'Jobs Press. max':>16}  {'':30}")
    lines.append("  " + _hline(56))
    for year, p_avg in zip(years, press_vals):
        bar = _bar(p_avg, press_max, width=30) if press_max > 0 else " " * 30
        lines.append(f"  {year:<6} {p_avg:>16.2f}  {bar}")

    lines.append("=" * 78)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPARE SNAPSHOTS — межрегиональный баланс + миграция
# ══════════════════════════════════════════════════════════════════════════════

def compare_snapshots(
    snapshots: dict,
    tick_stats: list,
    all_action_log: Optional[List[dict]] = None,
    detail: bool = False,
) -> str:
    """
    Межрегиональный баланс населения с экономическими Δ.

    Для каждого снимка показывает:
      - Ключевые цифры (население, безработица, средняя зарплата)
      - Таблицу регионов с Δ населения, безработицы, зарплаты

    Затем migration_summary с трендами.
    При detail=True — agent_behavior_audit.
    """
    ticks = sorted(snapshots.keys())
    lines = [
        "\n" + "=" * 78,
        "  МЕЖРЕГИОНАЛЬНЫЙ БАЛАНС НАСЕЛЕНИЯ (Краи Словакии)",
        "=" * 78,
    ]

    # ── Сводная таблица по тикам ────────────────────────────────────────────
    first_tick = ticks[0]
    last_tick  = ticks[-1]

    if "region" in snapshots[first_tick].columns:
        all_regions = sorted(snapshots[first_tick]["region"].unique())
    else:
        all_regions = []

    # Заголовок с тремя метриками на тик: население, безработица, зарплата
    header = f"  {'Регион':<22}"
    for t in ticks:
        lbl = "Старт" if t == 0 else f"Тик {t}"
        header += f"  {lbl:>9}"
    header += f"  {'Δ нас.':>8}  {'Δ безраб.':>10}  {'Δ зарпл.':>10}"
    lines.append(header)
    lines.append("  " + _hline(22 + len(ticks) * 11 + 34))

    for region in all_regions:
        name = REGION_NAMES.get(region, region)
        row = f"  {name:<22}"
        counts = []
        unemp_rates = []
        avg_wages = []
        for t in ticks:
            snap = snapshots[t]
            reg_df = snap[snap["region"] == region]
            count = len(reg_df)
            counts.append(count)
            row += f"  {count:>9,}"

            # Безработица
            if "status" in reg_df.columns:
                unemp_r = (reg_df["status"] == "unemployed").mean()
            else:
                unemp_r = 0
            unemp_rates.append(unemp_r)

            # Зарплата
            if "wage" in reg_df.columns:
                w = reg_df[reg_df["wage"] > 0]["wage"].mean()
            else:
                w = 0
            avg_wages.append(w)

        # Δ населения
        delta_pop = counts[-1] - counts[0]
        sign_pop  = "+" if delta_pop >= 0 else ""
        # Δ безработицы (в процентных пунктах)
        delta_unemp = (unemp_rates[-1] - unemp_rates[0]) * 100
        sign_unemp  = "+" if delta_unemp >= 0 else ""
        # Δ зарплаты
        delta_wage = avg_wages[-1] - avg_wages[0]
        sign_wage  = "+" if delta_wage >= 0 else ""

        row += f"  {sign_pop}{delta_pop:>7,}  {sign_unemp}{delta_unemp:>8.1f}pp  {sign_wage}{delta_wage:>8,.0f}€"
        lines.append(row)

    # ── Строка ИТОГО ────────────────────────────────────────────────────────
    first_df = snapshots[first_tick]
    last_df  = snapshots[last_tick]
    total_first = len(first_df)
    total_last  = len(last_df)
    u_first = (first_df["status"] == "unemployed").mean() if "status" in first_df.columns else 0
    u_last  = (last_df["status"] == "unemployed").mean() if "status" in last_df.columns else 0
    w_first = first_df[first_df["wage"] > 0]["wage"].mean() if "wage" in first_df.columns else 0
    w_last  = last_df[last_df["wage"] > 0]["wage"].mean() if "wage" in last_df.columns else 0

    lines.append("  " + _hline(22 + len(ticks) * 11 + 34))
    row_total = f"  {'ИТОГО':<22}"
    for t in ticks:
        row_total += f"  {len(snapshots[t]):>9,}"
    d_pop = total_last - total_first
    d_unemp = (u_last - u_first) * 100
    d_wage = w_last - w_first
    sp = "+" if d_pop >= 0 else ""
    su = "+" if d_unemp >= 0 else ""
    sw = "+" if d_wage >= 0 else ""
    row_total += f"  {sp}{d_pop:>7,}  {su}{d_unemp:>8.1f}pp  {sw}{d_wage:>8,.0f}€"
    lines.append(row_total)

    lines.append("=" * 78)

    # ── Миграционная сводка с трендами ──────────────────────────────────────
    lines.append(migration_summary(tick_stats))

    # ── Поведенческий аудит (только detail) ─────────────────────────────────
    if detail:
        lines.append(agent_behavior_audit(all_action_log, sample_size=30))

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY REPORT — обёртка для полного отчёта
# ══════════════════════════════════════════════════════════════════════════════

def summary_report(
    df_final: pd.DataFrame,
    tick_stats: list,
    all_action_log: Optional[List[dict]] = None,
    snapshots: Optional[dict] = None,
    detail: bool = False,
) -> str:
    """
    Итоговый отчёт: демографический портрет финального состояния +
    сводка динамики + опционально межрегиональный баланс и аудит.

    Режимы:
      detail=False (default) — критические и важные метрики, ~2-3 экрана.
      detail=True — полный отчёт с аудитом и дополнительными метриками.
    """
    parts = []

    # 1. Демографический портрет финального состояния
    parts.append(demographic_portrait(
        df_final,
        label="ФИНАЛ",
        tick_num=None,
        exclude_students=True,
        detail=detail,
    ))

    # 2. Сводка динамики
    parts.append(migration_summary(tick_stats))

    # 3. Межрегиональный баланс (если есть снимки)
    if snapshots and len(snapshots) >= 2:
        parts.append(compare_snapshots(snapshots, tick_stats, all_action_log, detail=detail))
    elif detail and all_action_log:
        parts.append(agent_behavior_audit(all_action_log, sample_size=30))

    return "\n\n".join(parts)

