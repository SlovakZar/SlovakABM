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


def _append_dynamic_vars_section(lines: list, df: pd.DataFrame) -> None:
    """v2: Добавляет секцию динамических переменных сигнальной системы."""
    dyn_vars = [
        ("econ_penalty",             "Econ Penalty",           "штраф к D_econ"),
        ("infra_bonus",              "Infra Bonus",            "бонус к инфраструктуре"),
        ("inertia_mobility_penalty", "Inertia Mobility Penalty", "штраф к инерции от соседей"),
        ("jobloss_econ_gap_bonus",   "Jobloss Econ Gap Bonus", "ramp-бонус к econ_gap от LOST_JOB"),
        ("migration_pressure",       "Migration Pressure",     "накопленное давление к миграции"),
    ]

    available = []
    for col, label, desc in dyn_vars:
        if col in df.columns:
            available.append((col, label, desc))

    if not available:
        return

    lines.append(_section("ДИНАМИЧЕСКИЕ ПЕРЕМЕННЫЕ СИГНАЛЬНОЙ СИСТЕМЫ v2"))
    lines.append(f"  {'Переменная':<28} {'Среднее':>10}  {'Медиана':>10}  {'Q75':>10}  {'Max':>10}  {'Доля>0':>8}")
    lines.append("  " + _hline(82))

    for col, label, desc in available:
        vals = df[col].dropna()
        if len(vals) == 0:
            lines.append(f"  {label:<28} {'—':>10}  {'—':>10}  {'—':>10}  {'—':>10}  {'—':>8}")
            continue
        m = vals.mean()
        med = vals.median()
        q75 = vals.quantile(0.75)
        vmax = vals.max()
        pos_share = (vals > 0.001).mean()
        lines.append(f"  {label:<28} {m:>10.4f}  {med:>10.4f}  {q75:>10.4f}  {vmax:>10.4f}  {pos_share:>7.1%}")

    # Дополнительно: средние по статусам занятости для econ_penalty и jobloss_econ_gap_bonus
    if "status" in df.columns:
        statuses = ["stay", "commute", "unemployed"]
        sl = {"stay": "Stay    ", "commute": "Commute ", "unemployed": "Unemp   "}
        key_vars = [c for c, _, _ in available if c in ("econ_penalty", "jobloss_econ_gap_bonus")]
        if key_vars:
            lines.append(f"\n  ── По статусам занятости ──")
            lines.append(f"  {'Статус':<10} " + " ".join(f"{col:>12}" for col in key_vars))
            lines.append("  " + _hline(14 + len(key_vars) * 13))
            for st in statuses:
                sub = df[df["status"] == st]
                if sub.empty:
                    continue
                row = f"  {sl.get(st, st):<10}"
                for col in key_vars:
                    row += f" {sub[col].mean():>11.4f}"
                lines.append(row)


def _hline(width=78, char="─") -> str:
    return char * width


def _section(title: str, width=78) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def industry_jobs_snapshot(G, df=None, top_n: int = 12) -> str:
    """
    v3: Снимок industry_jobs — занятые и вакантные места по отраслям.

    Показывает для топ-N районов (по общей ёмкости) разбивку:
      occupied — занятые места (агенты с workplace=district)
      vacant   — открытые вакансии
      pressure — occupied / (occupied + vacant)

    Если передан df, то occupied считается по фактическим агентам.
    """
    lines = []
    lines.append(_section("INDUSTRY JOBS: ЗАНЯТЫЕ И ВАКАНТНЫЕ МЕСТА ПО ОТРАСЛЯМ (v3)"))

    if G is None:
        lines.append("  [Граф не передан]")
        return "\n".join(lines)

    # Собираем статистику по районам
    district_stats = []
    for district in G.nodes:
        ind_jobs = G.nodes[district].get("industry_jobs", {})
        if not ind_jobs:
            continue
        total_occ = sum(v["occupied"] for v in ind_jobs.values())
        total_vac = sum(v["vacant"] for v in ind_jobs.values())
        total_cap = total_occ + total_vac
        if total_cap == 0:
            continue
        # Фактическое число занятых агентов (если df передан)
        actual_occ = total_occ
        if df is not None:
            actual_occ = int(
                (df["workplace_district"] == district).sum()
                if "workplace_district" in df.columns
                else total_occ
            )
        district_stats.append((
            district, total_occ, total_vac, total_cap,
            actual_occ / max(total_cap, 1)
        ))

    if not district_stats:
        lines.append("  [Нет данных industry_jobs в графе]")
        return "\n".join(lines)

    # Сортируем по общей ёмкости
    district_stats.sort(key=lambda x: -x[3])
    district_stats = district_stats[:top_n]

    lines.append(f"  {'Район':<30} {'Occupied':>10} {'Vacant':>10} "
                 f"{'Всего':>10} {'Pressure':>9}")
    lines.append("  " + _hline(75))

    for d, occ, vac, cap, press in district_stats:
        name = d.replace("District of ", "")[:28]
        lines.append(f"  {name:<30} {occ:>10,} {vac:>10,} {cap:>10,} {press:>8.3f}")

    # ── Детально по отраслям для топ-3 районов ──────────────────────────
    lines.append(f"\n  ДЕТАЛЬНО ПО ОТРАСЛЯМ (топ-3 района):")
    for d, _, _, _, _ in district_stats[:3]:
        name = d.replace("District of ", "")
        ind_jobs = G.nodes[d].get("industry_jobs", {})
        if not ind_jobs:
            continue

        # Считаем фактически занятых агентов по отраслям
        actual_by_ind = {}
        if df is not None and "workplace_district" in df.columns and "industry" in df.columns:
            sub = df[df["workplace_district"] == d]
            actual_by_ind = sub.groupby("industry")["id"].count().to_dict()

        lines.append(f"\n  ── {name} ──")
        lines.append(f"  {'Отрасль':<45} {'Occ':>8} {'Vac':>8} "
                     f"{'Всего':>8} {'Факт':>8} {'Press':>7}")
        lines.append("  " + _hline(90))

        # Сортируем отрасли по общей ёмкости
        sorted_inds = sorted(
            ind_jobs.items(),
            key=lambda x: -(x[1]["occupied"] + x[1]["vacant"])
        )
        for ind, jobs in sorted_inds[:8]:
            occ = jobs["occupied"]
            vac = jobs["vacant"]
            cap = occ + vac
            actual = actual_by_ind.get(ind, occ)
            press = actual / max(cap, 1)
            ind_short = str(ind)[:43]
            lines.append(f"  {ind_short:<45} {occ:>8,} {vac:>8,} "
                         f"{cap:>8,} {actual:>8,} {press:>6.3f}")

    return "\n".join(lines)


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

    # ═══ v2: ДИНАМИЧЕСКИЕ ПЕРЕМЕННЫЕ СИГНАЛЬНОЙ СИСТЕМЫ ═══════════════════
    _append_dynamic_vars_section(lines, work_df)

    # ═══ ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ (detail=True) ═══════════════════════════════
    if detail:
        lines.append(_section("ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ (detail=True)"))

        # Психологические параметры
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
        if available_psych:
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
        f"  {'ID':<6} | {'Домен':<10} | {'Решение':<9} | "
        f"{'Жильё до→после':<28} | {'Работа до→после':<28} | "
        f"{'Зарплата':>8} | {'Отрасль':<18} | {'Надбавка':>8}"
    )
    lines.append("  " + "─" * 97)

    for ag in sampled:
        ag_id       = str(ag.get("id", "?"))[:6]
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
# 5. AGENT PARAMETERS TABLE — матрица параметров агентов с динамикой
# ══════════════════════════════════════════════════════════════════════════════

# Константы (синхронизированы с engine.py)
_NATIONAL_AVG_WAGE = 1614.0
_EDU_MAP = {"low": 0.25, "medium": 0.55, "high": 0.85}
_HOUSING_BUDGET = 0.35


def _compute_capabilities(agent_row) -> float:
    """capabilities = (income_index + education_index + weak_ties) / 3."""
    wage = float(agent_row.get("wage", 0))
    income_index = min(wage / (1.5 * _NATIONAL_AVG_WAGE), 1.0)
    edu = str(agent_row.get("education", "medium"))
    education_index = _EDU_MAP.get(edu, 0.55)
    weak_ties = float(agent_row.get("weak_ties_utility", 0.0))
    return round((income_index + education_index + weak_ties) / 3.0, 4)


def _compute_dynamic_inertia(agent_row) -> float:
    """v3: Динамическая инерция барьера 2 = inertia × max(0.15, 1 − social_boost)."""
    inertia = float(agent_row.get("inertia", 0.5))
    social_boost = float(agent_row.get("social_boost", 0.0))
    return round(inertia * max(0.15, 1.0 - social_boost), 4)


def _compute_dynamic_threshold_stage1(agent_row) -> float:
    """v3: Динамический порог барьера 1 = (internal_mig_thr + inertia_mob_penalty) × max(0.15, 1 − signal_reduction)."""
    internal_thr = float(agent_row.get("internal_mig_thr", 0.5))
    inertia_mob_pen = float(agent_row.get("inertia_mobility_penalty", 0.0))
    signal_red = float(agent_row.get("signal_reduction", 0.0))
    return round((internal_thr + inertia_mob_pen) * max(0.15, 1.0 - signal_red), 4)


def _industry_wage_in_district_report(G, district: str, industry: str) -> float:
    """Отраслевая зарплата в узле графа; fallback → avg_wage → NATIONAL_AVG_WAGE."""
    if G is None:
        return _NATIONAL_AVG_WAGE
    attr = G.nodes.get(district, {})
    sal = attr.get("salary_by_industry", {})
    if sal:
        return float(sal.get(industry, attr.get("avg_wage", _NATIONAL_AVG_WAGE)))
    return float(attr.get("avg_wage", _NATIONAL_AVG_WAGE))


def _compute_d_components(agent_row, G=None) -> dict:
    """
    Вычисляет компоненты D_instant v2 (зеркало engine._compute_d_instant).

    Возвращает словарь:
      D_econ, wage_pressure, D_place, place_reality, affordability, D_instant
    """
    wage = float(agent_row.get("wage", 0))
    industry = str(agent_row.get("industry", "Other"))
    workplace = str(agent_row.get("workplace_district", ""))
    residence = str(agent_row.get("district", ""))
    econ_gap = float(agent_row.get("econ_gap", 0.5))
    job_flex = float(agent_row.get("job_flexibility", 0.5))
    housing = float(agent_row.get("housing_price_m2", 1800.0))
    domain_future_place = float(agent_row.get("domain_future_place", 0.3))
    w_econ = float(agent_row.get("w_economic", 0.3))
    w_future = float(agent_row.get("w_future", 0.3))
    econ_penalty = float(agent_row.get("econ_penalty", 0.0))
    infra_bonus = float(agent_row.get("infra_bonus", 0.0))

    # wage_pressure: насколько зарплата агента отстаёт от отраслевой в районе работы
    industry_avg_wp = _industry_wage_in_district_report(G, workplace, industry)
    if wage > 0 and industry_avg_wp > 0:
        wage_pressure = industry_avg_wp / wage
    else:
        wage_pressure = 1.0  # безработный → максимальное давление

    # v3: econ_penalty — прямая прибавка к D_econ (не сглаживается формулой)
    D_econ = w_econ * wage_pressure * (econ_gap / max(job_flex, 0.01)) + econ_penalty

    # place_reality: качество жилья и инфраструктуры (0–1) — v2
    monthly_cost = housing * 50 * 0.004
    burden = monthly_cost / max(wage, 1.0)
    affordability = max(0.0, 1.0 - burden / _HOUSING_BUDGET)

    # infrastructure_score — из графа если есть, иначе 0.5
    if G is not None:
        infra = float(G.nodes.get(residence, {}).get("infrastructure_score", 0.5))
    else:
        infra = 0.5

    # v2: infra_component = 0.3 * (1 - infra + infra_bonus)
    infra_component = 0.3 * (1.0 - infra + infra_bonus)
    place_reality = 0.7 * affordability + infra_component

    gap = max(0.0, domain_future_place - place_reality)
    place_ratio = domain_future_place / max(place_reality, 0.001)
    amplifier = max(1.0, place_ratio)
    place_penalty = float(agent_row.get("place_deficit_penalty", 0.0))
    D_place = w_future * gap * amplifier * (1.0 + place_penalty)
    D_instant = float(np.clip(D_econ + D_place, 0.0, 1.0))

    return {
        "D_econ":           round(D_econ, 4),
        "wage_pressure":    round(wage_pressure, 4),
        "D_place":          round(D_place, 4),
        "place_reality":    round(place_reality, 4),
        "affordability":    round(affordability, 4),
        "D_instant":        round(D_instant, 4),
    }


def _fmt_arrow(v0, v6, fmt_spec=".3f", width=10) -> str:
    """Форматирует значение tick0→tick6 с выравниванием."""
    s0 = f"{v0:{fmt_spec}}"
    s6 = f"{v6:{fmt_spec}}"
    arrow = f"{s0}→{s6}"
    return f"{arrow:>{width}}"


def _fmt_val(v, fmt_spec=".3f", width=10) -> str:
    """Форматирует одиночное значение с выравниванием."""
    return f"{v:{fmt_spec}}".rjust(width)


def _fmt_bool_arrow(v0, v6, width=8) -> str:
    """Форматирует булево значение ✓/✗."""
    s0 = "✓" if v0 else "✗"
    s6 = "✓" if v6 else "✗"
    arrow = f"{s0}→{s6}"
    return f"{arrow:>{width}}"


def agent_parameters_table(
    snapshots: dict,
    G=None,
    n_show: int = 20,
    tick_a: int = 0,
    tick_b: int = 6,
    seed: int = 42,
) -> str:
    """
    Матрица параметров агентов: aspirations, D_econ/D_place и их составные,
    capabilities, inertia, dynamic_inertia, TPB, threshold, signal_reduction.

    Сравнивает tick_a (начало) и tick_b (после N тиков) для одних и тех же агентов.
    Показывает динамику изменений в формате «значение_0 → значение_N».

    Параметры:
      snapshots — словарь {tick: DataFrame}
      G         — граф (networkx.DiGraph) для вычисления отраслевых зарплат и инфраструктуры
      n_show    — сколько агентов показать (default 20)
      tick_a    — начальный тик (default 0)
      tick_b    — конечный тик (default 6)
      seed      — для воспроизводимой выборки
    """
    lines = []

    # Проверяем наличие снимков
    if tick_a not in snapshots:
        return f"\n  [agent_parameters_table] Снимок для тика {tick_a} отсутствует."
    if tick_b not in snapshots:
        return f"\n  [agent_parameters_table] Снимок для тика {tick_b} отсутствует."

    df_a = snapshots[tick_a]
    df_b = snapshots[tick_b]

    # Находим общих агентов по ID
    ids_a = set(df_a["id"].values)
    ids_b = set(df_b["id"].values)
    common_ids = sorted(ids_a & ids_b)

    if len(common_ids) == 0:
        return "\n  [agent_parameters_table] Нет общих агентов между снимками."

    # Выборка агентов
    rng = np.random.default_rng(seed)
    n_sample = min(n_show, len(common_ids))
    sampled_ids = sorted(rng.choice(list(common_ids), n_sample, replace=False))

    # Готовим отсортированные выборки
    sampled_df_a = df_a[df_a["id"].isin(sampled_ids)].sort_values("id").reset_index(drop=True)
    sampled_df_b = df_b[df_b["id"].isin(sampled_ids)].sort_values("id").reset_index(drop=True)

    # ── Заголовок ────────────────────────────────────────────────────────
    header = "═" * 160
    title = f"  МАТРИЦА ПАРАМЕТРОВ АГЕНТОВ: Динамика Тик {tick_a} → Тик {tick_b} (n={n_sample})"
    lines += [header, title, header]

    # ── Легенда ──────────────────────────────────────────────────────────
    lines.append("  ═══ БАРЬЕР 1 — Потенциал миграции (Aspirations × Capabilities vs Dynamic Inertia) ═══")
    lines.append("  Aspirations — EWMA-накопление D_instant. Старт=0 (холодный), на тике 1 = D_instant.")
    lines.append("  D_econ      — экономическая неудовлетворённость: w_econ × wage_pressure × econ_gap × (1−job_flex)")
    lines.append("  wage_pr     — wage_pressure: отставание зарплаты от отраслевой в районе работы (0–1)")
    lines.append("  D_place     — жилищная неудовлетворённость: w_future × gap × (dfp/pr) × (1+penalty)")
    lines.append("  place_r     — place_reality: 0.6×affordability + 0.4×infrastructure_score")
    lines.append("  PlacePen    — place_deficit_penalty (накопленный штраф)")
    lines.append("  EPen/IBonus — v2: econ_penalty / infra_bonus (динамические сигнальные переменные)")
    lines.append("  InMobPen    — v2: inertia_mobility_penalty (штраф к инерции от переездов соседей)")
    lines.append("  JlBonus     — v2: jobloss_econ_gap_bonus (ramp-бонус от LOST_JOB)")
    lines.append("  Capab.      — capabilities: (income_index + education_index + weak_ties) / 3")
    lines.append("  Inertia     — базовая инерция агента")
    lines.append("  ═══ БАРЬЕР 1: Потенциал vs Динамический порог ═══")
    lines.append("  DynThr1     — динамический порог: (internal_mig_thr + InMobPen) × max(0.15, 1 − signal_reduction)")
    lines.append("  Thr_mig     — internal_mig_threshold (базовый порог барьера 1)")
    lines.append("  SignRed     — signal_reduction (накопленный эффект сигналов, снижающий порог)")
    lines.append("  ═══ БАРЬЕР 2: D_perceived vs Динамическая инерция ═══")
    lines.append("  D_perc      — D_perceived = D_instant × Attribution × SocialCalibration")
    lines.append("  Attrib      — Attribution = PC × (1 − helplessness)")
    lines.append("  Help        — helplessness = clip(1 − PC − weak_ties × 0.3, 0, 1)")
    lines.append("  SocCal      — SocialCalibration = 1 + net_signal_susc × soc_calibration_signal")
    lines.append("  DynInert    — динамическая инерция S2 = inertia × max(0.15, 1 − social_boost)")
    lines.append("  MigrPress   — v4: накопленное давление к миграции (0–2)")
    lines.append("  TPB         — флаг активности / счётчик задержки намерения")
    lines.append("  IntState    — intention_state (none | seeking_work | seeking_residence)")

    # ── Шапка таблицы ────────────────────────────────────────────────────
    lines.append("")
    lines.append(
        f"  {'ID':>5} {'Тип':<11} {'Статус':<17} "
        f"{'Aspirations':>13} {'D_econ':>10} {'wage_pr':>8} {'D_place':>10} {'place_r':>8} {'PlacePen':>9} "
        f"{'EPen':>8} {'IBonus':>8} {'InMobPen':>9} {'JlBonus':>8} {'SocCalSig':>9} "
        f"{'Capab.':>10} {'Inertia':>13} {'DynThr1':>13} "
        f"{'D_perc':>10} {'Attrib':>8} {'Help':>8} {'SocCal':>8} {'DynInert':>13} {'MigrPress':>10} "
        f"{'TPB(акт/з)':>13} {'Thr_mig':>8} {'SignRed':>13} {'IntState':<18}"
    )
    lines.append("  " + "─" * 210)

    # ── Строки агентов ──────────────────────────────────────────────────
    for i in range(len(sampled_df_a)):
        ra = sampled_df_a.iloc[i]
        rb = sampled_df_b.iloc[i]
        agent_id = int(ra["id"])

        def _get(row, col, default=""):
            try:
                return row[col]
            except (KeyError, TypeError):
                return default

        # Базовые поля
        status_a   = str(_get(ra, "status", "?"))
        status_b   = str(_get(rb, "status", "?"))

        # ── Барьер 1: Aspirations и D-компоненты ─────────────────────────
        aspirations_a = float(_get(ra, "aspirations", 0))
        aspirations_b = float(_get(rb, "aspirations", 0))

        # D-компоненты: вычисляем для обоих тиков
        d_a = _compute_d_components(ra, G)
        d_b = _compute_d_components(rb, G)

        capabilities_a = _compute_capabilities(ra)
        capabilities_b = _compute_capabilities(rb)

        inertia_a = float(_get(ra, "inertia", 0))
        inertia_b = float(_get(rb, "inertia", 0))

        # v3: Барьер 1 — динамический порог
        dyn_thr1_a = _compute_dynamic_threshold_stage1(ra)
        dyn_thr1_b = _compute_dynamic_threshold_stage1(rb)

        # v3: Барьер 2 — D_perceived модель
        pc_a = float(_get(ra, "perceived_control", 0.5))
        pc_b = float(_get(rb, "perceived_control", 0.5))
        wt_a = float(_get(ra, "weak_ties_utility", 0.0))
        wt_b = float(_get(rb, "weak_ties_utility", 0.0))
        nss_a = float(_get(ra, "net_signal_susc", 0.5))
        nss_b = float(_get(rb, "net_signal_susc", 0.5))
        scs_a = float(_get(ra, "soc_calibration_signal", 0.0))
        scs_b = float(_get(rb, "soc_calibration_signal", 0.0))
        sb_a  = float(_get(ra, "social_boost", 0.0))
        sb_b  = float(_get(rb, "social_boost", 0.0))

        # helplessness = clip(1 − PC − weak_ties × 0.3, 0, 1)
        help_a = float(np.clip(1.0 - pc_a - wt_a * 0.3, 0.0, 1.0))
        help_b = float(np.clip(1.0 - pc_b - wt_b * 0.3, 0.0, 1.0))
        # Attribution = PC × (1 − helplessness)
        attr_a = pc_a * (1.0 - help_a)
        attr_b = pc_b * (1.0 - help_b)
        # SocialCalibration = 1 + net_signal_susc × soc_calibration_signal
        soccal_a = 1.0 + nss_a * scs_a
        soccal_b = 1.0 + nss_b * scs_b
        # D_perceived = D_instant × Attribution × SocialCalibration
        D_perc_a = d_a["D_instant"] * attr_a * soccal_a
        D_perc_b = d_b["D_instant"] * attr_b * soccal_b

        dyn_inertia_a = _compute_dynamic_inertia(ra)
        dyn_inertia_b = _compute_dynamic_inertia(rb)

        # ── Барьер 2: TPB ────────────────────────────────────────────────
        tpb_active_a = bool(_get(ra, "tpb_active", False))
        tpb_active_b = bool(_get(rb, "tpb_active", False))
        tpb_delay_a  = int(_get(ra, "intention_delay", 0))
        tpb_delay_b  = int(_get(rb, "intention_delay", 0))

        thr_mig_a = float(_get(ra, "internal_mig_thr", 0))
        thr_mig_b = float(_get(rb, "internal_mig_thr", 0))

        sign_red_a = float(_get(ra, "signal_reduction", 0))
        sign_red_b = float(_get(rb, "signal_reduction", 0))

        int_state_a = str(_get(ra, "intention_state", "none"))
        int_state_b = str(_get(rb, "intention_state", "none"))

        # Форматирование
        id_str      = f"{agent_id:>5}"
        status_str  = f"{status_a}→{status_b}"
        status_str  = f"{status_str:<17}"

        aspir_str   = _fmt_arrow(aspirations_a, aspirations_b, ".3f", 13)
        d_econ_str  = _fmt_arrow(d_a["D_econ"], d_b["D_econ"], ".3f", 10)
        wp_str      = _fmt_arrow(d_a["wage_pressure"], d_b["wage_pressure"], ".3f", 8)
        d_place_str = _fmt_arrow(d_a["D_place"], d_b["D_place"], ".3f", 10)
        pr_str      = _fmt_arrow(d_a["place_reality"], d_b["place_reality"], ".3f", 8)
        place_pen_a = float(_get(ra, "place_deficit_penalty", 0.0))
        place_pen_b = float(_get(rb, "place_deficit_penalty", 0.0))
        pp_str      = _fmt_arrow(place_pen_a, place_pen_b, ".2f", 9)

        # v2: динамические переменные
        ep_a  = float(_get(ra, "econ_penalty", 0.0))
        ep_b  = float(_get(rb, "econ_penalty", 0.0))
        ib_a  = float(_get(ra, "infra_bonus", 0.0))
        ib_b  = float(_get(rb, "infra_bonus", 0.0))
        imp_a = float(_get(ra, "inertia_mobility_penalty", 0.0))
        imp_b = float(_get(rb, "inertia_mobility_penalty", 0.0))
        jlb_a = float(_get(ra, "jobloss_econ_gap_bonus", 0.0))
        jlb_b = float(_get(rb, "jobloss_econ_gap_bonus", 0.0))
        ep_str  = _fmt_arrow(ep_a, ep_b, ".3f", 8)
        ib_str  = _fmt_arrow(ib_a, ib_b, ".3f", 8)
        imp_str = _fmt_arrow(imp_a, imp_b, ".3f", 9)
        jlb_str = _fmt_arrow(jlb_a, jlb_b, ".3f", 8)
        scs_str = _fmt_arrow(scs_a, scs_b, ".3f", 9)

        capab_str   = _fmt_arrow(capabilities_a, capabilities_b, ".3f", 10)
        inertia_str = _fmt_arrow(inertia_a, inertia_b, ".3f", 13)
        dyn_thr1_str = _fmt_arrow(dyn_thr1_a, dyn_thr1_b, ".3f", 13)

        D_perc_str  = _fmt_arrow(D_perc_a, D_perc_b, ".3f", 10)
        attr_str    = _fmt_arrow(attr_a, attr_b, ".3f", 8)
        help_str    = _fmt_arrow(help_a, help_b, ".3f", 8)
        soccal_str  = _fmt_arrow(soccal_a, soccal_b, ".3f", 8)
        dyn_str     = _fmt_arrow(dyn_inertia_a, dyn_inertia_b, ".3f", 13)

        # v4: migration_pressure
        migr_a = float(_get(ra, "migration_pressure", 0.0))
        migr_b = float(_get(rb, "migration_pressure", 0.0))
        migr_str = _fmt_arrow(migr_a, migr_b, ".3f", 10)

        tpb_str = f"{_fmt_bool_arrow(tpb_active_a, tpb_active_b, 6)} {tpb_delay_a}→{tpb_delay_b}"
        tpb_str = f"{tpb_str:>13}"

        thr_str     = _fmt_arrow(thr_mig_a, thr_mig_b, ".3f", 8)
        sign_str    = _fmt_arrow(sign_red_a, sign_red_b, ".3f", 13)
        int_state_s = f"{int_state_a}→{int_state_b}"
        int_state_s = f"{int_state_s:<18}"

        lines.append(
            f"  {id_str} {'':<11} {status_str} "
            f"{aspir_str} {d_econ_str} {wp_str} {d_place_str} {pr_str} {pp_str} "
            f"{ep_str} {ib_str} {imp_str} {jlb_str} {scs_str} "
            f"{capab_str} {inertia_str} {dyn_thr1_str} "
            f"{D_perc_str} {attr_str} {help_str} {soccal_str} {dyn_str} {migr_str} "
            f"{tpb_str} {thr_str} {sign_str} {int_state_s}"
        )

    # ── Сводная статистика по выборке ────────────────────────────────────
    lines.append("  " + "─" * 210)

    def _col_mean(df_sub, col):
        if col not in df_sub.columns:
            return 0.0
        return float(df_sub[col].mean())

    lines.append("  СВОДКА ПО ВЫБОРКЕ (средние):")
    lines.append(
        f"  {'':>5} {'':11} {'':17} "
        f"{'Aspirations':>13} {'D_econ':>10} {'wage_pr':>8} {'D_place':>10} {'place_r':>8} "
        f"{'EPen':>8} {'IBonus':>8} {'InMobPen':>9} {'JlBonus':>8} "
        f"{'Capab.':>10} {'Inertia':>13} {'DynInert':>13} {'MigrPress':>10} "
        f"{'TPB(акт/з)':>13} {'Thr_mig':>8} {'SignRed':>13} {'IntState':<18}"
    )

    # Барьер 1 — средние
    m_asp_a = _col_mean(sampled_df_a, "aspirations")
    m_asp_b = _col_mean(sampled_df_b, "aspirations")
    m_in_a  = _col_mean(sampled_df_a, "inertia")
    m_in_b  = _col_mean(sampled_df_b, "inertia")
    m_sr_a  = _col_mean(sampled_df_a, "signal_reduction")
    m_sr_b  = _col_mean(sampled_df_b, "signal_reduction")
    m_th_a  = _col_mean(sampled_df_a, "internal_mig_thr")
    m_th_b  = _col_mean(sampled_df_b, "internal_mig_thr")
    m_tpb_a = _col_mean(sampled_df_a, "tpb_active")
    m_tpb_b = _col_mean(sampled_df_b, "tpb_active")
    m_del_a = _col_mean(sampled_df_a, "intention_delay")
    m_del_b = _col_mean(sampled_df_b, "intention_delay")

    # D-компоненты: средние по выборке
    d_vals_a = [_compute_d_components(sampled_df_a.iloc[i], G) for i in range(len(sampled_df_a))]
    d_vals_b = [_compute_d_components(sampled_df_b.iloc[i], G) for i in range(len(sampled_df_b))]
    m_de_a  = np.mean([d["D_econ"] for d in d_vals_a])
    m_de_b  = np.mean([d["D_econ"] for d in d_vals_b])
    m_wp_a  = np.mean([d["wage_pressure"] for d in d_vals_a])
    m_wp_b  = np.mean([d["wage_pressure"] for d in d_vals_b])
    m_dp_a  = np.mean([d["D_place"] for d in d_vals_a])
    m_dp_b  = np.mean([d["D_place"] for d in d_vals_b])
    m_pr_a  = np.mean([d["place_reality"] for d in d_vals_a])
    m_pr_b  = np.mean([d["place_reality"] for d in d_vals_b])

    caps_a_vals = [_compute_capabilities(sampled_df_a.iloc[i]) for i in range(len(sampled_df_a))]
    caps_b_vals = [_compute_capabilities(sampled_df_b.iloc[i]) for i in range(len(sampled_df_b))]
    m_cap_a = np.mean(caps_a_vals) if caps_a_vals else 0.0
    m_cap_b = np.mean(caps_b_vals) if caps_b_vals else 0.0

    dyn_a_vals = [_compute_dynamic_inertia(sampled_df_a.iloc[i]) for i in range(len(sampled_df_a))]
    dyn_b_vals = [_compute_dynamic_inertia(sampled_df_b.iloc[i]) for i in range(len(sampled_df_b))]
    m_dyn_a = np.mean(dyn_a_vals) if dyn_a_vals else 0.0
    m_dyn_b = np.mean(dyn_b_vals) if dyn_b_vals else 0.0

    # v2: средние динамических переменных
    m_ep_a  = _col_mean(sampled_df_a, "econ_penalty")
    m_ep_b  = _col_mean(sampled_df_b, "econ_penalty")
    m_ib_a  = _col_mean(sampled_df_a, "infra_bonus")
    m_ib_b  = _col_mean(sampled_df_b, "infra_bonus")
    m_imp_a = _col_mean(sampled_df_a, "inertia_mobility_penalty")
    m_imp_b = _col_mean(sampled_df_b, "inertia_mobility_penalty")
    m_jlb_a = _col_mean(sampled_df_a, "jobloss_econ_gap_bonus")
    m_jlb_b = _col_mean(sampled_df_b, "jobloss_econ_gap_bonus")

    # v4: migration_pressure
    m_migr_a = _col_mean(sampled_df_a, "migration_pressure")
    m_migr_b = _col_mean(sampled_df_b, "migration_pressure")

    # Форматирование сводной строки
    m_asp_str = _fmt_arrow(m_asp_a, m_asp_b, ".3f", 13)
    m_de_str  = _fmt_arrow(m_de_a, m_de_b, ".3f", 10)
    m_wp_str  = _fmt_arrow(m_wp_a, m_wp_b, ".3f", 8)
    m_dp_str  = _fmt_arrow(m_dp_a, m_dp_b, ".3f", 10)
    m_pr_str  = _fmt_arrow(m_pr_a, m_pr_b, ".3f", 8)
    m_ep_s    = _fmt_arrow(m_ep_a, m_ep_b, ".3f", 8)
    m_ib_s    = _fmt_arrow(m_ib_a, m_ib_b, ".3f", 8)
    m_imp_s   = _fmt_arrow(m_imp_a, m_imp_b, ".3f", 9)
    m_jlb_s   = _fmt_arrow(m_jlb_a, m_jlb_b, ".3f", 8)
    m_cap_str = _fmt_arrow(m_cap_a, m_cap_b, ".3f", 10)
    m_in_str  = _fmt_arrow(m_in_a, m_in_b, ".3f", 13)
    m_dyn_s   = _fmt_arrow(m_dyn_a, m_dyn_b, ".3f", 13)
    m_migr_s  = _fmt_arrow(m_migr_a, m_migr_b, ".3f", 10)

    m_tpb_s = f"{_fmt_bool_arrow(m_tpb_a > 0.5, m_tpb_b > 0.5, 6)} {m_del_a:.1f}→{m_del_b:.1f}"
    m_tpb_s = f"{m_tpb_s:>13}"

    m_th_str = _fmt_arrow(m_th_a, m_th_b, ".3f", 8)
    m_sr_str = _fmt_arrow(m_sr_a, m_sr_b, ".3f", 13)

    # Статусы
    st_a_counts = sampled_df_a["status"].value_counts().to_dict() if "status" in sampled_df_a.columns else {}
    st_b_counts = sampled_df_b["status"].value_counts().to_dict() if "status" in sampled_df_b.columns else {}
    st_a_top = max(st_a_counts, key=st_a_counts.get) if st_a_counts else "?"
    st_b_top = max(st_b_counts, key=st_b_counts.get) if st_b_counts else "?"
    m_st_str = f"{'СРЕДН':>5} {'—':11} {st_a_top+'→'+st_b_top:<17}"

    # Intention states
    is_a_counts = sampled_df_a["intention_state"].value_counts().to_dict() if "intention_state" in sampled_df_a.columns else {}
    is_b_counts = sampled_df_b["intention_state"].value_counts().to_dict() if "intention_state" in sampled_df_b.columns else {}
    is_a_top = max(is_a_counts, key=is_a_counts.get) if is_a_counts else "none"
    is_b_top = max(is_b_counts, key=is_b_counts.get) if is_b_counts else "none"
    m_is_str = f"{is_a_top}→{is_b_top}"

    lines.append(
        f"  {m_st_str} "
        f"{m_asp_str} {m_de_str} {m_wp_str} {m_dp_str} {m_pr_str} "
        f"{m_ep_s} {m_ib_s} {m_imp_s} {m_jlb_s} "
        f"{m_cap_str} {m_in_str} {m_dyn_s} {m_migr_s} "
        f"{m_tpb_s} {m_th_str} {m_sr_str} {m_is_str}"
    )

    # ── Анализ изменений ─────────────────────────────────────────────────
    lines.append("")
    lines.append("  АНАЛИЗ ДИНАМИКИ:")

    n_asp_up = int(
        (sampled_df_b["aspirations"].values > sampled_df_a["aspirations"].values).sum()
    ) if "aspirations" in sampled_df_a.columns else 0

    tpb_a_arr = sampled_df_a["tpb_active"].values.astype(bool) if "tpb_active" in sampled_df_a.columns else np.zeros(n_sample, dtype=bool)
    tpb_b_arr = sampled_df_b["tpb_active"].values.astype(bool) if "tpb_active" in sampled_df_b.columns else np.zeros(n_sample, dtype=bool)
    n_tpb_new = int((~tpb_a_arr & tpb_b_arr).sum())

    is_a_arr = sampled_df_a["intention_state"].values if "intention_state" in sampled_df_a.columns else np.full(n_sample, "none")
    is_b_arr = sampled_df_b["intention_state"].values if "intention_state" in sampled_df_b.columns else np.full(n_sample, "none")
    n_state_changed = int((is_a_arr != is_b_arr).sum())

    st_a_arr = sampled_df_a["status"].values if "status" in sampled_df_a.columns else np.full(n_sample, "?")
    st_b_arr = sampled_df_b["status"].values if "status" in sampled_df_b.columns else np.full(n_sample, "?")
    n_status_changed = int((st_a_arr != st_b_arr).sum())

    # Динамика D-компонент
    d_inst_a = np.array([d["D_instant"] for d in d_vals_a])
    d_inst_b = np.array([d["D_instant"] for d in d_vals_b])
    n_d_up = int((d_inst_b > d_inst_a).sum())

    lines.append(f"  Агентов с ростом aspirations:              {n_asp_up}/{n_sample}")
    lines.append(f"  Агентов с ростом D_instant:                {n_d_up}/{n_sample}")
    lines.append(f"  Новых TPB-активаций:                        {n_tpb_new}/{n_sample}")
    lines.append(f"  Изменивших intention_state:                 {n_state_changed}/{n_sample}")
    lines.append(f"  Изменивших статус занятости:                {n_status_changed}/{n_sample}")

    # Средние D-компонент
    lines.append(f"  Среднее D_econ (тик {tick_b}):                     {m_de_b:.4f}")
    lines.append(f"  Среднее wage_pressure (тик {tick_b}):              {m_wp_b:.4f}")
    lines.append(f"  Среднее D_place (тик {tick_b}):                    {m_dp_b:.4f}")
    lines.append(f"  Среднее place_reality (тик {tick_b}):              {m_pr_b:.4f}")

    # v2: динамические переменные
    lines.append(f"  Среднее econ_penalty (тик {tick_b}):              {m_ep_b:.4f}")
    lines.append(f"  Среднее infra_bonus (тик {tick_b}):               {m_ib_b:.4f}")
    lines.append(f"  Среднее inertia_mobility_penalty (тик {tick_b}):  {m_imp_b:.4f}")
    lines.append(f"  Среднее jobloss_econ_gap_bonus (тик {tick_b}):    {m_jlb_b:.4f}")

    lines.append("═" * 160)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MASTER DISTRICT TABLE — мастер-таблица по 79 районам (окресам)
# ══════════════════════════════════════════════════════════════════════════════

def master_district_table(
    tick_stats: list,
    G,
    all_action_log: Optional[List[dict]] = None,
    snapshots: Optional[dict] = None,
) -> str:
    """
    Мастер-таблица по 79 районам Словакии (okresy, не края).

    Строки: 79 районов (из DISTRICT_TO_REGION_CODE в agents.py).
    Столбцы: для каждого тика — количество агентов в районе и стоимость жилья.
    Финальные столбцы:
      - Δ econ  — въехавшие в район по экономическому мотиву за всю симуляцию
      - Δ place — въехавшие в район по мотиву места за всю симуляцию

    Использует:
      - tick_stats[i]["district_counts"] для поколичества агентов по тикам
      - G.nodes[district]["housing_price_m2"] для стоимости жилья
      - all_action_log для подсчёта мотивированных переездов
      - snapshots[0] для данных тика 0 (если есть)
    """
    # Все районы из графа (79 okresov)
    all_districts = sorted(G.nodes)

    # Подготавливаем списки тиков
    n_ticks = len(tick_stats)
    tick_nums = list(range(1, n_ticks + 1))

    # ── Агенты на тике 0 (из snapshots) ──────────────────────────────────
    t0_counts = {}
    if snapshots and 0 in snapshots:
        t0_counts = snapshots[0].groupby("district")["id"].count().to_dict()

    # ── Агенты на каждом тике (из tick_stats) ────────────────────────────
    tick_counts = {}  # {district: [count_t1, count_t2, ...]}
    for district in all_districts:
        tick_counts[district] = []
    for s in tick_stats:
        dc = s.get("district_counts", {})
        for district in all_districts:
            tick_counts[district].append(dc.get(district, 0))

    # ── Стоимость жилья: эффективная цена из графа ──
    # G.nodes[district]["effective_housing_price_m2"] уже предвычислена
    # в update_graph() каждый тик с учётом housing_remaining и sensitivity.

    # Эффективная цена на каждом тике (читаем из графа, который обновляется каждый тик)
    # Но в отчёте у нас нет графа «на каждый тик» — используем housing_remaining из tick_stats
    # и формулу как в graph.py для воспроизведения.
    _AGENT_FOOTPRINT = 1.1  # AGENT_HOUSING_FOOTPRINT из graph.py
    _REMAINING_FLOOR = 1.5  # HOUSING_REMAINING_FLOOR из graph.py

    # Базовая цена и чувствительность из графа
    base_prices = {}
    sensitivities = {}
    for district in all_districts:
        base_prices[district] = G.nodes[district].get("housing_price_m2", 1800.0)
        sensitivities[district] = G.nodes[district].get("housing_market_sensitivity", 1.0)

    # Эффективная цена на каждом тике (используем housing_remaining из tick_stats)
    housing_by_tick = {}  # {district: [price_t0, price_t1, ...]}
    for district in all_districts:
        housing_by_tick[district] = []

    # Тик 0: нет данных housing_remaining → используем базовую цену
    if snapshots and 0 in snapshots:
        for district in all_districts:
            housing_by_tick[district].append(base_prices[district])

    # Тики 1..N: из tick_stats district_housing_remaining
    for s in tick_stats:
        hr = s.get("district_housing_remaining", {})
        for district in all_districts:
            remaining = hr.get(district, _REMAINING_FLOOR)
            bp = base_prices[district]
            sens = sensitivities[district]
            if remaining > 0.01:
                delta = bp * (_AGENT_FOOTPRINT / max(remaining, _REMAINING_FLOOR)) * sens
                effective = bp + delta
            else:
                effective = bp * 100.0  # жильё закончилось
            housing_by_tick[district].append(effective)

    # ── Подсчёт переездов по мотивам из all_action_log ──────────────────
    econ_inflow = {d: 0 for d in all_districts}
    place_inflow = {d: 0 for d in all_districts}
    if all_action_log:
        for entry in all_action_log:
            decision = entry.get("decision", "")
            if decision in ("move", "satellite_move"):
                new_res = entry.get("new_residence", "")
                domain = entry.get("activation_domain", "")
                if new_res in econ_inflow:
                    if domain == "economic":
                        econ_inflow[new_res] += 1
                    elif domain == "place":
                        place_inflow[new_res] += 1

    # ── Формирование таблицы ─────────────────────────────────────────────
    lines = []
    lines.append(_section("МАСТЕР-ТАБЛИЦА ПО 79 РАЙОНАМ (OKRESY)"))

    # Пояснение
    has_t0 = (snapshots and 0 in snapshots)
    lines.append(f"  Всего районов: {len(all_districts)} | Тиков: {n_ticks}")
    lines.append(f"  Таблица 1: количество агентов в районе на каждом тике")
    lines.append(f"  Таблица 2: эффективная стоимость жилья (€/м²) с учётом остатка квартир")
    lines.append(f"  Δ econ / Δ place: суммарный въезд в район по мотиву за всю симуляцию")
    lines.append("")

    # ── ПОДТАБЛИЦА 1: Количество агентов ─────────────────────────────────
    lines.append(_section("АГЕНТОВ В РАЙОНЕ ПО ТИКАМ"))
    header1 = f"  {'Район':<30}"
    if has_t0:
        header1 += f" {'T0':>6}"
    for tn in tick_nums:
        header1 += f" {'T' + str(tn):>6}"
    header1 += f" {'Δ econ':>8} {'Δ place':>8}"
    lines.append(header1)

    col_count = (1 if has_t0 else 0) + n_ticks
    line_w1 = 32 + col_count * 7 + 9 + 9
    lines.append("  " + _hline(line_w1, "─"))

    for district in all_districts:
        name = district.replace("District of ", "")[:28]
        row = f"  {name:<30}"
        if has_t0:
            row += f" {t0_counts.get(district, 0):>6,}"
        for count in tick_counts[district]:
            row += f" {count:>6,}"
        row += f" {econ_inflow[district]:>8,} {place_inflow[district]:>8,}"
        lines.append(row)

    # ИТОГО для агентов
    lines.append("  " + _hline(line_w1, "─"))
    total_row1 = f"  {'ИТОГО':<30}"
    if has_t0:
        total_row1 += f" {sum(t0_counts.values()):>6,}"
    for i in range(n_ticks):
        t_total = sum(tick_counts[d][i] for d in all_districts)
        total_row1 += f" {t_total:>6,}"
    total_econ = sum(econ_inflow.values())
    total_place = sum(place_inflow.values())
    total_row1 += f" {total_econ:>8,} {total_place:>8,}"
    lines.append(total_row1)

    # ── ПОДТАБЛИЦА 2: Эффективная стоимость жилья ────────────────────────
    lines.append("")
    lines.append(_section("ЭФФЕКТИВНАЯ СТОИМОСТЬ ЖИЛЬЯ ПО ТИКАМ (€/м²)"))
    lines.append(f"  Формула: базовая_цена × (1 + 1.1 / остаток_квартир × чувствительность)")
    lines.append(f"  Чем меньше остаток квартир → тем выше эффективная цена (конкуренция).")
    lines.append("")

    header2 = f"  {'Район':<30}"
    if has_t0:
        header2 += f" {'T0':>9}"
    for tn in tick_nums:
        header2 += f" {'T' + str(tn):>9}"
    lines.append(header2)

    line_w2 = 32 + col_count * 10
    lines.append("  " + _hline(line_w2, "─"))

    for district in all_districts:
        name = district.replace("District of ", "")[:28]
        row = f"  {name:<30}"
        for hp in housing_by_tick[district]:
            row += f" {hp:>8,.0f}€"
        lines.append(row)

    # ИТОГО для жилья (среднее)
    lines.append("  " + _hline(line_w2, "─"))
    total_row2 = f"  {'СРЕДНЕЕ':<30}"
    for tick_idx in range(len(tick_nums) + (1 if has_t0 else 0)):
        avg = sum(housing_by_tick[d][tick_idx] for d in all_districts) / max(len(all_districts), 1)
        total_row2 += f" {avg:>8,.0f}€"
    lines.append(total_row2)

    # ── Статистика по переездам ──────────────────────────────────────────
    lines.append("")
    lines.append(f"  Всего въездов по экономическому мотиву: {total_econ:,}")
    lines.append(f"  Всего въездов по мотиву места:         {total_place:,}")

    # Топ-10 районов по притоку
    if total_econ > 0:
        top_econ = sorted(econ_inflow.items(), key=lambda x: -x[1])[:5]
        lines.append(f"  Топ-5 районов по econ-притоку: "
                     + ", ".join(f"{d.replace('District of ', '')}({c})" for d, c in top_econ if c > 0))
    if total_place > 0:
        top_place = sorted(place_inflow.items(), key=lambda x: -x[1])[:5]
        lines.append(f"  Топ-5 районов по place-притоку: "
                     + ", ".join(f"{d.replace('District of ', '')}({c})" for d, c in top_place if c > 0))

    lines.append("=" * 78)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY REPORT — обёртка для полного отчёта
# ══════════════════════════════════════════════════════════════════════════════

def summary_report(
    df_final: pd.DataFrame,
    tick_stats: list,
    all_action_log: Optional[List[dict]] = None,
    snapshots: Optional[dict] = None,
    detail: bool = False,
    G=None,
) -> str:
    """
    Итоговый отчёт: демографический портрет финального состояния +
    сводка динамики + опционально межрегиональный баланс и аудит.

    Режимы:
      detail=False (default) — критические и важные метрики, ~2-3 экрана.
      detail=True — полный отчёт с аудитом и дополнительными метриками.
    """
    parts = []

    # 0. МАТРИЦА ПАРАМЕТРОВ АГЕНТОВ (если есть снимки с тиками 0 и 6)
    if snapshots and 0 in snapshots and 6 in snapshots:
        parts.append(agent_parameters_table(snapshots, G=G, n_show=20, tick_a=0, tick_b=6))

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

    # 4. МАСТЕР-ТАБЛИЦА ПО 79 РАЙОНАМ (всегда в конце)
    if G is not None and tick_stats:
        parts.append(master_district_table(tick_stats, G, all_action_log, snapshots=snapshots))

    return "\n\n".join(parts)

