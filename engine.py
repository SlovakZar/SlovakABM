"""
engine.py v10 — Двухбарьерная модель v4 + Унифицированный эвристический поиск

Архитектура принятия решений:

  БАРЬЕР 1 — Потенциал миграции против динамического порога:
    aspirations (EWMA от D_instant) × capabilities > dynamic_threshold → tpb_active.
    dynamic_threshold = (internal_mig_thr + inertia_mob_pen) × max(0.15, 1 − signal_reduction).

  БАРЬЕР 2 — v4: Накопительное давление + вероятностный триггер:
    Каждый тик пока tpb_active: D_perceived = D_instant × Attribution × SocialCalibration.
    gap = max(0, D_perceived − dynamic_inertia_s2); migration_pressure += gap.
    P(act) = clip(pressure / max(inertia, DIVISOR), P_MIN, P_MAX).
    Монетка → seeking_work / seeking_residence. Давление сбрасывается при действии.

  ЭВРИСТИЧЕСКИЙ ПОИСК (унифицированный):
    После активации агент формирует список кандидатов (работа/жильё)
    с учётом info_quality и отраслевого давления.
    Цикл по кандидатам: commute → move → satellite.
    При успехе → действие + PC↑. При провале → stay + адаптация + PC↓.

  ОБНОВЛЕНИЕ ДОМЕНОВ:
    Social (Блок A): target = 0.5 + social_boost, сглаживание α=0.88.
    Family (Блок F): commute-давление.
    Economic: от workplace_district, Place: от residence_district.

  СОБЫТИЙНЫЕ СИГНАЛЫ (Блок B/C):
    social_boost затухает ×0.8/тик.
    signal_reduction: decay ×0.85 + новые сигналы (unemployed, соседи).
    soc_calibration_signal: decay ×0.85 + сигналы от AGENT_MOVED/COMMUTE/JOB.
    econ_penalty: прямая прибавка к D_econ, decay −0.01/тик.
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional

from graph import (update_graph, get_awareness_set, update_industry_pressure,
                    update_industry_pressure_delta, init_housing_remaining,
                    get_effective_housing_price, AGENT_HOUSING_FOOTPRINT)
from signals import EventBus, Event, EventType, Dispatcher, set_settlement_map

# ── Константы ─────────────────────────────────────────────────────────────────

# Фильтр 2 — дерево занятости
MAX_JOBS_PRESSURE        = 1.20   # район считается перегруженным выше этого порога
MAX_WORK_CANDIDATES      = 12     # максимум районов для скрининга

HOUSING_BUDGET_RATIO     = 0.35   # жильё не должно превышать X доли зарплаты (×100м²)
MOVE_STRESS_FACTOR       = 0.80   # satisfaction после переезда × этот множитель

# ── Динамический трекер жилья ─────────────────────────────────────────────────
# Перенесён в graph.py:
#   housing_remaining — в G.nodes[district]["housing_remaining"]
#   effective_housing_price_m2 — в G.nodes[district]["effective_housing_price_m2"]
#   AGENT_HOUSING_FOOTPRINT — импортируется из graph.py

# Adapt
ADAPT_FLEX_THRESHOLD     = 0.65   # минимальная job_flexibility для адаптации
ADAPT_SAT_BOOST          = 0.06   # прирост sat_economic при адаптации

# Heuristic gates (commute/satellite)
COMMUTER_GATE_REF        = 0.50   # порог сравнения commuter_threshold (глобальное среднее)
JOB_FLEX_GATE_REF        = 0.50   # порог сравнения job_flexibility (глобальное среднее)

# Фильтр 2 — поведенческая эвристика зарплатных ожиданий
BASE_APPETITE_MIN        = 0.10   # базовый аппетит к росту зарплаты (при PC=0)
BASE_APPETITE_MAX        = 0.20   # добавка за econ_perceived_control (при PC=1 аппетит=0.30)
MIN_DESIRED_RAISE        = 0.05   # минимальная желаемая надбавка (при высокой desperation)
UNEMPLOYED_WAGE_FLOOR    = 0.70   # мин. доля от нац. средней для безработных (при PC=0)
UNEMPLOYED_WAGE_CEIL     = 0.20   # добавка за econ_perceived_control (при PC=1 доля=0.90)

# Обновление доменов
SAT_SMOOTHING            = 0.88
NATIONAL_AVG_WAGE        = 1614.0

# ── Двухбарьерная модель: константы ──────────────────────────────────────────
ASPIRATIONS_ALPHA        = 0.08   # скорость EWMA-накопления aspirations из D_instant
SIGNAL_DECAY             = 0.85   # затухание signal_reduction за тик

# v4: Накопительное давление и вероятностный триггер (замена жёсткой задержки)
MIGRATION_PRESSURE_P_MIN    = 0.03   # минимальный шанс сорваться в тик
MIGRATION_PRESSURE_P_MAX    = 0.80   # максимальный шанс (не 1.0 — элемент случайности)
MIGRATION_PRESSURE_DIVISOR  = 0.12   # делитель для перевода давления в вероятность
# P(act) = clip(pressure / (inertia + DIVISOR), P_MIN, P_MAX)
GAP_ADAPT_LAMBDA         = 0.05   # скорость адаптации econ_gap и domain_future_place
HUB_WEAK_TIES_BONUS      = 0.005  # прирост weak_ties_utility за тик в хабах
MOVE_WEAK_TIES_PENALTY   = -0.10  # сброс weak_ties при переезде

# Хабы: районные центры с повышенной социальной динамикой
HUB_DISTRICTS = {
    "District of Bratislava I", "District of Bratislava II",
    "District of Bratislava III", "District of Bratislava IV",
    "District of Bratislava V",
    "District of Košice I", "District of Košice II",
    "District of Košice III", "District of Košice IV",
    "District of Trnava", "District of Nitra", "District of Žilina",
    "District of Banská Bystrica", "District of Prešov",
    "District of Trenčín",
}


# ── Вспомогательные ───────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-4.0 * x))





def _housing_affordable(
    agent_wage: float,
    housing_price_m2: float,
    district: str = "",
    G: "nx.DiGraph | None" = None,
    budget_ratio: float = HOUSING_BUDGET_RATIO,
) -> bool:
    """
    Жильё доступно, если ежемесячный платёж (аренда ~0.4% эффективной цены 50м²)
    не превышает budget_ratio от зарплаты.

    Использует эффективную цену из графа (G.nodes[district]["effective_housing_price_m2"]).
    Если G или district не указан — используется номинальная цена.
    """
    if agent_wage <= 0:
        return False

    if district and G is not None and district in G.nodes:
        effective_price = get_effective_housing_price(G, district)
    else:
        effective_price = housing_price_m2

    monthly_cost = effective_price * 50 * 0.004
    return monthly_cost <= agent_wage * budget_ratio


def _compute_avg_dissatisfaction(df: pd.DataFrame) -> float:
    """
    Облегчённая версия: средняя dissatisfaction для статистики.
    Не обновляет sat_* домены — считает по текущим значениям (меняются
    только при _execute_move / _execute_adapt).
    """
    domains = [
        ("sat_economic", "w_economic", "thr_economic"),
        ("sat_social",   "w_social",   "thr_social"),
        ("sat_family",   "w_family",   "thr_family"),
        ("sat_place",    "w_future",   "thr_place"),
    ]
    dissat = np.zeros(len(df))
    for val_col, w_col, thr_col in domains:
        val = df[val_col].values
        w   = df[w_col].values
        thr = df[thr_col].values
        gap = np.maximum(0, thr - val) / np.maximum(thr, 0.01)
        dissat += (w * gap) ** 2
    return float(np.clip(np.sqrt(dissat), 0.0, 1.0).mean())


def _compute_jobs_pressure(df: pd.DataFrame, jobs_capacity: dict,
                           G: nx.DiGraph = None) -> dict:
    """
    v3: jobs_pressure[district] = число занятых агентов с workplace=district
                                  / (occupied + vacant по всем отраслям).

    Использует G.nodes[district]["industry_jobs"] для получения occupied+vacant,
    если доступно. Иначе fallback на jobs_capacity.

    Значение > 1.0 означает перегрузку рынка труда.
    """
    wp_counts = (df[df["is_employed"]]
                 .groupby("workplace_district")["id"]
                 .count()
                 .to_dict())
    pressure = {}

    if G is not None:
        for district in G.nodes:
            cnt = wp_counts.get(district, 0)
            ind_jobs = G.nodes[district].get("industry_jobs", {})
            if ind_jobs:
                total_cap = sum(
                    v["occupied"] + v["vacant"]
                    for v in ind_jobs.values()
                )
            else:
                total_cap = jobs_capacity.get(district, G.nodes[district].get("jobs_capacity", 1))
            pressure[district] = cnt / max(total_cap, 1)
    else:
        for d, cap in jobs_capacity.items():
            cnt = wp_counts.get(d, 0)
            pressure[d] = cnt / max(cap, 1)

    return pressure


# ── Двухбарьерная модель: Барьер 1 — Потенциал против динамической инерции ──

def _compute_d_instant_vectorized(
    agent_wages: np.ndarray,
    industry_avg_wages_wp: np.ndarray,
    econ_gaps: np.ndarray,
    job_flexs: np.ndarray,
    housing_prices: np.ndarray,
    infra_scores: np.ndarray,
    domain_futures: np.ndarray,
    w_econs: np.ndarray,
    w_futures: np.ndarray,
    place_deficit_penalties: np.ndarray,
    econ_penalties: np.ndarray,
    infra_bonuses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Векторизованная версия _compute_d_instant — все параметры массивы длины n.

    v3: econ_penalty — прямая прибавка к D_econ; infra_bonus — к инфраструктуре.

    Returns (D_instant, D_econ, D_place) — массивы np.ndarray длины n.
    """
    # Экономическая компонента
    safe_w = np.maximum(agent_wages, 0.0)
    safe_ind = np.maximum(industry_avg_wages_wp, 0.0)
    wage_pressure = np.where(
        (safe_w > 0) & (safe_ind > 0),
        safe_ind / safe_w,
        1.0  # безработный — максимальное давление
    )
    D_econ = w_econs * wage_pressure * (econ_gaps / np.maximum(job_flexs, 0.01)) + econ_penalties

    # Жилищная компонента
    monthly_cost = housing_prices * 50.0 * 0.004
    burden = monthly_cost / np.maximum(safe_w, 1.0)
    affordability = np.maximum(0.0, 1.0 - burden / 0.35)
    infra_component = 0.3 * (1.0 - infra_scores + infra_bonuses)
    place_reality = 0.7 * affordability + infra_component

    gap = np.maximum(0.0, domain_futures - place_reality)
    place_ratio = domain_futures / np.maximum(place_reality, 0.001)
    amplifier = np.maximum(1.0, place_ratio)
    D_place = w_futures * gap * amplifier * (1.0 + place_deficit_penalties)

    D_instant = np.clip(D_econ + D_place, 0.0, 1.0)
    return D_instant, np.clip(D_econ, 0.0, 1.0), np.clip(D_place, 0.0, 1.0)


def _two_barrier_activation(
    df: pd.DataFrame,
    G: nx.DiGraph,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Двухбарьерная модель активации (Aspirations×Capabilities → D_perceived → Эвристический поиск).

    Барьер 1 — Потенциал миграции против динамического порога (internal_mig_threshold):
      Обновляет aspirations (EWMA от D_instant).
      Вычисляет capabilities (income + education + weak_ties).
      Вычисляет динамический порог: (internal_mig_thr + inertia_mob_penalty) × max(0.15, 1 − signal_reduction).
      Если aspirations × capabilities > dynamic_threshold → tpb_active = True.

    Барьер 2 — v4: Накопительное давление + вероятностный триггер:
      Каждый тик пока tpb_active: D_perceived = D_instant × Attribution × SocialCalibration.
      gap = max(0, D_perceived − dynamic_inertia_s2); migration_pressure += gap.
      P(act) = clip(pressure / max(inertia, DIVISOR), P_MIN, P_MAX).
      Если random < P(act) → intention_state = seeking_work/seeking_residence.
      Давление сбрасывается при исполнении действия (_execute_*).
      Если барьер 1 не пройден → tpb_active=False, давление медленно затухает (−0.02/тик).

    Возвращает обновлённый DataFrame.
    """
    df = df.copy()
    n = len(df)

    # ── Массивы для векторизации ──────────────────────────────────────────
    ages           = df["age"].values
    educations     = df["education"].values
    wages          = df["wage"].values
    statuses       = df["status"].values
    workplaces     = df["workplace_district"].values
    residences     = df["district"].values
    industries     = df["industry"].values

    aspirations    = df["aspirations"].values.copy()
    signal_red     = df["signal_reduction"].values.copy()
    tpb_active     = df["tpb_active"].values.copy()
    econ_gaps      = df["econ_gap"].values.copy()
    domain_future  = df["domain_future_place"].values.copy()

    w_econs        = df["w_economic"].values
    w_futures      = df["w_future"].values
    job_flexs      = df["job_flexibility"].values
    inertias       = df["inertia"].values
    percontrols    = df["perceived_control"].values
    econ_percontrols = df["econ_perceived_control"].values
    weak_ties      = df["weak_ties_utility"].values
    net_susc       = df["net_signal_susc"].values
    family_mods    = df["family_weight_mod"].values
    social_boosts  = df["social_boost"].values
    maritals       = df["marital"].values
    internal_thrs  = df["internal_mig_thr"].values
    moved_ticks    = df["moved_ticks"].values
    intention_states = df["intention_state"].values.copy()
    migr_pressure  = df["migration_pressure"].values.copy()
    econ_penalties = df["econ_penalty"].values
    infra_bonuses  = df["infra_bonus"].values
    place_penalties = df["place_deficit_penalty"].values
    soc_cal_signals = df["soc_calibration_signal"].values
    mob_penalties  = df["inertia_mobility_penalty"].values

    # Доминантный домен (заполним позже для активированных)
    activation_domains = df["activation_domain"].values.copy()

    # ── Предвычисление graph-атрибутов для ВСЕХ агентов (один проход) ──
    # Строим словари: district → значение (быстрее чем G.nodes.get в цикле)
    all_districts = set(G.nodes)
    housing_cache   = {d: float(G.nodes[d].get("effective_housing_price_m2",
                         G.nodes[d].get("housing_price_m2", 1800.0))) for d in all_districts}
    infra_cache     = {d: float(G.nodes[d].get("infrastructure_score", 0.5)) for d in all_districts}

    # Отраслевая зарплата: кэш (district, industry) → wage
    ind_wage_cache: dict[tuple, float] = {}
    # Уникальные пары (workplace, industry) для быстрого заполнения кэша
    unique_wp_ind = set(zip(workplaces, industries))
    for wp, ind in unique_wp_ind:
        ind_wage_cache[(wp, ind)] = _industry_wage_in_district(G, wp, ind)

    # Собираем массивы для всех агентов
    housing_prices_arr = np.array([housing_cache.get(res, 1800.0) for res in residences], dtype=float)
    infra_scores_arr   = np.array([infra_cache.get(res, 0.5) for res in residences], dtype=float)
    industry_wages_arr = np.array([
        ind_wage_cache.get((wp, ind), NATIONAL_AVG_WAGE)
        for wp, ind in zip(workplaces, industries)
    ], dtype=float)

    # ── Векторизованный D_instant для ВСЕХ агентов ───────────────────────
    D_inst_all, D_econ_all, D_place_all = _compute_d_instant_vectorized(
        agent_wages=wages,
        industry_avg_wages_wp=industry_wages_arr,
        econ_gaps=econ_gaps,
        job_flexs=job_flexs,
        housing_prices=housing_prices_arr,
        infra_scores=infra_scores_arr,
        domain_futures=domain_future,
        w_econs=w_econs,
        w_futures=w_futures,
        place_deficit_penalties=place_penalties,
        econ_penalties=econ_penalties,
        infra_bonuses=infra_bonuses,
    )

    # ── Поагентный цикл (только stateful-логика барьеров) ────────────────
    for i in range(n):
        # Пропускаем студентов
        if statuses[i] == "student":
            continue

        # Пропускаем агентов с недавним переездом (< 9 тиков)
        if moved_ticks[i] < 9:
            tpb_active[i] = False
            migr_pressure[i] = 0.0
            continue

        # Пропускаем слишком старых/молодых
        if ages[i] < 18 or ages[i] > 62:
            tpb_active[i] = False
            migr_pressure[i] = 0.0
            continue

        # Берём предвычисленные D-значения для этого агента
        D_inst = float(D_inst_all[i])
        D_econ = float(D_econ_all[i])
        D_place = float(D_place_all[i])

        # ── Обновление aspirations (EWMA, с холодным стартом) ────────────
        if aspirations[i] < 0.01:
            aspirations[i] = D_inst
        else:
            aspirations[i] = ASPIRATIONS_ALPHA * D_inst + (1.0 - ASPIRATIONS_ALPHA) * aspirations[i]

        # ── Capabilities ──────────────────────────────────────────────────
        income_index = min(wages[i] / (1.5 * NATIONAL_AVG_WAGE), 1.0)
        edu_map = {"low": 0.25, "medium": 0.55, "high": 0.85}
        education_index = edu_map.get(educations[i], 0.55)
        capabilities = income_index + (education_index + weak_ties[i]) / 2.0

        # ── Динамический порог (v3: internal_mig_thr вместо inertia, граница 0.15) ─
        dynamic_threshold = (internal_thrs[i] + mob_penalties[i]) * max(0.15, 1.0 - signal_red[i])

        # ── БАРЬЕР 1: Потенциал vs Динамический порог ────────────────────
        if aspirations[i] * capabilities > dynamic_threshold:
            # ── БАРЬЕР 2: D_perceived — накопительное давление + монетка ─
            if not tpb_active[i]:
                # Первый вход — определяем доминантный домен
                tpb_active[i] = True
                if D_econ >= D_place:
                    activation_domains[i] = "economic"
                else:
                    activation_domains[i] = "place"

            # Каждый тик пока active — вычисляем D_perceived и копим давление
            if tpb_active[i]:
                # Attribution
                helplessness = float(np.clip(
                    1.0 - percontrols[i] - weak_ties[i] * 0.3, 0.0, 1.0
                ))
                attribution = percontrols[i] * (1.0 - helplessness)

                # SocialCalibration
                social_calibration = 1.0 + net_susc[i] * soc_cal_signals[i]

                # D_perceived
                D_perceived = D_inst * attribution * social_calibration

                # Динамическая инерция
                dynamic_inertia_s2 = inertias[i] * max(0.15, 1.0 - social_boosts[i])

                # Копим давление: зазор между воспринятым дискомфортом и инерцией
                gap = max(0.0, D_perceived - dynamic_inertia_s2)
                migr_pressure[i] += gap

                # Вероятность «сорваться»: давление / инерция, с границами
                p_act = np.clip(
                    migr_pressure[i] / max(inertias[i], MIGRATION_PRESSURE_DIVISOR),
                    MIGRATION_PRESSURE_P_MIN,
                    MIGRATION_PRESSURE_P_MAX,
                )

                # Бросаем монетку
                if rng.random() < p_act:
                    dom = activation_domains[i]
                    if dom == "economic":
                        intention_states[i] = "seeking_work"
                    elif dom == "place":
                        intention_states[i] = "seeking_residence"
                    # Давление сбрасывается при исполнении (_execute_*)
                # Иначе — остаёмся active, давление копится дальше
        else:
            # Барьер 1 не пройден — сбрасываем состояние
            tpb_active[i] = False
            migr_pressure[i] = max(0.0, migr_pressure[i] - 0.02)  # медленное затухание

    # ── Запись обратно в DataFrame ────────────────────────────────────────
    df["aspirations"]        = np.clip(aspirations, 0.0, 1.0)
    df["signal_reduction"]   = np.clip(signal_red, 0.0, 1.0)
    df["tpb_active"]         = tpb_active
    df["migration_pressure"] = np.clip(migr_pressure, 0.0, 2.0)
    df["econ_gap"]           = np.clip(econ_gaps, 0.0, 1.0)
    df["domain_future_place"] = np.clip(domain_future, 0.0, 1.0)
    df["intention_state"]    = intention_states
    df["activation_domain"]  = activation_domains

    return df


# ── Исполнение решений ────────────────────────────────────────────────────────

def _execute_commute(
    df: pd.DataFrame,
    idx: int,
    new_workplace: str,
    G: nx.DiGraph,
    rng: np.random.Generator,
):
    """
    Агент меняет место работы без смены жительства.
    
    v4: Обновляет industry_pressure при смене workplace.
    """
    old_wp = df.at[idx, "workplace_district"]
    agent_industry = str(df.at[idx, "industry"])
    
    # v4: Вычитаем давление из старого района/отрасли
    if old_wp and old_wp in G.nodes and df.at[idx, "is_employed"]:
        ind_jobs = G.nodes[old_wp].get("industry_jobs", {})
        if ind_jobs and agent_industry in ind_jobs:
            vacant_old = ind_jobs[agent_industry].get("vacant", 1)
            delta_old = -1.0 / max(vacant_old, 1)
            update_industry_pressure_delta(G, old_wp, agent_industry, delta_old)
    
    df.at[idx, "workplace_district"] = new_workplace
    df.at[idx, "status"]             = "commute"
    df.at[idx, "intention_state"]    = "none"
    df.at[idx, "dst_work"]           = ""
    df.at[idx, "moved_ticks"]        = 0
    df.at[idx, "tpb_active"]         = False
    df.at[idx, "intention_delay"]    = 0

    # Сброс aspirations — агент удовлетворил потребность, EWMA обнуляется
    df.at[idx, "aspirations"] = 0.0
    df.at[idx, "place_deficit_penalty"] = 0.0

    # v2: сброс динамических переменных сигнальной системы
    df.at[idx, "econ_penalty"] = 0.0
    df.at[idx, "infra_bonus"] = 0.0
    df.at[idx, "inertia_mobility_penalty"] = 0.0
    df.at[idx, "jobloss_econ_gap_bonus"] = 0.0
    df.at[idx, "soc_calibration_signal"] = 0.0
    df.at[idx, "migration_pressure"] = 0.0

    # v8: отраслевая зарплата в новом районе работы
    wp_attr = G.nodes.get(new_workplace, {})
    wp_salary_by_ind = wp_attr.get("salary_by_industry", {})
    base_wage = wp_salary_by_ind.get(agent_industry, wp_attr.get("avg_wage", df.at[idx, "wage"]))
    new_wage = float(max(0, rng.normal(base_wage, base_wage * 0.18)))
    df.at[idx, "wage"] = new_wage

    # Weak ties растут — агент заводит знакомства в новом месте работы
    df.at[idx, "weak_ties_utility"] = float(np.clip(
        df.at[idx, "weak_ties_utility"] + 0.04, 0.0, 1.0
    ))
    
    # v4: Добавляем давление в новый район/отрасль
    if new_workplace in G.nodes:
        ind_jobs = G.nodes[new_workplace].get("industry_jobs", {})
        if ind_jobs and agent_industry in ind_jobs:
            vacant_new = ind_jobs[agent_industry].get("vacant", 1)
            delta_new = 1.0 / max(vacant_new, 1)
            update_industry_pressure_delta(G, new_workplace, agent_industry, delta_new)


def _execute_move(
    df: pd.DataFrame,
    idx: int,
    new_residence: str,
    new_workplace: str,
    G: nx.DiGraph,
    rng: np.random.Generator,
):
    """
    Агент меняет место жительства (и возможно место работы).
    Стресс переезда: satisfaction временно снижается.
    Inertia пересчитывается: стаж и место сброшены.

    v11: Обновляет housing_remaining в графе — при выезде из старого района
    жильё освобождается (+AGENT_HOUSING_FOOTPRINT), при въезде в новый —
    занимается (−AGENT_HOUSING_FOOTPRINT).

    v4: Обновляет industry_pressure при смене workplace.
    """

    # Сохраняем старые значения до изменений
    old_residence = str(df.at[idx, "residence_district"])
    old_workplace = str(df.at[idx, "workplace_district"])
    agent_industry = str(df.at[idx, "industry"])
    
    # v4: Вычитаем давление из старого района/отрасли
    if old_workplace and old_workplace in G.nodes and df.at[idx, "is_employed"]:
        ind_jobs = G.nodes[old_workplace].get("industry_jobs", {})
        if ind_jobs and agent_industry in ind_jobs:
            vacant_old = ind_jobs[agent_industry].get("vacant", 1)
            delta_old = -1.0 / max(vacant_old, 1)
            update_industry_pressure_delta(G, old_workplace, agent_industry, delta_old)

    df.at[idx, "district"]           = new_residence
    df.at[idx, "residence_district"] = new_residence
    df.at[idx, "workplace_district"] = new_workplace
    df.at[idx, "region"]             = G.nodes[new_residence].get("region",
                                        df.at[idx, "region"])

    status = "stay" if new_residence == new_workplace else "commute"
    df.at[idx, "status"]          = status
    df.at[idx, "intention_state"] = "none"
    df.at[idx, "dst_work"]        = ""
    df.at[idx, "moved_ticks"]     = 0
    df.at[idx, "tenure"]          = 0
    df.at[idx, "tpb_active"]      = False
    df.at[idx, "intention_delay"] = 0

    # Сброс aspirations — агент переехал, EWMA обнуляется
    df.at[idx, "aspirations"] = 0.0
    df.at[idx, "place_deficit_penalty"] = 0.0

    # v2: сброс динамических переменных сигнальной системы
    df.at[idx, "econ_penalty"] = 0.0
    df.at[idx, "infra_bonus"] = 0.0
    df.at[idx, "inertia_mobility_penalty"] = 0.0
    df.at[idx, "jobloss_econ_gap_bonus"] = 0.0
    df.at[idx, "soc_calibration_signal"] = 0.0
    df.at[idx, "migration_pressure"] = 0.0

    # Сброс части слабых связей при переезде
    df.at[idx, "weak_ties_utility"] = float(np.clip(
        df.at[idx, "weak_ties_utility"] + MOVE_WEAK_TIES_PENALTY, 0.0, 1.0
    ))

    # v8: отраслевая зарплата от нового workplace
    wp_attr = G.nodes.get(new_workplace, {})
    wp_salary_by_ind = wp_attr.get("salary_by_industry", {})
    base_wage = wp_salary_by_ind.get(agent_industry, wp_attr.get("avg_wage", df.at[idx, "wage"]))
    new_wage = float(max(0, rng.normal(base_wage, base_wage * 0.18)))
    df.at[idx, "wage"] = new_wage
    
    # v4: Добавляем давление в новый район/отрасль
    if new_workplace in G.nodes:
        ind_jobs = G.nodes[new_workplace].get("industry_jobs", {})
        if ind_jobs and agent_industry in ind_jobs:
            vacant_new = ind_jobs[agent_industry].get("vacant", 1)
            delta_new = 1.0 / max(vacant_new, 1)
            update_industry_pressure_delta(G, new_workplace, agent_industry, delta_new)

    # Стресс переезда
    for col in ["sat_economic", "sat_social", "sat_family", "sat_place"]:
        df.at[idx, col] = float(np.clip(df.at[idx, col] * MOVE_STRESS_FACTOR, 0.05, 0.95))

    # Форсированный рывок sat_place к target_place нового района
    new_h = G.nodes[new_residence].get("housing_price_m2", 1800)
    new_i = G.nodes[new_residence].get("infrastructure_score", 0.5)
    new_target = _sigmoid(0.5 * (1800 - new_h) / 1800 + 0.5 * (new_i - 0.5))
    df.at[idx, "sat_place"] = float(np.clip(
        df.at[idx, "sat_place"] * 0.5 + new_target * 0.5, 0.05, 0.95
    ))

    # Inertia сбрасывается частично: социальный компонент остаётся
    new_inertia = float(np.clip(
        df.at[idx, "inertia_social"] * 0.30 + 0.10,
        0.05, 0.90
    ))
    df.at[idx, "inertia"] = new_inertia

    # Жильё: берём цену нового места проживания
    df.at[idx, "housing_price_m2"] = float(
        G.nodes[new_residence].get("housing_price_m2", df.at[idx, "housing_price_m2"])
    )

    # v11: Обновление динамического трекера жилья
    # Освобождаем жильё в старом районе в графе
    # Освобождаем жильё в старом районе
    if old_residence and old_residence != new_residence and old_residence in G.nodes:
        G.nodes[old_residence]["housing_remaining"] = (
            G.nodes[old_residence].get("housing_remaining", 0.0) + AGENT_HOUSING_FOOTPRINT
        )
    # Занимаем жильё в новом районе
    if new_residence in G.nodes:
        G.nodes[new_residence]["housing_remaining"] = max(
            0.0,
            G.nodes[new_residence].get("housing_remaining", 0.0) - AGENT_HOUSING_FOOTPRINT
        )

def _execute_adapt(df: pd.DataFrame, idx: int, domain: str = "economic"):
    """
    Агент адаптируется на месте: снижает притязания, экономит.
    domain = "economic" → boost sat_economic (требуется job_flexibility)
    domain = "place"    → boost sat_place (снижение жилищных ожиданий)
    """
    flex = df.at[idx, "job_flexibility"]
    if domain == "economic":
        df.at[idx, "sat_economic"] = float(np.clip(
            df.at[idx, "sat_economic"] + flex * ADAPT_SAT_BOOST, 0.0, 1.0
        ))
    elif domain == "place":
        df.at[idx, "sat_place"] = float(np.clip(
            df.at[idx, "sat_place"] + flex * ADAPT_SAT_BOOST * 0.7, 0.0, 1.0
        ))
        # Также снижаем порог — адаптация «места» это не только рост sat, но и снижение планки
        df.at[idx, "thr_place"] = float(np.clip(
            df.at[idx, "thr_place"] - ADAPT_SAT_BOOST * 0.5, 0.10, 0.85
        ))
    df.at[idx, "intention_state"] = "none"
    df.at[idx, "dst_work"]        = ""
    df.at[idx, "tpb_active"]      = False
    df.at[idx, "intention_delay"] = 0

    # Сброс aspirations — агент адаптировался, EWMA обнуляется
    df.at[idx, "aspirations"] = 0.0
    df.at[idx, "place_deficit_penalty"] = 0.0
    df.at[idx, "migration_pressure"] = 0.0


# ── Унифицированный эвристический поиск ──────────────────────────────────────

def _industry_wage_in_district(G: nx.DiGraph, district: str, industry: str) -> float:
    """Отраслевая зарплата в узле; fallback → avg_wage → NATIONAL_AVG_WAGE."""
    attr = G.nodes.get(district, {})
    sal = attr.get("salary_by_industry", {})
    if sal:
        return float(sal.get(industry, attr.get("avg_wage", NATIONAL_AVG_WAGE)))
    return float(attr.get("avg_wage", NATIONAL_AVG_WAGE))


def _compute_target_wage(agent_wage, econ_perceived_control, sat_economic, thr_economic,
                         G, district, agent_industry) -> float:
    """Целевая зарплата: поведенческая эвристика.

    Для занятых: текущая_зарплата × (1 + желаемая_надбавка).
    Для безработных: max(национальный_пол, отраслевая×0.85) × (1 + желаемая_надбавка).
      Национальный пол = (UNEMPLOYED_WAGE_FLOOR + PC × UNEMPLOYED_WAGE_CEIL) × NATIONAL_AVG_WAGE.
    """
    base_appetite = BASE_APPETITE_MIN + econ_perceived_control * BASE_APPETITE_MAX
    desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE)

    if agent_wage > 0:
        return agent_wage * (1.0 + desired_raise)
    else:
        home_ind_wage = _industry_wage_in_district(G, district, agent_industry)
        # Национальный пол зарплатных ожиданий безработного
        nat_floor = (UNEMPLOYED_WAGE_FLOOR + econ_perceived_control * UNEMPLOYED_WAGE_CEIL) * NATIONAL_AVG_WAGE
        base = max(nat_floor, home_ind_wage * 0.85)
        return base * (1.0 + desired_raise)


def _form_work_candidates(
    residence: str, agent_industry: str, target_wage: float,
    G: nx.DiGraph, network_location: bool, perceived_control: float,
    info_quality: float, rng: np.random.Generator,
) -> list:
    """
    Формирует список кандидатов-районов для работы (economic-driven).

    Базовые фильтры:
      - Отраслевая зарплата в dst ≥ target_wage
      - industry_pressure[dst][industry] < MAX_JOBS_PRESSURE

    Возвращает перемешанный список подходящих районов.
    """
    candidates_raw = get_awareness_set(
        G, residence,
        network_location=network_location,
        perceived_control=perceived_control,
        info_quality=info_quality,
        max_candidates=MAX_WORK_CANDIDATES,
        mode="work",
    )
    # Включаем домашний район (может работа там уже подходит)
    candidates_raw = list(set(candidates_raw) | {residence})

    filtered = []
    for dst in candidates_raw:
        dst_attr = G.nodes.get(dst, {})
        ind_wage = _industry_wage_in_district(G, dst, agent_industry)
        if ind_wage < target_wage:
            continue
        ind_pressure = dst_attr.get("industry_pressure", {}).get(agent_industry, 0.0)
        if ind_pressure >= MAX_JOBS_PRESSURE:
            continue
        filtered.append(dst)

    # Взвешенная стохастическая сортировка: кандидаты с более высокой
    # отраслевой зарплатой имеют больше шансов оказаться в начале списка.
    if len(filtered) > 1:
        scores = np.array([
            _industry_wage_in_district(G, d, agent_industry) / max(target_wage, 1.0)
            for d in filtered
        ])
        weights = np.exp(np.clip(scores - 1.0, -2.0, 2.0) * 3.0)
        weights /= weights.sum()
        noisy_keys = -np.log(np.maximum(rng.random(len(filtered)), 1e-9)) / np.maximum(weights, 1e-9)
        order = np.argsort(noisy_keys)
        filtered = [filtered[o] for o in order]

    return filtered


def _form_residence_candidates(
    residence: str, workplace: str, agent_wage: float,
    G: nx.DiGraph, network_location: bool, perceived_control: float,
    info_quality: float, rng: np.random.Generator,
) -> list:
    """
    Формирует список кандидатов-районов для жилья (place-driven).

    Базовые фильтры:
      - Доступность жилья (_housing_affordable)
      - place_reality кандидата > place_reality текущего residence (зазор > 2%)
get_effective_housing_price(G, residence
    Возвращает перемешанный список подходящих районов.
    """
    # Текущая place_reality резиденции
    res_attr = G.nodes.get(residence, {})
    cur_h = res_attr.get("housing_price_m2", 1800.0)
    cur_i = res_attr.get("infrastructure_score", 0.5)
    monthly_cost = cur_h * 50 * 0.004
    burden = monthly_cost / max(agent_wage, 1.0)
    cur_afford = max(0.0, 1.0 - burden / 0.35)
    cur_place_reality = 0.6 * cur_afford + 0.4 * cur_i

    # Кандидаты из awareness_set вокруг residence + workplace
    candidates_raw = get_awareness_set(
        G, residence,
        network_location=network_location,
        perceived_control=perceived_control,
        info_quality=info_quality,
        max_candidates=MAX_WORK_CANDIDATES,
        mode="residence",
    )
    candidates_raw = list(set(candidates_raw) | {workplace, residence})

    filtered = []
    for dst in candidates_raw:
        if dst == residence:
            continue
        dst_attr = G.nodes.get(dst, {})
        housing_price = dst_attr.get("housing_price_m2", 9999.0)
        if not _housing_affordable(agent_wage, housing_price, district=dst, G=G):
            continue
        # place_reality кандидата
        mc = get_effective_housing_price(G, dst) * 50 * 0.004
        b = mc / max(agent_wage, 1.0)
        aff = max(0.0, 1.0 - b / 0.35)
        dst_infra = dst_attr.get("infrastructure_score", 0.5)
        dst_place_reality = 0.6 * aff + 0.4 * dst_infra
        if dst_place_reality <= cur_place_reality + 0.02:
            continue
        filtered.append(dst)

    # Взвешенная стохастическая сортировка: кандидаты с лучшим place_reality
    # имеют больше шансов оказаться в начале списка.
    if len(filtered) > 1:
        scores = np.array([
            (G.nodes.get(d, {}).get("infrastructure_score", 0.5) +
             max(0.0, 1.0 - (get_effective_housing_price(G, d) * 50 * 0.004)
                 / max(agent_wage, 1.0) / 0.35)) / 2.0
            for d in filtered
        ])
        weights = np.exp(np.clip(scores - np.mean(scores), -2.0, 2.0) * 2.0)
        weights /= weights.sum()
        noisy_keys = -np.log(np.maximum(rng.random(len(filtered)), 1e-9)) / np.maximum(weights, 1e-9)
        order = np.argsort(noisy_keys)
        filtered = [filtered[o] for o in order]

    return filtered


def _find_job_near(
    residence: str, agent_industry: str, target_wage: float,
    G: nx.DiGraph, network_location: bool, perceived_control: float,
    info_quality: float, rng: np.random.Generator,
    commuter_threshold_min: float,
) -> Optional[str]:
    """
    Ищет работу в районе residence или его ближайших соседях (satisficing).

    Проверяет сам residence и его awareness_set:
      - Отраслевая зарплата ≥ target_wage
      - industry_pressure < MAX_JOBS_PRESSURE
      - Время commute из residence ≤ commuter_threshold_min

    Возвращает dst_work или None.
    """
    # Сначала проверяем сам residence
    res_attr = G.nodes.get(residence, {})
    ind_wage = _industry_wage_in_district(G, residence, agent_industry)
    ind_press = res_attr.get("industry_pressure", {}).get(agent_industry, 0.0)
    if ind_wage >= target_wage and ind_press < MAX_JOBS_PRESSURE:
        return residence

    candidates = get_awareness_set(
        G, residence,
        network_location=network_location,
        perceived_control=perceived_control,
        info_quality=info_quality,
        max_candidates=MAX_WORK_CANDIDATES,
        mode="work",
    )
    # Взвешенная стохастическая сортировка: кандидаты ближе по commute
    # и с более высокой отраслевой зарплатой — в начале.
    if len(candidates) > 1:
        scores = np.array([
            _industry_wage_in_district(G, d, agent_industry) / max(target_wage, 1.0)
            for d in candidates
        ])
        weights = np.exp(np.clip(scores - 1.0, -2.0, 2.0) * 2.0)
        weights /= weights.sum()
        noisy_keys = -np.log(np.maximum(rng.random(len(candidates)), 1e-9)) / np.maximum(weights, 1e-9)
        order = np.argsort(noisy_keys)
        candidates = [candidates[o] for o in order]

    for dst in candidates:
        dst_attr = G.nodes.get(dst, {})
        dst_ind_wage = _industry_wage_in_district(G, dst, agent_industry)
        if dst_ind_wage < target_wage:
            continue
        dst_press = dst_attr.get("industry_pressure", {}).get(agent_industry, 0.0)
        if dst_press >= MAX_JOBS_PRESSURE:
            continue
        # Проверяем время commute
        if G.has_edge(residence, dst):
            tt = G[residence][dst].get("travel_time_min", 999)
            if tt <= commuter_threshold_min:
                return dst
    return None


def _unified_heuristic_search(
    df: pd.DataFrame, idx: int,
    G: nx.DiGraph, jobs_pressure: dict,
    rng: np.random.Generator,
) -> tuple[str, dict]:
    """
    Унифицированный эвристический поиск: цикл по кандидатам + цепочка стратегий.

    Для economic-доминанты:
      Формирует work_candidates → для каждого: commute → move → satellite.
      Первый успех — выход.

    Для place-доминанты:
      Формирует residence_candidates → для каждого: find_job_near → keep_old_job → satellite_job.
      Первый успех — выход.

    Если все кандидаты исчерпаны → stay + адаптация PC.

    Возвращает (decision, snap_data).
    decision: "commute" | "move" | "satellite_move" | "adapt" | "stay"
    """
    row = df.iloc[idx]
    residence     = str(row["residence_district"])
    workplace     = str(row["workplace_district"])
    agent_wage    = float(row["wage"])
    agent_ind     = str(row["industry"])
    epc           = float(row["econ_perceived_control"])
    pc            = float(row["perceived_control"])
    sat_econ      = float(row["sat_economic"])
    thr_econ      = float(row["thr_economic"])
    net_loc       = bool(row["network_location"])
    info_q        = float(row["info_quality"])
    comm_thr_norm = float(row["commuter_threshold"])
    comm_thr_min  = 30.0 + 90.0 * comm_thr_norm
    sat_place     = float(row["sat_place"])
    thr_place     = float(row["thr_place"])
    activation_dom = str(row["activation_domain"])
    current_status = str(row["status"])
    job_flex       = float(row["job_flexibility"])
    flex_bonus     = max(0.0, job_flex - JOB_FLEX_GATE_REF)

    target_wage = _compute_target_wage(
        agent_wage, epc, sat_econ, thr_econ, G, residence, agent_ind
    )

    def _snap(decision, new_res, new_wp, wage_val, desired_raise_val=0.0):
        thr_e = max(thr_econ, 0.01)
        return {
            "id": int(row["id"]),
            "activation_domain": activation_dom,
            "prev_residence": residence,
            "prev_workplace": workplace,
            "industry": agent_ind,
            "domain_economic_gap": round(float(np.clip((thr_e - sat_econ) / thr_e, 0.0, 1.0)), 4),
            "decision": decision,
            "new_residence": new_res,
            "new_workplace": new_wp,
            "wage": wage_val,
            "desired_raise": round(desired_raise_val, 4),
        }

    # ── ECONOMIC-DRIVEN ──────────────────────────────────────────────────
    if activation_dom == "economic":
        work_candidates = _form_work_candidates(
            residence, agent_ind, target_wage, G,
            net_loc, pc, info_q, rng,
        )
        if not work_candidates:
            return "stay", _snap("stay", residence, workplace, agent_wage)

        for dst_work in work_candidates:
            # 1. Commute
            if G.has_edge(residence, dst_work):
                tt = G[residence][dst_work].get("travel_time_min", 999)
                # Гейт: пропускаем если уже commute, иначе commuter_threshold > 0.5
                if current_status == "commute" or comm_thr_norm > COMMUTER_GATE_REF:
                    effective_thr = comm_thr_min * (1.0 + flex_bonus)
                    if tt <= effective_thr:
                        desired_raise = (target_wage / max(agent_wage, 1.0)) - 1.0 if agent_wage > 0 else 0.0
                        _execute_commute(df, idx, dst_work, G, rng)
                        new_w = float(df.at[idx, "wage"])
                        return "commute", _snap("commute", residence, dst_work, new_w, desired_raise)

            # 2. Direct move
            dst_attr = G.nodes.get(dst_work, {})
            housing_dst = dst_attr.get("housing_price_m2", 9999.0)
            if _housing_affordable(agent_wage, housing_dst, district=dst_work, G=G):
                desired_raise = (target_wage / max(agent_wage, 1.0)) - 1.0 if agent_wage > 0 else 0.0
                _execute_move(df, idx, dst_work, dst_work, G, rng)
                new_w = float(df.at[idx, "wage"])
                return "move", _snap("move", dst_work, dst_work, new_w, desired_raise)

            # 3. Satellite move: ищем спутники (входящие потоки в dst_work)
            satellites = get_awareness_set(
                G, dst_work,
                network_location=False,
                perceived_control=pc,
                info_quality=info_q,
                max_candidates=MAX_WORK_CANDIDATES,
                mode="satellite",
            )
            # Взвешенная сортировка спутников: дешевле жильё = выше в списке
            if len(satellites) > 1:
                sat_scores = np.array([
                    1.0 / max(get_effective_housing_price(G, s), 100.0)
                    for s in satellites
                ])
                sat_weights = np.exp(np.clip(sat_scores / max(sat_scores.max(), 1e-9) - 0.5, -2.0, 2.0) * 3.0)
                sat_weights /= sat_weights.sum()
                sat_noisy = -np.log(np.maximum(rng.random(len(satellites)), 1e-9)) / np.maximum(sat_weights, 1e-9)
                satellites = [satellites[o] for o in np.argsort(sat_noisy)]
            for sat in satellites:
                if sat == residence or sat == dst_work:
                    continue
                sat_attr = G.nodes.get(sat, {})
                sat_housing = sat_attr.get("housing_price_m2", 9999.0)
                if not _housing_affordable(agent_wage, sat_housing, district=sat, G=G):
                    continue
                # Время от спутника до работы
                if G.has_edge(sat, dst_work):
                    tt_sat = G[sat][dst_work].get("travel_time_min", 999)
                    # Гейт: boosted commuter_threshold > 0.5
                    adjusted_commuter = comm_thr_norm * (1.0 + flex_bonus)
                    if adjusted_commuter > COMMUTER_GATE_REF and tt_sat <= comm_thr_min * (1.0 + flex_bonus):
                        desired_raise = (target_wage / max(agent_wage, 1.0)) - 1.0 if agent_wage > 0 else 0.0
                        _execute_move(df, idx, sat, dst_work, G, rng)
                        new_w = float(df.at[idx, "wage"])
                        return "satellite_move", _snap("satellite_move", sat, dst_work, new_w, desired_raise)

        # Все кандидаты исчерпаны
        return "stay", _snap("stay", residence, workplace, agent_wage)

    # ── PLACE-DRIVEN ─────────────────────────────────────────────────────
    else:
        res_candidates = _form_residence_candidates(
            residence, workplace, agent_wage, G,
            net_loc, pc, info_q, rng,
        )
        if not res_candidates:
            return "stay", _snap("stay", residence, workplace, agent_wage)

        for new_res in res_candidates:
            # 1. Поиск работы только в new_res (без соседей)
            res_attr = G.nodes.get(new_res, {})
            ind_wage = _industry_wage_in_district(G, new_res, agent_ind)
            ind_press = res_attr.get("industry_pressure", {}).get(agent_ind, 0.0)
            if ind_wage >= target_wage and ind_press < MAX_JOBS_PRESSURE:
                desired_raise = (target_wage / max(agent_wage, 1.0)) - 1.0 if agent_wage > 0 else 0.0
                _execute_move(df, idx, new_res, new_res, G, rng)
                new_w = float(df.at[idx, "wage"])
                return "move", _snap("move", new_res, new_res, new_w, desired_raise)

            # 2. Сохранение старой работы (маятник: те же правила)
            if G.has_edge(new_res, workplace):
                tt_old = G[new_res][workplace].get("travel_time_min", 999)
                adjusted_commuter = comm_thr_norm * (1.0 + flex_bonus)
                if (current_status == "commute" or adjusted_commuter > COMMUTER_GATE_REF) and tt_old <= comm_thr_min * (1.0 + flex_bonus):
                    _execute_move(df, idx, new_res, workplace, G, rng)
                    new_w = float(df.at[idx, "wage"])
                    return "move", _snap("move", new_res, workplace, new_w)

            # 3. Поиск работы в сателлитах new_res
            sat_work_candidates = get_awareness_set(
                G, new_res,
                network_location=net_loc,
                perceived_control=pc,
                info_quality=info_q,
                max_candidates=MAX_WORK_CANDIDATES,
                mode="work",
            )
            # Взвешенная сортировка: кандидаты с более высокой отраслевой зарплатой — в начале
            if len(sat_work_candidates) > 1:
                sw_scores = np.array([
                    _industry_wage_in_district(G, d, agent_ind) / max(target_wage, 1.0)
                    for d in sat_work_candidates
                ])
                sw_weights = np.exp(np.clip(sw_scores - 1.0, -2.0, 2.0) * 2.0)
                sw_weights /= sw_weights.sum()
                sw_noisy = -np.log(np.maximum(rng.random(len(sat_work_candidates)), 1e-9)) / np.maximum(sw_weights, 1e-9)
                sat_work_candidates = [sat_work_candidates[o] for o in np.argsort(sw_noisy)]
            for sat_work in sat_work_candidates:
                sat_attr = G.nodes.get(sat_work, {})
                sat_ind_wage = _industry_wage_in_district(G, sat_work, agent_ind)
                if sat_ind_wage < target_wage:
                    continue
                sat_press = sat_attr.get("industry_pressure", {}).get(agent_ind, 0.0)
                if sat_press >= MAX_JOBS_PRESSURE:
                    continue
                if G.has_edge(new_res, sat_work):
                    tt_sw = G[new_res][sat_work].get("travel_time_min", 999)
                    adjusted_commuter = comm_thr_norm * (1.0 + flex_bonus)
                    if (current_status == "commute" or adjusted_commuter > COMMUTER_GATE_REF) and tt_sw <= comm_thr_min * (1.0 + flex_bonus):
                        desired_raise = (target_wage / max(agent_wage, 1.0)) - 1.0 if agent_wage > 0 else 0.0
                        _execute_move(df, idx, new_res, sat_work, G, rng)
                        new_w = float(df.at[idx, "wage"])
                        return "move", _snap("move", new_res, sat_work, new_w, desired_raise)

        # Все кандидаты исчерпаны
        return "stay", _snap("stay", residence, workplace, agent_wage)


# ── Главный FFT pipeline ──────────────────────────────────────────────────────

def _fft_pipeline(
    df: pd.DataFrame,
    G: nx.DiGraph,
    jobs_pressure: dict,
    rng: np.random.Generator,
    bus: Optional[EventBus] = None,
    tick_num: int = 0,
) -> tuple[pd.DataFrame, dict, list]:
    """
    Полный FFT pipeline за один тик.

    Проход 1: Двухбарьерная активация (Aspirations×Capabilities → TPB).
      После задержки → intention_state = "seeking_work" | "seeking_residence".

    Проход 2: Унифицированный эвристический поиск (_unified_heuristic_search):
      Для каждого активированного агента — цикл по кандидатам + цепочка стратегий.
      При успехе → commute/move/satellite_move. При провале → stay + адаптация PC.

    bus, tick_num — для эмиссии событий в сигнальную шину (шаги 2+).

    Возвращает (df, stats, action_log).
    """
    df = df.copy()
    stats = {
        "moves": 0, "commutes": 0, "adapts": 0,
        "activated": 0, "dst_found": 0, "satellite_moves": 0,
        "econ_driven_moves": 0, "place_driven_moves": 0,
        "econ_activated": 0, "place_activated": 0,
    }
    action_log = []

    # ── Проход 1: Двухбарьерная активация ────────────────────────────────
    intention_before = df["intention_state"].values.copy()
    df = _two_barrier_activation(df, G, rng)

    # Статистика активации
    tpb_active_count = int(df["tpb_active"].sum())
    stats["activated"] = tpb_active_count

    new_seeking = (
        (df["intention_state"].values != intention_before) &
        (df["intention_state"].values != "none")
    )
    if new_seeking.any():
        new_idx = np.where(new_seeking)[0]
        for idx in new_idx:
            dom = df.at[idx, "activation_domain"]
            if dom == "economic":
                stats["econ_activated"] += 1
            elif dom == "place":
                stats["place_activated"] += 1

    # ── Проход 2: Унифицированный эвристический поиск ────────────────────
    active_idx = np.where(df["intention_state"].values != "none")[0]

    for idx in active_idx:
        # Захват состояния ДО исполнения решения
        residence_before_i = str(df.at[idx, "residence_district"])
        workplace_before_i = str(df.at[idx, "workplace_district"])
        wage_before_i      = float(df.at[idx, "wage"])
        status_before_i    = str(df.at[idx, "status"])

        decision, snap_data = _unified_heuristic_search(df, idx, G, jobs_pressure, rng)

        if decision in ("commute", "move", "satellite_move"):
            # Успех: обновляем perceived_control вверх
            df.at[idx, "perceived_control"] = float(np.clip(
                df.at[idx, "perceived_control"] + 0.02, 0.05, 1.0
            ))
            df.at[idx, "econ_perceived_control"] = float(np.clip(
                df.at[idx, "econ_perceived_control"] + 0.02, 0.05, 1.0
            ))

            # Обновляем snap данными из фактического состояния агента
            snap_data["new_residence"] = str(df.at[idx, "residence_district"])
            snap_data["new_workplace"] = str(df.at[idx, "workplace_district"])
            snap_data["wage"] = float(df.at[idx, "wage"])
            action_log.append(snap_data)

            # ── Эмиссия событий в сигнальную шину ──────────────────────────
            if bus is not None:
                residence_after = str(df.at[idx, "residence_district"])
                workplace_after = str(df.at[idx, "workplace_district"])
                wage_after      = float(df.at[idx, "wage"])
                status_after    = str(df.at[idx, "status"])
                agent_id        = int(df.at[idx, "id"])
                motivation      = str(df.at[idx, "activation_domain"])
                settlement      = str(df.at[idx, "district"])  # residence district

                # AGENT_MOVED: изменился residence
                if residence_before_i != residence_after:
                    bus.emit(Event(
                        event_type=EventType.AGENT_MOVED,
                        tick_emitted=tick_num,
                        source_agent_id=agent_id,
                        source_district=residence_before_i,
                        target_district=residence_after,
                        motivation=motivation,
                        magnitude=min(1.0, abs(wage_after - wage_before_i) / max(wage_before_i, 1.0)),
                    ))

                # JOB_CHANGED: изменился workplace с ростом зарплаты >20%
                if workplace_before_i != workplace_after:
                    if wage_before_i > 0 and wage_after > wage_before_i * 1.20:
                        bus.emit(Event(
                            event_type=EventType.JOB_CHANGED,
                            tick_emitted=tick_num,
                            source_agent_id=agent_id,
                            source_district=workplace_after,  # новый workplace — source сигнала
                            industry=str(df.at[idx, "industry"]),
                            magnitude=min(1.0, (wage_after / max(wage_before_i, 1.0) - 1.0)),
                        ))

                # AGENT_COMMUTE_STARTED: стал маятником (был stay → commute)
                if status_before_i == "stay" and status_after == "commute":
                    bus.emit(Event(
                        event_type=EventType.AGENT_COMMUTE_STARTED,
                        tick_emitted=tick_num,
                        source_agent_id=agent_id,
                        source_district=residence_after,
                        magnitude=0.5,
                    ))

            if decision == "commute":
                stats["commutes"] += 1
            elif decision == "satellite_move":
                stats["moves"] += 1
                stats["satellite_moves"] += 1
                stats["econ_driven_moves"] += 1
            elif decision == "move":
                stats["moves"] += 1
                dom = df.at[idx, "activation_domain"]
                if dom == "economic":
                    stats["econ_driven_moves"] += 1
                elif dom == "place":
                    stats["place_driven_moves"] += 1

        else:
            # Stay: адаптация — снижаем perceived_control
            df.at[idx, "perceived_control"] = float(np.clip(
                df.at[idx, "perceived_control"] - 0.03, 0.05, 1.0
            ))
            df.at[idx, "econ_perceived_control"] = float(np.clip(
                df.at[idx, "econ_perceived_control"] - 0.03, 0.05, 1.0
            ))
            df.at[idx, "intention_state"] = "none"
            df.at[idx, "dst_work"] = ""
            df.at[idx, "tpb_active"] = False
            df.at[idx, "intention_delay"] = 0

            # Адаптация если job_flexibility позволяет
            flex = float(df.at[idx, "job_flexibility"])
            if flex > ADAPT_FLEX_THRESHOLD:
                dom = df.at[idx, "activation_domain"]
                _execute_adapt(df, idx, domain=dom if dom in ("economic", "place") else "economic")
                snap_data["decision"] = "adapt"
                snap_data["new_residence"] = str(df.at[idx, "residence_district"])
                snap_data["new_workplace"] = str(df.at[idx, "workplace_district"])
                snap_data["wage"] = float(df.at[idx, "wage"])
                action_log.append(snap_data)
                stats["adapts"] += 1
            else:
                df.at[idx, "inertia"] = float(np.clip(
                    df.at[idx, "inertia"] + 0.03, 0.05, 0.95
                ))

    return df, stats, action_log


# ── Исполнение сценарных событий ──────────────────────────────────────────────

def _execute_factory_closed(
    df: pd.DataFrame,
    scenario_event: "ScenarioEvent",
    G: nx.DiGraph,
    rng: np.random.Generator,
    bus: "Optional[EventBus]" = None,
    tick_num: int = 0,
) -> None:
    """
    Прямое увольнение агентов для FACTORY_CLOSED.

    v3: Граф-модификация (occupied_jobs) теперь идёт через signals (CLOSED_EMPLOYER),
    здесь только выбор N случайных агентов и перевод в unemployed + LOST_JOB.
    """
    import numpy as np

    district = scenario_event.district
    industry = scenario_event.industry
    n_target = scenario_event.n_agents_affected

    # Маска: занятые агенты в нужной отрасли и районе
    mask = (
        (df["workplace_district"].values == district) &
        (df["industry"].values == industry) &
        (df["status"].values != "unemployed") &
        (df["status"].values != "student")
    )
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return

    n_actual = min(n_target, len(candidates))
    chosen = rng.choice(candidates, size=n_actual, replace=False)

    for idx in chosen:
        agent_id = int(df.at[idx, "id"])
        residence = str(df.at[idx, "district"])
        agent_industry = str(df.at[idx, "industry"])
        old_workplace = str(df.at[idx, "workplace_district"])
        
        # v4: Вычитаем давление из старого района/отрасли перед потерей работы
        if old_workplace and old_workplace in G.nodes and df.at[idx, "is_employed"]:
            ind_jobs = G.nodes[old_workplace].get("industry_jobs", {})
            if ind_jobs and agent_industry in ind_jobs:
                vacant_old = ind_jobs[agent_industry].get("vacant", 1)
                delta_old = -1.0 / max(vacant_old, 1)
                update_industry_pressure_delta(G, old_workplace, agent_industry, delta_old)

        df.at[idx, "status"] = "unemployed"
        df.at[idx, "is_employed"] = False
        df.at[idx, "intention_state"] = "seeking_work"
        df.at[idx, "tpb_active"] = False
        df.at[idx, "intention_delay"] = 0
        df.at[idx, "workplace_district"] = df.at[idx, "district"]

        # v2: эмиссия LOST_JOB для сигнальной системы
        if bus is not None:
            bus.emit(Event(
                event_type=EventType.LOST_JOB,
                tick_emitted=tick_num,
                source_agent_id=agent_id,
                source_district=residence,
                industry=industry,
                magnitude=0.8,
            ))
            df.at[idx, "jobloss_econ_gap_bonus"] = 0.25


def _execute_housing_shock(
    district: str,
    magnitude: float,
    G: "nx.DiGraph | None" = None,
    tick_num: int = 0,
) -> None:
    """
    Применяет HOUSING_SHOCK к динамическому трекеру жилья в графе.

    magnitude > 0 → жильё сокращается (стихийное бедствие, снос, аварийный фонд).
    magnitude < 0 → жильё добавляется (новая застройка, реновация).

    Например, magnitude=0.15 при remaining=100 сокращает доступное жильё на 15 единиц.
    """
    if G is None or district not in G.nodes:
        return

    current = G.nodes[district].get("housing_remaining", 0.0)
    delta = current * magnitude  # доля от текущего остатка
    G.nodes[district]["housing_remaining"] = max(0.0, current - delta)

    if abs(delta) > 0.5:
        direction = "сократилось" if magnitude > 0 else "увеличилось"
        print(f"  [tick {tick_num}] HOUSING_SHOCK {district}: "
              f"жильё {direction} на {abs(delta):.1f} "
              f"(magnitude={magnitude:.2f}, остаток={G.nodes[district]['housing_remaining']:.1f})")


# ── Главный tick ──────────────────────────────────────────────────────────────

# v2: Константы decay
SB_MOVE_DECAY_PER_TICK = 0.01     # затухание social_boost MOVE за тик
SB_MOVE_TOTAL_TICKS    = 6        # длительность MOVE decay
SB_COMMUTE_TOTAL_TICKS = 3        # длительность COMMUTE до сброса
ECON_PENALTY_DECAY_PER_TICK = 0.01  # затухание econ_penalty за тик
INERTIA_MOB_DECAY_PER_TICK = 0.01  # затухание inertia_mobility_penalty за тик
JOBLOSS_RAMP_UP_TICKS   = 3       # тиков роста econ_gap после LOST_JOB
JOBLOSS_RAMP_DOWN_TICKS = 3       # тиков возврата econ_gap
JOBLOSS_RAMP_STEP       = 0.05    # шаг ramp


def _process_sb_pending(df: pd.DataFrame) -> None:
    """v2: Обрабатывает очередь sb_pending — затухание social_boost.

    Формат sb_pending: "M5,M3,C2" — M= MOVE (remaining_ticks), C= COMMUTE (remaining_ticks).
    M-decay: -0.01/тик на каждый активный M-поток.
    C-decay: полный сброс +0.02 через 3 тика.
    """
    sb_pending = df["sb_pending"].values
    sb = df["social_boost"].values.copy()

    for i in range(len(df)):
        val = str(sb_pending[i])
        if not val or val == "nan" or val == "":
            continue

        parts = val.split(",")
        new_parts = []

        for p in parts:
            if not p or len(p) < 2:
                continue
            typ = p[0]
            try:
                rem = int(p[1:])
            except ValueError:
                continue

            if typ == 'M':
                # Линейный спад: -0.01/тик
                sb[i] = max(0.0, sb[i] - SB_MOVE_DECAY_PER_TICK)
                if rem > 1:
                    new_parts.append(f'M{rem - 1}')
                # rem == 1: последний тик, decay применён, поток удаляется
            elif typ == 'C':
                if rem == 1:
                    # Сброс всего буста через 3 тика
                    sb[i] = max(0.0, sb[i] - 0.02)
                    # поток удаляется
                else:
                    new_parts.append(f'C{rem - 1}')

        df.at[i, "sb_pending"] = ",".join(new_parts) if new_parts else ""

    df["social_boost"] = np.clip(sb, 0.0, 1.0)


def _decay_dynamic_vars(df: pd.DataFrame) -> None:
    """v3: Затухание динамических переменных сигнальной системы.

    econ_penalty:              -0.01/тик (до 0)
    infra_bonus:               без автоматического decay (управляется сигналами)
    inertia_mobility_penalty:  затухание к 0 с обеих сторон (0.01/тик)
    soc_calibration_signal:    ×0.85/тик (затухание как signal_reduction)
    jobloss_econ_gap_bonus:    ramp up/down (обрабатывается в _process_jobloss_ramp)
    """
    n = len(df)

    # econ_penalty: линейный спад к 0
    ep = df["econ_penalty"].values.copy()
    ep = np.maximum(0.0, ep - ECON_PENALTY_DECAY_PER_TICK)
    df["econ_penalty"] = ep

    # inertia_mobility_penalty: затухание к 0 с обеих сторон
    # Положительные → уменьшаются, отрицательные → увеличиваются (стремятся к 0)
    imp = df["inertia_mobility_penalty"].values.copy()
    positive = imp > 0
    negative = imp < 0
    imp[positive] = np.maximum(0.0, imp[positive] - INERTIA_MOB_DECAY_PER_TICK)
    imp[negative] = np.minimum(0.0, imp[negative] + INERTIA_MOB_DECAY_PER_TICK)
    df["inertia_mobility_penalty"] = imp

    # soc_calibration_signal: мультипликативное затухание (как signal_reduction)
    scs = df["soc_calibration_signal"].values.copy()
    scs = scs * SIGNAL_DECAY
    df["soc_calibration_signal"] = np.clip(scs, 0.0, 1.0)


def _process_jobloss_ramp(df: pd.DataFrame) -> None:
    """v2: Обрабатывает ramp econ_gap после LOST_JOB.

    jobloss_econ_gap_bonus > 0 → фаза ramp-up (+0.05/тик × 3)
    jobloss_econ_gap_bonus < 0 → фаза ramp-down (-0.05/тик × 3)
    """
    bonus = df["jobloss_econ_gap_bonus"].values
    econ_gap = df["econ_gap"].values

    for i in range(len(df)):
        b = bonus[i]
        if b > 0.001:
            # Фаза ramp-up: econ_gap растёт
            step = min(JOBLOSS_RAMP_STEP, b)
            econ_gap[i] = min(1.0, econ_gap[i] + step)
            bonus[i] = max(0.0, b - step)
            # После исчерпания ramp-up переходим в ramp-down
            if bonus[i] < 0.001:
                bonus[i] = -JOBLOSS_RAMP_DOWN_TICKS * JOBLOSS_RAMP_STEP
        elif b < -0.001:
            # Фаза ramp-down: econ_gap возвращается
            step = min(JOBLOSS_RAMP_STEP, -b)
            econ_gap[i] = max(0.0, econ_gap[i] - step)
            bonus[i] = min(0.0, b + step)
            if bonus[i] > -0.001:
                bonus[i] = 0.0

    df["econ_gap"] = econ_gap
    df["jobloss_econ_gap_bonus"] = bonus


# ── Главный tick ──────────────────────────────────────────────────────────────

SOCIAL_BOOST_DECAY = 0.80   # множитель затухания social_boost за тик


def tick(
    df: pd.DataFrame,
    G: nx.DiGraph,
    jobs_capacity: dict,
    tick_num: int,
    rng: np.random.Generator,
    init_dists: dict = None,
    bus: Optional[EventBus] = None,
    scenario: Optional[object] = None,
) -> tuple[pd.DataFrame, dict]:
    """Один шаг симуляции. Возвращает обновлённый DataFrame и статистику.

    bus      — сигнальная шина (EventBus), опционально
    scenario — сценарные события (Scenario), опционально
    """

    n_agents = len(df)

    df = df.copy()

    # ── COLLECT: сценарные события ──────────────────────────────────────────
    if scenario is not None:
        for se in scenario.get_events(tick_num):
            # v3: Все события — в шину. Signals обрабатывает и агентов (df),
            # и среду (G — industry_jobs, vacant/occupied).
            if bus is not None:
                bus.emit(se.to_event(tick_num))

            # Прямое увольнение агентов для FACTORY_CLOSED (batch-операция —
            # выбор N случайных агентов не укладывается в модель «сигнал-маска»).
            if se.event_type == "FACTORY_CLOSED" and se.n_agents_affected > 0:
                _execute_factory_closed(df, se, G, rng, bus=bus, tick_num=tick_num)

            # v11: Прямое изменение жилья для HOUSING_SHOCK
            if se.event_type == "HOUSING_SHOCK":
                _execute_housing_shock(
                    district=se.district,
                    magnitude=se.magnitude,
                    G=G,
                    tick_num=tick_num,
                )

    # 0. v2: Обработка sb_pending — затухание social_boost по новой схеме
    _process_sb_pending(df)

    # 0b. v2: Затухание динамических переменных сигнальной системы
    _decay_dynamic_vars(df)

    # 1. Время
    df["age"]         = df["age"] + 1 / 12
    df["tenure"]      = df["tenure"] + 1
    df["moved_ticks"] = df["moved_ticks"] + 1

    # 1b. Graduation: пакетный выпуск раз в 12 тиков по когортам
    if tick_num in (12, 24, 36, 48):
        cohort = tick_num // 12  # 1, 2, 3, 4
        grad_mask = (
            (df["status"].values == "student") &
            (df["graduation_cohort"].values == cohort)
        )
        if grad_mask.any():
            grad_idx = np.where(grad_mask)[0]
            for idx in grad_idx:
                age_at_grad = df.at[idx, "age"]
                residence   = str(df.at[idx, "residence_district"])
                agent_id    = int(df.at[idx, "id"])
                industry    = str(df.at[idx, "industry"])

                df.at[idx, "status"]          = "unemployed"
                df.at[idx, "is_employed"]     = False
                df.at[idx, "graduation_cohort"] = -1
                # v8: сэмплируем отрасль из распределения района проживания
                if init_dists:
                    res_data = init_dists.get(residence, {})
                    ind_dist = res_data.get("industry", {})
                    if ind_dist:
                        keys = list(ind_dist.keys())
                        weights = np.array([ind_dist[k] for k in keys], dtype=float)
                        if weights.sum() > 0:
                            df.at[idx, "industry"] = keys[rng.choice(
                                len(keys), p=weights / weights.sum()
                            )]
                            industry = str(df.at[idx, "industry"])
                # Выпускники вузов (>= 22 лет) → high education
                if age_at_grad >= 22 and df.at[idx, "education"] != "high":
                    df.at[idx, "education"] = "high"
                # Инерция снижается — выпускник мобилен
                df.at[idx, "inertia"] = float(np.clip(
                    df.at[idx, "inertia"] * 0.60, 0.05, 0.90
                ))
                # Эмиссия в сигнальную шину
                if bus is not None:
                    bus.emit(Event(
                        event_type=EventType.GRADUATED,
                        tick_emitted=tick_num,
                        source_agent_id=agent_id,
                        source_district=residence,
                        industry=industry,
                        magnitude=0.8,
                    ))
                    bus.emit(Event(
                        event_type=EventType.LOST_JOB,
                        tick_emitted=tick_num,
                        source_agent_id=agent_id,
                        source_district=residence,
                        industry=industry,
                        magnitude=0.7,
                    ))
                    # v2: устанавливаем jobloss_econ_gap_bonus для ramp
                    df.at[idx, "jobloss_econ_gap_bonus"] = 0.25
            n_grad = int(grad_mask.sum())
            if n_grad > 0:
                print(f"  [tick {tick_num}] Выпустилось студентов (cohort={cohort}): {n_grad}")

    # ── DISPATCH + APPLY (фаза 1): обработка graduation-событий ────────────
    if bus is not None:
        signals = bus.process(tick_num, df, G)
        if signals:
            df = bus.flush(df, signals, G)

    # 1c. Ежетиковые обновления перед двухбарьерной проверкой
    # ── signal_reduction: затухание ───────────────────────────────────────
    df["signal_reduction"] = df["signal_reduction"].values * SIGNAL_DECAY

    # ── weak_ties_utility: бонус за нахождение в хабе ─────────────────────
    hub_mask = df["workplace_district"].isin(HUB_DISTRICTS).values
    if hub_mask.any():
        df.loc[hub_mask, "weak_ties_utility"] = np.clip(
            df.loc[hub_mask, "weak_ties_utility"] + HUB_WEAK_TIES_BONUS, 0.0, 1.0
        )

    # ── econ_gap: адаптация восприятия к реальности (отраслевая зарплата) ─
    # Векторизовано: собираем отраслевые зарплаты для workplace каждого агента
    n = len(df)
    wp_districts = df["workplace_district"].values
    agent_wages = df["wage"].values
    agent_inds = df["industry"].values
    old_econ_gaps = df["econ_gap"].values

    # Build mapping: district → {industry: wage}
    # Pre-compute per-agent industry wage in their workplace
    industry_wages_wp = np.full(n, NATIONAL_AVG_WAGE, dtype=float)
    for i in range(n):
        wp = wp_districts[i]
        industry_wages_wp[i] = _industry_wage_in_district(G, wp, agent_inds[i])

    target_econ_gap = np.where(
        (agent_wages > 0) & (industry_wages_wp > 0),
        np.maximum(0.0, 1.0 - agent_wages / industry_wages_wp),
        1.0
    )
    new_econ_gaps = np.clip(
        (1.0 - GAP_ADAPT_LAMBDA) * old_econ_gaps + GAP_ADAPT_LAMBDA * target_econ_gap,
        0.0, 1.0
    )
    df["econ_gap"] = new_econ_gaps

    # v2: Обработка LOST_JOB ramp (econ_gap + jobloss_econ_gap_bonus)
    _process_jobloss_ramp(df)

    # ── place_deficit_penalty: накопление штрафа за неудовлетворённость местом ─
    res_districts = df["district"].values
    agent_wages = df["wage"].values

    old_penalties = df["place_deficit_penalty"].values
    new_penalties = old_penalties.copy()

    for i in range(n):
        res_attr = G.nodes.get(res_districts[i], {})
        housing_price = get_effective_housing_price(G, res_districts[i])
        infra_score = res_attr.get("infrastructure_score", 0.5)

        monthly_cost = housing_price * 50 * 0.004
        burden = monthly_cost / max(agent_wages[i], 1.0)
        affordability = max(0.0, 1.0 - burden / 0.35)
        place_reality = 0.6 * affordability + 0.4 * infra_score

        dfp = df.at[i, "domain_future_place"]
        if dfp > place_reality:
            gap_pct = (dfp - place_reality) / max(place_reality, 0.001)
            new_penalties[i] = np.clip(old_penalties[i] + gap_pct * 0.02 / 6.0, 0.0, 0.2)
        else:
            # Затухание если место устраивает
            new_penalties[i] = max(0.0, old_penalties[i] - 0.01)

    df["place_deficit_penalty"] = new_penalties

    # 2. Давление рынка труда (v3: использует G.nodes industry_jobs, fallback на jobs_capacity)
    jobs_pressure = _compute_jobs_pressure(df, jobs_capacity, G)

    # 3. FFT pipeline (эмитит агентные события в шину)
    df, fft_stats, action_log = _fft_pipeline(df, G, jobs_pressure, rng,
                                                 bus=bus, tick_num=tick_num)

    # ── DISPATCH + APPLY (фаза 2): обработка агентных событий движения ───
    if bus is not None:
        signals = bus.process(tick_num, df, G)
        if signals:
            df = bus.flush(df, signals, G)

    # 7. Реакция среды (residence counts для жилья, workplace counts для зарплат)
    residence_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, residence_counts, n_agents)

    # 9. Статистика
    n_unemployed = int((df["status"] == "unemployed").sum())
    n_commuters  = int((df["status"] == "commute").sum())
    n_stay       = int((df["status"] == "stay").sum())
    n_students   = int((df["status"] == "student").sum())

    stats = {
        "tick":              tick_num,
        "moves":             fft_stats["moves"],
        "commutes":          fft_stats["commutes"],
        "adapts":            fft_stats["adapts"],
        "activated":         fft_stats["activated"],
        "dst_found":         fft_stats["dst_found"],
        "satellite_moves":   fft_stats["satellite_moves"],
        "econ_driven_moves": fft_stats["econ_driven_moves"],
        "place_driven_moves": fft_stats["place_driven_moves"],
        "econ_activated":    fft_stats["econ_activated"],
        "place_activated":   fft_stats["place_activated"],
        "tpb_active_count":  int(df["tpb_active"].sum()),
        "avg_aspirations":   round(float(df["aspirations"].mean()), 4),
        "avg_signal_red":    round(float(df["signal_reduction"].mean()), 4),
        "n_unemployed":      n_unemployed,
        "n_commuters":       n_commuters,
        "n_stay":            n_stay,
        "n_students":        n_students,
        "move_rate_pct":     round(fft_stats["moves"] / n_agents * 100, 3),
        "avg_age":           round(float(df["age"].mean()), 2),
        "avg_wage":          round(float(df[df["wage"] > 0]["wage"].mean()), 2),
        "avg_dissat":        round(_compute_avg_dissatisfaction(df), 4),
        "avg_inertia":       round(float(df["inertia"].mean()), 4),
        "district_counts":   residence_counts,
        "jobs_pressure_max": round(max(jobs_pressure.values()) if jobs_pressure else 0, 2),
        "action_log":        action_log,
        "district_housing_remaining": {
            d: G.nodes[d].get("housing_remaining", 0.0) for d in G.nodes
        },
    }

    return df, stats


# ── Симуляция ─────────────────────────────────────────────────────────────────

def run_simulation(
    df: pd.DataFrame,
    G: nx.DiGraph,
    jobs_capacity: dict,
    n_ticks: int = 60,
    snapshot_ticks: Optional[list] = None,
    seed: int = 42,
    verbose: bool = True,
    init_dists: dict = None,
    bus: Optional[EventBus] = None,
    scenario: Optional[object] = None,
) -> tuple[pd.DataFrame, dict, list, list]:
    """
    Главный цикл симуляции.

    Принимает jobs_capacity явно — строится в agents.py из commuting-матрицы
    и передаётся через run.py.
    init_dists — распределения из agent_init_distributions.json (для graduation).

    bus      — сигнальная шина (EventBus), опционально
    scenario — сценарные события (Scenario), опционально

    Возвращает (df_final, snapshots, tick_stats, all_action_log).
    """
    rng = np.random.default_rng(seed)

    if snapshot_ticks is None:
        snapshot_ticks = [0, n_ticks // 4, n_ticks // 2, n_ticks]

    # Инициализируем граф реальными counts
    residence_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, residence_counts, len(df))

    # v11: Инициализируем динамический трекер жилья в графе
    init_housing_remaining(G, len(df))

    snapshots  = {}
    tick_stats = []
    all_action_log = []  # агрегированный лог решений за все тики

    if 0 in snapshot_ticks:
        snapshots[0] = df.copy()

    if verbose:
        print(f"\nСимуляция: {n_ticks} тиков | {len(df):,} агентов | {G.number_of_nodes()} районов")
        print(f"{'Тик':>5} {'Год':>4} "
              f"{'Акт':>6} {'EconA':>6} {'PlaceA':>7} {'Commute':>8} {'Move':>6} "
              f"{'EconM':>6} {'PlaceM':>7} {'Satel':>6} {'Adapt':>6} "
              f"{'Безраб':>8} {'Dissat':>8}")
        print("─" * 98)

    for t in range(1, n_ticks + 1):
        df, stats = tick(df, G, jobs_capacity, t, rng, init_dists=init_dists,
                          bus=bus, scenario=scenario)
        tick_stats.append(stats)
        all_action_log.extend(stats.get("action_log", []))

        if t in snapshot_ticks:
            snapshots[t] = df.copy()

        if verbose and (t % 6 == 0 or t == 1 or t == n_ticks):
            yr = t // 12
            mo = t % 12 or 12
            print(
                f"  {t:3d} г{yr}м{mo:02d}"
                f"  {stats['activated']:>5}"
                f"  {stats['econ_activated']:>5}"
                f"  {stats['place_activated']:>6}"
                f"  {stats['commutes']:>7}"
                f"  {stats['moves']:>5}"
                f"  {stats['econ_driven_moves']:>5}"
                f"  {stats['place_driven_moves']:>6}"
                f"  {stats['satellite_moves']:>5}"
                f"  {stats['adapts']:>5}"
                f"  {stats['n_unemployed']:>7,}"
                f"  {stats['avg_dissat']:>7.4f}"
            )

    if verbose:
        print("─" * 98)
        total_moves    = sum(s["moves"]   for s in tick_stats)
        total_commutes = sum(s["commutes"] for s in tick_stats)
        total_adapts   = sum(s["adapts"]   for s in tick_stats)
        total_sat      = sum(s["satellite_moves"] for s in tick_stats)
        total_econ_m   = sum(s["econ_driven_moves"] for s in tick_stats)
        total_place_m  = sum(s["place_driven_moves"] for s in tick_stats)
        total_econ_a   = sum(s["econ_activated"] for s in tick_stats)
        total_place_a  = sum(s["place_activated"] for s in tick_stats)
        print(f"\n  Итого активаций:        {sum(s['activated'] for s in tick_stats):,}")
        print(f"    economic-driven:       {total_econ_a:,}")
        print(f"    place-driven:          {total_place_a:,}")
        print(f"  Итого переездов:        {total_moves:,}")
        print(f"    economic-driven:       {total_econ_m:,}")
        print(f"    place-driven:          {total_place_m:,}")
        print(f"  Итого новых commute:    {total_commutes:,}")
        print(f"  Итого спутник-переездов:{total_sat:,}")
        print(f"  Итого адаптаций:        {total_adapts:,}")
        print(f"  Безработных в конце:    {tick_stats[-1]['n_unemployed']:,}")

    return df, snapshots, tick_stats, all_action_log
