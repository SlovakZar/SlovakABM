"""
engine.py v10 — Two-barrier model v4 + Unified heuristic search

Decision-making architecture:

  BARRIER 1 — Migration potential vs dynamic threshold:
    aspirations (EWMA of D_instant) × capabilities > dynamic_threshold → tpb_active.
    dynamic_threshold = (internal_mig_thr + inertia_mob_pen) × max(0.15, 1 − signal_reduction).

  BARRIER 2 — v4: Accumulative pressure + probabilistic trigger:
    Each tick while tpb_active: D_perceived = D_instant × Attribution × SocialCalibration.
    gap = max(0, D_perceived − dynamic_inertia_s2); migration_pressure += gap.
    P(act) = clip(pressure / max(inertia, DIVISOR), P_MIN, P_MAX).
    Coin toss → seeking_work / seeking_residence. Pressure resets on action.

  HEURISTIC SEARCH (unified):
    After activation, the agent forms a candidate list (job/housing)
    accounting for info_quality and industry pressure.
    Candidate loop: commute → move → satellite.
    On success → action + PC↑. On failure → stay + adapt + PC↓.

  DOMAIN UPDATE:
    Social (Block A): target = 0.5 + social_boost, smoothing α=0.88.
    Family (Block F): commute pressure.
    Economic: from workplace_district, Place: from residence_district.

  EVENT SIGNALS (Block B/C):
    social_boost decays ×0.8/tick.
    signal_reduction: decay ×0.85 + new signals (unemployed, neighbors).
    soc_calibration_signal: decay ×0.85 + signals from AGENT_MOVED/COMMUTE/JOB.
    econ_penalty: direct addition to D_econ, decay −0.01/тик.
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional

from graph import (update_graph, get_awareness_set, update_industry_pressure,
                    update_industry_pressure_delta, init_housing_remaining,
                    get_effective_housing_price, get_effective_infrastructure,
                    AGENT_HOUSING_FOOTPRINT)
from signals import EventBus, Event, EventType, Dispatcher, set_settlement_map

# ── Constants ─────────────────────────────────────────────────────────────────

# Filter 2 — employment tree
MAX_JOBS_PRESSURE        = 1.00   # v5: ratio-порог (>1.0 = переполнен)
MAX_WORK_CANDIDATES      = 6      # максимум districts для скрининга

HOUSING_BUDGET_RATIO     = 0.35   # housing should not exceed X share of wage (×100м²)
MOVE_STRESS_FACTOR       = 0.80   # satisfaction after move × этот множитель

# ── Dynamic housing tracker ─────────────────────────────────────────────────
# Moved to graph.py:
#   housing_remaining — in G.nodes[district]["housing_remaining"]
#   effective_housing_price_m2 — in G.nodes[district]["effective_housing_price_m2"]
#   AGENT_HOUSING_FOOTPRINT — imported from graph.py

# Adapt
ADAPT_FLEX_THRESHOLD     = 0.65   # minимальная job_flexibility для адаптации
ADAPT_SAT_BOOST          = 0.06   # прирост sat_economic при адаптации

# Heuristic gates (commute/satellite)
COMMUTER_GATE_REF        = 0.50   # порог сравнения commuter_threshold (глобальное среднее)
JOB_FLEX_GATE_REF        = 0.50   # порог сравнения job_flexibility (глобальное среднее)

# Фильтр 2 — behavioral heuristic of wage expectations
BASE_APPETITE_MIN        = 0.10   # базовый аппетит к росту wage (при PC=0)
BASE_APPETITE_MAX        = 0.20   # добавка за econ_perceived_control (при PC=1 аппетит=0.30)
MIN_DESIRED_RAISE        = 0.05   # minимальная желаемая надбавка (при высокой desperation)
UNEMPLOYED_WAGE_FLOOR    = 0.70   # min. доля от нац. средней для безработных (при PC=0)
UNEMPLOYED_WAGE_CEIL     = 0.20   # добавка за econ_perceived_control (при PC=1 доля=0.90)

# Domain update
SAT_SMOOTHING            = 0.88
NATIONAL_AVG_WAGE        = 1614.0

# ── Two-barrier model: constants ──────────────────────────────────────────
ASPIRATIONS_ALPHA        = 0.15   # скорость EWMA-накопления aspirations из D_instant
SIGNAL_DECAY             = 0.70   # затухание signal_reduction за тик
MIGRATION_COOLDOWN_TICKS = 9      # тиков задержки после переезда до новой активации

# v4: Accumulative pressure and probabilistic trigger (replaces hard delay)
MIGRATION_PRESSURE_P_MIN    = 0.03   # minимальный шанс сорваться в тик
MIGRATION_PRESSURE_P_MAX    = 0.80   # максимальный шанс (не 1.0 — элемент случайности)
MIGRATION_PRESSURE_DIVISOR  = 0.12   # делитель для перевода давления в вероятность
# P(act) = clip(pressure / (inertia + DIVISOR), P_MIN, P_MAX)

PC_D_PERCEIVED_MODIFIER    = 2.0    # множитель контроля (PC) в расчёте D_perceived
GAP_ADAPT_LAMBDA         = 0.05   # скорость адаптации econ_gap и domain_future_place
HUB_WEAK_TIES_BONUS      = 0.005  # прирост weak_ties_utility за тик в хабах
MOVE_WEAK_TIES_PENALTY   = -0.10  # reset weak_ties при переезде

# Hubs: district centers with elevated social dynamics
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


# ── Helpers ───────────────────────────────────────────────────────────

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
    Housing is affordable if monthly payment (аренда ~0.4% эффективной цены 50м²)
    не превышает budget_ratio от wage.

    Uses effective price from graph (G.nodes[district]["effective_housing_price_m2"]).
    If G or district not specified — nominal price is used.
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
    Lightweight version: средняя dissatisfaction для статистики.
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
    v3: jobs_pressure[district] = число занятых agents с workplace=district
                                  / (occupied + vacant по всем отраслям).

    Uses G.nodes[district]["industry_jobs"] для получения occupied+vacant,
    if available. Otherwise fallback to jobs_capacity.

    Value > 1.0 means labor market overload.
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


# ── Two-barrier model: Barrier 1 — Potential vs dynamic inertia ──

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
    Vectorized version _compute_d_instant — all parameters are arrays of length n.

    v3: econ_penalty — direct addition to D_econ; infra_bonus — к инфраструктуре.

    Returns (D_instant, D_econ, D_place) — массивы np.ndarray длины n.
    """
    # Economic component
    safe_w = np.maximum(agent_wages, 0.0)
    safe_ind = np.maximum(industry_avg_wages_wp, 0.0)
    # Use np.divide with where to avoid division-by-zero warning
    wage_pressure = np.where(
        (safe_w > 0) & (safe_ind > 0),
        np.divide(safe_ind, safe_w, where=(safe_w > 0), out=np.zeros_like(safe_ind)),
        1.0  # безработный — максимальное давление
    )
    D_econ = w_econs * wage_pressure * (econ_gaps / np.maximum(job_flexs, 0.01)) + econ_penalties

    # Housing component
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
    Two-barrier activation model (Aspirations×Capabilities → D_perceived → Эвристический поиск).

    Барьер 1 — Потенциал миграции против динамического порога (internal_mig_threshold):
      Обновляет aspirations (EWMA от D_instant).
      Вычисляет capabilities (income + education + weak_ties).
      Вычисляет dynamic threshold: (internal_mig_thr + inertia_mob_penalty) × max(0.15, 1 − signal_reduction).
      Если aspirations × capabilities > dynamic_threshold → tpb_active = True.

    Барьер 2 — v4: Накопительное давление + вероятностный триггер:
      Each tick while tpb_active: D_perceived = D_instant × Attribution × SocialCalibration.
      gap = max(0, D_perceived − dynamic_inertia_s2); migration_pressure += gap.
      P(act) = clip(pressure / max(inertia, DIVISOR), P_MIN, P_MAX).
      Если random < P(act) → intention_state = seeking_work/seeking_residence.
      Давление сбрасывается при исполнении действия (_execute_*).
      Если barrier 1 не пройден → tpb_active=False, давление медленно затухает (−0.02/тик).

    Returns updated DataFrame.
    """
    df = df.copy()
    n = len(df)

    # ── Arrays for vectorization ──────────────────────────────────────────
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

    # Dominant domain (filled later for activated agents)
    activation_domains = df["activation_domain"].values.copy()

    # ── Precompute graph attributes for ALL agents (one pass) ──
    # Build dictionaries: district → value (faster than G.nodes.get in a loop)
    all_districts = set(G.nodes)
    housing_cache   = {d: float(G.nodes[d].get("effective_housing_price_m2",
                         G.nodes[d].get("housing_price_m2", 1800.0))) for d in all_districts}
    infra_cache     = {d: get_effective_infrastructure(G, d) for d in all_districts}

    # Industry wage: cache (district, industry) → wage
    ind_wage_cache: dict[tuple, float] = {}
    # Unique (workplace, industry) pairs for fast cache filling
    unique_wp_ind = set(zip(workplaces, industries))
    for wp, ind in unique_wp_ind:
        ind_wage_cache[(wp, ind)] = _industry_wage_in_district(G, wp, ind)

    # Collect arrays for all agents
    housing_prices_arr = np.array([housing_cache.get(res, 1800.0) for res in residences], dtype=float)
    infra_scores_arr   = np.array([infra_cache.get(res, 0.5) for res in residences], dtype=float)
    industry_wages_arr = np.array([
        ind_wage_cache.get((wp, ind), NATIONAL_AVG_WAGE)
        for wp, ind in zip(workplaces, industries)
    ], dtype=float)

    # ── Vectorized D_instant for ALL agents ───────────────────────
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

    # ── VECTORIZED two-barrier logic (replaces per-agent loop) ──
    # Step 1: exclusion masks
    skip_student = (statuses == "student")
    skip_recent  = (moved_ticks < MIGRATION_COOLDOWN_TICKS)
    skip_age     = (ages < 18) | (ages > 62)
    skip_mask    = skip_student | skip_recent | skip_age
    active_mask  = ~skip_mask

    # Reset state for excluded agents
    tpb_active[skip_mask] = False
    migr_pressure[skip_mask] = 0.0

    if active_mask.any():
        # Active agent indices
        act_idx = np.where(active_mask)[0]

        # Step 2: aspirations EWMA (vectorized)
        asp_act = aspirations[act_idx]
        D_inst_act = D_inst_all[act_idx]
        aspirations[act_idx] = np.where(
            asp_act < 0.01,
            D_inst_act,
            ASPIRATIONS_ALPHA * D_inst_act + (1.0 - ASPIRATIONS_ALPHA) * asp_act,
        )

        # Step 3: capabilities (vectorized)
        income_index = np.minimum(wages[act_idx] / (2.0 * NATIONAL_AVG_WAGE), 1.0)
        edu_vals = np.full(len(act_idx), 0.55, dtype=float)
        edu_vals[educations[act_idx] == "low"]    = 0.25
        edu_vals[educations[act_idx] == "medium"] = 0.55
        edu_vals[educations[act_idx] == "high"]   = 0.85
        capabilities = income_index + (edu_vals + weak_ties[act_idx]) / 2.0

        # Step 4: dynamic threshold (vectorized)
        dynamic_threshold = (internal_thrs[act_idx] + mob_penalties[act_idx]) * np.maximum(0.15, 1.0 - signal_red[act_idx])

        # Step 5: BARRIER 1 — passed mask
        b1_pass = (aspirations[act_idx] * capabilities) > dynamic_threshold

        # Barrier 1 not passed → reset
        fail_idx = act_idx[~b1_pass]
        if len(fail_idx) > 0:
            tpb_active[fail_idx] = False
            migr_pressure[fail_idx] = np.maximum(0.0, migr_pressure[fail_idx] - 0.02)

        # Step 6: BARRIER 2 — only for those who passed barrier 1
        pass_idx = act_idx[b1_pass]
        n_pass = len(pass_idx)

        if n_pass > 0:
            # 6a. First entry: determine dominant domain
            just_activated = ~tpb_active[pass_idx]
            tpb_active[pass_idx] = True

            if just_activated.any():
                ja_idx = pass_idx[just_activated]
                activation_domains[ja_idx] = np.where(
                    D_econ_all[ja_idx] >= D_place_all[ja_idx], "economic", "place"
                )

            # 6b. Attribution (vectorized)
            pc_scaled = percontrols[pass_idx] * PC_D_PERCEIVED_MODIFIER
            helplessness = np.clip(1.0 - pc_scaled - weak_ties[pass_idx] * 0.3, 0.0, 1.0)
            attribution = percontrols[pass_idx] * (1.0 - helplessness)

            # 6c. SocialCalibration
            social_calibration = 1.0 + net_susc[pass_idx] * soc_cal_signals[pass_idx]

            # 6d. D_perceived
            D_perceived = D_inst_all[pass_idx] * attribution * social_calibration

            # 6e. Dynamic inertia (s2)
            dynamic_inertia_s2 = inertias[pass_idx] * np.maximum(0.15, 1.0 - social_boosts[pass_idx])

            # 6f. Pressure accumulation
            gap = np.maximum(0.0, D_perceived - dynamic_inertia_s2)
            migr_pressure[pass_idx] += gap

            # 6g. Action probability
            p_act = np.clip(
                migr_pressure[pass_idx] / np.maximum(inertias[pass_idx], MIGRATION_PRESSURE_DIVISOR),
                MIGRATION_PRESSURE_P_MIN,
                MIGRATION_PRESSURE_P_MAX,
            )

            # 6h. Flip coins (single rng.random call for all)
            do_act = rng.random(n_pass) < p_act

            if do_act.any():
                act_now_idx = pass_idx[do_act]
                intention_states[act_now_idx] = np.where(
                    activation_domains[act_now_idx] == "economic",
                    "seeking_work", "seeking_residence"
                )

    # ── Write back to DataFrame ────────────────────────────────────────
    df["aspirations"]        = np.clip(aspirations, 0.0, 1.0)
    df["signal_reduction"]   = np.clip(signal_red, 0.0, 1.0)
    df["tpb_active"]         = tpb_active
    df["migration_pressure"] = np.clip(migr_pressure, 0.0, 2.0)
    df["econ_gap"]           = np.clip(econ_gaps, 0.0, 1.0)
    df["domain_future_place"] = np.clip(domain_future, 0.0, 1.0)
    df["intention_state"]    = intention_states
    df["activation_domain"]  = activation_domains

    return df


# ── Decision execution ────────────────────────────────────────────────────────

def _execute_commute(
    df: pd.DataFrame,
    idx: int,
    new_workplace: str,
    G: nx.DiGraph,
    rng: np.random.Generator,
):
    """
    Agent changes workplace without moving residence.
    
    v4: Updates industry_pressure on workplace change.
    """
    old_wp = df.at[idx, "workplace_district"]
    agent_industry = str(df.at[idx, "industry"])
    
    # v5: Subtract pressure from old district/industry (capacity-based)
    if old_wp and old_wp in G.nodes and df.at[idx, "is_employed"]:
        ind_jobs = G.nodes[old_wp].get("industry_jobs", {})
        if ind_jobs and agent_industry in ind_jobs:
            cap_old = ind_jobs[agent_industry].get("capacity",
                       ind_jobs[agent_industry].get("occupied", 0) + ind_jobs[agent_industry].get("vacant", 1))
            delta_old = -1.0 / max(cap_old, 1)
            update_industry_pressure_delta(G, old_wp, agent_industry, delta_old)
    
    df.at[idx, "workplace_district"] = new_workplace
    df.at[idx, "status"]             = "commute"
    df.at[idx, "intention_state"]    = "none"
    df.at[idx, "dst_work"]           = ""
    df.at[idx, "moved_ticks"]        = 0
    df.at[idx, "tpb_active"]         = False
    df.at[idx, "intention_delay"]    = 0

    # Reset aspirations — agent satisfied the need, EWMA resets
    df.at[idx, "aspirations"] = 0.0
    df.at[idx, "place_deficit_penalty"] = 0.0

    # v2: reset dynamic signal system variables
    df.at[idx, "econ_penalty"] = 0.0
    df.at[idx, "infra_bonus"] = 0.0
    df.at[idx, "inertia_mobility_penalty"] = 0.0
    df.at[idx, "jobloss_econ_gap_bonus"] = 0.0
    df.at[idx, "soc_calibration_signal"] = 0.0
    df.at[idx, "migration_pressure"] = 0.0

    # v8: industry wage in new work district
    wp_attr = G.nodes.get(new_workplace, {})
    wp_salary_by_ind = wp_attr.get("salary_by_industry", {})
    base_wage = wp_salary_by_ind.get(agent_industry, wp_attr.get("avg_wage", df.at[idx, "wage"]))
    new_wage = float(max(0, rng.normal(base_wage, base_wage * 0.18)))
    df.at[idx, "wage"] = new_wage

    # Weak ties grow — agent makes acquaintances at new workplace
    df.at[idx, "weak_ties_utility"] = float(np.clip(
        df.at[idx, "weak_ties_utility"] + 0.04, 0.0, 1.0
    ))
    
    # v5: Add pressure to new district/industry (capacity-based)
    if new_workplace in G.nodes:
        ind_jobs = G.nodes[new_workplace].get("industry_jobs", {})
        if ind_jobs and agent_industry in ind_jobs:
            cap_new = ind_jobs[agent_industry].get("capacity",
                       ind_jobs[agent_industry].get("occupied", 0) + ind_jobs[agent_industry].get("vacant", 1))
            delta_new = 1.0 / max(cap_new, 1)
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
    Agent changes residence (и возможно moто работы).
    Move stress: satisfaction temporarily decreases.
    Inertia recalculated: tenure and location reset.

    v11: Updates housing_remaining in graph — при выезде из старого района
    жильё освобождается (+AGENT_HOUSING_FOOTPRINT), при въезде в новый —
    занимается (−AGENT_HOUSING_FOOTPRINT).

    v4: Updates industry_pressure on workplace change.
    """

    # Save old values before changes
    old_residence = str(df.at[idx, "residence_district"])
    old_workplace = str(df.at[idx, "workplace_district"])
    agent_industry = str(df.at[idx, "industry"])
    
    # v5: Subtract pressure from old district/industry (capacity-based)
    if old_workplace and old_workplace in G.nodes and df.at[idx, "is_employed"]:
        ind_jobs = G.nodes[old_workplace].get("industry_jobs", {})
        if ind_jobs and agent_industry in ind_jobs:
            cap_old = ind_jobs[agent_industry].get("capacity",
                       ind_jobs[agent_industry].get("occupied", 0) + ind_jobs[agent_industry].get("vacant", 1))
            delta_old = -1.0 / max(cap_old, 1)
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

    # Reset aspirations — agent moved, EWMA resets
    df.at[idx, "aspirations"] = 0.0
    df.at[idx, "place_deficit_penalty"] = 0.0

    # v2: reset dynamic signal system variables
    df.at[idx, "econ_penalty"] = 0.0
    df.at[idx, "infra_bonus"] = 0.0
    df.at[idx, "inertia_mobility_penalty"] = 0.0
    df.at[idx, "jobloss_econ_gap_bonus"] = 0.0
    df.at[idx, "soc_calibration_signal"] = 0.0
    df.at[idx, "migration_pressure"] = 0.0

    # Reset some weak ties on move
    df.at[idx, "weak_ties_utility"] = float(np.clip(
        df.at[idx, "weak_ties_utility"] + MOVE_WEAK_TIES_PENALTY, 0.0, 1.0
    ))

    # v8: industry wage from new workplace
    wp_attr = G.nodes.get(new_workplace, {})
    wp_salary_by_ind = wp_attr.get("salary_by_industry", {})
    base_wage = wp_salary_by_ind.get(agent_industry, wp_attr.get("avg_wage", df.at[idx, "wage"]))
    new_wage = float(max(0, rng.normal(base_wage, base_wage * 0.18)))
    df.at[idx, "wage"] = new_wage
    
    # v5: Add pressure to new district/industry (capacity-based)
    if new_workplace in G.nodes:
        ind_jobs = G.nodes[new_workplace].get("industry_jobs", {})
        if ind_jobs and agent_industry in ind_jobs:
            cap_new = ind_jobs[agent_industry].get("capacity",
                       ind_jobs[agent_industry].get("occupied", 0) + ind_jobs[agent_industry].get("vacant", 1))
            delta_new = 1.0 / max(cap_new, 1)
            update_industry_pressure_delta(G, new_workplace, agent_industry, delta_new)

    # Move stress
    for col in ["sat_economic", "sat_social", "sat_family", "sat_place"]:
        df.at[idx, col] = float(np.clip(df.at[idx, col] * MOVE_STRESS_FACTOR, 0.05, 0.95))

    # Forced jump of sat_place towards target_place of new district
    new_h = G.nodes[new_residence].get("housing_price_m2", 1800)
    new_i = get_effective_infrastructure(G, new_residence)
    new_target = _sigmoid(0.5 * (1800 - new_h) / 1800 + 0.5 * (new_i - 0.5))
    df.at[idx, "sat_place"] = float(np.clip(
        df.at[idx, "sat_place"] * 0.5 + new_target * 0.5, 0.05, 0.95
    ))

    # Inertia partially reset: social component remains
    new_inertia = float(np.clip(
        df.at[idx, "inertia_social"] * 0.30 + 0.10,
        0.05, 0.90
    ))
    df.at[idx, "inertia"] = new_inertia

    # Housing: take price of new residence
    df.at[idx, "housing_price_m2"] = float(
        G.nodes[new_residence].get("housing_price_m2", df.at[idx, "housing_price_m2"])
    )

    # v11: Dynamic housing tracker update
    # Free up housing in old district in graph
    # Free up housing in old district
    if old_residence and old_residence != new_residence and old_residence in G.nodes:
        G.nodes[old_residence]["housing_remaining"] = (
            G.nodes[old_residence].get("housing_remaining", 0.0) + AGENT_HOUSING_FOOTPRINT
        )
    # Occupy housing in new district
    if new_residence in G.nodes:
        G.nodes[new_residence]["housing_remaining"] = max(
            0.0,
            G.nodes[new_residence].get("housing_remaining", 0.0) - AGENT_HOUSING_FOOTPRINT
        )

def _execute_adapt(df: pd.DataFrame, idx: int, domain: str = "economic"):
    """
    Agent adapts in place: снижает притязания, экономит.
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
        # Also lower the threshold — place adaptation is not only sat growth but also threshold reduction
        df.at[idx, "thr_place"] = float(np.clip(
            df.at[idx, "thr_place"] - ADAPT_SAT_BOOST * 0.5, 0.10, 0.85
        ))
    df.at[idx, "intention_state"] = "none"
    df.at[idx, "dst_work"]        = ""
    df.at[idx, "tpb_active"]      = False
    df.at[idx, "intention_delay"] = 0

    # Reset aspirations — agent adapted, EWMA resets
    df.at[idx, "aspirations"] = 0.0
    df.at[idx, "place_deficit_penalty"] = 0.0
    df.at[idx, "migration_pressure"] = 0.0


# ── Unified heuristic search ──────────────────────────────────────

def _industry_wage_in_district(G: nx.DiGraph, district: str, industry: str) -> float:
    """Industry wage in node; fallback → avg_wage → NATIONAL_AVG_WAGE."""
    attr = G.nodes.get(district, {})
    sal = attr.get("salary_by_industry", {})
    if sal:
        return float(sal.get(industry, attr.get("avg_wage", NATIONAL_AVG_WAGE)))
    return float(attr.get("avg_wage", NATIONAL_AVG_WAGE))


def _compute_target_wage(agent_wage, econ_perceived_control, sat_economic, thr_economic,
                         G, district, agent_industry) -> float:
    """Target wage: behavioral heuristic.

    For employed: current_wage × (1 + желаемая_надбавка).
    For unemployed: max(national_floor, отраслевая×0.85) × (1 + желаемая_надбавка).
      Национальный пол = (UNEMPLOYED_WAGE_FLOOR + PC × UNEMPLOYED_WAGE_CEIL) × NATIONAL_AVG_WAGE.
    """
    base_appetite = BASE_APPETITE_MIN + econ_perceived_control * BASE_APPETITE_MAX
    desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE)

    if agent_wage > 0:
        return agent_wage * (1.0 + desired_raise)
    else:
        home_ind_wage = _industry_wage_in_district(G, district, agent_industry)
        # National floor of unemployed wage expectations
        nat_floor = (UNEMPLOYED_WAGE_FLOOR + econ_perceived_control * UNEMPLOYED_WAGE_CEIL) * NATIONAL_AVG_WAGE
        base = max(nat_floor, home_ind_wage * 0.85)
        return base * (1.0 + desired_raise)


def _form_work_candidates(
    residence: str, agent_industry: str, target_wage: float,
    G: nx.DiGraph, network_location: bool, perceived_control: float,
    info_quality: float, rng: np.random.Generator,
) -> list:
    """
    Forms candidate districts for work (economic-driven).

    Basic filters:
      - Отраслевая зарплата в dst ≥ target_wage
      - industry_pressure[dst][industry] < MAX_JOBS_PRESSURE

    Returns shuffled list of suitable districts.
    """
    candidates_raw = get_awareness_set(
        G, residence,
        network_location=network_location,
        perceived_control=perceived_control,
        info_quality=info_quality,
        max_candidates=MAX_WORK_CANDIDATES,
        mode="work",
    )
    # Include home district (job there may already fit)
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

    # Weighted stochastic sorting: candidates with higher
    # отраслевой зарплатой have higher chance to be at the top of the list.
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
    Forms candidate districts for housing (place-driven).

    Basic filters:
      - Доступность жилья (_housing_affordable)
      - place_reality candidate > place_reality текущего residence (зазор > 2%)
get_effective_housing_price(G, residence
    Returns shuffled list of suitable districts.
    """
    # Current place_reality of residence
    res_attr = G.nodes.get(residence, {})
    cur_h = res_attr.get("housing_price_m2", 1800.0)
    cur_i = get_effective_infrastructure(G, residence)
    monthly_cost = cur_h * 50 * 0.004
    burden = monthly_cost / max(agent_wage, 1.0)
    cur_afford = max(0.0, 1.0 - burden / 0.35)
    cur_place_reality = 0.6 * cur_afford + 0.4 * cur_i

    # Candidates from awareness_set around residence + workplace
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
        # place_reality candidate
        mc = get_effective_housing_price(G, dst) * 50 * 0.004
        b = mc / max(agent_wage, 1.0)
        aff = max(0.0, 1.0 - b / 0.35)
        dst_infra = get_effective_infrastructure(G, dst)
        dst_place_reality = 0.6 * aff + 0.4 * dst_infra
        if dst_place_reality <= cur_place_reality + 0.02:
            continue
        filtered.append(dst)

    # Weighted stochastic sorting: candidates with better place_reality
    # have higher chance to be at the top of the list.
    if len(filtered) > 1:
        scores = np.array([
            (get_effective_infrastructure(G, d) +
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
    Searches for job in residence district или его ближайших соседях (satisficing).

    Проверяет сам residence и его awareness_set:
      - Отраслевая зарплата ≥ target_wage
      - industry_pressure < MAX_JOBS_PRESSURE
      - Commute time from residence ≤ commuter_threshold_min

    Возвращает dst_work или None.
    """
    # First check residence itself
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
    # Weighted stochastic sorting: candidates with shorter commute
    # and with higher industry wage — first.
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
        # Check commute time
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
    Unified heuristic search: цикл по candidateм + цепочка стратегий.

    Для economic-доminанты:
      Формирует work_candidates → для каждого: commute → move → satellite.
      Первый успех — выход.

    Для place-доminанты:
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
                # Gate: pass if already commute, else commuter_threshold > 0.5
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

            # 3. Satellite move: look for satellites (incoming flows to dst_work)
            satellites = get_awareness_set(
                G, dst_work,
                network_location=False,
                perceived_control=pc,
                info_quality=info_q,
                max_candidates=MAX_WORK_CANDIDATES,
                mode="satellite",
            )
            # Weighted satellite sorting: cheaper housing = higher in list
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
                # Travel time from satellite to work
                if G.has_edge(sat, dst_work):
                    tt_sat = G[sat][dst_work].get("travel_time_min", 999)
                    # Gate: boosted commuter_threshold > 0.5
                    adjusted_commuter = comm_thr_norm * (1.0 + flex_bonus)
                    if adjusted_commuter > COMMUTER_GATE_REF and tt_sat <= comm_thr_min * (1.0 + flex_bonus):
                        desired_raise = (target_wage / max(agent_wage, 1.0)) - 1.0 if agent_wage > 0 else 0.0
                        _execute_move(df, idx, sat, dst_work, G, rng)
                        new_w = float(df.at[idx, "wage"])
                        return "satellite_move", _snap("satellite_move", sat, dst_work, new_w, desired_raise)

        # All candidates exhausted
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
            # 1. Job search only in new_res (no neighbors)
            res_attr = G.nodes.get(new_res, {})
            ind_wage = _industry_wage_in_district(G, new_res, agent_ind)
            ind_press = res_attr.get("industry_pressure", {}).get(agent_ind, 0.0)
            if ind_wage >= target_wage and ind_press < MAX_JOBS_PRESSURE:
                desired_raise = (target_wage / max(agent_wage, 1.0)) - 1.0 if agent_wage > 0 else 0.0
                _execute_move(df, idx, new_res, new_res, G, rng)
                new_w = float(df.at[idx, "wage"])
                return "move", _snap("move", new_res, new_res, new_w, desired_raise)

            # 2. Keep old job (commuter: same rules)
            if G.has_edge(new_res, workplace):
                tt_old = G[new_res][workplace].get("travel_time_min", 999)
                adjusted_commuter = comm_thr_norm * (1.0 + flex_bonus)
                if (current_status == "commute" or adjusted_commuter > COMMUTER_GATE_REF) and tt_old <= comm_thr_min * (1.0 + flex_bonus):
                    _execute_move(df, idx, new_res, workplace, G, rng)
                    new_w = float(df.at[idx, "wage"])
                    return "move", _snap("move", new_res, workplace, new_w)

            # 3. Job search in satellites of new_res
            sat_work_candidates = get_awareness_set(
                G, new_res,
                network_location=net_loc,
                perceived_control=pc,
                info_quality=info_q,
                max_candidates=MAX_WORK_CANDIDATES,
                mode="work",
            )
            # Weighted sorting: candidates with higher industry wage — first
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

        # All candidates exhausted
        return "stay", _snap("stay", residence, workplace, agent_wage)


# ── Main FFT pipeline ──────────────────────────────────────────────────────

def _fft_pipeline(
    df: pd.DataFrame,
    G: nx.DiGraph,
    jobs_pressure: dict,
    rng: np.random.Generator,
    bus: Optional[EventBus] = None,
    tick_num: int = 0,
) -> tuple[pd.DataFrame, dict, list]:
    """
    Full FFT pipeline for one tick.

    Pass 1: Two-barrier activation (Aspirations×Capabilities → TPB).
      После задержки → intention_state = "seeking_work" | "seeking_residence".

    Pass 2: Unified heuristic search (_unified_heuristic_search):
      Для каждого активированного агента — цикл по candidateм + цепочка стратегий.
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

    # ── Pass 1: Two-barrier activation ────────────────────────────────
    intention_before = df["intention_state"].values.copy()
    df = _two_barrier_activation(df, G, rng)

    # Activation statistics
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

    # ── Pass 2: Unified heuristic search ────────────────────
    active_idx = np.where(df["intention_state"].values != "none")[0]

    for idx in active_idx:
        # Capture state BEFORE decision execution
        residence_before_i = str(df.at[idx, "residence_district"])
        workplace_before_i = str(df.at[idx, "workplace_district"])
        wage_before_i      = float(df.at[idx, "wage"])
        status_before_i    = str(df.at[idx, "status"])

        decision, snap_data = _unified_heuristic_search(df, idx, G, jobs_pressure, rng)

        if decision in ("commute", "move", "satellite_move"):
            # Success: bump perceived_control up
            df.at[idx, "perceived_control"] = float(np.clip(
                df.at[idx, "perceived_control"] + 0.02, 0.05, 1.0
            ))
            df.at[idx, "econ_perceived_control"] = float(np.clip(
                df.at[idx, "econ_perceived_control"] + 0.02, 0.05, 1.0
            ))

            # Update snap with actual agent state
            snap_data["new_residence"] = str(df.at[idx, "residence_district"])
            snap_data["new_workplace"] = str(df.at[idx, "workplace_district"])
            snap_data["wage"] = float(df.at[idx, "wage"])
            action_log.append(snap_data)

            # ── Emit events to signal bus ──────────────────────────
            if bus is not None:
                residence_after = str(df.at[idx, "residence_district"])
                workplace_after = str(df.at[idx, "workplace_district"])
                wage_after      = float(df.at[idx, "wage"])
                status_after    = str(df.at[idx, "status"])
                agent_id        = int(df.at[idx, "id"])
                motivation      = str(df.at[idx, "activation_domain"])
                settlement      = str(df.at[idx, "district"])  # residence district

                # AGENT_MOVED: residence changed
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

                # JOB_CHANGED: workplace changed with wage increase >20%
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

                # AGENT_COMMUTE_STARTED: became commuter (was stay → commute)
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
            # Stay: adaptation — lower perceived_control
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

            # Adapt if job_flexibility allows
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


# ── Scenario event execution ──────────────────────────────────────────────

def _execute_new_employer(
    district: str,
    industry: str,
    size: str,
    G: nx.DiGraph,
    industry_shares: dict,
    tick_num: int = 0,
) -> int:
    """
    v5: New employer opening.

    1. +1 компания размера size в G.nodes[district][\"business\"]
    2. Пересчёт ёмкости (capacity растёт → vacant растёт)
    3. Returns approximate number добавленных рабочих moт.
    """
    from graph import change_company_count
    n_jobs = change_company_count(G, district, size, +1, industry_shares)
    return n_jobs


def _execute_closed_employer(
    df: pd.DataFrame,
    district: str,
    industry: str,
    size: str,
    G: nx.DiGraph,
    industry_shares: dict,
    rng: np.random.Generator,
    bus: "Optional[EventBus]" = None,
    tick_num: int = 0,
    n_to_fire: int = 0,
) -> int:
    """
    v7: Employer closure.

    1. Agent layoffs в district × industry
       — если n_to_fire > 0: n_target = n_to_fire × agent_scale
       — иначе: SIZE_EMPLOYEES[size] × agent_scale (старое поведение)
    2. −1 компания размера size → пересчёт ёмкости (capacity падает)
    3. LOST_JOB каждому уволенному
    4. Возвращает число уволенных.
    """
    from graph import change_company_count, SIZE_EMPLOYEES

    scale = G.graph.get("agent_scale", 1.0)
    if n_to_fire > 0:
        n_target = max(1, int(n_to_fire * scale))
    else:
        n_target = max(1, int(SIZE_EMPLOYEES.get(size, 25) * scale))

    # Mask: employed agents in the target industry and district
    mask = (
        (df["workplace_district"].values == district) &
        (df["industry"].values == industry) &
        (df["status"].values != "unemployed") &
        (df["status"].values != "student")
    )
    candidates = np.where(mask)[0]
    n_fired = 0

    if len(candidates) > 0:
        n_actual = min(n_target, len(candidates))
        chosen = rng.choice(candidates, size=n_actual, replace=False)

        for idx in chosen:
            agent_id = int(df.at[idx, "id"])
            residence = str(df.at[idx, "district"])
            agent_industry = str(df.at[idx, "industry"])
            old_workplace = str(df.at[idx, "workplace_district"])

            # v5: Subtract pressure (capacity-based)
            if old_workplace and old_workplace in G.nodes and df.at[idx, "is_employed"]:
                ind_jobs = G.nodes[old_workplace].get("industry_jobs", {})
                if ind_jobs and agent_industry in ind_jobs:
                    cap_old = ind_jobs[agent_industry].get("capacity",
                               ind_jobs[agent_industry].get("occupied", 0) + ind_jobs[agent_industry].get("vacant", 1))
                    delta_old = -1.0 / max(cap_old, 1)
                    update_industry_pressure_delta(G, old_workplace, agent_industry, delta_old)

            df.at[idx, "status"] = "unemployed"
            df.at[idx, "is_employed"] = False
            df.at[idx, "intention_state"] = "seeking_work"
            df.at[idx, "tpb_active"] = False
            df.at[idx, "intention_delay"] = 0
            df.at[idx, "workplace_district"] = df.at[idx, "district"]

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

            n_fired += 1

    # Remove company → capacity decreases
    change_company_count(G, district, size, -1, industry_shares)

    return n_fired


def _execute_housing_shock(
    district: str,
    magnitude: float,
    G: "nx.DiGraph | None" = None,
    tick_num: int = 0,
) -> None:
    """
    Applies HOUSING_SHOCK to dynamic housing tracker in graph.

    magnitude > 0 → housing decreases (natural disaster, demolition, dilapidated stock).
    magnitude < 0 → housing increases (new construction, renovation).

    Например, magnitude=0.15 при remaining=100 сокращает доступное жильё на 15 единиц.
    """
    if G is None or district not in G.nodes:
        return

    current = G.nodes[district].get("housing_remaining", 0.0)
    delta = current * magnitude  # share of current remaining
    G.nodes[district]["housing_remaining"] = max(0.0, current - delta)

    if abs(delta) > 0.5:
        direction = "сократилось" if magnitude > 0 else "увеличилось"
        print(f"  [tick {tick_num}] HOUSING_SHOCK {district}: "
              f"жильё {direction} на {abs(delta):.1f} "
              f"(magnitude={magnitude:.2f}, остаток={G.nodes[district]['housing_remaining']:.1f})")


# ── v5: company size → jobs mapping ────────────────────────────
_SIZE_LABEL_TO_JOBS = {"small": 25, "medium": 130, "big": 400}


# ── Main tick ───────────────────────────────────────────────────────────

# v2: Constants decay
SB_MOVE_DECAY_PER_TICK = 0.005    # затухание social_boost MOVE за тик
SB_MOVE_TOTAL_TICKS    = 6        # длительность MOVE decay
SB_COMMUTE_TOTAL_TICKS = 3        # длительность COMMUTE до resetа
ECON_PENALTY_DECAY_PER_TICK = 0.01  # затухание econ_penalty за тик
INERTIA_MOB_DECAY_PER_TICK = 0.01  # затухание inertia_mobility_penalty за тик
JOBLOSS_RAMP_UP_TICKS   = 3       # тиков роста econ_gap после LOST_JOB
JOBLOSS_RAMP_DOWN_TICKS = 3       # тиков возврата econ_gap
JOBLOSS_RAMP_STEP       = 0.05    # шаг ramp


def _process_sb_pending(df: pd.DataFrame) -> None:
    """v2: Processes sb_pending queue — social_boost decay (ВЕКТОРИЗОВАНО).

    Формат sb_pending: "M5,M3,C2" — M= MOVE (remaining_ticks), C= COMMUTE (remaining_ticks).
    M-decay: -0.01/тик на каждый активный M-поток.
    C-decay: полный reset +0.02 через 3 тика.
    """
    sb_pending = df["sb_pending"].values
    sb = df["social_boost"].values.copy()
    n = len(df)

    # Collect indices of non-empty pending
    non_empty_mask = np.array([
        (v is not None and str(v) not in ("", "nan", "None"))
        for v in sb_pending
    ], dtype=bool)

    if not non_empty_mask.any():
        return

    indices = np.where(non_empty_mask)[0]
    new_pending = np.empty(n, dtype=object)

    for i in indices:
        val = str(sb_pending[i])
        parts = val.split(",")
        new_parts = []
        m_count = 0  # счётчик активных M-потоков для этого агента

        for p in parts:
            if not p or len(p) < 2:
                continue
            typ = p[0]
            try:
                rem = int(p[1:])
            except ValueError:
                continue

            if typ == 'M':
                m_count += 1
                if rem > 1:
                    new_parts.append(f'M{rem - 1}')
            elif typ == 'C':
                if rem == 1:
                    # Reset after 3 ticks: -0.02
                    sb[i] = max(0.0, sb[i] - 0.02)
                else:
                    new_parts.append(f'C{rem - 1}')

        # Apply accumulated M-decay: -0.01 per active M-flow
        if m_count > 0:
            sb[i] = max(0.0, sb[i] - m_count * SB_MOVE_DECAY_PER_TICK)

        new_pending[i] = ",".join(new_parts) if new_parts else ""

    # For inactive agents — empty string (vectorized)
    new_pending[~non_empty_mask] = ""

    df["sb_pending"] = new_pending
    df["social_boost"] = np.clip(sb, 0.0, 1.0)


def _decay_dynamic_vars(df: pd.DataFrame) -> None:
    """v3: Decay of dynamic signal system variables.

    econ_penalty:              -0.01/тик (до 0)
    infra_bonus:               без автоматического decay (управляется сигналами)
    inertia_mobility_penalty:  decay to 0 from both sides (0.01/тик)
    soc_calibration_signal:    ×0.85/тик (затухание как signal_reduction)
    jobloss_econ_gap_bonus:    ramp up/down (обрабатывается в _process_jobloss_ramp)
    """
    n = len(df)

    # econ_penalty: linear decay to 0
    ep = df["econ_penalty"].values.copy()
    ep = np.maximum(0.0, ep - ECON_PENALTY_DECAY_PER_TICK)
    df["econ_penalty"] = ep

    # inertia_mobility_penalty: decay to 0 from both sides
    # Positive → decrease, negative → increase (trend toward 0)
    imp = df["inertia_mobility_penalty"].values.copy()
    positive = imp > 0
    negative = imp < 0
    imp[positive] = np.maximum(0.0, imp[positive] - INERTIA_MOB_DECAY_PER_TICK)
    imp[negative] = np.minimum(0.0, imp[negative] + INERTIA_MOB_DECAY_PER_TICK)
    df["inertia_mobility_penalty"] = imp

    # soc_calibration_signal: multiplicative decay (как signal_reduction)
    scs = df["soc_calibration_signal"].values.copy()
    scs = scs * SIGNAL_DECAY
    df["soc_calibration_signal"] = np.clip(scs, 0.0, 1.0)


def _process_jobloss_ramp(df: pd.DataFrame) -> None:
    """v2: Processes econ_gap ramp after LOST_JOB (ВЕКТОРИЗОВАНО).

    jobloss_econ_gap_bonus > 0 → фаза ramp-up (+0.05/тик × 3)
    jobloss_econ_gap_bonus < 0 → фаза ramp-down (-0.05/тик × 3)
    """
    bonus = df["jobloss_econ_gap_bonus"].values.copy()
    econ_gap = df["econ_gap"].values.copy()

    # ── Ramp-up phase: bonus > 0.001 ─────────────────────────────────────
    ramp_up = bonus > 0.001
    if ramp_up.any():
        step_up = np.minimum(JOBLOSS_RAMP_STEP, bonus[ramp_up])
        econ_gap[ramp_up] = np.minimum(1.0, econ_gap[ramp_up] + step_up)
        bonus[ramp_up] = np.maximum(0.0, bonus[ramp_up] - step_up)
        # After ramp-up exhausted, transition to ramp-down
        exhausted = bonus < 0.001
        bonus[ramp_up & exhausted] = -JOBLOSS_RAMP_DOWN_TICKS * JOBLOSS_RAMP_STEP

    # ── Ramp-down phase: bonus < -0.001 ──────────────────────────────────
    ramp_down = bonus < -0.001
    if ramp_down.any():
        step_down = np.minimum(JOBLOSS_RAMP_STEP, -bonus[ramp_down])
        econ_gap[ramp_down] = np.maximum(0.0, econ_gap[ramp_down] - step_down)
        bonus[ramp_down] = np.minimum(0.0, bonus[ramp_down] + step_down)
        # Reset on completion
        done = bonus > -0.001
        bonus[ramp_down & done] = 0.0

    df["econ_gap"] = econ_gap
    df["jobloss_econ_gap_bonus"] = bonus


# ── Main tick ───────────────────────────────────────────────────────────

SOCIAL_BOOST_DECAY = 0.60   # множитель затухания social_boost за тик


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
    """One simulation step. Returns updated DataFrame and statistics.

    bus      — сигнальная шина (EventBus), опционально
    scenario — scenario events (Scenario), опционально
    """

    n_agents = len(df)

    df = df.copy()

    # ── COLLECT: scenario events ──────────────────────────────────────────
    if scenario is not None:
        # v5: industry_shares for capacity recalculation (из init_dists)
        industry_shares_map = {}
        if init_dists:
            for d, data in init_dists.items():
                shares = data.get("industry", {})
                if shares:
                    industry_shares_map[d] = shares

        for se in scenario.get_events(tick_num):
            # v5: All events — to bus (агент-сигналы: social_boost, aspirations, etc.)
            if bus is not None:
                bus.emit(se.to_event(tick_num))

            # v5: NEW_EMPLOYER — direct graph handler
            if se.event_type == "NEW_EMPLOYER":
                size_label = se.size or "small"
                shares = industry_shares_map.get(se.district, {})
                n_jobs = _execute_new_employer(
                    district=se.district,
                    industry=se.industry,
                    size=size_label,
                    G=G,
                    industry_shares=shares,
                    tick_num=tick_num,
                )
                print(f"  [tick {tick_num}] NEW_EMPLOYER  {se.district}: "
                      f"industry={se.industry}, размер={size_label}, "
                      f"+{n_jobs} рабочих moт")

            # v5: CLOSED_EMPLOYER — direct handler: layoffs + graph
            if se.event_type == "CLOSED_EMPLOYER":
                size_label = se.size or "small"
                shares = industry_shares_map.get(se.district, {})
                n_fired = _execute_closed_employer(
                    df=df,
                    district=se.district,
                    industry=se.industry,
                    size=size_label,
                    G=G,
                    industry_shares=shares,
                    rng=rng,
                    bus=bus,
                    tick_num=tick_num,
                    n_to_fire=se.n_agents_affected,
                )
                if se.n_agents_affected > 0:
                    n_fire_str = f"n_agents={se.n_agents_affected}"
                else:
                    n_fire_str = f"размер={size_label}"
                print(f"  [tick {tick_num}] CLOSED_EMPLOYER {se.district}: "
                      f"industry={se.industry}, {n_fire_str}, "
                      f"−{n_fired} уволено")

            # v11: Direct housing change for HOUSING_SHOCK
            if se.event_type == "HOUSING_SHOCK":
                _execute_housing_shock(
                    district=se.district,
                    magnitude=se.magnitude,
                    G=G,
                    tick_num=tick_num,
                )

    # 0. v2: sb_pending processing — social_boost decay by new scheme
    _process_sb_pending(df)

    # 0b. v2: Decay of dynamic signal system variables
    _decay_dynamic_vars(df)

    # 1. Time
    df["age"]         = df["age"] + 1 / 12
    df["tenure"]      = df["tenure"] + 1
    df["moved_ticks"] = df["moved_ticks"] + 1

    # 1b. Graduation: batch graduation every 12 ticks by cohorts
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
                # v8: sample industry from residence district distribution
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
                # University graduates (>= 22 years) → high education
                if age_at_grad >= 22 and df.at[idx, "education"] != "high":
                    df.at[idx, "education"] = "high"
                # Inertia decreases — graduate is mobile
                df.at[idx, "inertia"] = float(np.clip(
                    df.at[idx, "inertia"] * 0.60, 0.05, 0.90
                ))
                # Emit to signal bus
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
                    # v2: set jobloss_econ_gap_bonus for ramp
                    df.at[idx, "jobloss_econ_gap_bonus"] = 0.25
            n_grad = int(grad_mask.sum())
            if n_grad > 0:
                print(f"  [tick {tick_num}] Выпустилось студентов (cohort={cohort}): {n_grad}")

    # ── DISPATCH + APPLY (фаза 1): processing graduation events ────────────
    if bus is not None:
        signals = bus.process(tick_num, df, G)
        if signals:
            df = bus.flush(df, signals, G)

    # 1c. Per-tick updates before two-barrier check
    # ── signal_reduction: decay ─────────────────────────────────────────
    df["signal_reduction"] = df["signal_reduction"].values * SIGNAL_DECAY

    # ── weak_ties_utility: bonus for being in a hub ──────────────────────
    hub_mask = df["workplace_district"].isin(HUB_DISTRICTS).values
    if hub_mask.any():
        df.loc[hub_mask, "weak_ties_utility"] = np.clip(
            df.loc[hub_mask, "weak_ties_utility"] + HUB_WEAK_TIES_BONUS, 0.0, 1.0
        )

    # ── econ_gap: perception adaptation to reality (industry wage) ─────
    # Векторизовано: collect industry wages for each agent workplace
    n = len(df)
    wp_districts = df["workplace_district"].values
    agent_wages = df["wage"].values
    agent_inds = df["industry"].values
    old_econ_gaps = df["econ_gap"].values

    # Build mapping: district → {industry: wage}
    # Pre-compute per-agent industry wage in their workplace (ВЕКТОРИЗОВАНО)
    industry_wages_wp = np.full(n, NATIONAL_AVG_WAGE, dtype=float)
    # Build dictionary for unique (district, industry) pairs
    unique_pairs = set(zip(wp_districts, agent_inds))
    wage_lookup = {}
    for wp, ind in unique_pairs:
        wage_lookup[(wp, ind)] = _industry_wage_in_district(G, wp, ind)
    # Vectorized fill via list comprehension (faster than .apply)
    industry_wages_wp = np.array([
        wage_lookup.get((wp_districts[i], agent_inds[i]), NATIONAL_AVG_WAGE)
        for i in range(n)
    ], dtype=float)

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

    # v2: LOST_JOB ramp processing (econ_gap + jobloss_econ_gap_bonus)
    _process_jobloss_ramp(df)

    # ── place_deficit_penalty: penalty accumulation for place dissatisfaction (ВЕКТОРИЗОВАНО) ─
    res_districts = df["district"].values
    agent_wages = df["wage"].values
    domain_futures = df["domain_future_place"].values
    old_penalties = df["place_deficit_penalty"].values.copy()

    # Precompute housing_price and infra_score for all unique districts
    unique_res = set(res_districts)
    hp_map = {d: get_effective_housing_price(G, d) for d in unique_res}
    infra_map = {d: get_effective_infrastructure(G, d) for d in unique_res}

    # Vectorized place_reality calculation for all agents
    housing_prices = np.array([hp_map[d] for d in res_districts], dtype=float)
    infra_scores = np.array([infra_map[d] for d in res_districts], dtype=float)
    safe_wages = np.maximum(agent_wages, 1.0)
    monthly_costs = housing_prices * 50.0 * 0.004
    burdens = monthly_costs / safe_wages
    affordability = np.maximum(0.0, 1.0 - burdens / 0.35)
    place_reality = 0.6 * affordability + 0.4 * infra_scores

    # Penalty when domain_future > place_reality
    deficit_mask = domain_futures > place_reality
    new_penalties = old_penalties.copy()

    if deficit_mask.any():
        gap_pct = (domain_futures[deficit_mask] - place_reality[deficit_mask]) / np.maximum(place_reality[deficit_mask], 0.001)
        new_penalties[deficit_mask] = np.clip(
            old_penalties[deficit_mask] + gap_pct * 0.02 / 6.0, 0.0, 0.2
        )

    # Decay if place is satisfactory
    satisfied_mask = ~deficit_mask
    if satisfied_mask.any():
        new_penalties[satisfied_mask] = np.maximum(0.0, old_penalties[satisfied_mask] - 0.01)

    df["place_deficit_penalty"] = new_penalties

    # 2. Labor market pressure (v3: использует G.nodes industry_jobs, fallback на jobs_capacity)
    jobs_pressure = _compute_jobs_pressure(df, jobs_capacity, G)

    # 3. FFT pipeline (emits agent events to bus)
    df, fft_stats, action_log = _fft_pipeline(df, G, jobs_pressure, rng,
                                                 bus=bus, tick_num=tick_num)

    # ── DISPATCH + APPLY (фаза 2): processing agent movement events ───
    if bus is not None:
        signals = bus.process(tick_num, df, G)
        if signals:
            df = bus.flush(df, signals, G)

    # 7. Environment response (residence counts for housing, workplace counts for wages)
    residence_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, residence_counts, n_agents)

    # 9. Statistics
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


# ── Simulation ──────────────────────────────────────────────────────────────

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

    Принимает jobs_capacity явно — строится в agents.py from commuting matrix
    и передаётся через run.py.
    init_dists — распределения из agent_init_distributions.json (для graduation).

    bus      — сигнальная шина (EventBus), опционально
    scenario — scenario events (Scenario), опционально

    Возвращает (df_final, snapshots, tick_stats, all_action_log).
    """
    rng = np.random.default_rng(seed)

    if snapshot_ticks is None:
        snapshot_ticks = [0, n_ticks // 4, n_ticks // 2, n_ticks]

    # Initialize graph with real counts
    residence_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, residence_counts, len(df))

    # v11: Инициализируем динамический трекер жилья in graph
    init_housing_remaining(G, len(df))

    snapshots  = {}
    tick_stats = []
    all_action_log = []  # агрегированный лог решений за все тики

    if 0 in snapshot_ticks:
        snapshots[0] = df.copy()

    if verbose:
        print(f"\nSimulation: {n_ticks} ticks | {len(df):,} agents | {G.number_of_nodes()} districts")
        print(f"{'Tick':>5} {'Year':>4} "
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
        print(f"  Итого new commutes:    {total_commutes:,}")
        print(f"  Итого спутник-переездов:{total_sat:,}")
        print(f"  Итого адаптаций:        {total_adapts:,}")
        print(f"  Unemployed в конце:    {tick_stats[-1]['n_unemployed']:,}")

    return df, snapshots, tick_stats, all_action_log
