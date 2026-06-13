"""
engine.py v7.1 — Two-Barrier Activation + FFT Decision Trees (векторизованные обновления econ_gap / domain_future_place)

Архитектура принятия решений (v7):

  ДВУХБАРЬЕРНАЯ МОДЕЛЬ АКТИВАЦИИ:
    Барьер 1 — Потенциал миграции против динамической инерции:
      Aspirations (EWMA от D_instant) × Capabilities > dynamic_inertia.
    Барьер 2 — Теория запланированного поведения (TPB):
      - Attitude = D_instant (нормированная мгновенная неудовлетворённость economic+place)
      - Subjective norm = f(social_boost, family_weight_modifier)
        Social и family домены влияют исключительно через subjective_norm
        и НЕ могут самостоятельно активировать миграцию.
      - Perceived behavioral control = (econ_perceived_control + perceived_control) / 2
      - Intention = (Attitude + SubjectiveNorm + PBC) / 3
      Если Intention > internal_mig_threshold → intention_delay = 1–3 тика.
      Доминантный домен (economic / place) определяется по компонентам D_econ и D_place.
      После истечения задержки → intention_state = "seeking_work" | "seeking_residence".

  FFT ДЕРЕВЬЯ РЕШЕНИЙ (после активации):
    ФИЛЬТР 2 — поиск работы (economic-driven):
      Эвристика зарплатных ожиданий с отраслевой привязкой.
      Скрининг awareness_set по двум аспектам:
        А: отраслевая зарплата > target_wage
        Б: jobs_pressure < MAX_JOBS_PRESSURE
      → dst_work найден / не найден.
      При неудаче возможна адаптация на месте.

    ФИЛЬТР 3a — локация для economic-driven (после фильтра 2):
      Путь А — COMMUTE, Путь Б — MOVE, Путь В — SATELLITE MOVE.
      Выбор спутника – стохастический satisficing среди доступных вариантов.

    ФИЛЬТР 3b — place-driven поиск жилья (минуя фильтр 2):
      Расширенный радиус поездки, сравнение target_place.
      Переезд с сохранением текущего места работы.

  ОБНОВЛЕНИЕ ДОМЕНОВ:
    Economic: от workplace_district.
    Place: от residence_district.
    Social: target = 0.5 + social_boost, сглаживание.
    Family: commute-давление.
    Dissatisfaction вычисляется для мониторинга, но не используется при активации.

  СОБЫТИЙНЫЕ СИГНАЛЫ (Блок B):
    social_boost затухает ×0.8/тик.
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional

from graph import update_graph, get_awareness_set

# ── Константы ─────────────────────────────────────────────────────────────────

# Фильтр 2 — дерево занятости
MAX_JOBS_PRESSURE        = 1.20   # район считается перегруженным выше этого порога
MAX_WORK_CANDIDATES      = 12     # максимум районов для скрининга

# Фильтр 3 — дерево локации
SATELLITE_SEARCH_RADIUS  = 90.0   # мин. — радиус поиска района-спутника от dst_work
HOUSING_BUDGET_RATIO     = 0.35   # жильё не должно превышать X доли зарплаты (×100м²)
MOVE_STRESS_FACTOR       = 0.80   # satisfaction после переезда × этот множитель
PLACE_DRIVEN_TRAVEL_BUF  = 1.40   # множитель к commuter_threshold для place-driven поиска

# Adapt
ADAPT_FLEX_THRESHOLD     = 0.65   # минимальная job_flexibility для адаптации
ADAPT_SAT_BOOST          = 0.06   # прирост sat_economic при адаптации

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
GAP_ADAPT_LAMBDA         = 0.05   # скорость адаптации econ_gap и domain_future_place
HUB_WEAK_TIES_BONUS      = 0.005  # прирост weak_ties_utility за тик в хабах
MOVE_WEAK_TIES_PENALTY   = -0.10  # сброс weak_ties при переезде
UNEMPLOYED_SIGNAL        = 0.35   # добавка к signal_reduction при потере работы
NEIGHBOR_SIGNAL_COEF     = 0.04   # коэфф. сигнала от переехавшего соседа

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
    budget_ratio: float = HOUSING_BUDGET_RATIO,
) -> bool:
    """
    Жильё доступно если ежемесячный платёж (аренда ~0.4% цены 50м²)
    не превышает budget_ratio от зарплаты.
    Грубая эвристика — агент не считает ипотеку, он смотрит на порядок цифр.
    """
    if agent_wage <= 0:
        return False
    monthly_cost = housing_price_m2 * 50 * 0.004
    return monthly_cost <= agent_wage * budget_ratio


# ── Обновление доменов ────────────────────────────────────────────────────────

def update_domain_satisfaction(
    df: pd.DataFrame,
    G: nx.DiGraph,
    jobs_pressure: dict,
) -> pd.DataFrame:
    """
    Обновляет satisfaction по четырём доменам.

    Ключевое изменение v4:
      Economic домен считается от workplace_district агента,
      а не от residence_district. Агент из Сеницы работающий в Братиславе
      получает экономический сигнал Братиславы.
      Place домен — от residence_district (где живёт, там и среда).
    """
    sat_econ   = df["sat_economic"].values.copy()
    sat_social = df["sat_social"].values.copy()
    sat_family = df["sat_family"].values.copy()
    sat_place  = df["sat_place"].values.copy()

    wages      = df["wage"].values
    statuses   = df["status"].values
    workplaces = df["workplace_district"].values
    residences = df["district"].values   # residence

    for i in range(len(df)):
        wp = workplaces[i]
        res = residences[i]

        wp_attr  = G.nodes.get(wp, {})
        res_attr = G.nodes.get(res, {})

        avg_wage_wp  = wp_attr.get("avg_wage", NATIONAL_AVG_WAGE)
        housing_res  = res_attr.get("housing_price_m2", 1800)
        infra_res    = res_attr.get("infrastructure_score", 0.5)
        pressure_wp  = jobs_pressure.get(wp, 1.0)

        w = wages[i]

        # ── Economic (от workplace) ───────────────────────────────────────────
        if statuses[i] == "student":
            sat_econ[i] = float(np.clip(
                SAT_SMOOTHING * sat_econ[i] + (1 - SAT_SMOOTHING) * 0.48,
                0.0, 1.0
            ))
        else:
            q_wage    = (w - avg_wage_wp) / max(avg_wage_wp, 1)
            q_employ  = float(np.clip(1.0 - (pressure_wp - 1.0), -0.5, 0.5))
            raw_econ  = 0.65 * q_wage + 0.35 * q_employ
            target_econ = _sigmoid(raw_econ)
            sat_econ[i] = float(np.clip(
                SAT_SMOOTHING * sat_econ[i] + (1 - SAT_SMOOTHING) * target_econ,
                0.0, 1.0
            ))

        # ── Social (Блок A: target=0.5+boost, сглаживание α=0.88) ──────────
        social_boost = float(df["social_boost"].values[i])
        target_social = float(np.clip(0.50 + social_boost, 0.05, 0.95))
        sat_social[i] = float(np.clip(
            SAT_SMOOTHING * sat_social[i] + (1 - SAT_SMOOTHING) * target_social,
            0.0, 1.0
        ))

        # ── Family (Блок F: commute-давление) ────────────────────────────────
        target_family = 0.50
        if statuses[i] == "commute":
            res = residences[i]
            wp  = workplaces[i]
            travel_time = 999
            if G.has_edge(res, wp):
                travel_time = G[res][wp].get("travel_time_min", 999)
            comm_thr_norm = float(df["commuter_threshold"].values[i])
            comm_thr_min  = 30.0 + 90.0 * comm_thr_norm
            if travel_time > comm_thr_min:
                excess_ratio = min(1.0, (travel_time - comm_thr_min) / comm_thr_min)
                target_family = float(np.clip(0.50 - excess_ratio * 0.25, 0.10, 0.90))
        sat_family[i] = float(np.clip(
            SAT_SMOOTHING * sat_family[i] + (1 - SAT_SMOOTHING) * target_family,
            0.0, 1.0
        ))

        # ── Place (от residence: жильё + инфраструктура) ──────────────────────
        q_housing = (1800 - housing_res) / 1800
        target_place = _sigmoid(0.5 * q_housing + 0.5 * (infra_res - 0.5))
        sat_place[i] = float(np.clip(
            SAT_SMOOTHING * sat_place[i] + (1 - SAT_SMOOTHING) * target_place,
            0.0, 1.0
        ))

    df = df.copy()
    df["sat_economic"] = np.clip(sat_econ, 0.0, 1.0)
    df["sat_social"]   = np.clip(sat_social, 0.0, 1.0)
    df["sat_family"]   = np.clip(sat_family, 0.0, 1.0)
    df["sat_place"]    = np.clip(sat_place, 0.0, 1.0)
    return df


def compute_dissatisfaction(df: pd.DataFrame) -> np.ndarray:
    """
    Взвешенная dissatisfaction по активным доменам.
    Домен активен если value < threshold.
    (Используется только для мониторинга, не для активации.)
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
    return np.clip(np.sqrt(dissat), 0.0, 1.0)


def _compute_jobs_pressure(df: pd.DataFrame, jobs_capacity: dict) -> dict:
    """
    jobs_pressure[district] = число занятых агентов с workplace=district
                              / jobs_capacity[district]
    Значение > 1.0 означает перегрузку рынка труда.
    """
    wp_counts = (df[df["is_employed"]]
                 .groupby("workplace_district")["id"]
                 .count()
                 .to_dict())
    pressure = {}
    for d, cap in jobs_capacity.items():
        cnt = wp_counts.get(d, 0)
        pressure[d] = cnt / max(cap, 1)
    return pressure


# ── Двухбарьерная модель: Барьер 1 — Потенциал против динамической инерции ──

def _compute_d_instant(
    agent_wage: float,
    industry_avg_wage_wp: float,
    econ_gap: float,
    job_flexibility: float,
    housing_price_m2: float,
    infrastructure_score: float,
    domain_future_place: float,
    w_econ: float,
    w_future: float,
) -> tuple[float, float, float]:
    """
    Вычисляет мгновенную неудовлетворённость D_instant.

    Returns (D_instant, D_econ, D_place).
    """
    if agent_wage > 0 and industry_avg_wage_wp > 0:
        wage_pressure = max(0.0, 1.0 - agent_wage / industry_avg_wage_wp)
    else:
        wage_pressure = 1.0
    D_econ = w_econ * wage_pressure * econ_gap * (1.0 - job_flexibility)

    monthly_cost = housing_price_m2 * 50 * 0.004
    burden = monthly_cost / max(agent_wage, 1.0)
    affordability = max(0.0, 1.0 - burden / HOUSING_BUDGET_RATIO)
    place_reality = 0.6 * affordability + 0.4 * infrastructure_score
    D_place = w_future * max(0.0, domain_future_place - place_reality)

    D_instant = D_econ + D_place
    return float(np.clip(D_instant, 0.0, 1.0)), float(D_econ), float(D_place)


def _two_barrier_activation(
    df: pd.DataFrame,
    G: nx.DiGraph,
    dissatisfaction: np.ndarray,  # сохраняется для совместимости; не используется внутри
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Двухбарьерная модель активации (Aspirations×Capabilities → TPB → Эвристический поиск).

    Social и family домены влияют ТОЛЬКО через subjective_norm
    и не могут самостоятельно активировать миграционное намерение.

    Барьер 1 — Потенциал миграции против динамической инерции.
    Барьер 2 — Теория запланированного поведения (TPB).

    Возвращает обновлённый DataFrame.
    """
    df = df.copy()
    n = len(df)

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
    intention_del  = df["intention_delay"].values.copy()
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
    family_mods    = df["family_weight_modifier"].values
    social_boosts  = df["social_boost"].values
    maritals       = df["marital"].values
    internal_thrs  = df["internal_mig_thr"].values
    moved_ticks    = df["moved_ticks"].values
    intention_states = df["intention_state"].values.copy()

    activation_domains = df["activation_domain"].values.copy()

    for i in range(n):
        if statuses[i] == "student":
            continue
        if moved_ticks[i] < 9:
            tpb_active[i] = False
            intention_del[i] = 0
            continue
        if ages[i] < 18 or ages[i] > 62:
            tpb_active[i] = False
            intention_del[i] = 0
            continue

        wp = workplaces[i]
        res = residences[i]
        wp_attr  = G.nodes.get(wp, {})
        res_attr = G.nodes.get(res, {})

        industry_avg_wp = wp_attr.get("avg_wage", NATIONAL_AVG_WAGE)
        housing_price   = res_attr.get("housing_price_m2", 1800.0)
        infra_score     = res_attr.get("infrastructure_score", 0.5)

        D_inst, D_econ, D_place = _compute_d_instant(
            agent_wage=wages[i],
            industry_avg_wage_wp=industry_avg_wp,
            econ_gap=econ_gaps[i],
            job_flexibility=job_flexs[i],
            housing_price_m2=housing_price,
            infrastructure_score=infra_score,
            domain_future_place=domain_future[i],
            w_econ=w_econs[i],
            w_future=w_futures[i],
        )

        aspirations[i] = ASPIRATIONS_ALPHA * D_inst + (1.0 - ASPIRATIONS_ALPHA) * aspirations[i]

        income_index = min(wages[i] / (1.5 * NATIONAL_AVG_WAGE), 1.0)
        edu_map = {"low": 0.25, "medium": 0.55, "high": 0.85}
        education_index = edu_map.get(educations[i], 0.55)
        capabilities = (income_index + education_index + weak_ties[i]) / 3.0

        dynamic_inertia = inertias[i] * max(0.3, 1.0 - signal_red[i])

        if aspirations[i] * capabilities > dynamic_inertia:
            if not tpb_active[i]:
                tpb_active[i] = True

                # Доминантный домен: только economic или place.
                # Social / family не участвуют в определении домена.
                if D_econ >= D_place:
                    activation_domains[i] = "economic"
                else:
                    activation_domains[i] = "place"

                # Attitude — нормированная мгновенная неудовлетворённость
                attitude = D_inst

                # Subjective norm: влияние social и family доменов
                social_pressure = net_susc[i] * social_boosts[i]
                family_pressure_val = -family_mods[i] * (1.0 if maritals[i] == "married" else 0.5)
                subjective_norm = float(np.clip(0.5 + social_pressure + family_pressure_val, 0.0, 1.0))

                pbc = (econ_percontrols[i] + percontrols[i]) / 2.0
                intention = (attitude + subjective_norm + pbc) / 3.0

                if intention > internal_thrs[i]:
                    intention_del[i] = max(1, 3 - int(2.0 * percontrols[i]))
                else:
                    tpb_active[i] = False
                    intention_del[i] = 0
        else:
            tpb_active[i] = False
            intention_del[i] = 0

        if tpb_active[i] and intention_del[i] > 0:
            intention_del[i] -= 1
            if intention_del[i] == 0:
                dom = activation_domains[i]
                if dom == "economic":
                    intention_states[i] = "seeking_work"
                elif dom == "place":
                    intention_states[i] = "seeking_residence"
                else:
                    intention_states[i] = "none"

    df["aspirations"]        = np.clip(aspirations, 0.0, 1.0)
    df["signal_reduction"]   = np.clip(signal_red, 0.0, 1.0)
    df["tpb_active"]         = tpb_active
    df["intention_delay"]    = intention_del
    df["econ_gap"]           = np.clip(econ_gaps, 0.0, 1.0)
    df["domain_future_place"] = np.clip(domain_future, 0.0, 1.0)
    df["intention_state"]    = intention_states
    df["activation_domain"]  = activation_domains

    return df


# ── FFT: Фильтр 2 — Дерево занятости ─────────────────────────────────────────

def _fft_filter2_find_work(
    district: str,
    agent_wage: float,
    agent_industry: str,
    econ_perceived_control: float,
    sat_economic: float,
    thr_economic: float,
    G: nx.DiGraph,
    jobs_pressure: dict,
    network_location: bool,
    perceived_control: float,
    rng: np.random.Generator,
) -> Optional[str]:
    """
    Ищет dst_work — район где агент может найти работу лучше текущей.

    v8: зарплатные ожидания привязаны к ОТРАСЛИ агента.
    Целевая зарплата — поведенческая эвристика.
    """
    def _industry_wage(node_attr: dict, industry: str) -> float:
        sal = node_attr.get("salary_by_industry", {})
        if sal:
            return float(sal.get(industry, node_attr.get("avg_wage", NATIONAL_AVG_WAGE)))
        return float(node_attr.get("avg_wage", NATIONAL_AVG_WAGE))

    if agent_wage > 0:
        base_appetite = BASE_APPETITE_MIN + econ_perceived_control * BASE_APPETITE_MAX
        thr = max(thr_economic, 0.01)
        desperation = float(np.clip((thr_economic - sat_economic) / thr, 0.0, 1.0))
        desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE) * (1.0 - desperation)
        target_wage = agent_wage * (1.0 + desired_raise)
    else:
        home_attr = G.nodes.get(district, {})
        home_ind_wage = _industry_wage(home_attr, agent_industry)
        base_appetite = BASE_APPETITE_MIN + econ_perceived_control * BASE_APPETITE_MAX
        thr = max(thr_economic, 0.01)
        desperation = float(np.clip((thr_economic - sat_economic) / thr, 0.0, 1.0))
        desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE) * (1.0 - desperation)
        target_wage = (home_ind_wage * 0.85) * (1.0 + desired_raise)

    home_attr = G.nodes.get(district, {})
    home_wage = _industry_wage(home_attr, agent_industry)
    home_pressure = jobs_pressure.get(district, 0)
    if home_wage >= target_wage and home_pressure < MAX_JOBS_PRESSURE:
        return district

    candidates = get_awareness_set(
        G, district,
        network_location=network_location,
        perceived_control=perceived_control,
        max_candidates=MAX_WORK_CANDIDATES,
    )
    candidates = list(set(candidates) | {district})
    rng.shuffle(candidates)

    for dst in candidates:
        dst_attr = G.nodes.get(dst, {})
        dst_wage = _industry_wage(dst_attr, agent_industry)
        if dst_wage < target_wage:
            continue
        if jobs_pressure.get(dst, 0) >= MAX_JOBS_PRESSURE:
            continue
        return dst
    return None


# ── Place-driven search: поиск жилья при доминанте place ───────────────────

def _place_driven_search(
    residence: str,
    workplace: str,
    agent_wage: float,
    G: nx.DiGraph,
    network_location: bool,
    perceived_control: float,
    commuter_threshold_min: float,
    rng: np.random.Generator,
) -> Optional[str]:
    """
    Ищет новый район проживания, сохраняя текущее место работы.
    """
    expanded_threshold = commuter_threshold_min * PLACE_DRIVEN_TRAVEL_BUF

    current_attr = G.nodes.get(residence, {})
    cur_h = current_attr.get("housing_price_m2", 1800)
    cur_i = current_attr.get("infrastructure_score", 0.5)
    cur_target = _sigmoid(0.5 * (1800 - cur_h) / 1800 + 0.5 * (cur_i - 0.5))

    candidates = get_awareness_set(
        G, workplace,
        network_location=network_location,
        perceived_control=perceived_control,
        max_candidates=MAX_WORK_CANDIDATES,
    )
    candidates = list(set(candidates) | {residence, workplace})
    rng.shuffle(candidates)

    for dst in candidates:
        if dst == residence:
            continue
        dst_attr = G.nodes.get(dst, {})
        housing_price = dst_attr.get("housing_price_m2", 9999)
        if not _housing_affordable(agent_wage, housing_price):
            continue
        if G.has_edge(dst, workplace):
            tt = G[dst][workplace].get("travel_time_min", 999)
            if tt > expanded_threshold:
                continue
        else:
            continue
        dst_infra = dst_attr.get("infrastructure_score", 0.5)
        dst_target = _sigmoid(0.5 * (1800 - housing_price) / 1800 + 0.5 * (dst_infra - 0.5))
        if dst_target <= cur_target + 0.02:
            continue
        return dst
    return None


# ── FFT: Фильтр 3 — Дерево локации ───────────────────────────────────────────

def _fft_filter3_find_residence(
    residence: str,
    dst_work: str,
    agent_wage: float,
    G: nx.DiGraph,
    rng: np.random.Generator,
    commuter_threshold_min: float,
    sat_place: float,
    thr_place: float,
):
    """
    Возвращает (new_residence, outcome):
      outcome: "commute" | "move" | "satellite_move" | "none"

    Путь А — COMMUTE, Путь Б — MOVE, Путь В — SATELLITE MOVE (стохастический выбор).
    """
    if G.has_edge(residence, dst_work):
        tt = G[residence][dst_work].get("travel_time_min", 999)
    else:
        tt = 999

    place_ok = sat_place >= thr_place * 0.80
    if tt <= commuter_threshold_min and place_ok:
        return residence, "commute"

    dst_attr = G.nodes.get(dst_work, {})
    housing_dst = dst_attr.get("housing_price_m2", 9999)
    if _housing_affordable(agent_wage, housing_dst):
        return dst_work, "move"

    satellites = []
    for _, neighbor, attr in G.out_edges(dst_work, data=True):
        if neighbor == dst_work:
            continue
        tt_to_work = attr.get("travel_time_min", 999)
        if tt_to_work > commuter_threshold_min:
            continue
        neighbor_attr = G.nodes.get(neighbor, {})
        housing_neighbor = neighbor_attr.get("housing_price_m2", 9999)
        if _housing_affordable(agent_wage, housing_neighbor):
            satellites.append((neighbor, tt_to_work, housing_neighbor))

    if satellites:
        # Стохастический satisficing: перемешиваем подходящие спутники
        rng.shuffle(satellites)
        best_satellite = satellites[0][0]
        return best_satellite, "satellite_move"

    return residence, "none"


# ── Исполнение решений ────────────────────────────────────────────────────────

def _execute_commute(
    df: pd.DataFrame,
    idx: int,
    new_workplace: str,
    G: nx.DiGraph,
    rng: np.random.Generator,
):
    """Агент меняет место работы без смены жительства.
       moved_ticks НЕ сбрасывается — переезда не было."""
    df.at[idx, "workplace_district"] = new_workplace
    df.at[idx, "status"]             = "commute"
    df.at[idx, "intention_state"]    = "none"
    df.at[idx, "dst_work"]           = ""
    df.at[idx, "tpb_active"]         = False
    df.at[idx, "intention_delay"]    = 0

    agent_industry = str(df.at[idx, "industry"])
    wp_attr = G.nodes.get(new_workplace, {})
    wp_salary_by_ind = wp_attr.get("salary_by_industry", {})
    base_wage = wp_salary_by_ind.get(agent_industry, wp_attr.get("avg_wage", df.at[idx, "wage"]))
    new_wage = float(max(0, rng.normal(base_wage, base_wage * 0.18)))
    df.at[idx, "wage"] = new_wage

    df.at[idx, "weak_ties_utility"] = float(np.clip(
        df.at[idx, "weak_ties_utility"] + 0.04, 0.0, 1.0
    ))


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
    Стресс переезда, сброс инерции, moved_ticks=0.
    """
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

    df.at[idx, "weak_ties_utility"] = float(np.clip(
        df.at[idx, "weak_ties_utility"] + MOVE_WEAK_TIES_PENALTY, 0.0, 1.0
    ))

    agent_industry = str(df.at[idx, "industry"])
    wp_attr = G.nodes.get(new_workplace, {})
    wp_salary_by_ind = wp_attr.get("salary_by_industry", {})
    base_wage = wp_salary_by_ind.get(agent_industry, wp_attr.get("avg_wage", df.at[idx, "wage"]))
    new_wage = float(max(0, rng.normal(base_wage, base_wage * 0.18)))
    df.at[idx, "wage"] = new_wage

    for col in ["sat_economic", "sat_social", "sat_family", "sat_place"]:
        df.at[idx, col] = float(np.clip(df.at[idx, col] * MOVE_STRESS_FACTOR, 0.05, 0.95))

    new_h = G.nodes[new_residence].get("housing_price_m2", 1800)
    new_i = G.nodes[new_residence].get("infrastructure_score", 0.5)
    new_target = _sigmoid(0.5 * (1800 - new_h) / 1800 + 0.5 * (new_i - 0.5))
    df.at[idx, "sat_place"] = float(np.clip(
        df.at[idx, "sat_place"] * 0.5 + new_target * 0.5, 0.05, 0.95
    ))

    new_inertia = float(np.clip(
        df.at[idx, "inertia_social"] * 0.30 + 0.10,
        0.05, 0.90
    ))
    df.at[idx, "inertia"] = new_inertia

    df.at[idx, "housing_price_m2"] = float(
        G.nodes[new_residence].get("housing_price_m2", df.at[idx, "housing_price_m2"])
    )


def _execute_adapt(df: pd.DataFrame, idx: int, domain: str = "economic"):
    """
    Агент адаптируется на месте: снижает притязания, экономит.
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
        df.at[idx, "thr_place"] = float(np.clip(
            df.at[idx, "thr_place"] - ADAPT_SAT_BOOST * 0.5, 0.10, 0.85
        ))
    df.at[idx, "intention_state"] = "none"
    df.at[idx, "dst_work"]        = ""
    df.at[idx, "tpb_active"]      = False
    df.at[idx, "intention_delay"] = 0


# ── Главный FFT pipeline ──────────────────────────────────────────────────────

def _fft_pipeline(
    df: pd.DataFrame,
    G: nx.DiGraph,
    dissatisfaction: np.ndarray,
    jobs_pressure: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict, list]:
    """
    Полный FFT pipeline за один тик.

    Проход 1: Двухбарьерная активация.
    Проход 2: Фильтр 2 (поиск работы).
    Проход 3a: Фильтр 3 для economic-driven.
    Проход 3b: Place-driven поиск жилья.
    """
    df = df.copy()
    stats = {
        "moves": 0, "commutes": 0, "adapts": 0,
        "activated": 0, "dst_found": 0, "satellite_moves": 0,
        "econ_driven_moves": 0, "place_driven_moves": 0,
        "econ_activated": 0, "place_activated": 0,
    }
    action_log = []

    def _snapshot(idx):
        row = df.iloc[idx]
        thr_e = max(float(row["thr_economic"]), 0.01)
        sat_e = float(row["sat_economic"])
        return {
            "id":                  int(row["id"]),
            "agent_type":          str(row["agent_type"]),
            "activation_domain":   str(row["activation_domain"]),
            "prev_residence":      str(row["residence_district"]),
            "prev_workplace":      str(row["workplace_district"]),
            "industry":            str(row["industry"]),
            "domain_economic_gap": round(float(np.clip((thr_e - sat_e) / thr_e, 0.0, 1.0)), 4),
        }

    # Проход 1: Двухбарьерная активация
    intention_before = df["intention_state"].values.copy()
    df = _two_barrier_activation(df, G, dissatisfaction, rng)

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

    # Проход 2: Фильтр 2 (seeking_work)
    seeking_work_idx = np.where(df["intention_state"].values == "seeking_work")[0]
    for idx in seeking_work_idx:
        row      = df.iloc[idx]
        district = str(row["residence_district"])
        dst_work = _fft_filter2_find_work(
            district              = district,
            agent_wage            = float(row["wage"]),
            agent_industry        = str(row["industry"]),
            econ_perceived_control = float(row["econ_perceived_control"]),
            sat_economic          = float(row["sat_economic"]),
            thr_economic          = float(row["thr_economic"]),
            G                     = G,
            jobs_pressure         = jobs_pressure,
            network_location      = bool(row["network_location"]),
            perceived_control     = float(row["perceived_control"]),
            rng                   = rng,
        )

        if dst_work is None:
            primary_econ_pressure = (
                float(row["w_economic"]) *
                max(0, float(row["thr_economic"]) - float(row["sat_economic"])) /
                max(float(row["thr_economic"]), 0.01)
            )
            other_pressure = dissatisfaction[idx] - primary_econ_pressure
            if (primary_econ_pressure > other_pressure and
                    float(row["job_flexibility"]) > ADAPT_FLEX_THRESHOLD):
                snap = _snapshot(idx)
                _execute_adapt(df, idx, domain="economic")
                snap["decision"] = "adapt"
                snap["new_residence"] = str(df.at[idx, "residence_district"])
                snap["new_workplace"] = str(df.at[idx, "workplace_district"])
                snap["wage"] = float(df.at[idx, "wage"])
                snap["desired_raise"] = 0.0
                action_log.append(snap)
                stats["adapts"] += 1
            else:
                df.at[idx, "intention_state"] = "none"
                df.at[idx, "tpb_active"]      = False
                df.at[idx, "intention_delay"] = 0
                df.at[idx, "inertia"] = float(np.clip(
                    df.at[idx, "inertia"] + 0.03, 0.05, 0.95
                ))
        else:
            df.at[idx, "dst_work"]        = dst_work
            df.at[idx, "intention_state"] = "seeking_residence"
            stats["dst_found"] += 1

    # Проход 3a: Фильтр 3 для economic-driven
    seeking_res_idx = np.where(
        (df["intention_state"].values == "seeking_residence") &
        (df["dst_work"].values != "")
    )[0]
    for idx in seeking_res_idx:
        row       = df.iloc[idx]
        residence = str(row["residence_district"])
        dst_work  = str(row["dst_work"])

        if not dst_work:
            df.at[idx, "intention_state"] = "none"
            df.at[idx, "tpb_active"]      = False
            df.at[idx, "intention_delay"] = 0
            continue

        comm_thr_norm = float(row["commuter_threshold"])
        comm_thr_min  = 30.0 + 90.0 * comm_thr_norm

        new_residence, outcome = _fft_filter3_find_residence(
            residence              = residence,
            dst_work               = dst_work,
            agent_wage             = float(row["wage"]),
            G                      = G,
            rng                    = rng,
            commuter_threshold_min = comm_thr_min,
            sat_place              = float(row["sat_place"]),
            thr_place              = float(row["thr_place"]),
        )

        if outcome == "commute":
            snap = _snapshot(idx)
            agent_w = float(row["wage"])
            thr_e = max(float(row["thr_economic"]), 0.01)
            sat_e = float(row["sat_economic"])
            epc   = float(row["econ_perceived_control"])
            if agent_w > 0:
                base_appetite = BASE_APPETITE_MIN + epc * BASE_APPETITE_MAX
                desperation = float(np.clip((thr_e - sat_e) / thr_e, 0.0, 1.0))
                desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE) * (1.0 - desperation)
            else:
                desired_raise = 0.0
            _execute_commute(df, idx, dst_work, G, rng)
            snap["decision"]      = "commute"
            snap["new_residence"] = str(df.at[idx, "residence_district"])
            snap["new_workplace"] = str(df.at[idx, "workplace_district"])
            snap["wage"]          = float(df.at[idx, "wage"])
            snap["desired_raise"] = round(desired_raise, 4)
            action_log.append(snap)
            stats["commutes"] += 1

        elif outcome in ("move", "satellite_move"):
            snap = _snapshot(idx)
            agent_w = float(row["wage"])
            thr_e = max(float(row["thr_economic"]), 0.01)
            sat_e = float(row["sat_economic"])
            epc   = float(row["econ_perceived_control"])
            if agent_w > 0:
                base_appetite = BASE_APPETITE_MIN + epc * BASE_APPETITE_MAX
                desperation = float(np.clip((thr_e - sat_e) / thr_e, 0.0, 1.0))
                desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE) * (1.0 - desperation)
            else:
                desired_raise = 0.0
            _execute_move(df, idx, new_residence, dst_work, G, rng)
            snap["decision"]      = outcome
            snap["new_residence"] = str(df.at[idx, "residence_district"])
            snap["new_workplace"] = str(df.at[idx, "workplace_district"])
            snap["wage"]          = float(df.at[idx, "wage"])
            snap["desired_raise"] = round(desired_raise, 4)
            action_log.append(snap)
            stats["moves"] += 1
            stats["econ_driven_moves"] += 1
            if outcome == "satellite_move":
                stats["satellite_moves"] += 1

        else:
            if float(row["job_flexibility"]) > ADAPT_FLEX_THRESHOLD:
                snap = _snapshot(idx)
                _execute_adapt(df, idx, domain="economic")
                snap["decision"]      = "adapt"
                snap["new_residence"] = str(df.at[idx, "residence_district"])
                snap["new_workplace"] = str(df.at[idx, "workplace_district"])
                snap["wage"]          = float(df.at[idx, "wage"])
                snap["desired_raise"] = 0.0
                action_log.append(snap)
                stats["adapts"] += 1
            else:
                df.at[idx, "intention_state"] = "none"
                df.at[idx, "dst_work"]        = ""
                df.at[idx, "tpb_active"]      = False
                df.at[idx, "intention_delay"] = 0

    # Проход 3b: Place-driven поиск жилья
    place_driven_idx = np.where(
        (df["intention_state"].values == "seeking_residence") &
        (df["dst_work"].values == "")
    )[0]
    for idx in place_driven_idx:
        row       = df.iloc[idx]
        residence = str(row["residence_district"])
        workplace = str(row["workplace_district"])

        comm_thr_norm = float(row["commuter_threshold"])
        comm_thr_min  = 30.0 + 90.0 * comm_thr_norm

        new_residence = _place_driven_search(
            residence              = residence,
            workplace              = workplace,
            agent_wage             = float(row["wage"]),
            G                      = G,
            network_location       = bool(row["network_location"]),
            perceived_control      = float(row["perceived_control"]),
            commuter_threshold_min = comm_thr_min,
            rng                    = rng,
        )

        if new_residence is not None and new_residence != residence:
            snap = _snapshot(idx)
            _execute_move(df, idx, new_residence, workplace, G, rng)
            snap["decision"]      = "move"
            snap["new_residence"] = str(df.at[idx, "residence_district"])
            snap["new_workplace"] = str(df.at[idx, "workplace_district"])
            snap["wage"]          = float(df.at[idx, "wage"])
            snap["desired_raise"] = 0.0
            action_log.append(snap)
            stats["moves"] += 1
            stats["place_driven_moves"] += 1
        else:
            if float(row["job_flexibility"]) > ADAPT_FLEX_THRESHOLD:
                snap = _snapshot(idx)
                _execute_adapt(df, idx, domain="place")
                snap["decision"]      = "adapt"
                snap["new_residence"] = str(df.at[idx, "residence_district"])
                snap["new_workplace"] = str(df.at[idx, "workplace_district"])
                snap["wage"]          = float(df.at[idx, "wage"])
                snap["desired_raise"] = 0.0
                action_log.append(snap)
                stats["adapts"] += 1
            else:
                df.at[idx, "intention_state"] = "none"
                df.at[idx, "tpb_active"]      = False
                df.at[idx, "intention_delay"] = 0

    return df, stats, action_log


# ── Главный tick ──────────────────────────────────────────────────────────────

EVENT_SOCIAL_BOOST = 0.08
SOCIAL_BOOST_DECAY = 0.80


def tick(
    df: pd.DataFrame,
    G: nx.DiGraph,
    jobs_capacity: dict,
    tick_num: int,
    rng: np.random.Generator,
    init_dists: dict = None,
) -> tuple[pd.DataFrame, dict]:
    """Один шаг симуляции."""

    n_agents = len(df)
    df = df.copy()

    # 0. Блок B: затухание social_boost
    df["social_boost"] = df["social_boost"].values * SOCIAL_BOOST_DECAY

    # 1. Время
    df["age"]         = df["age"] + 1 / 12
    df["tenure"]      = df["tenure"] + 1
    df["moved_ticks"] = df["moved_ticks"] + 1

    # 1b. Graduation с soft-стартом (2 тика задержки)
    student_mask = df["status"].values == "student"
    if student_mask.any():
        df.loc[student_mask, "graduation_tick"] = (
            df.loc[student_mask, "graduation_tick"] - 1
        )
        graduating = student_mask & (df["graduation_tick"].values <= 0)
        if graduating.any():
            grad_idx = np.where(graduating)[0]
            for idx in grad_idx:
                age_at_grad = df.at[idx, "age"]
                residence   = str(df.at[idx, "residence_district"])
                df.at[idx, "status"]          = "unemployed"
                df.at[idx, "is_employed"]     = False
                df.at[idx, "graduation_tick"] = -1

                # Soft-старт: включаем TPB с задержкой 2 тика
                df.at[idx, "tpb_active"] = True
                df.at[idx, "activation_domain"] = "economic"
                df.at[idx, "intention_delay"] = 2
                df.at[idx, "intention_state"] = "none"
                # Гарантируем, что moved_ticks не заблокирует активацию
                df.at[idx, "moved_ticks"] = max(df.at[idx, "moved_ticks"], 9)

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
                if age_at_grad >= 22 and df.at[idx, "education"] != "high":
                    df.at[idx, "education"] = "high"
                df.at[idx, "inertia"] = float(np.clip(
                    df.at[idx, "inertia"] * 0.60, 0.05, 0.90
                ))
            n_grad = int(graduating.sum())
            if n_grad > 0:
                print(f"  [tick {tick_num}] Выпустилось студентов: {n_grad}")

    # 1c. Ежетиковые обновления (ускоренная векторизованная версия)
    df["signal_reduction"] = df["signal_reduction"].values * SIGNAL_DECAY

    hub_mask = df["workplace_district"].isin(HUB_DISTRICTS).values
    if hub_mask.any():
        df.loc[hub_mask, "weak_ties_utility"] = np.clip(
            df.loc[hub_mask, "weak_ties_utility"] + HUB_WEAK_TIES_BONUS, 0.0, 1.0
        )

    # ── Обновление econ_gap (векторизовано) ──────────────────────────────
    wages = df["wage"].values
    workplaces = df["workplace_district"].values
    # Сбор средней зарплаты по рабочему району каждого агента
    industry_avg = np.array([
        G.nodes.get(wp, {}).get("avg_wage", NATIONAL_AVG_WAGE)
        for wp in workplaces
    ])
    target_gap = np.where(
        (wages > 0) & (industry_avg > 0),
        np.maximum(0.0, 1.0 - wages / industry_avg),
        1.0
    )
    df["econ_gap"] = np.clip(
        (1 - GAP_ADAPT_LAMBDA) * df["econ_gap"].values + GAP_ADAPT_LAMBDA * target_gap,
        0.0, 1.0
    )

    # ── Обновление domain_future_place (векторизовано) ────────────────────
    residences = df["district"].values
    housing_prices = np.array([
        G.nodes.get(res, {}).get("housing_price_m2", 1800.0)
        for res in residences
    ])
    infra_scores = np.array([
        G.nodes.get(res, {}).get("infrastructure_score", 0.5)
        for res in residences
    ])
    monthly_cost = housing_prices * 50 * 0.004
    burden = monthly_cost / np.maximum(wages, 1.0)
    affordability = np.maximum(0.0, 1.0 - burden / HOUSING_BUDGET_RATIO)
    place_reality = 0.6 * affordability + 0.4 * infra_scores

    old_dfp = df["domain_future_place"].values
    df["domain_future_place"] = np.clip(
        (1 - GAP_ADAPT_LAMBDA) * old_dfp + GAP_ADAPT_LAMBDA * place_reality,
        0.0, 1.0
    )

    # 2. Давление рынка труда
    jobs_pressure = _compute_jobs_pressure(df, jobs_capacity)

    # 3. Обновление доменов (economic от workplace, place от residence)
    df = update_domain_satisfaction(df, G, jobs_pressure)

    # 4. Dissatisfaction (для статистики)
    dissatisfaction = compute_dissatisfaction(df)

    # 5. FFT pipeline
    residence_before  = df["district"].values.copy()
    workplace_before  = df["workplace_district"].values.copy()
    wage_before       = df["wage"].values.copy()
    status_before     = df["status"].values.copy()

    df, fft_stats, action_log = _fft_pipeline(df, G, dissatisfaction, jobs_pressure, rng)

    # 5b. Событийные сигналы
    social_boosts = df["social_boost"].values.copy()
    for i in range(len(df)):
        if residence_before[i] != df.at[i, "district"]:
            old_dist = residence_before[i]
            new_dist = df.at[i, "district"]
            mask_old = df["district"].values == old_dist
            social_boosts[mask_old] = np.clip(
                social_boosts[mask_old] + EVENT_SOCIAL_BOOST * 0.6, 0.0, 1.0
            )
            mask_new = df["district"].values == new_dist
            social_boosts[mask_new] = np.clip(
                social_boosts[mask_new] + EVENT_SOCIAL_BOOST, 0.0, 1.0
            )
        if workplace_before[i] != df.at[i, "workplace_district"]:
            old_wage = wage_before[i]
            new_wage = df.at[i, "wage"]
            if old_wage > 0 and new_wage > old_wage * 1.20:
                wp = df.at[i, "workplace_district"]
                mask_wp = df["workplace_district"].values == wp
                social_boosts[mask_wp] = np.clip(
                    social_boosts[mask_wp] + EVENT_SOCIAL_BOOST * 0.8, 0.0, 1.0
                )
        if status_before[i] == "stay" and df.at[i, "status"] == "commute":
            res = df.at[i, "district"]
            mask_res = df["district"].values == res
            social_boosts[mask_res] = np.clip(
                social_boosts[mask_res] + EVENT_SOCIAL_BOOST * 0.5, 0.0, 1.0
            )
    df["social_boost"] = social_boosts

    # 6. Обновление signal_reduction
    signal_reds = df["signal_reduction"].values.copy()
    status_new   = df["status"].values
    net_suscs    = df["net_signal_susc"].values
    for i in range(len(df)):
        new_signals = 0.0
        if status_before[i] != "unemployed" and status_new[i] == "unemployed":
            new_signals += UNEMPLOYED_SIGNAL
        if residence_before[i] != df.at[i, "district"]:
            old_res = residence_before[i]
            mask_neighbors = df["district"].values == old_res
            signal_reds[mask_neighbors] = np.clip(
                signal_reds[mask_neighbors] + NEIGHBOR_SIGNAL_COEF * net_suscs[mask_neighbors],
                0.0, 1.0
            )
        signal_reds[i] = float(np.clip(signal_reds[i] + new_signals, 0.0, 1.0))
    df["signal_reduction"] = signal_reds

    # 7. Реакция среды
    residence_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, residence_counts, n_agents)

    # 8. Статистика
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
        "avg_dissat":        round(float(dissatisfaction.mean()), 4),
        "avg_inertia":       round(float(df["inertia"].mean()), 4),
        "district_counts":   residence_counts,
        "jobs_pressure_max": round(max(jobs_pressure.values()) if jobs_pressure else 0, 2),
        "action_log":        action_log,
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
) -> tuple[pd.DataFrame, dict, list, list]:
    """
    Главный цикл симуляции.
    """
    rng = np.random.default_rng(seed)

    if snapshot_ticks is None:
        snapshot_ticks = [0, n_ticks // 4, n_ticks // 2, n_ticks]

    residence_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, residence_counts, len(df))

    snapshots  = {}
    tick_stats = []
    all_action_log = []

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
        df, stats = tick(df, G, jobs_capacity, t, rng, init_dists=init_dists)
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
