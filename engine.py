"""
engine.py v6 — Dominant Domain Activation + Split Strategies (Economic / Place)

Архитектура принятия решений:

  ФИЛЬТР 1 — Активация по доминантному домену (Блок D+G)
    gap = (thr - sat) / max(thr, 0.01) для каждого из 4 доменов.
    dominant_domain = argmax(gap).
    Активация: max_gap > ACTIVATION_THRESHOLD (0.20) + базовые проверки.
    Назначение intention_state по доминанте:
      economic → "seeking_work"
      place    → "seeking_residence"
      social / family → "none" (зарезервировано)
    Inertia-задержка общая для всех доменов.

  ФИЛЬТР 2 — Дерево занятости (economic-driven)
    intention_state = "seeking_work"
    Целевая зарплата — поведенческая эвристика:
      base_appetite = 0.10 + econ_perceived_control × 0.20
      desperation = clamp((thr - sat) / thr, 0, 1)
      desired_raise = 0.05 + (base_appetite - 0.05) × (1 - desperation)
      target_wage = agent_wage × (1 + desired_raise)
    Приоритет домашнего района → скрининг awareness_set:
      Аспект А: avg_wage[dst] > target_wage
      Аспект Б: jobs_pressure[dst] < 1.2

  ФИЛЬТР 3a — Локация для economic-driven (после фильтра 2)
    intention_state = "seeking_residence" (с dst_work)
    Путь А — COMMUTE: travel_time ≤ threshold И sat_place ≥ thr_place×0.8
    Путь Б — MOVE: жильё в dst_work доступно
    Путь В — SATELLITE: спутник в радиусе 90 мин с доступным жильём

  ФИЛЬТР 3b — Place-driven поиск жилья (минуя фильтр 2)
    intention_state = "seeking_residence" (без dst_work)
    _place_driven_search: ищем район с доступным жильём в пределах
    расширенной досягаемости до текущего workplace.
    → переезд с сохранением workplace.

  ОБНОВЛЕНИЕ ДОМЕНОВ:
    Social (Блок A): target = 0.5 + social_boost, сглаживание α=0.88.
    Family (Блок F): commute-давление.
    Economic: от workplace_district, Place: от residence_district.

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

# Фильтр 1
MIN_ECON_PC_TO_ACTIVATE  = 0.40   # минимальный econ_perceived_control для активации
ACTIVATION_THRESHOLD     = 0.20   # минимальный gap домена для активации (0–1)

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
            # Студенты: economic — нейтральный с медленным дрейфом,
            # они не сравнивают свою «зарплату» с рынком труда.
            # Лёгкий позитивный дрейф — стипендия/поддержка семьи.
            target_econ = 0.48
            sat_econ[i] = float(np.clip(
                SAT_SMOOTHING * sat_econ[i] + (1 - SAT_SMOOTHING) * target_econ,
                0.0, 1.0
            ))
        else:
            q_wage    = (w - avg_wage_wp) / max(avg_wage_wp, 1)
            # Давление рынка труда: перегруженный рынок → сложнее найти работу
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
        # Базовая цель family — 0.5; commute снижает её пропорционально
        # превышению времени в пути над порогом агента.
        target_family = 0.50
        if statuses[i] == "commute":
            res = residences[i]
            wp  = workplaces[i]
            travel_time = 999
            if G.has_edge(res, wp):
                travel_time = G[res][wp].get("travel_time_min", 999)
            # commuter_threshold из [0,1] в минуты: 30–120
            comm_thr_norm = float(df["commuter_threshold"].values[i])
            comm_thr_min  = 30.0 + 90.0 * comm_thr_norm
            if travel_time > comm_thr_min:
                # excess_ratio ∈ [0, 1] при превышении до 2×
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


# ── FFT: Фильтр 1 — Страж ворот ──────────────────────────────────────────────

def _fft_filter1(
    df: pd.DataFrame,
    dissatisfaction: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Фильтр 1 — активация по ведущему (доминантному) домену.

    Для каждого агента вычисляется относительный дефицит по 4 доменам:
      gap = (thr - sat) / max(thr, 0.01), только положительные.
    dominant_domain = argmax(gap).

    Активация: max_gap > ACTIVATION_THRESHOLD (и базовые проверки пройдены).

    Назначение intention_state по доминантному домену:
      "economic" → "seeking_work"
      "place"    → "seeking_residence"
      "social" / "family" → пока "none" (зарезервировано)

    Inertia-задержка и activation_timer — без изменений, общие для всех доменов.
    """
    df = df.copy()
    n = len(df)

    # ── Вычисляем gap для каждого домена ──────────────────────────────────
    domain_specs = [
        ("economic", "sat_economic", "thr_economic"),
        ("social",   "sat_social",   "thr_social"),
        ("family",   "sat_family",   "thr_family"),
        ("place",    "sat_place",    "thr_place"),
    ]
    gaps = np.zeros((n, 4))
    for j, (_, sat_col, thr_col) in enumerate(domain_specs):
        sat = df[sat_col].values
        thr = df[thr_col].values
        gaps[:, j] = np.maximum(0, thr - sat) / np.maximum(thr, 0.01)

    max_gap = gaps.max(axis=1)
    dominant_idx = gaps.argmax(axis=1)  # 0=economic, 1=social, 2=family, 3=place
    domain_names = np.array(["economic", "social", "family", "place"])
    dominant_domain = domain_names[dominant_idx]
    df["activation_domain"] = dominant_domain

    # Базовые маски
    none_mask    = df["intention_state"].values == "none"
    age_mask     = (df["age"].values >= 18) & (df["age"].values <= 62)
    moved_mask   = df["moved_ticks"].values >= 9
    econ_pc_mask = df["econ_perceived_control"].values > MIN_ECON_PC_TO_ACTIVATE
    unemployed   = df["status"].values == "unemployed"
    student_mask = df["status"].values == "student"

    # Агент под давлением: max_gap > порог ИЛИ безработный
    under_pressure = (max_gap > ACTIVATION_THRESHOLD) | (unemployed & econ_pc_mask)

    # ── Обновляем activation_timer ────────────────────────────────────────────
    timers    = df["activation_timer"].values.copy()
    inertia   = df["inertia"].values
    shock_sens = df["shock_sensitivity"].values

    eligible = none_mask & age_mask & moved_mask & econ_pc_mask & ~student_mask
    timers = np.where(eligible & under_pressure, timers + 1, timers)
    timers = np.where(eligible & ~under_pressure, 0, timers)

    df["activation_timer"] = timers

    # ── Вычисляем задержку и активируем ──────────────────────────────────────
    delay = np.clip((inertia * 12.0).astype(int), 1, 11)
    delay = np.maximum(1, delay - (shock_sens * 6.0).astype(int))

    ready = timers >= delay
    activated = eligible & ready

    # ── Назначаем intention_state по доминантному домену ─────────────────────
    # Делаем это для ВСЕХ агентов под давлением (не только активированных),
    # чтобы dominant_domain был актуален на момент срабатывания таймера.
    # Но intention_state меняем только у активированных.
    if activated.any():
        act_idx = np.where(activated)[0]
        for idx in act_idx:
            dom = dominant_domain[idx]
            if dom == "economic":
                df.at[idx, "intention_state"] = "seeking_work"
            elif dom == "place":
                df.at[idx, "intention_state"] = "seeking_residence"
            else:
                # social / family — пока оставляем none
                df.at[idx, "intention_state"] = "none"
            df.at[idx, "activation_timer"] = 0

    return df, activated


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
      - Для занятых: target_wage = agent_wage × (1 + desired_raise)
      - Для безработных: ориентир = отраслевая зарплата дома × 0.85 × (1 + desired_raise)
      - При скрининге кандидатов используется отраслевая зарплата dst_района.

    Целевая зарплата — поведенческая эвристика:
      base_appetite = BASE_APPETITE_MIN + econ_perceived_control × BASE_APPETITE_MAX
      desperation   = clamp((thr_economic - sat_economic) / thr_economic, 0, 1)
      desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE) × (1 - desperation)

    Приоритет домашнего района: если residence_district проходит оба аспекта —
    сразу возвращаем его без перебора awareness_set.

    Скрининг (Elimination by Aspects):
      Аспект А: отраслевая_зарплата[dst] > target_wage
      Аспект Б: jobs_pressure[dst] < MAX_JOBS_PRESSURE

    Satisficing: после shuffle берём первый прошедший оба фильтра.
    awareness_set расширяется через network_location и perceived_control.
    """
    # ── Вспомогательная: отраслевая зарплата в районе ────────────────────
    def _industry_wage(node_attr: dict, industry: str) -> float:
        """Отраслевая зарплата в узле; fallback → avg_wage → NATIONAL_AVG_WAGE."""
        sal = node_attr.get("salary_by_industry", {})
        if sal:
            return float(sal.get(industry, node_attr.get("avg_wage", NATIONAL_AVG_WAGE)))
        return float(node_attr.get("avg_wage", NATIONAL_AVG_WAGE))

    # ── Расчёт целевой зарплаты ──────────────────────────────────────────
    if agent_wage > 0:
        # Занятый: целевая зарплата = текущая × (1 + желаемая надбавка)
        base_appetite = BASE_APPETITE_MIN + econ_perceived_control * BASE_APPETITE_MAX
        thr = max(thr_economic, 0.01)
        desperation = float(np.clip((thr_economic - sat_economic) / thr, 0.0, 1.0))
        desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE) * (1.0 - desperation)
        target_wage = agent_wage * (1.0 + desired_raise)
    else:
        # Безработный: ориентируется на отраслевую зарплату в районе проживания
        home_attr = G.nodes.get(district, {})
        home_ind_wage = _industry_wage(home_attr, agent_industry)
        # Дисконт 0.85 — безработный готов на меньшее
        base_appetite = BASE_APPETITE_MIN + econ_perceived_control * BASE_APPETITE_MAX
        thr = max(thr_economic, 0.01)
        desperation = float(np.clip((thr_economic - sat_economic) / thr, 0.0, 1.0))
        desired_raise = MIN_DESIRED_RAISE + (base_appetite - MIN_DESIRED_RAISE) * (1.0 - desperation)
        target_wage = (home_ind_wage * 0.85) * (1.0 + desired_raise)

    # ── Приоритет домашнего района (с отраслевой зарплатой) ──────────────
    home_attr = G.nodes.get(district, {})
    home_wage = _industry_wage(home_attr, agent_industry)
    home_pressure = jobs_pressure.get(district, 0)
    if home_wage >= target_wage and home_pressure < MAX_JOBS_PRESSURE:
        return district

    # ── Кандидаты из awareness_set ────────────────────────────────────────
    candidates = get_awareness_set(
        G, district,
        network_location=network_location,
        perceived_control=perceived_control,
        max_candidates=MAX_WORK_CANDIDATES,
    )
    # Включаем текущий район — может зарплата там уже поднялась
    candidates = list(set(candidates) | {district})

    rng.shuffle(candidates)

    for dst in candidates:
        dst_attr = G.nodes.get(dst, {})
        dst_wage = _industry_wage(dst_attr, agent_industry)

        # Аспект А: отраслевая зарплата
        if dst_wage < target_wage:
            continue

        # Аспект Б: рынок труда не перегружен
        if jobs_pressure.get(dst, 0) >= MAX_JOBS_PRESSURE:
            continue

        return dst

    return None  # Не нашёл подходящего района


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

    Скрининг кандидатов из awareness_set вокруг workplace:
      1. Доступность жилья: _housing_affordable(agent_wage, housing_price)
      2. Время в пути до workplace ≤ commuter_threshold_min × PLACE_DRIVEN_TRAVEL_BUF
         (расширенный порог — мотивация места высока)
      3. target_place нового района строго лучше текущего (зазор > 2%)

    Satisficing: первый подходящий после shuffle.
    Возвращает new_residence или None.
    """
    # Расширенный порог: place-мотивация допускает более долгий commute
    expanded_threshold = commuter_threshold_min * PLACE_DRIVEN_TRAVEL_BUF

    # Текущий target_place — опорная точка для сравнения
    current_attr = G.nodes.get(residence, {})
    cur_h = current_attr.get("housing_price_m2", 1800)
    cur_i = current_attr.get("infrastructure_score", 0.5)
    cur_target = _sigmoid(0.5 * (1800 - cur_h) / 1800 + 0.5 * (cur_i - 0.5))

    # Кандидаты из awareness_set вокруг workplace
    candidates = get_awareness_set(
        G, workplace,
        network_location=network_location,
        perceived_control=perceived_control,
        max_candidates=MAX_WORK_CANDIDATES,
    )
    # Включаем текущий residence и workplace
    candidates = list(set(candidates) | {residence, workplace})

    rng.shuffle(candidates)

    for dst in candidates:
        if dst == residence:
            continue  # нет смысла переезжать туда же

        dst_attr = G.nodes.get(dst, {})
        housing_price = dst_attr.get("housing_price_m2", 9999)

        # Критерий 1: доступность жилья
        if not _housing_affordable(agent_wage, housing_price):
            continue

        # Критерий 2: время в пути до текущей работы
        if G.has_edge(dst, workplace):
            tt = G[dst][workplace].get("travel_time_min", 999)
            if tt > expanded_threshold:
                continue
        else:
            continue  # нет прямого пути — пропускаем

        # Критерий 3: target_place строго лучше текущего (зазор 2% против джиттера)
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

    Путь А — COMMUTE:
      travel_time[residence → dst_work] ≤ commuter_threshold_min
      И sat_place ≥ thr_place × 0.8 (место не слишком давит)

    Путь Б — MOVE (прямой):
      Жильё в dst_work доступно по бюджету → переезд.

    Путь В — SATELLITE MOVE:
      Жильё в dst_work не доступно.
      Ищем район-спутник в радиусе SATELLITE_SEARCH_RADIUS от dst_work,
      с доступным жильём и временем езды ≤ commuter_threshold_min до dst_work.
      → переезд в спутник + commute в dst_work.

    Путь Г — none:
      Ничего не подошло.
    """
    # Путь А: commute из текущего места жительства
    if G.has_edge(residence, dst_work):
        tt = G[residence][dst_work].get("travel_time_min", 999)
    else:
        tt = 999

    # Commute разрешён только если travel_time в норме И sat_place не слишком низкий
    place_ok = sat_place >= thr_place * 0.80
    if tt <= commuter_threshold_min and place_ok:
        return residence, "commute"

    # Путь Б: прямой переезд в dst_work
    dst_attr = G.nodes.get(dst_work, {})
    housing_dst = dst_attr.get("housing_price_m2", 9999)
    if _housing_affordable(agent_wage, housing_dst):
        return dst_work, "move"

    # Путь В: поиск района-спутника
    # Ищем соседей dst_work в радиусе SATELLITE_SEARCH_RADIUS
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
        # Выбираем спутник: сначала по доступности жилья, потом по времени
        satellites.sort(key=lambda x: (x[2], x[1]))
        best_satellite = satellites[0][0]
        return best_satellite, "satellite_move"

    # Путь Г: ничего не нашли
    return residence, "none"


# ── Исполнение решений ────────────────────────────────────────────────────────

def _execute_commute(
    df: pd.DataFrame,
    idx: int,
    new_workplace: str,
    G: nx.DiGraph,
    rng: np.random.Generator,
):
    """Агент меняет место работы без смены жительства."""
    old_wp = df.at[idx, "workplace_district"]
    df.at[idx, "workplace_district"] = new_workplace
    df.at[idx, "status"]             = "commute"
    df.at[idx, "intention_state"]    = "none"
    df.at[idx, "dst_work"]           = ""
    df.at[idx, "moved_ticks"]        = 0

    # v8: отраслевая зарплата в новом районе работы
    agent_industry = str(df.at[idx, "industry"])
    wp_attr = G.nodes.get(new_workplace, {})
    wp_salary_by_ind = wp_attr.get("salary_by_industry", {})
    base_wage = wp_salary_by_ind.get(agent_industry, wp_attr.get("avg_wage", df.at[idx, "wage"]))
    new_wage = float(max(0, rng.normal(base_wage, base_wage * 0.18)))
    df.at[idx, "wage"] = new_wage

    # Weak ties растут — агент заводит знакомства в новом месте работы
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
    Стресс переезда: satisfaction временно снижается.
    Inertia пересчитывается: стаж и место сброшены.
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

    # v8: отраслевая зарплата от нового workplace
    agent_industry = str(df.at[idx, "industry"])
    wp_attr = G.nodes.get(new_workplace, {})
    wp_salary_by_ind = wp_attr.get("salary_by_industry", {})
    base_wage = wp_salary_by_ind.get(agent_industry, wp_attr.get("avg_wage", df.at[idx, "wage"]))
    new_wage = float(max(0, rng.normal(base_wage, base_wage * 0.18)))
    df.at[idx, "wage"] = new_wage

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

    Проход 1: Фильтр 1 → активация по доминантному домену:
      economic → "seeking_work"
      place    → "seeking_residence"
      social / family → "none"

    Проход 2 (только seeking_work): Фильтр 2 → ищем dst_work
      → "seeking_residence" | adapt | none

    Проход 3a (economic-driven, seeking_residence после фильтра 2):
      Фильтр 3 → commute | move | satellite_move | none

    Проход 3b (place-driven, seeking_residence из фильтра 1):
      _place_driven_search → move (с сохранением workplace) | adapt | none

    Возвращает (df, stats, action_log).
    action_log — список словарей с деталями каждого совершённого действия.
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
        """Снимает слепок агента ДО изменений для лога."""
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

    # ── Проход 1: Фильтр 1 (доминантный домен + активация) ───────────────
    df, activate_mask = _fft_filter1(df, dissatisfaction)
    n_activate = int(activate_mask.sum())

    if n_activate > 0:
        activate_idx = np.where(activate_mask)[0]
        for idx in activate_idx:
            dom = df.at[idx, "activation_domain"]
            if dom == "economic":
                stats["econ_activated"] += 1
            elif dom == "place":
                stats["place_activated"] += 1
            # intention_state уже установлен в _fft_filter1
        stats["activated"] = n_activate

    # ── Проход 2: Фильтр 2 — дерево занятости (только seeking_work) ──────
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
                df.at[idx, "inertia"] = float(np.clip(
                    df.at[idx, "inertia"] + 0.03, 0.05, 0.95
                ))
        else:
            df.at[idx, "dst_work"]        = dst_work
            df.at[idx, "intention_state"] = "seeking_residence"
            stats["dst_found"] += 1

    # ── Проход 3a: Фильтр 3 — локация для economic-driven ─────────────────
    # Ищем агентов, которые стали seeking_residence через фильтр 2
    # (у них заполнен dst_work)
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
            old_wp = str(df.at[idx, "workplace_district"])
            # Вычисляем desired_raise до исполнения (те же формулы что в фильтре 2)
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
            snap["decision"]      = outcome  # "move" или "satellite_move"
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

    # ── Проход 3b: Place-driven — поиск жилья с сохранением работы ────────
    place_driven_idx = np.where(
        (df["intention_state"].values == "seeking_residence") &
        (df["dst_work"].values == "")  # нет dst_work → place-driven
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
            # Переезд с сохранением workplace
            snap = _snapshot(idx)
            _execute_move(df, idx, new_residence, workplace, G, rng)
            snap["decision"]      = "move"
            snap["new_residence"] = str(df.at[idx, "residence_district"])
            snap["new_workplace"] = str(df.at[idx, "workplace_district"])
            snap["wage"]          = float(df.at[idx, "wage"])
            snap["desired_raise"] = 0.0  # place-driven: нет поиска работы
            action_log.append(snap)
            stats["moves"] += 1
            stats["place_driven_moves"] += 1
        else:
            # Fallback: адаптация места или возврат в none
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

    return df, stats, action_log


# ── Главный tick ──────────────────────────────────────────────────────────────

EVENT_SOCIAL_BOOST = 0.08   # прирост social_boost от события
SOCIAL_BOOST_DECAY = 0.80   # множитель затухания за тик


def tick(
    df: pd.DataFrame,
    G: nx.DiGraph,
    jobs_capacity: dict,
    tick_num: int,
    rng: np.random.Generator,
    init_dists: dict = None,
) -> tuple[pd.DataFrame, dict]:
    """Один шаг симуляции. Возвращает обновлённый DataFrame и статистику."""

    n_agents = len(df)

    df = df.copy()

    # 0. Блок B: затухание social_boost (×0.8 каждый тик)
    df["social_boost"] = df["social_boost"].values * SOCIAL_BOOST_DECAY

    # 1. Время
    df["age"]         = df["age"] + 1 / 12
    df["tenure"]      = df["tenure"] + 1
    df["moved_ticks"] = df["moved_ticks"] + 1

    # 1b. Graduation: студенты взрослеют → выпуск
    student_mask = df["status"].values == "student"
    if student_mask.any():
        # Декремент тиков до выпуска
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
                df.at[idx, "intention_state"] = "seeking_work"
                df.at[idx, "is_employed"]     = False
                df.at[idx, "graduation_tick"] = -1
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
                # Выпускники вузов (>= 22 лет) → high education
                if age_at_grad >= 22 and df.at[idx, "education"] != "high":
                    df.at[idx, "education"] = "high"
                # Инерция снижается — выпускник мобилен
                df.at[idx, "inertia"] = float(np.clip(
                    df.at[idx, "inertia"] * 0.60, 0.05, 0.90
                ))
            n_grad = int(graduating.sum())
            if n_grad > 0:
                print(f"  [tick {tick_num}] Выпустилось студентов: {n_grad}")

    # 2. Давление рынка труда
    jobs_pressure = _compute_jobs_pressure(df, jobs_capacity)

    # 3. Обновление доменов (economic от workplace, place от residence)
    df = update_domain_satisfaction(df, G, jobs_pressure)

    # 4. Dissatisfaction
    dissatisfaction = compute_dissatisfaction(df)

    # 5. FFT pipeline
    residence_before  = df["district"].values.copy()
    workplace_before  = df["workplace_district"].values.copy()
    wage_before       = df["wage"].values.copy()
    status_before     = df["status"].values.copy()
    df, fft_stats, action_log = _fft_pipeline(df, G, dissatisfaction, jobs_pressure, rng)

    # 5b. Блок B: событийные сигналы (social_boost)
    # Триггер 1: Переезд → shock старому району, positive новому
    # Триггер 2: Смена работы с ростом >20% → positive коллегам
    # Триггер 3: Становление маятником → positive соседям
    social_boosts = df["social_boost"].values.copy()

    for i in range(len(df)):
        # Триггер 1: переезд (residence изменился)
        if residence_before[i] != df.at[i, "district"]:
            old_dist = residence_before[i]
            new_dist = df.at[i, "district"]
            # shock старому району
            mask_old = df["district"].values == old_dist
            social_boosts[mask_old] = np.clip(
                social_boosts[mask_old] + EVENT_SOCIAL_BOOST * 0.6, 0.0, 1.0
            )
            # positive новому району
            mask_new = df["district"].values == new_dist
            social_boosts[mask_new] = np.clip(
                social_boosts[mask_new] + EVENT_SOCIAL_BOOST, 0.0, 1.0
            )

        # Триггер 2: смена workplace с ростом зарплаты >20%
        if workplace_before[i] != df.at[i, "workplace_district"]:
            old_wage = wage_before[i]
            new_wage = df.at[i, "wage"]
            if old_wage > 0 and new_wage > old_wage * 1.20:
                wp = df.at[i, "workplace_district"]
                mask_wp = df["workplace_district"].values == wp
                social_boosts[mask_wp] = np.clip(
                    social_boosts[mask_wp] + EVENT_SOCIAL_BOOST * 0.8, 0.0, 1.0
                )

        # Триггер 3: стал маятником (был stay → стал commute)
        if status_before[i] == "stay" and df.at[i, "status"] == "commute":
            res = df.at[i, "district"]
            mask_res = df["district"].values == res
            social_boosts[mask_res] = np.clip(
                social_boosts[mask_res] + EVENT_SOCIAL_BOOST * 0.5, 0.0, 1.0
            )

    df["social_boost"] = social_boosts

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

    Принимает jobs_capacity явно — строится в agents.py из commuting-матрицы
    и передаётся через run.py.
    init_dists — распределения из agent_init_distributions.json (для graduation).

    Возвращает (df_final, snapshots, tick_stats, all_action_log).
    """
    rng = np.random.default_rng(seed)

    if snapshot_ticks is None:
        snapshot_ticks = [0, n_ticks // 4, n_ticks // 2, n_ticks]

    # Инициализируем граф реальными counts
    residence_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, residence_counts, len(df))

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
