"""
engine.py v5 — Fast-and-Frugal Trees (FFT) + Inertia Delay + Event Signals

Архитектура принятия решений:

  ФИЛЬТР 1 — Inertia-задержка (Блок D+G)
    2+ домена из 4 ниже индивидуальных порогов → начало отсчёта задержки.
    Задержка = inertia × 12 месяцев (1–11), сокращается shock_sensitivity.
    activation_timer считает тики под давлением; при timer ≥ delay → активация.
    Безработные активируются сразу (если верят в изменения).

  ФИЛЬТР 2 — Дерево занятости (где работать?)
    intention_state = "seeking_work"
    Скрининг доступных workplace-районов по двум жёстким аспектам:
      Аспект А: avg_wage[dst] > wage[agent] × (1 + domain_economic_gap)
      Аспект Б: jobs_pressure[dst] < 1.2 (район не перегружен)
    Satisficing: берём первый район прошедший оба фильтра (после shuffle).

  ФИЛЬТР 3 — Дерево локации (где жить?)
    intention_state = "seeking_residence"
    Путь А — COMMUTE: travel_time[residence → dst_work] ≤ commuter_threshold мин
    Путь Б — MOVE: жильё в dst_work доступно → переезд.
    Путь В — SATELLITE: спутник в радиусе 90 мин с доступным жильём.

  ОБНОВЛЕНИЕ ДОМЕНОВ:
    Social (Блок A): target = 0.5 + social_boost, сглаживание α=0.88.
    Family (Блок F): commute-давление — длинный commute снижает target.
    Economic: от workplace_district, Place: от residence_district.

  DISSATISFACTION (Блок C): евклидова норма (sqrt суммы квадратов) вместо линейной.

  СОБЫТИЙНЫЕ СИГНАЛЫ (Блок B):
    social_boost затухает ×0.8/тик. Три триггера:
    - Переезд → shock старому району, positive новому
    - Смена работы с ростом >20% → positive коллегам
    - Становление маятником → positive соседям
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

# Фильтр 2 — дерево занятости
WAGE_GAP_MULTIPLIER      = 1.0    # dst_wage > agent_wage × (1 + econ_gap × этот множитель)
MAX_JOBS_PRESSURE        = 1.20   # район считается перегруженным выше этого порога
MAX_WORK_CANDIDATES      = 12     # максимум районов для скрининга

# Фильтр 3 — дерево локации
SATELLITE_SEARCH_RADIUS  = 90.0   # мин. — радиус поиска района-спутника от dst_work
HOUSING_BUDGET_RATIO     = 0.35   # жильё не должно превышать X доли зарплаты (×100м²)
MOVE_STRESS_FACTOR       = 0.80   # satisfaction после переезда × этот множитель

# Adapt
ADAPT_FLEX_THRESHOLD     = 0.65   # минимальная job_flexibility для адаптации
ADAPT_SAT_BOOST          = 0.06   # прирост sat_economic при адаптации

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
    Блок D+G: новый Фильтр 1 — inertia как задержка активации.

    Условие активации:
      2+ домена из 4 ниже индивидуальных порогов (thr_*).

    Inertia задаёт задержку: delay = int(inertia × 12) месяцев (1–11).
    shock_sensitivity сокращает задержку: delay = max(1, delay - shock_sens×6).

    activation_timer считает тики:
      - Если 2+ доменов ниже порогов → timer += 1
      - Иначе → timer = 0
      - Когда timer >= delay → агент активируется

    Дополнительные условия:
      - intention_state == "none"
      - возраст 18–62
      - moved_ticks >= 6
      - econ_perceived_control > MIN_ECON_PC_TO_ACTIVATE
      - безработные активируются автоматически (если верят в изменения)
      - студенты исключены
    """
    df = df.copy()
    n = len(df)

    # ── Считаем число доменов ниже порогов ───────────────────────────────────
    below_econ   = df["sat_economic"].values < df["thr_economic"].values
    below_social = df["sat_social"].values   < df["thr_social"].values
    below_family = df["sat_family"].values   < df["thr_family"].values
    below_place  = df["sat_place"].values    < df["thr_place"].values

    domains_below = (
        below_econ.astype(int) +
        below_social.astype(int) +
        below_family.astype(int) +
        below_place.astype(int)
    )

    # Базовые маски
    none_mask    = df["intention_state"].values == "none"
    age_mask     = (df["age"].values >= 18) & (df["age"].values <= 62)
    moved_mask   = df["moved_ticks"].values >= 6
    econ_pc_mask = df["econ_perceived_control"].values > MIN_ECON_PC_TO_ACTIVATE
    unemployed   = df["status"].values == "unemployed"
    student_mask = df["status"].values == "student"

    # Агент под давлением: 2+ доменов ниже порогов ИЛИ безработный
    under_pressure = (domains_below >= 2) | (unemployed & econ_pc_mask)

    # ── Обновляем activation_timer ────────────────────────────────────────────
    timers    = df["activation_timer"].values.copy()
    inertia   = df["inertia"].values
    shock_sens = df["shock_sensitivity"].values

    # Увеличиваем таймер если под давлением и проходит базовые проверки
    eligible = none_mask & age_mask & moved_mask & econ_pc_mask & ~student_mask
    timers = np.where(eligible & under_pressure, timers + 1, timers)
    # Сбрасываем если НЕ под давлением
    timers = np.where(eligible & ~under_pressure, 0, timers)

    df["activation_timer"] = timers

    # ── Вычисляем задержку и активируем ──────────────────────────────────────
    delay = np.clip((inertia * 12.0).astype(int), 1, 11)
    # shock_sensitivity сокращает задержку (высокая shock_sens = быстрая реакция)
    delay = np.maximum(1, delay - (shock_sens * 6.0).astype(int))

    ready = timers >= delay
    activated = eligible & ready

    return df, activated


# ── FFT: Фильтр 2 — Дерево занятости ─────────────────────────────────────────

def _fft_filter2_find_work(
    district: str,
    agent_wage: float,
    econ_gap: float,
    G: nx.DiGraph,
    jobs_pressure: dict,
    network_location: bool,
    perceived_control: float,
    rng: np.random.Generator,
) -> Optional[str]:
    """
    Ищет dst_work — район где агент может найти работу лучше текущей.

    Скрининг (Elimination by Aspects):
      Аспект А: avg_wage[dst] > agent_wage × (1 + econ_gap)
                Если агент безработный (wage=0), порог = национальная средняя × 0.8
      Аспект Б: jobs_pressure[dst] < MAX_JOBS_PRESSURE
                Район не должен быть перегружен рабочей силой

    Satisficing: после shuffle берём первый прошедший оба фильтра.
    awareness_set расширяется через network_location и perceived_control.
    """
    # Минимальная целевая зарплата
    if agent_wage > 0:
        target_wage = agent_wage * (1.0 + econ_gap * WAGE_GAP_MULTIPLIER)
    else:
        target_wage = NATIONAL_AVG_WAGE * 0.80

    # Кандидаты из awareness_set
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
        dst_wage = dst_attr.get("avg_wage", 0)

        # Аспект А: зарплата
        if dst_wage < target_wage:
            continue

        # Аспект Б: рынок труда не перегружен
        if jobs_pressure.get(dst, 0) >= MAX_JOBS_PRESSURE:
            continue

        return dst

    return None  # Не нашёл подходящего района


# ── FFT: Фильтр 3 — Дерево локации ───────────────────────────────────────────

def _fft_filter3_find_residence(
    residence: str,
    dst_work: str,
    agent_wage: float,
    G: nx.DiGraph,
    rng: np.random.Generator,
    commuter_threshold_min: float,
) -> tuple[str, str]:
    """
    Определяет стратегию локации после нахождения dst_work.

    Возвращает (new_residence, outcome):
      outcome: "commute" | "move" | "satellite_move" | "none"

    Путь А — COMMUTE:
      travel_time[residence → dst_work] ≤ commuter_threshold_min
      Агент остаётся жить где живёт, только меняет workplace.

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

    if tt <= commuter_threshold_min:
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

    # Зарплата в новом месте работы
    target_wage = G.nodes[new_workplace].get("avg_wage", df.at[idx, "wage"])
    new_wage    = float(max(0, rng.normal(target_wage, target_wage * 0.18)))
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

    # Зарплата от нового workplace
    target_wage = G.nodes[new_workplace].get("avg_wage", df.at[idx, "wage"])
    new_wage    = float(max(0, rng.normal(target_wage, target_wage * 0.18)))
    df.at[idx, "wage"] = new_wage

    # Стресс переезда
    for col in ["sat_economic", "sat_social", "sat_family"]:
        df.at[idx, col] = float(np.clip(df.at[idx, col] * MOVE_STRESS_FACTOR, 0.05, 0.95))

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


def _execute_adapt(df: pd.DataFrame, idx: int):
    """
    Агент адаптируется на месте: снижает притязания, экономит.
    Только при высокой job_flexibility и экономическом давлении.
    """
    flex = df.at[idx, "job_flexibility"]
    df.at[idx, "sat_economic"]    = float(np.clip(
        df.at[idx, "sat_economic"] + flex * ADAPT_SAT_BOOST, 0.0, 1.0
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
) -> tuple[pd.DataFrame, dict]:
    """
    Полный FFT pipeline за один тик.

    Три прохода:
      Проход 1: Фильтр 1 → активируем агентов (none → seeking_work)
      Проход 2: Фильтр 2 → ищем dst_work (seeking_work → seeking_residence | adapt | none)
      Проход 3: Фильтр 3 → выбираем локацию (seeking_residence → commute | move | none)

    Три прохода вместо одного цикла — векторизованный Фильтр 1,
    индивидуальные Фильтры 2 и 3.
    """
    df = df.copy()
    stats = {
        "moves": 0, "commutes": 0, "adapts": 0,
        "activated": 0, "dst_found": 0, "satellite_moves": 0,
    }

    # ── Проход 1: Фильтр 1 (inertia-задержка + активация) ──────────────────
    df, activate_mask = _fft_filter1(df, dissatisfaction)
    n_activate = int(activate_mask.sum())

    if n_activate > 0:
        activate_idx = np.where(activate_mask)[0]
        for idx in activate_idx:
            df.at[idx, "intention_state"] = "seeking_work"
            # Сбрасываем activation_timer после активации
            df.at[idx, "activation_timer"] = 0
        stats["activated"] = n_activate

    # ── Проход 2: Фильтр 2 — дерево занятости ────────────────────────────────
    seeking_work_idx = np.where(df["intention_state"].values == "seeking_work")[0]

    for idx in seeking_work_idx:
        row      = df.iloc[idx]
        district = str(row["residence_district"])

        dst_work = _fft_filter2_find_work(
            district        = district,
            agent_wage      = float(row["wage"]),
            econ_gap        = float(row["thr_economic"]),
            G               = G,
            jobs_pressure   = jobs_pressure,
            network_location= bool(row["network_location"]),
            perceived_control = float(row["perceived_control"]),
            rng             = rng,
        )

        if dst_work is None:
            # Нет подходящего района для работы
            # Проверяем возможность адаптации
            primary_econ_pressure = (
                float(row["w_economic"]) *
                max(0, float(row["thr_economic"]) - float(row["sat_economic"])) /
                max(float(row["thr_economic"]), 0.01)
            )
            other_pressure = dissatisfaction[idx] - primary_econ_pressure

            if (primary_econ_pressure > other_pressure and
                    float(row["job_flexibility"]) > ADAPT_FLEX_THRESHOLD):
                _execute_adapt(df, idx)
                stats["adapts"] += 1
            else:
                # Возврат в none — не нашёл выхода, продолжает жить с неудовлетворённостью
                df.at[idx, "intention_state"] = "none"
                # Восстанавливаем инерцию частично
                df.at[idx, "inertia"] = float(np.clip(
                    df.at[idx, "inertia"] + 0.03, 0.05, 0.95
                ))
        else:
            df.at[idx, "dst_work"]        = dst_work
            df.at[idx, "intention_state"] = "seeking_residence"
            stats["dst_found"] += 1

    # ── Проход 3: Фильтр 3 — дерево локации ──────────────────────────────────
    seeking_res_idx = np.where(df["intention_state"].values == "seeking_residence")[0]

    for idx in seeking_res_idx:
        row      = df.iloc[idx]
        residence = str(row["residence_district"])
        dst_work  = str(row["dst_work"])

        if not dst_work:
            df.at[idx, "intention_state"] = "none"
            continue

        # Конвертируем commuter_threshold из [0,1] в минуты
        # Глобальная средняя = 0.534 → ~60 мин; 0.0 → 30 мин; 1.0 → 120 мин
        comm_thr_norm = float(row["commuter_threshold"])
        comm_thr_min  = 30.0 + 90.0 * comm_thr_norm

        new_residence, outcome = _fft_filter3_find_residence(
            residence              = residence,
            dst_work               = dst_work,
            agent_wage             = float(row["wage"]),
            G                      = G,
            rng                    = rng,
            commuter_threshold_min = comm_thr_min,
        )

        if outcome == "commute":
            _execute_commute(df, idx, dst_work, G, rng)
            stats["commutes"] += 1

        elif outcome in ("move", "satellite_move"):
            _execute_move(df, idx, new_residence, dst_work, G, rng)
            stats["moves"] += 1
            if outcome == "satellite_move":
                stats["satellite_moves"] += 1

        else:
            # none: не смог найти ни commute ни жильё
            # Проверяем адаптацию как последний fallback
            if float(row["job_flexibility"]) > ADAPT_FLEX_THRESHOLD:
                _execute_adapt(df, idx)
                stats["adapts"] += 1
            else:
                df.at[idx, "intention_state"] = "none"
                df.at[idx, "dst_work"]        = ""

    return df, stats


# ── Главный tick ──────────────────────────────────────────────────────────────

EVENT_SOCIAL_BOOST = 0.08   # прирост social_boost от события
SOCIAL_BOOST_DECAY = 0.80   # множитель затухания за тик


def tick(
    df: pd.DataFrame,
    G: nx.DiGraph,
    jobs_capacity: dict,
    tick_num: int,
    rng: np.random.Generator,
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
                df.at[idx, "status"]          = "unemployed"
                df.at[idx, "intention_state"] = "seeking_work"
                df.at[idx, "is_employed"]     = False
                df.at[idx, "graduation_tick"] = -1
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
    df, fft_stats     = _fft_pipeline(df, G, dissatisfaction, jobs_pressure, rng)

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
) -> tuple[pd.DataFrame, dict, list]:
    """
    Главный цикл симуляции.

    Принимает jobs_capacity явно — строится в agents.py из commuting-матрицы
    и передаётся через run.py.
    """
    rng = np.random.default_rng(seed)

    if snapshot_ticks is None:
        snapshot_ticks = [0, n_ticks // 4, n_ticks // 2, n_ticks]

    # Инициализируем граф реальными counts
    residence_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, residence_counts, len(df))

    snapshots  = {}
    tick_stats = []

    if 0 in snapshot_ticks:
        snapshots[0] = df.copy()

    if verbose:
        print(f"\nСимуляция: {n_ticks} тиков | {len(df):,} агентов | {G.number_of_nodes()} районов")
        print(f"{'Тик':>5} {'Год':>4} "
              f"{'Акт':>6} {'Найд':>6} {'Commute':>8} {'Move':>6} "
              f"{'Satel':>6} {'Adapt':>6} "
              f"{'Безраб':>8} {'Dissat':>8} {'MaxPres':>8}")
        print("─" * 82)

    for t in range(1, n_ticks + 1):
        df, stats = tick(df, G, jobs_capacity, t, rng)
        tick_stats.append(stats)

        if t in snapshot_ticks:
            snapshots[t] = df.copy()

        if verbose and (t % 6 == 0 or t == 1 or t == n_ticks):
            yr = t // 12
            mo = t % 12 or 12
            print(
                f"  {t:3d} г{yr}м{mo:02d}"
                f"  {stats['activated']:>5}"
                f"  {stats['dst_found']:>5}"
                f"  {stats['commutes']:>7}"
                f"  {stats['moves']:>5}"
                f"  {stats['satellite_moves']:>5}"
                f"  {stats['adapts']:>5}"
                f"  {stats['n_unemployed']:>7,}"
                f"  {stats['avg_dissat']:>7.4f}"
                f"  {stats['jobs_pressure_max']:>7.2f}"
            )

    if verbose:
        print("─" * 82)
        total_moves    = sum(s["moves"]   for s in tick_stats)
        total_commutes = sum(s["commutes"] for s in tick_stats)
        total_adapts   = sum(s["adapts"]   for s in tick_stats)
        total_sat      = sum(s["satellite_moves"] for s in tick_stats)
        print(f"\n  Итого переездов:        {total_moves:,}")
        print(f"  Итого новых commute:    {total_commutes:,}")
        print(f"  Итого спутник-переездов:{total_sat:,}")
        print(f"  Итого адаптаций:        {total_adapts:,}")
        print(f"  Безработных в конце:    {tick_stats[-1]['n_unemployed']:,}")

    return df, snapshots, tick_stats
