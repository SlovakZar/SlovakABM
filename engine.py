"""
engine.py v3 — tick-loop с переработанным TPB pipeline.

Изменения относительно v2:
  1. perceived_control — вероятностный фильтр перехода forming → strong.
     Агент с низким pc чаще «передумывает» и возвращается в none.
     Убран из get_awareness_set (там был просто радиус поиска).

  2. _tpb_step разбит на три явные фазы внутри strong → решение:
     Фаза А: диагностика давления доменов → primary_driver
     Фаза Б: оценка вариантов через расширенную utility
     Фаза В: выбор исхода (move / commute / adapt / stay)

  3. Расширенная utility U(dst):
     u_economic  — зарплата, жильё, занятость
                   модифицируется econ_perceived_control агента
     u_place     — инфраструктура района + перспективность (зарплата vs нац. среднее)
                   взвешивается на w_future агента
     u_social    — бонус за наличие контактов в dst (network_location + signal)
                   взвешивается на weak_ties_utility
     u_distance  — штраф времени в пути
     u_inertia   — стоимость разрыва социальных связей (inertia_social)

  4. internal_mig_thr — вероятностный фильтр готовности агента к переезду,
     не порог utility. Агент с высоким thr реже проходит этот фильтр.
     Маятниковая миграция НЕ зависит от internal_mig_thr — commute проще.

  5. adapt — fallback в конце, только если:
     — нет позитивного варианта (best_util <= 0)
     — primary_driver == economic (адаптация не решает place/social проблему)
     — job_flexibility > 0.80

  6. place и social домены теперь входят в utility направления,
     а не только в dissatisfaction агрегат.
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional

from graph import update_graph, get_awareness_set

# ── Константы utility функции ─────────────────────────────────────────────────

# Веса компонентов utility переезда
U_WAGE_W    = 0.45   # зарплатный стимул
U_HOUSING_W = 0.20   # жилищный стимул (affordability)
U_EMPLOY_W  = 0.15   # стимул занятости
U_PLACE_W   = 0.20   # инфраструктура + перспективность района

# Сетевой бонус
U_NETWORK_POSITIVE = 0.18  # network_location=True + signal=positive
U_NETWORK_NEUTRAL  = 0.07  # network_location=True, signal=neutral

# Стоимость разрыва социальных связей при переезде
SOCIAL_COST_WEIGHT = 0.20

# Штраф времени в пути (за каждую минуту)
TIME_PENALTY = 0.0018      # 60 мин → -0.108

# Национальное среднее зарплаты для нормировки u_place
NATIONAL_AVG_WAGE = 1614.0

# ── Параметры обновления доменов ──────────────────────────────────────────────
SAT_SMOOTHING = 0.90        # инерция EMA для satisfaction

# ── Параметры network_signal ──────────────────────────────────────────────────
SIGNAL_DECAY         = 0.15
SIGNAL_POSITIVE_BOOST = 0.08


# ── Обновление доменов ────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-4.0 * x))


def update_domain_satisfaction(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """
    Обновляет value каждого из четырёх доменов для всех агентов.

    Economic: личная зарплата vs средняя в районе + affordability жилья + занятость.
    Social:   реагирует на network_signal от контактов.
    Family:   медленный стохастический дрейф (жизненные события).
    Place:    aspirational gap — функция экономики района + инфраструктуры.
    """
    sat_econ   = df["sat_economic"].values.copy()
    sat_social = df["sat_social"].values.copy()
    sat_family = df["sat_family"].values.copy()
    sat_place  = df["sat_place"].values.copy()

    for district in df["district"].unique():
        mask = (df["district"] == district).values
        if not mask.any():
            continue

        attr     = G.nodes.get(district, {})
        avg_wage = attr.get("avg_wage", 1200)
        housing  = attr.get("housing_price_m2", 1800)
        jobs_cap = attr.get("jobs_capacity", 1000)
        real_pop = attr.get("real_population", 10000)
        infra    = attr.get("infrastructure_score", 0.0)

        idxs  = np.where(mask)[0]
        wages = df.iloc[idxs]["wage"].values

        for i, idx in enumerate(idxs):
            w = wages[i]

            # ── Economic domain ───────────────────────────────────────────────
            q_wage    = (w - avg_wage) / max(avg_wage, 1)
            q_housing = (1800 - housing) / 1800
            q_employ  = math.log1p(jobs_cap / max(real_pop, 1) * 100) / 5
            raw_econ  = 0.5 * q_wage + 0.3 * q_housing + 0.2 * q_employ
            target_econ = _sigmoid(raw_econ)
            sat_econ[idx] = (SAT_SMOOTHING * sat_econ[idx] +
                             (1 - SAT_SMOOTHING) * target_econ)

            # ── Social domain ─────────────────────────────────────────────────
            signal = df.iloc[idx]["network_signal"]
            susc   = float(df.iloc[idx]["net_signal_susc"])
            if signal == "positive":
                delta_social = -susc * SIGNAL_POSITIVE_BOOST
            elif signal == "shock":
                delta_social = -susc * 0.15
            else:
                delta_social = 0.0
            sat_social[idx] = float(np.clip(
                SAT_SMOOTHING * sat_social[idx]
                + (1 - SAT_SMOOTHING) * sat_social[idx]
                + delta_social,
                0.0, 1.0
            ))

            # ── Family domain ─────────────────────────────────────────────────
            sat_family[idx] = float(np.clip(
                sat_family[idx] + np.random.normal(0, 0.005), 0.0, 1.0
            ))

            # ── Place domain ──────────────────────────────────────────────────
            # Зависит от экономики района + инфраструктуры
            # infra нормирован в [0,1] через infrastructure_score из графа
            target_place = _sigmoid(
                0.6 * (target_econ - 0.5) +      # экономика тянет место
                0.4 * (infra - 0.5)               # инфраструктура
            )
            sat_place[idx] = float(np.clip(
                SAT_SMOOTHING * sat_place[idx] +
                (1 - SAT_SMOOTHING) * target_place,
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
    Взвешенная dissatisfaction: сумма вкладов активных (ниже порога) доменов.
    Домен активируется если value < threshold.
    Вклад = weight × (threshold - value) / threshold
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
        dissat += w * gap
    return np.clip(dissat, 0.0, 1.0)


# ── Utility функция для выбора района ─────────────────────────────────────────

def _compute_move_utility(
    src: str,
    dst: str,
    G: nx.DiGraph,
    agent_wage: float,
    info_quality: float,
    econ_perceived_control: float,
    weak_ties: float,
    network_location: bool,
    network_signal: str,
    w_future: float,
    inertia_social: float,
    rng: np.random.Generator,
) -> float:
    """
    Многомерная utility переезда из src в dst.

    Компоненты:
      u_economic  — разница в зарплате, жилье и занятости между районами.
                    Модифицируется econ_perceived_control: агент с высоким
                    econ_pc менее чувствителен к зарплатной разнице
                    (уверен что найдёт себя), с низким — более.
      u_place     — инфраструктура + перспективность района назначения.
                    Взвешивается на w_future агента.
      u_social    — бонус за наличие знакомых в dst (network_location + signal).
                    Взвешивается на weak_ties_utility агента.
      u_inertia   — штраф за разрыв социальных связей при переезде.
      u_distance  — штраф времени в пути.

    info_quality искажает знание о dst: низкое качество → шум + позитивное смещение.
    """
    if src == dst:
        return 0.0

    a_src = G.nodes.get(src, {})
    a_dst = G.nodes.get(dst, {})

    # ── Экономические характеристики ─────────────────────────────────────────
    w_src = a_src.get("avg_wage", NATIONAL_AVG_WAGE)
    w_dst = a_dst.get("avg_wage", NATIONAL_AVG_WAGE)
    h_src = a_src.get("housing_price_m2", 1800)
    h_dst = a_dst.get("housing_price_m2", 1800)
    j_src = a_src.get("jobs_capacity", 1) / max(a_src.get("real_population", 1), 1)
    j_dst = a_dst.get("jobs_capacity", 1) / max(a_dst.get("real_population", 1), 1)

    # Искажение информации о dst
    noise_sd = (1 - info_quality) * 0.10
    prior_wage_bias    = 1.06
    prior_housing_bias = 0.96
    obs_wage    = (w_dst * info_quality
                   + w_dst * prior_wage_bias * (1 - info_quality)
                   + rng.normal(0, w_dst * noise_sd))
    obs_housing = (h_dst * info_quality
                   + h_dst * prior_housing_bias * (1 - info_quality)
                   + rng.normal(0, h_dst * noise_sd))

    # econ_perceived_control модифицирует чувствительность к зарплатной разнице:
    # высокий econ_pc → уверен что справится → меньше реагирует на чужую зарплату
    econ_sensitivity = 0.5 + 0.5 * (1.0 - econ_perceived_control)

    u_wage    = U_WAGE_W    * econ_sensitivity * (obs_wage - w_src) / max(w_src, 1)
    u_housing = U_HOUSING_W * (h_src - obs_housing) / max(h_src, 1)
    u_employ  = U_EMPLOY_W  * (j_dst - j_src) / max(j_src, 0.001)
    u_economic = u_wage + u_housing + u_employ

    # ── Place utility ─────────────────────────────────────────────────────────
    # Инфраструктура нормирована в [0,1] через infrastructure_score в graph.py
    infra_dst  = a_dst.get("infrastructure_score", 0.0)
    # Перспективность: насколько зарплата в dst выше нац. среднего
    wage_ratio = obs_wage / NATIONAL_AVG_WAGE - 1.0   # >0 если выше среднего
    u_place = U_PLACE_W * w_future * (0.5 * infra_dst + 0.5 * wage_ratio)

    # ── Social utility ────────────────────────────────────────────────────────
    # Бонус за наличие знакомых в dst.
    # network_location — есть ли у агента хотя бы один удалённый контакт.
    # network_signal — получил ли он позитивный сигнал от переехавших.
    # Мы не знаем точно из каких районов пришёл сигнал, поэтому:
    # — signal=positive и network_location=True: предполагаем связь с dst через сеть
    # — network_location=True без сигнала: меньший бонус (просто знает кого-то там)
    if network_location and network_signal == "positive":
        u_social = weak_ties * U_NETWORK_POSITIVE
    elif network_location:
        u_social = weak_ties * U_NETWORK_NEUTRAL
    else:
        u_social = 0.0

    # ── Стоимость разрыва социальных связей ──────────────────────────────────
    # inertia_social — глубина укоренённости. Высокая → дороже уезжать.
    u_inertia = -SOCIAL_COST_WEIGHT * inertia_social

    # ── Штраф расстояния ─────────────────────────────────────────────────────
    if G.has_edge(src, dst):
        travel_t = G[src][dst].get("travel_time_min", 60)
    else:
        travel_t = 120.0
    u_distance = -TIME_PENALTY * travel_t

    return u_economic + u_place + u_social + u_inertia + u_distance


# ── TPB Pipeline ──────────────────────────────────────────────────────────────

def _tpb_step(
    df: pd.DataFrame,
    G: nx.DiGraph,
    dissatisfaction: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """
    TPB pipeline: три стадии намерения + трёхфазное принятие решения.

    none → forming:
      Условие: dissat ≥ inertia AND age ≥ 18 AND moved_ticks ≥ 6

    forming → strong:
      Условие: forming_ticks ≥ forming_duration
      + вероятностный фильтр perceived_control:
        агент с низким pc чаще «передумывает» и возвращается в none.
      + откат в none если dissat упала ниже inertia*0.85

    strong → решение (три фазы):
      Фаза А: какой домен давит сильнее всего (primary_driver)
      Фаза Б: оценка всех кандидатов через расширенную utility
      Фаза В: выбор исхода — move / commute / adapt / stay
        — internal_mig_thr: вероятностный фильтр готовности к переезду
        — commute: если лучший dst близко (≤60 мин) И primary_driver=economic
        — adapt: только fallback, если нет позитивного варианта И primary_driver=economic
    """
    df = df.copy()
    stats = {"moves": 0, "commutes": 0, "adapts": 0, "forming_new": 0, "strong_new": 0}

    age_arr     = df["age"].values
    mt_arr      = df["moved_ticks"].values
    inertia_arr = df["inertia"].values
    intent_arr  = df["intention_state"].values

    # ── none → forming ────────────────────────────────────────────────────────
    none_mask = (
        (intent_arr == "none") &
        (age_arr >= 18) &
        (mt_arr >= 6) &
        (dissatisfaction >= inertia_arr)
    )
    for idx in np.where(none_mask)[0]:
        df.iloc[idx, df.columns.get_loc("intention_state")] = "forming"
        df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0
        stats["forming_new"] += 1

    # ── forming: инкремент / переход в strong / откат ─────────────────────────
    for idx in np.where((df["intention_state"] == "forming").values)[0]:
        new_ft = int(df.iloc[idx]["forming_ticks"]) + 1
        df.iloc[idx, df.columns.get_loc("forming_ticks")] = new_ft

        # Откат если dissat упала
        if dissatisfaction[idx] < df.iloc[idx]["inertia"] * 0.85:
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0
            continue

        if new_ft >= int(df.iloc[idx]["forming_duration"]):
            # Вероятностный фильтр perceived_control:
            # низкий pc → агент не верит что переезд для него реален → none
            pc = float(df.iloc[idx]["perceived_control"])
            if rng.random() < pc:
                df.iloc[idx, df.columns.get_loc("intention_state")] = "strong"
                stats["strong_new"] += 1
            else:
                df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
                df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0

    # ── strong → решение ──────────────────────────────────────────────────────
    for idx in np.where((df["intention_state"] == "strong").values)[0]:
        row      = df.iloc[idx]
        district = row["district"]
        dissat_i = dissatisfaction[idx]

        # Извлекаем параметры агента
        pc       = float(row["perceived_control"])
        econ_pc  = float(row["econ_perceived_control"])
        net_loc  = bool(row["network_location"])
        net_sig  = str(row["network_signal"])
        iq       = float(row["info_quality"])
        flex     = float(row["job_flexibility"])
        weak_ties= float(row["weak_ties_utility"])
        int_thr  = float(row["internal_mig_thr"])
        comm_thr = float(row["commuter_threshold"])
        w_future = float(row["w_future"])
        inertia_s= float(row["inertia_social"])

        sat_e = float(row["sat_economic"])
        sat_s = float(row["sat_social"])
        sat_f = float(row["sat_family"])
        sat_p = float(row["sat_place"])
        thr_e = float(row["thr_economic"])
        thr_s = float(row["thr_social"])
        thr_f = float(row["thr_family"])
        thr_p = float(row["thr_place"])
        w_e   = float(row["w_economic"])
        w_s   = float(row["w_social"])
        w_f   = float(row["w_family"])
        w_p   = float(row["w_future"])

        # ── Фаза А: диагностика давления доменов ─────────────────────────────
        pressures = {
            "economic": w_e * max(0, thr_e - sat_e) / max(thr_e, 0.01),
            "social":   w_s * max(0, thr_s - sat_s) / max(thr_s, 0.01),
            "family":   w_f * max(0, thr_f - sat_f) / max(thr_f, 0.01),
            "place":    w_p * max(0, thr_p - sat_p) / max(thr_p, 0.01),
        }
        primary_driver = max(pressures, key=pressures.get)

        # ── Фаза Б: оценка вариантов ─────────────────────────────────────────
        # perceived_control определяет awareness_set через get_awareness_set.
        # Высокий pc → шире радиус поиска (уже реализовано в graph.py).
        candidates = get_awareness_set(
            G, district,
            network_location=net_loc,
            perceived_control=pc
        )
        if not candidates:
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0
            continue

        # Оцениваем utility каждого кандидата
        # Satisficing: перемешиваем и берём первый вариант выше нуля
        shuffled = list(candidates)
        rng.shuffle(shuffled)

        best_dst  = None
        best_util = -999.0

        for candidate in shuffled:
            u = _compute_move_utility(
                district, candidate, G,
                float(row["wage"]), iq, econ_pc,
                weak_ties, net_loc, net_sig,
                w_future, inertia_s, rng
            )
            if u > best_util:
                best_util = u
                best_dst  = candidate

        # ── Фаза В: выбор исхода ─────────────────────────────────────────────

        # Фильтр готовности к переезду.
        # internal_mig_thr — психологический барьер: чем выше, тем реже агент
        # решается на переезд вообще. Маятник не зависит от этого фильтра.
        open_to_move = (rng.random() > int_thr)

        if best_util > 0 and open_to_move:
            # Есть позитивный вариант и агент готов двигаться.

            # Commute: если primary_driver экономический И dst близко
            travel = G[district][best_dst].get("travel_time_min", 999) \
                if G.has_edge(district, best_dst) else 999
            if primary_driver == "economic" and travel <= 60:
                # Проверяем что зарплата в dst реально выше
                dst_wage = G.nodes[best_dst].get("avg_wage", 0)
                src_wage = G.nodes[district].get("avg_wage", 0)
                if dst_wage > src_wage:
                    df.iloc[idx, df.columns.get_loc("intention_state")]  = "none"
                    df.iloc[idx, df.columns.get_loc("forming_ticks")]    = 0
                    df.iloc[idx, df.columns.get_loc("weak_ties_utility")] = float(np.clip(
                        weak_ties + 0.03, 0.0, 1.0
                    ))
                    stats["commutes"] += 1
                    continue

            # Move
            _execute_move(df, idx, best_dst, G, row, rng)
            stats["moves"] += 1

        elif best_util > 0 and not open_to_move:
            # Есть позитивный вариант, но агент психологически не готов.
            # Остаётся в none — возможно снова войдёт в forming позже.
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0

        elif primary_driver == "economic" and flex > 0.80:
            # Нет позитивного варианта, но можно адаптироваться на месте.
            # Adapt — только экономические проблемы, только высокая гибкость.
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0
            df.iloc[idx, df.columns.get_loc("sat_economic")]    = float(np.clip(
                sat_e + flex * 0.06, 0.0, 1.0
            ))
            stats["adapts"] += 1

        else:
            # Нет позитивного варианта, адаптация невозможна или нерелевантна.
            # Возвращаемся в none — агент продолжает жить с неудовлетворённостью.
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0

    return df, stats


def _execute_move(
    df: pd.DataFrame,
    idx: int,
    dst: str,
    G: nx.DiGraph,
    row: pd.Series,
    rng: np.random.Generator,
):
    """Выполняет переезд агента в dst. Изменяет df in-place."""
    df.iloc[idx, df.columns.get_loc("district")]        = dst
    df.iloc[idx, df.columns.get_loc("region")]          = \
        G.nodes[dst].get("region", row["region"])
    df.iloc[idx, df.columns.get_loc("moved_ticks")]     = 0
    df.iloc[idx, df.columns.get_loc("tenure")]          = 0
    df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
    df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0

    # Зарплата корректируется к среднему нового района
    target_wage = G.nodes[dst].get("avg_wage", float(row["wage"]))
    new_wage    = float(rng.normal(target_wage, target_wage * 0.18))
    df.iloc[idx, df.columns.get_loc("wage")] = max(0.0, new_wage)

    # Satisfaction кратковременно снижается — стресс переезда
    for col in ["sat_economic", "sat_social", "sat_family"]:
        df.iloc[idx, df.columns.get_loc(col)] = float(np.clip(
            df.iloc[idx][col] * 0.80, 0.05, 0.95
        ))

    # Inertia пересчитывается: стаж и место сброшены, social component остаётся
    new_inertia = float(np.clip(
        float(row["inertia_social"]) * 0.35 + 0.10,
        0.05, 0.90
    ))
    df.iloc[idx, df.columns.get_loc("inertia")] = new_inertia


# ── Network signal update ─────────────────────────────────────────────────────

def update_network_signals(
    df: pd.DataFrame,
    moved_from_districts: dict,
) -> pd.DataFrame:
    """
    Обновляет network_signal агентов на основе реальных переездов в этом тике.

    moved_from_districts: {district: count_of_agents_who_left}
    Агенты с network_location=True в районах с оттоком получают positive signal.
    """
    df     = df.copy()
    signals = df["network_signal"].values.copy()

    for idx in range(len(df)):
        current = signals[idx]

        # Затухание positive сигнала
        if current == "positive" and rng_global.random() < SIGNAL_DECAY:
            signals[idx] = "neutral"
            continue

        # Новый сигнал: если в районе агента был отток и агент имеет удалённые контакты
        if current == "neutral" and bool(df.iloc[idx]["network_location"]):
            district = df.iloc[idx]["district"]
            outflow  = moved_from_districts.get(district, 0)
            if outflow > 0:
                susc = float(df.iloc[idx]["net_signal_susc"])
                if rng_global.random() < susc * 0.20:
                    signals[idx] = "positive"

    df["network_signal"] = signals
    return df


# Глобальный RNG для update_network_signals (инициализируется в run_simulation)
rng_global = np.random.default_rng(42)


# ── Главный tick ──────────────────────────────────────────────────────────────

def tick(
    df: pd.DataFrame,
    G: nx.DiGraph,
    tick_num: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Один шаг симуляции. Возвращает обновлённый DataFrame и статистику."""
    global rng_global
    rng_global = rng

    n_agents = len(df)

    # 1. Время
    df = df.copy()
    df["age"]         = df["age"] + 1 / 12
    df["tenure"]      = df["tenure"] + 1
    df["moved_ticks"] = df["moved_ticks"] + 1

    # 2. Обновление доменов
    df = update_domain_satisfaction(df, G)

    # 3. Dissatisfaction
    dissatisfaction = compute_dissatisfaction(df)

    # 4. TPB pipeline
    district_before = df["district"].copy()
    df, move_stats  = _tpb_step(df, G, dissatisfaction, rng)

    # 5. Считаем реальный отток из каждого района
    moved_from = {}
    for i, (d_before, d_after) in enumerate(zip(district_before, df["district"])):
        if d_before != d_after:
            moved_from[d_before] = moved_from.get(d_before, 0) + 1

    # 6. Network signals — на основе реального оттока
    df = update_network_signals(df, moved_from)

    # 7. Реакция среды
    district_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, district_counts, n_agents)

    stats = {
        "tick":          tick_num,
        "moves":         move_stats["moves"],
        "commutes":      move_stats["commutes"],
        "adapts":        move_stats["adapts"],
        "forming_new":   move_stats["forming_new"],
        "strong_new":    move_stats["strong_new"],
        "move_rate_pct": round(move_stats["moves"] / n_agents * 100, 3),
        "avg_age":       round(float(df["age"].mean()), 2),
        "avg_wage":      round(float(df[df["wage"] > 0]["wage"].mean()), 2),
        "avg_dissat":    round(float(dissatisfaction.mean()), 4),
        "avg_inertia":   round(float(df["inertia"].mean()), 4),
        "district_counts": district_counts,
    }

    return df, stats


def run_simulation(
    df: pd.DataFrame,
    G: nx.DiGraph,
    n_ticks: int = 60,
    snapshot_ticks: Optional[list] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, list]:

    global rng_global
    rng = np.random.default_rng(seed)
    rng_global = rng

    if snapshot_ticks is None:
        snapshot_ticks = [0, n_ticks // 4, n_ticks // 2, n_ticks]

    district_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, district_counts, len(df))

    snapshots  = {}
    tick_stats = []

    if 0 in snapshot_ticks:
        snapshots[0] = df.copy()

    if verbose:
        print(f"\nЗапуск симуляции: {n_ticks} тиков | {len(df):,} агентов | {G.number_of_nodes()} районов")
        print(f"{'Тик':>5} {'Год':>5} {'Переезды':>9} {'Маятник':>8} {'Адапт':>7}"
              f" {'Forming':>8} {'Ср.dissat':>10} {'Ср.з/п':>8}")
        print("-" * 70)

    for t in range(1, n_ticks + 1):
        df, stats = tick(df, G, t, rng)
        tick_stats.append(stats)

        if t in snapshot_ticks:
            snapshots[t] = df.copy()

        if verbose and (t % 12 == 0 or t == n_ticks or t == 1):
            yr = t // 12
            mo = t % 12 or 12
            print(
                f"  {t:3d} (г{yr}м{mo:02d})"
                f"  {stats['moves']:>7} ({stats['move_rate_pct']:>4.2f}%)"
                f"  {stats['commutes']:>7}"
                f"  {stats['adapts']:>6}"
                f"  {stats['forming_new']:>7}"
                f"  {stats['avg_dissat']:>9.4f}"
                f"  {stats['avg_wage']:>7,.0f}€"
            )

    if verbose:
        print("-" * 70)
        print("Симуляция завершена.")

    return df, snapshots, tick_stats
