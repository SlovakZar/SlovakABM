"""
engine.py v2 — tick-loop с TPB pipeline и четырьмя доменами.

Логика тика:
  1. age += 1/12, tenure += 1, moved_ticks += 1
  2. update_domain_satisfaction() — оценивает каждый домен vs среда
  3. compute_dissatisfaction() — взвешенная сумма активных доменов
  4. TPB pipeline:
       none    → forming   если dissatisfaction ≥ inertia
       forming → strong    если forming_ticks ≥ forming_duration
       strong  → решение   satisficing из awareness_set
  5. Решение: move / commute / adapt / stay
  6. network_signal: переехавшие агенты обновляют сигнал контактам
  7. update_graph() — среда реагирует на новое распределение

Ключевые принципы из документа модели:
  - perceived_control фильтрует awareness_set
  - info_quality модифицирует точность знания о районе
  - inertia сравнивается с dissatisfaction как барьер
  - satisficing: агент выбирает первый вариант выше порога, не оптимум
  - network_signal: эндогенный — переехавшие передают сигнал контактам
  - Три исхода: move (переезд), commute (маятник), adapt (гибкость на месте)
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional

from graph import update_graph, get_awareness_set

# ── Веса доменов в utility-функции ────────────────────────────────────────────
# Относительный вклад каждого измерения в привлекательность района
U_WAGE_W    = 0.50   # зарплатный стимул
U_HOUSING_W = 0.25   # жилищный стимул
U_EMPLOY_W  = 0.25   # стимул занятости

# Штраф времени в пути (за каждую минуту)
TIME_PENALTY = 0.0018   # 60 мин → -0.108

# ── Параметры обновления доменов ──────────────────────────────────────────────
SAT_SMOOTHING = 0.90   # инерция EMA для satisfaction (медленная адаптация)

# ── Параметры network_signal ──────────────────────────────────────────────────
SIGNAL_DECAY    = 0.15   # доля агентов теряющих positive signal каждый тик
SIGNAL_POSITIVE_BOOST = 0.08   # как сильно positive signal снижает dissatisfaction


# ── Обновление доменов ────────────────────────────────────────────────────────

def _economic_value(agent_wage: float, district: str, G: nx.DiGraph) -> float:
    """
    Нормализованная экономическая ценность текущего места [0,1].
    Сравниваем личную зарплату и местный рынок труда со средним по Словакии.
    """
    attr = G.nodes[district]
    # Медиана по всем узлам (вычисляется один раз за тик — передаётся снаружи)
    return _sigmoid(
        0.5 * (agent_wage - attr["avg_wage"]) / max(attr["avg_wage"], 1) -
        0.3 * (attr["housing_price_m2"] - 1800) / 1800 +
        0.2 * math.log1p(attr["jobs_capacity"] / max(attr["real_population"], 1) * 100)
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-4.0 * x))


def update_domain_satisfaction(
    df: pd.DataFrame,
    G: nx.DiGraph,
) -> pd.DataFrame:
    """
    Обновляет value каждого из четырёх доменов для всех агентов.

    Economic: личная зарплата vs средняя в районе + affordability жилья
    Social:   реагирует на network_signal от контактов
    Family:   медленно дрейфует (стохастические жизненные события)
    Place:    aspirational gap + видимые сигналы среды
    """
    sat_econ   = df["sat_economic"].values.copy()
    sat_social = df["sat_social"].values.copy()
    sat_family = df["sat_family"].values.copy()
    sat_place  = df["sat_place"].values.copy()

    for district in df["district"].unique():
        mask = (df["district"] == district).values
        if not mask.any():
            continue
        attr = G.nodes.get(district, {})
        avg_wage = attr.get("avg_wage", 1200)
        housing  = attr.get("housing_price_m2", 1800)
        jobs_cap = attr.get("jobs_capacity", 1000)
        real_pop = attr.get("real_population", 10000)

        idxs = np.where(mask)[0]
        wages = df.iloc[idxs]["wage"].values

        for i, idx in enumerate(idxs):
            w = wages[i]

            # ── Economic domain ───────────────────────────────────────────────
            q_wage    = (w - avg_wage) / max(avg_wage, 1)
            q_housing = (1800 - housing) / 1800           # affordability
            q_employ  = math.log1p(jobs_cap / max(real_pop, 1) * 100) / 5
            raw_econ  = 0.5 * q_wage + 0.3 * q_housing + 0.2 * q_employ
            target_econ = _sigmoid(raw_econ)
            sat_econ[idx] = (SAT_SMOOTHING * sat_econ[idx] +
                             (1 - SAT_SMOOTHING) * target_econ)

            # ── Social domain ─────────────────────────────────────────────────
            signal = df.iloc[idx]["network_signal"]
            susc   = float(df.iloc[idx]["net_signal_susc"])
            if signal == "positive":
                # Положительный сигнал от переехавших → снижение social satisfaction
                # (знакомые уехали → ощущение потери + информация о лучшем месте)
                delta_social = -susc * SIGNAL_POSITIVE_BOOST
            elif signal == "shock":
                delta_social = -susc * 0.15
            else:
                delta_social = 0.0
            sat_social[idx] = float(np.clip(
                SAT_SMOOTHING * sat_social[idx] + (1 - SAT_SMOOTHING) * sat_social[idx] + delta_social,
                0.0, 1.0
            ))

            # ── Family domain ─────────────────────────────────────────────────
            # Медленный дрейф: ± случайное жизненное событие (упрощение)
            sat_family[idx] = float(np.clip(
                sat_family[idx] + np.random.normal(0, 0.005), 0.0, 1.0
            ))

            # ── Place domain ──────────────────────────────────────────────────
            # Aspirational gap: как далеко место от "желаемого будущего"
            # Упрощение: коррелирует с экономическим доменом + case place_aspiration
            place_aspiration = float(df.iloc[idx]["sat_place"])
            target_place = (target_econ * 0.6 + place_aspiration * 0.4)
            sat_place[idx] = float(np.clip(
                SAT_SMOOTHING * sat_place[idx] + (1 - SAT_SMOOTHING) * target_place,
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
    Взвешенная диссatisfaction: сумма вкладов активных (ниже порога) доменов.

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
    rng: np.random.Generator,
) -> float:
    """
    Utility переезда из src в dst с учётом качества информации.

    info_quality модифицирует точность знания о целевом районе:
      observed_value = true_value × quality + prior × (1 - quality)
    Prior: слегка положительное смещение (люди склонны идеализировать)
    """
    if src == dst:
        return 0.0

    a_src = G.nodes.get(src, {})
    a_dst = G.nodes.get(dst, {})

    w_src = a_src.get("avg_wage", 1200)
    w_dst = a_dst.get("avg_wage", 1200)
    h_src = a_src.get("housing_price_m2", 1800)
    h_dst = a_dst.get("housing_price_m2", 1800)
    j_src = a_src.get("jobs_capacity", 1) / max(a_src.get("real_population", 1), 1)
    j_dst = a_dst.get("jobs_capacity", 1) / max(a_dst.get("real_population", 1), 1)

    # Искажение информации через info_quality
    prior_wage_bias = 1.08   # агенты слегка переоценивают зарплату в других местах
    prior_housing_bias = 0.95
    observed_wage    = w_dst * info_quality + w_dst * prior_wage_bias * (1 - info_quality)
    observed_housing = h_dst * info_quality + h_dst * prior_housing_bias * (1 - info_quality)
    # Добавляем шум пропорциональный неопределённости
    noise_sd = (1 - info_quality) * 0.1
    observed_wage    += rng.normal(0, observed_wage * noise_sd)
    observed_housing += rng.normal(0, observed_housing * noise_sd)

    u_wage    = U_WAGE_W    * (observed_wage - w_src) / max(w_src, 1)
    u_housing = U_HOUSING_W * (h_src - observed_housing) / max(h_src, 1)
    u_employ  = U_EMPLOY_W  * (j_dst - j_src) / max(j_src, 0.001)

    # Штраф времени в пути
    if G.has_edge(src, dst):
        travel_t = G[src][dst].get("travel_time_min", 60)
    else:
        travel_t = 120.0  # нет прямой связи → большой штраф
    u_time = -TIME_PENALTY * travel_t

    return u_wage + u_housing + u_employ + u_time


# ── TPB Pipeline ──────────────────────────────────────────────────────────────

def _tpb_step(
    df: pd.DataFrame,
    G: nx.DiGraph,
    dissatisfaction: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """
    Применяет TPB pipeline для всех агентов.

    Состояния:
      none    → forming  если dissat ≥ inertia AND age ≥ 18 AND moved_ticks ≥ 6
      forming → strong   если forming_ticks ≥ forming_duration
      strong  → решение  satisficing из awareness_set

    Решение:
      adapt    если job_flexibility высокая (адаптируется на месте)
      commute  если commuter_threshold пройден И dissat в economic домене
      move     если internal_mig_thr пройден → переезд внутри страны
      (external: если external_mig_thr пройден — отдельный исход)
    """
    df = df.copy()
    stats = {"moves": 0, "commutes": 0, "adapts": 0, "forming_new": 0, "strong_new": 0}

    age_arr       = df["age"].values
    mt_arr        = df["moved_ticks"].values
    inertia_arr   = df["inertia"].values
    intent_arr    = df["intention_state"].values
    ft_arr        = df["forming_ticks"].values
    fd_arr        = df["forming_duration"].values

    # Индексы для каждой стадии
    # none → forming
    none_mask = (
        (intent_arr == "none") &
        (age_arr >= 18) &
        (mt_arr >= 6) &
        (dissatisfaction >= inertia_arr)
    )
    none_idx = np.where(none_mask)[0]
    for idx in none_idx:
        df.iloc[idx, df.columns.get_loc("intention_state")] = "forming"
        df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0
        stats["forming_new"] += 1

    # forming: инкремент или переход в strong
    forming_mask = (df["intention_state"] == "forming").values
    forming_idx  = np.where(forming_mask)[0]
    for idx in forming_idx:
        new_ft = int(df.iloc[idx]["forming_ticks"]) + 1
        df.iloc[idx, df.columns.get_loc("forming_ticks")] = new_ft
        if new_ft >= int(df.iloc[idx]["forming_duration"]):
            df.iloc[idx, df.columns.get_loc("intention_state")] = "strong"
            stats["strong_new"] += 1
        # Если dissat снова упала ниже inertia — возвращаемся в none
        elif dissatisfaction[idx] < df.iloc[idx]["inertia"] * 0.85:
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0

    # strong → решение
    strong_mask = (df["intention_state"] == "strong").values
    strong_idx  = np.where(strong_mask)[0]

    for idx in strong_idx:
        row = df.iloc[idx]
        age      = float(row["age"])
        district = row["district"]
        dissat_i = dissatisfaction[idx]
        pc       = float(row["perceived_control"])
        econ_pc  = float(row["econ_perceived_control"])
        net_loc  = bool(row["network_location"])
        iq       = float(row["info_quality"])
        flex     = float(row["job_flexibility"])
        comm_thr = float(row["commuter_threshold"])
        int_thr  = float(row["internal_mig_thr"])
        ext_thr  = float(row["external_mig_thr"])
        sat_econ = float(row["sat_economic"])
        thr_econ = float(row["thr_economic"])

        # ── Адаптация (третий выход) ──────────────────────────────────────────
        # Высокая гибкость позволяет снизить неудовлетворённость на месте
        if flex > 0.65 and dissat_i < 0.55:
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0
            # Небольшой прирост economic satisfaction от адаптации
            df.iloc[idx, df.columns.get_loc("sat_economic")] = float(np.clip(
                sat_econ + flex * 0.08, 0.0, 1.0
            ))
            stats["adapts"] += 1
            continue

        # ── Awareness set ─────────────────────────────────────────────────────
        candidates = get_awareness_set(G, district,
                                       network_location=net_loc,
                                       perceived_control=pc)
        if not candidates:
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            continue

        # ── Маятниковая миграция (commute) ────────────────────────────────────
        # Условие: economic домен под порогом, inertia высокая
        econ_stressed = sat_econ < thr_econ
        high_inertia  = float(row["inertia"]) > 0.55
        if econ_stressed and high_inertia and econ_pc > comm_thr:
            # Ищем ближайший район с лучшей зарплатой
            commute_candidates = [
                c for c in candidates
                if G.has_edge(district, c) and
                   G[district][c].get("travel_time_min", 999) <= 60 and
                   G.nodes[c].get("avg_wage", 0) > G.nodes[district].get("avg_wage", 0)
            ]
            if commute_candidates:
                df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
                df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0
                # Слабые связи накапливаются в месте работы
                df.iloc[idx, df.columns.get_loc("weak_ties_utility")] = float(np.clip(
                    float(row["weak_ties_utility"]) + 0.03, 0.0, 1.0
                ))
                stats["commutes"] += 1
                continue

        # ── Переезд (move) ────────────────────────────────────────────────────
        # Satisficing: выбираем первый вариант превышающий порог utility
        # Порог определяется internal или external threshold
        is_external = False
        move_threshold = int_thr  # порог для внутреннего переезда

        best_dst   = None
        best_util  = -999.0

        # Перемешиваем кандидатов (satisficing, не оптимизация)
        shuffled = list(candidates)
        rng.shuffle(shuffled)

        for candidate in shuffled:
            u = _compute_move_utility(district, candidate, G,
                                      float(row["wage"]), iq, rng)
            if u > best_util:
                best_util = u
                best_dst  = candidate
            # Satisficing: берём первый вариант выше порога
            if u > move_threshold:
                best_dst = candidate
                break

        if best_dst is not None and best_util > move_threshold:
            # ── Выполняем переезд ─────────────────────────────────────────────
            df.iloc[idx, df.columns.get_loc("district")]         = best_dst
            df.iloc[idx, df.columns.get_loc("region")]           = \
                G.nodes[best_dst].get("region", row["region"])
            df.iloc[idx, df.columns.get_loc("moved_ticks")]      = 0
            df.iloc[idx, df.columns.get_loc("tenure")]           = 0
            df.iloc[idx, df.columns.get_loc("intention_state")]  = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]    = 0

            # Зарплата корректируется к среднему нового района
            target_wage = G.nodes[best_dst].get("avg_wage", float(row["wage"]))
            new_wage    = float(rng.normal(target_wage, target_wage * 0.18))
            df.iloc[idx, df.columns.get_loc("wage")] = max(0.0, new_wage)

            # Satisfaction кратковременно снижается (стресс переезда)
            for sat_col in ["sat_economic", "sat_social", "sat_family"]:
                df.iloc[idx, df.columns.get_loc(sat_col)] = float(np.clip(
                    df.iloc[idx][sat_col] * 0.78, 0.05, 0.95
                ))

            # Inertia обновляется: стаж сброшен, social component остаётся
            new_inertia = float(np.clip(float(row["inertia_social"]) * 0.4 + 0.1, 0.05, 0.9))
            df.iloc[idx, df.columns.get_loc("inertia")] = new_inertia

            stats["moves"] += 1
        else:
            # Нет подходящего варианта → возвращаемся в none
            df.iloc[idx, df.columns.get_loc("intention_state")] = "none"
            df.iloc[idx, df.columns.get_loc("forming_ticks")]   = 0

    return df, stats


# ── Network signal update ─────────────────────────────────────────────────────

def update_network_signals(df: pd.DataFrame, movers_districts: dict) -> pd.DataFrame:
    """
    Обновляет network_signal агентов на основе переездов их контактов.

    Упрощённая логика: если в районе агента был большой отток —
    агенты с network_location=True получают positive signal.
    Decay: часть сигналов затухает каждый тик.
    """
    df = df.copy()
    signals = df["network_signal"].values.copy()

    for idx in range(len(df)):
        current_signal = signals[idx]

        # Затухание
        if current_signal == "positive" and np.random.random() < SIGNAL_DECAY:
            signals[idx] = "neutral"
            continue

        # Новый сигнал от переезда контактов
        if bool(df.iloc[idx]["network_location"]) and current_signal == "neutral":
            district = df.iloc[idx]["district"]
            if district in movers_districts and movers_districts[district] > 0:
                susc = float(df.iloc[idx]["net_signal_susc"])
                if np.random.random() < susc * 0.15:
                    signals[idx] = "positive"

    df["network_signal"] = signals
    return df


# ── Главный tick ──────────────────────────────────────────────────────────────

def tick(
    df: pd.DataFrame,
    G: nx.DiGraph,
    tick_num: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Один шаг симуляции. Возвращает обновлённый DataFrame и статистику."""

    n_agents = len(df)

    # 1. Время
    df = df.copy()
    df["age"]         = df["age"] + 1/12
    df["tenure"]      = df["tenure"] + 1
    df["moved_ticks"] = df["moved_ticks"] + 1

    # 2. Обновление доменов
    df = update_domain_satisfaction(df, G)

    # 3. Диссatisfaction
    dissatisfaction = compute_dissatisfaction(df)

    # 4. TPB pipeline
    df, move_stats = _tpb_step(df, G, dissatisfaction, rng)

    # 5. Network signals
    # Считаем откуда уехали агенты в этом тике
    # (approximation: из move_stats мы знаем общее число но не откуда)
    # Используем предыдущее district-распределение как прокси
    district_counts = df.groupby("district")["id"].count().to_dict()
    movers_proxy    = {d: max(0, district_counts.get(d, 0)) for d in G.nodes}
    df = update_network_signals(df, movers_proxy)

    # 6. Реакция среды
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
        "env_wages":   {d: G.nodes[d]["avg_wage"] for d in list(G.nodes)[:10]},
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

    rng = np.random.default_rng(seed)

    if snapshot_ticks is None:
        snapshot_ticks = [0, n_ticks // 4, n_ticks // 2, n_ticks]

    # Инициализируем agent_count в графе
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
