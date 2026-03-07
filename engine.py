"""
engine.py — главный цикл симуляции.

Каждый тик (= 1 месяц):
  1. age += 1/12 для всех агентов
  2. Для агентов старше 18 и не переехавших недавно:
     - вычисляем utility для всех доступных районов
     - если max(utility) > threshold + случайность → переезд
  3. Обновляем moved_ticks

Utility функция (примитивная, без эвристик):
  U(j) = w_wage    * (wage_j_avg - wage_i_avg) / wage_i_avg
       + w_housing * (housing_i - housing_j) / housing_i     ← дешевле = лучше
       + w_employ  * (employ_j - employ_i) / employ_i        ← больше работы = лучше
       + age_penalty(age)                                     ← старше → инертнее
       - distance_penalty * dist_ij_km                       ← дальше = хуже

Агент переезжает в район с максимальной utility, если она > MOVE_THRESHOLD.
"""

import random
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional


# ── веса утилитарной функции ──────────────────────────────────────────────────
W_WAGE     = 0.7    # вес зарплатного стимула (основной)
W_HOUSING  = 0.2    # вес жилищного стимула
W_EMPLOY   = 0.1    # вес занятости
DIST_PENALTY = 0.0005  # штраф за км расстояния (снижен: 50 км = 0.025)
MOVE_THRESHOLD = 0.04  # минимальная utility для решения о переезде
INERTIA_TICKS  = 12    # нельзя переехать снова ранее чем через N тиков
AGE_MOBILITY_PEAK = 28.0  # максимальная мобильность в этом возрасте
AGE_MOBILITY_DECAY = 0.025  # скорость спада мобильности с возрастом
MOVE_PROB_NOISE = 0.4  # случайность: доля от utility при принятии решения


def _age_mobility(age: float) -> float:
    """
    Коэффициент мобильности по возрасту [0, 1].
    Пик ~28 лет, логистический спад старше.
    """
    if age < 18:
        return 0.0
    # Gaussian-like пик вокруг 28, плавный спад
    return float(np.exp(-AGE_MOBILITY_DECAY * max(0.0, age - AGE_MOBILITY_PEAK) ** 1.5))


def _compute_utility(
    agent_district: str,
    candidate_district: str,
    G: nx.Graph,
    agent_wage: float,
) -> float:
    """
    Utility переезда из agent_district в candidate_district.
    """
    if agent_district == candidate_district:
        return 0.0

    attr_i = G.nodes[agent_district]
    attr_j = G.nodes[candidate_district]

    # Зарплатный стимул
    w_i = attr_i.get("avg_wage", 1) or 1
    w_j = attr_j.get("avg_wage", 1) or 1
    u_wage = W_WAGE * (w_j - w_i) / w_i

    # Жилищный стимул (чем дешевле в j, тем лучше)
    h_i = attr_i.get("housing_price_m2", 1) or 1
    h_j = attr_j.get("housing_price_m2", 1) or 1
    if h_i > 0:
        u_housing = W_HOUSING * (h_i - h_j) / h_i
    else:
        u_housing = 0.0

    # Стимул занятости (jobs per capita — нормализован)
    e_i = attr_i.get("employment_per_capita", 0.4) or 0.4
    e_j = attr_j.get("employment_per_capita", 0.4) or 0.4
    u_employ = W_EMPLOY * (e_j - e_i) / max(e_i, 0.01)

    # Штраф расстояния
    if G.has_edge(agent_district, candidate_district):
        dist_km = G[agent_district][candidate_district].get("distance_km", 50)
    else:
        # Если ребра нет — считаем через атрибуты узлов (все соединены)
        dist_km = 50
    u_distance = -DIST_PENALTY * dist_km

    return u_wage + u_housing + u_employ + u_distance


def tick(
    df: pd.DataFrame,
    G: nx.Graph,
    tick_num: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """
    Один шаг симуляции. Возвращает обновлённый DataFrame и статистику тика.

    Returns:
        df: обновлённый DataFrame агентов
        stats: dict с метриками тика
    """
    n_agents = len(df)

    # 1. Все стареют на 1 месяц
    df["age"] = df["age"] + 1 / 12

    # 2. Обновляем счётчик moved_ticks
    df["moved_ticks"] = df["moved_ticks"] + 1

    # 3. Миграционные решения
    districts = list(G.nodes())
    moves = 0

    # Отбираем кандидатов: 18+, прошло достаточно тиков с последнего переезда
    mask_eligible = (df["age"] >= 18) & (df["moved_ticks"] >= INERTIA_TICKS)
    eligible_idx = df.index[mask_eligible].tolist()

    # Для скорости — векторизуем атрибуты графа
    node_attrs = {d: G.nodes[d] for d in districts}

    for idx in eligible_idx:
        row = df.loc[idx]
        current = row["district"]
        age = row["age"]
        agent_wage = row["wage"]

        mob = _age_mobility(age)
        if mob < 0.01:
            continue

        # Вычисляем utility для всех других районов
        best_district = current
        best_utility = 0.0

        for candidate in districts:
            if candidate == current:
                continue
            u = _compute_utility(current, candidate, G, agent_wage)
            if u > best_utility:
                best_utility = u
                best_district = candidate

        # Принимаем решение о переезде с учётом мобильности и случайности
        if best_utility > MOVE_THRESHOLD:
            # Случайный шум снижает вероятность переезда
            noise = rng.uniform(0, MOVE_PROB_NOISE)
            effective_utility = best_utility * mob - noise
            if effective_utility > 0:
                df.at[idx, "district"] = best_district
                df.at[idx, "moved_ticks"] = 0
                # После переезда зарплата корректируется к среднему нового района
                target_wage = node_attrs[best_district].get("avg_wage", agent_wage) or agent_wage
                df.at[idx, "wage"] = float(
                    rng.normal(target_wage, target_wage * 0.25)
                )
                df.at[idx, "wage"] = max(0.0, df.at[idx, "wage"])
                moves += 1

    # Собираем статистику тика
    district_counts = df.groupby("district")["id"].count().to_dict()
    stats = {
        "tick": tick_num,
        "moves": moves,
        "move_rate_pct": round(moves / n_agents * 100, 3),
        "district_counts": district_counts,
        "avg_age": round(df["age"].mean(), 2),
        "avg_wage": round(df["wage"].mean(), 2),
    }

    return df, stats


def run_simulation(
    df: pd.DataFrame,
    G: nx.Graph,
    n_ticks: int = 60,
    snapshot_ticks: Optional[list] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list, dict]:
    """
    Запускает симуляцию на n_ticks тиков.

    Args:
        df: начальный DataFrame агентов
        G: граф районов
        n_ticks: количество тиков
        snapshot_ticks: тики для сохранения снимка состояния (None = авто)
        seed: seed для воспроизводимости
        verbose: печатать прогресс

    Returns:
        df_final: финальный DataFrame
        snapshots: список снимков на контрольных тиках
        tick_stats: статистика по каждому тику
    """
    rng = np.random.default_rng(seed)

    if snapshot_ticks is None:
        # Начало, середина, конец
        snapshot_ticks = [0, n_ticks // 2, n_ticks - 1]

    snapshots = {}
    tick_stats = []

    # Снимок начального состояния
    if 0 in snapshot_ticks:
        snapshots[0] = df.copy()

    if verbose:
        print(f"\nЗапуск симуляции: {n_ticks} тиков, {len(df):,} агентов")
        print("-" * 50)

    for t in range(1, n_ticks + 1):
        df, stats = tick(df, G, t, rng)
        tick_stats.append(stats)

        if t in snapshot_ticks:
            snapshots[t] = df.copy()

        if verbose and (t % 12 == 0 or t == n_ticks):
            year = t // 12
            month = t % 12 or 12
            print(
                f"  Тик {t:3d} (год {year}, мес {month:2d}) | "
                f"переездов: {stats['moves']:4d} ({stats['move_rate_pct']:.2f}%) | "
                f"ср. возраст: {stats['avg_age']:.1f} | "
                f"ср. зарплата: {stats['avg_wage']:.0f}€"
            )

    if verbose:
        print("-" * 50)
        print(f"Симуляция завершена.")

    return df, snapshots, tick_stats
