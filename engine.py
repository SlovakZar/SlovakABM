"""
engine.py v1 — tick loop с удовлетворённостью, якорями и реакцией среды.

Логика тика:
  1. age += 1/12, tenure += 1 для всех
  2. update_satisfaction(agent) — оценивает текущие условия vs региональные средние
  3. Агент рассматривает переезд ТОЛЬКО если satisfaction < (1 - move_threshold)
  4. Если рассматривает — вычисляет utility для всех районов + anchor_penalty
  5. Переезжает если max_utility > 0 (с учётом якорей)
  6. update_graph() — среда реагирует на новое распределение агентов
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional

from graph import update_graph, TRNAVA_DISTRICTS

# ── веса утилитарной функции ──────────────────────────────────────────────────
W_WAGE      = 0.55
W_HOUSING   = 0.25
W_EMPLOY    = 0.20
DIST_PENALTY = 0.0004      # штраф за расстояние: 50 км = -0.02

# ── якоря ─────────────────────────────────────────────────────────────────────
ANCHOR_MARRIED   = 0.06    # штраф utility для женатых
ANCHOR_TENURE    = 0.002   # штраф за каждый месяц стажа (насыщается через log)
ANCHOR_AGE_SLOPE = 0.008   # дополнительный штраф за каждый год старше 35

# ── satisfaction ──────────────────────────────────────────────────────────────
SAT_SMOOTHING    = 0.92    # инерция удовлетворённости (EMA)
SAT_WAGE_W       = 0.5     # вес зарплаты в локальном качестве
SAT_HOUSING_W    = 0.3     # вес жилья (дороже = хуже)
SAT_EMPLOY_W     = 0.2     # вес занятости

# ── мобильность по возрасту ───────────────────────────────────────────────────
AGE_PEAK         = 27.0
AGE_DECAY        = 0.028


def _age_mobility(age: float) -> float:
    if age < 18:
        return 0.0
    return float(np.exp(-AGE_DECAY * max(0.0, age - AGE_PEAK) ** 1.4))


def _local_quality(agent_wage: float, district: str, G: nx.Graph) -> float:
    """
    Нормализованное качество жизни в текущем районе агента [0, 1].
    Сравниваем с медианными показателями по всему краю.
    """
    all_wages    = [G.nodes[d]["avg_wage"] for d in TRNAVA_DISTRICTS]
    all_housing  = [G.nodes[d]["housing_price_m2"] for d in TRNAVA_DISTRICTS]
    all_jobs     = [G.nodes[d]["jobs_capacity"] for d in TRNAVA_DISTRICTS]

    med_wage    = float(np.median(all_wages))
    med_housing = float(np.median(all_housing))
    med_jobs    = float(np.median(all_jobs))

    attr = G.nodes[district]

    # Зарплата: личная зарплата относительно среднего по краю
    q_wage = (agent_wage - med_wage) / max(med_wage, 1)

    # Жильё: ниже медианы = лучше (больше affordability)
    q_housing = (med_housing - attr["housing_price_m2"]) / max(med_housing, 1)

    # Занятость: рабочие места относительно медианы
    q_employ = (attr["jobs_capacity"] - med_jobs) / max(med_jobs, 1)

    # Взвешенная сумма → сигмоид → [0, 1]
    raw = SAT_WAGE_W * q_wage + SAT_HOUSING_W * q_housing + SAT_EMPLOY_W * q_employ
    return float(1 / (1 + math.exp(-3.0 * raw)))  # мягкая сигмоида


def _anchor_penalty(age: float, marital: str, tenure: int) -> float:
    """Суммарный штраф за переезд (социальные якоря)."""
    penalty = 0.0
    if marital == "Married person":
        penalty += ANCHOR_MARRIED
    # Логарифмическое насыщение стажа: долго жить = привязан, но не бесконечно
    penalty += ANCHOR_TENURE * math.log1p(tenure / 12)
    # Возрастной штраф (только после 35)
    if age > 35:
        penalty += ANCHOR_AGE_SLOPE * (age - 35)
    return penalty


def _compute_utility(src: str, dst: str, G: nx.Graph, agent_wage: float) -> float:
    """Utility переезда из src в dst (без учёта якорей — они снаружи)."""
    if src == dst:
        return 0.0

    a_i = G.nodes[src]
    a_j = G.nodes[dst]

    # Зарплатный стимул: ожидаемая зарплата в dst vs src
    w_i = a_i.get("avg_wage", 1) or 1
    w_j = a_j.get("avg_wage", 1) or 1
    u_wage = W_WAGE * (w_j - w_i) / w_i

    # Жилищный стимул
    h_i = a_i.get("housing_price_m2", 1) or 1
    h_j = a_j.get("housing_price_m2", 1) or 1
    u_housing = W_HOUSING * (h_i - h_j) / max(h_i, 1)

    # Стимул занятости: jobs_capacity нормализован на реальное население
    pop_i = a_i.get("real_population", 1) or 1
    pop_j = a_j.get("real_population", 1) or 1
    e_i = a_i.get("jobs_capacity", 1) / pop_i
    e_j = a_j.get("jobs_capacity", 1) / pop_j
    u_employ = W_EMPLOY * (e_j - e_i) / max(e_i, 0.001)

    # Штраф расстояния
    dist_km = G[src][dst]["distance_km"] if G.has_edge(src, dst) else 80.0
    u_dist = -DIST_PENALTY * dist_km

    return u_wage + u_housing + u_employ + u_dist


def tick(
    df: pd.DataFrame,
    G: nx.Graph,
    tick_num: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Один шаг симуляции. Возвращает обновлённый DataFrame и статистику."""

    n_agents = len(df)
    districts = TRNAVA_DISTRICTS

    # 1. Все стареют и накапливают стаж
    df["age"]    = df["age"] + 1 / 12
    df["tenure"] = df["tenure"] + 1
    df["moved_ticks"] = df["moved_ticks"] + 1

    # 2. Обновляем satisfaction для всех агентов (векторизовано по районам)
    new_sat = df["satisfaction"].values.copy()
    for district in districts:
        mask = (df["district"] == district).values
        if not mask.any():
            continue
        wages_in_d = df.loc[mask, "wage"].values
        for i, idx in enumerate(np.where(mask)[0]):
            lq = _local_quality(wages_in_d[i], district, G)
            new_sat[idx] = SAT_SMOOTHING * new_sat[idx] + (1 - SAT_SMOOTHING) * lq
    df["satisfaction"] = np.clip(new_sat, 0.0, 1.0)

    # 3. Отбор кандидатов на переезд:
    #    - возраст >= 18
    #    - satisfaction < (1 - move_threshold)   ← КЛЮЧЕВОЕ условие
    #    - прошло >= 12 тиков с последнего переезда
    sat_arr = df["satisfaction"].values
    thr_arr = df["move_threshold"].values
    age_arr = df["age"].values
    mt_arr  = df["moved_ticks"].values

    eligible_mask = (
        (age_arr >= 18) &
        (sat_arr < (1 - thr_arr)) &
        (mt_arr >= 12)
    )
    eligible_idx = np.where(eligible_mask)[0]

    moves = 0

    for idx in eligible_idx:
        row = df.iloc[idx]
        current   = row["district"]
        age       = row["age"]
        mob       = _age_mobility(age)
        if mob < 0.02:
            continue

        anchor = _anchor_penalty(age, row["marital"], int(row["tenure"]))
        agent_wage = row["wage"]

        best_dst     = current
        best_net_u   = 0.0  # net utility после учёта якорей

        for candidate in districts:
            if candidate == current:
                continue
            raw_u = _compute_utility(current, candidate, G, agent_wage)
            net_u = raw_u - anchor  # якорь снижает ценность любого переезда
            if net_u > best_net_u:
                best_net_u = net_u
                best_dst   = candidate

        if best_dst != current and best_net_u > 0:
            # Финальная стохастичность: мобильность × случайность
            noise = rng.uniform(0, 0.3)
            if best_net_u * mob > noise:
                df.iloc[idx, df.columns.get_loc("district")]    = best_dst
                df.iloc[idx, df.columns.get_loc("moved_ticks")] = 0
                df.iloc[idx, df.columns.get_loc("tenure")]       = 0

                # Зарплата корректируется к среднему нового района
                target = G.nodes[best_dst]["avg_wage"] or agent_wage
                new_wage = float(rng.normal(target, target * 0.20))
                df.iloc[idx, df.columns.get_loc("wage")] = max(0.0, new_wage)

                # Satisfaction слегка падает сразу после переезда (стресс)
                df.iloc[idx, df.columns.get_loc("satisfaction")] = max(
                    0.1, row["satisfaction"] * 0.8
                )
                moves += 1

    # 4. Среда реагирует на новое распределение
    district_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, district_counts, n_agents)

    stats = {
        "tick": tick_num,
        "moves": moves,
        "move_rate_pct": round(moves / n_agents * 100, 3),
        "eligible": int(eligible_mask.sum()),
        "avg_age": round(float(df["age"].mean()), 2),
        "avg_wage": round(float(df["wage"].mean()), 2),
        "avg_satisfaction": round(float(df["satisfaction"].mean()), 3),
        "district_counts": district_counts,
        # динамика среды
        "env_wages":   {d: G.nodes[d]["avg_wage"] for d in districts},
        "env_housing": {d: G.nodes[d]["housing_price_m2"] for d in districts},
    }

    return df, stats


def run_simulation(
    df: pd.DataFrame,
    G: nx.Graph,
    n_ticks: int = 60,
    snapshot_ticks: Optional[list] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, list]:

    rng = np.random.default_rng(seed)

    if snapshot_ticks is None:
        snapshot_ticks = [0, n_ticks // 2, n_ticks]

    # Инициализируем agent_count в графе
    district_counts = df.groupby("district")["id"].count().to_dict()
    update_graph(G, district_counts, len(df))

    snapshots   = {}
    tick_stats  = []

    if 0 in snapshot_ticks:
        snapshots[0] = df.copy()

    if verbose:
        print(f"\nЗапуск симуляции: {n_ticks} тиков, {len(df):,} агентов")
        print(f"{'Тик':>5} {'Год':>4} {'Переездов':>10} {'Eligible':>9}"
              f" {'Ср.возраст':>10} {'Ср.з/п':>8} {'Ср.sat':>7}")
        print("-" * 62)

    for t in range(1, n_ticks + 1):
        df, stats = tick(df, G, t, rng)
        tick_stats.append(stats)

        if t in snapshot_ticks:
            snapshots[t] = df.copy()

        if verbose and (t % 12 == 0 or t == n_ticks):
            yr = t // 12
            mo = t % 12 or 12
            print(
                f"  {t:3d} (г{yr:1d}м{mo:02d})"
                f"  {stats['moves']:>7} ({stats['move_rate_pct']:>5.2f}%)"
                f"  {stats['eligible']:>7}"
                f"  {stats['avg_age']:>9.1f}"
                f"  {stats['avg_wage']:>7,.0f}€"
                f"  {stats['avg_satisfaction']:>6.3f}"
            )

    if verbose:
        print("-" * 62)
        print("Симуляция завершена.")

    return df, snapshots, tick_stats
