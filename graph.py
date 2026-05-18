"""
graph.py v3 — граф Словакии на основе реальных commuting потоков.

Источники:
  environment.json                — атрибуты узлов (SODB 2021 / NBS / ŠÚ SR)
  commuting_filtered_with_travel.csv — матрица потоков + время в пути (OSRM)

Структура графа:
  Узлы: 79 районов Словакии
  Рёбра: направленные (origin → dest) по реальным commuting потокам

Атрибуты узлов:
  avg_wage, avg_wage_base       — средняя зарплата (€/мес)
  housing_price_m2, ..._base    — цена жилья (€/м²)
  unemployment_rate             — уровень безработицы [0,1]
  jobs_capacity                 — суммарная занятость по district (сумма WC)
  real_population               — численность населения
  owner_share                   — доля собственников жилья [0,1]
  region                        — код региона (BA, TT, TN, NR, ZA, BB, PO, KE)
  agent_count                   — текущее число агентов (обновляется каждый тик)

Атрибуты рёбер:
  travel_time_min  — время в пути (мин)
  flow_work        — поток занятых
  total_flow       — суммарный поток
  flow_weight      — нормированный вес ребра [0,1]

Реакция среды (update_graph):
  housing_price реагирует на плотность агентов vs ожидаемую долю
  avg_wage реагирует на соотношение агентов и jobs_capacity
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# ── Параметры реакции среды ───────────────────────────────────────────────────
HOUSING_ALPHA = 0.03
WAGE_ALPHA    = 0.02

# Региональные центры для long-range рёбер в awareness_set
REGIONAL_CENTERS = {
    "BA": "District of Bratislava I",
    "TT": "District of Trnava",
    "TN": "District of Trenčín",
    "NR": "District of Nitra",
    "ZA": "District of Žilina",
    "BB": "District of Banská Bystrica",
    "PO": "District of Prešov",
    "KE": "District of Košice I",
}


def build_graph(
    env_path: str = "environment.json",
    commuting_path: str = "commuting_filtered_with_travel.csv",
) -> nx.DiGraph:
    """
    Строит направленный граф Словакии из commuting-матрицы.
    Атрибуты узлов загружаются из environment.json (создан build_environment.py).
    """
    # ── Загрузка environment.json ─────────────────────────────────────────────
    env_path_obj = Path(env_path)
    if not env_path_obj.exists():
        env_path_obj = Path(__file__).parent / env_path
    with open(env_path_obj, encoding="utf-8") as f:
        env = json.load(f)
    locations = env.get("locations", {})

    # ── Загрузка commuting матрицы ────────────────────────────────────────────
    comm_path_obj = Path(commuting_path)
    if not comm_path_obj.exists():
        comm_path_obj = Path(__file__).parent / commuting_path
    df_comm = pd.read_csv(comm_path_obj)

    # Только рёбра district → district
    mask = (
        df_comm["origin_district"].str.startswith("District of") &
        df_comm["destination_district"].str.startswith("District of")
    )
    df_comm = df_comm[mask].copy()

    # Self-loops → атрибут внутреннего потока
    self_loops = df_comm[df_comm["origin_district"] == df_comm["destination_district"]]
    edges_df   = df_comm[df_comm["origin_district"] != df_comm["destination_district"]]
    internal_flow = dict(zip(self_loops["origin_district"], self_loops["total_flow"]))

    max_flow = edges_df["total_flow"].max() if len(edges_df) > 0 else 1.0

    # ── Сбор всех районов ─────────────────────────────────────────────────────
    all_districts = (set(edges_df["origin_district"]) |
                     set(edges_df["destination_district"]) |
                     set(locations.keys()))

    # ── Строим граф ───────────────────────────────────────────────────────────
    G = nx.DiGraph()

    for district in sorted(all_districts):
        loc = locations.get(district, {})

        region = loc.get("region", "XX")
        population = loc.get("population", 10000)
        unemployment_rate = loc.get("unemployment_rate") or 0.08
        avg_wage = loc.get("avg_wage") or 1400.0
        owner_share = (loc.get("housing", {}) or {}).get("owner_share") or 0.65

        housing_m2 = (loc.get("housing", {}) or {}).get("price_m2") or 1500.0

        # jobs_capacity — сумма занятых по всем отраслям из environment
        salary_by_ind = loc.get("salary_by_industry", {})
        # Если нет детальных данных — используем population * занятость ~45%
        jobs_capacity = max(1, int(population * (1 - unemployment_rate) * 0.45))

        G.add_node(
            district,
            region=region,
            real_population=int(population),
            unemployment_rate=float(unemployment_rate),
            owner_share=float(owner_share),
            jobs_capacity=jobs_capacity,
            internal_flow=float(internal_flow.get(district, 0)),
            # Базовые (неизменяемые)
            avg_wage_base=float(avg_wage),
            housing_price_base=float(housing_m2),
            # Динамические (обновляются каждый тик)
            avg_wage=float(avg_wage),
            housing_price_m2=float(housing_m2),
            agent_count=0,
        )

    # ── Рёбра ────────────────────────────────────────────────────────────────
    for _, row in edges_df.iterrows():
        src, dst = row["origin_district"], row["destination_district"]
        if src not in G.nodes or dst not in G.nodes:
            continue
        G.add_edge(
            src, dst,
            travel_time_min=round(float(row.get("travel_time_min", 60)), 2),
            flow_work=float(row.get("flow_work", 0)),
            flow_school=float(row.get("flow_school", 0)),
            total_flow=float(row["total_flow"]),
            flow_weight=round(float(row["total_flow"]) / max_flow, 4),
        )

    print(f"  Граф: {G.number_of_nodes()} узлов | {G.number_of_edges()} рёбер")
    _print_top_edges(G)
    return G


def _print_top_edges(G: nx.DiGraph, n: int = 5):
    edges = sorted(G.edges(data=True),
                   key=lambda e: e[2].get("total_flow", 0), reverse=True)
    print(f"  Топ-{n} рёбер по потоку:")
    for src, dst, attr in edges[:n]:
        s = src.replace("District of ", "")
        d = dst.replace("District of ", "")
        print(f"    {s} → {d}: {attr['total_flow']:,.0f} чел | {attr['travel_time_min']:.0f} мин")


def get_awareness_set(
    G: nx.DiGraph,
    district: str,
    network_location: bool = False,
    perceived_control: float = 0.5,
    max_candidates: int = 15,
) -> list:
    """
    Формирует awareness_set — районы которые агент реально рассматривает.

      1. Соседи по commuting-графу в пределах time_limit
         time_limit растёт с perceived_control: 0.3 → 45 мин, 0.7 → 120 мин
      2. Если network_location=True — региональные центры других краёв
    """
    time_limit = 30.0 + 150.0 * perceived_control

    candidates = set()
    for _, dst, attr in G.out_edges(district, data=True):
        if attr.get("travel_time_min", 999) <= time_limit:
            candidates.add(dst)

    if network_location:
        current_region = G.nodes[district].get("region", "XX")
        for rcode, center in REGIONAL_CENTERS.items():
            if rcode != current_region and center in G.nodes:
                candidates.add(center)

    candidates.discard(district)

    if len(candidates) > max_candidates:
        def _weight(d):
            return G[district][d].get("flow_weight", 0) if G.has_edge(district, d) else 0
        candidates = sorted(candidates, key=_weight, reverse=True)[:max_candidates]
    else:
        candidates = list(candidates)

    return candidates


def update_graph(G: nx.DiGraph, agent_district_counts: dict, total_agents: int):
    """
    Обновляет динамические атрибуты узлов каждый тик.

    Жильё: растёт при притоке агентов, падает при оттоке.
    Зарплата: реагирует на соотношение рабочей силы и jobs_capacity.
    """
    total_real_pop = sum(G.nodes[d]["real_population"] for d in G.nodes)

    for district in G.nodes:
        attr = G.nodes[district]
        current_agents = agent_district_counts.get(district, 0)

        expected_share = attr["real_population"] / max(total_real_pop, 1)
        actual_share   = current_agents / max(total_agents, 1)
        density_ratio  = actual_share / max(expected_share, 0.001)

        # Жильё
        housing_pressure = density_ratio - 1.0
        new_housing = attr["housing_price_m2"] * (1 + HOUSING_ALPHA * housing_pressure)
        new_housing = float(np.clip(
            new_housing,
            attr["housing_price_base"] * 0.55,
            attr["housing_price_base"] * 2.5,
        ))

        # Зарплата
        labour_ratio = current_agents / max(attr["jobs_capacity"], 1)
        neutral      = attr["real_population"] / max(attr["jobs_capacity"] * 100, 1)
        wage_pressure = float(np.clip((neutral - labour_ratio) / max(neutral, 0.001), -0.5, 0.5))
        new_wage = attr["avg_wage"] * (1 + WAGE_ALPHA * wage_pressure)
        new_wage = float(np.clip(
            new_wage,
            attr["avg_wage_base"] * 0.65,
            attr["avg_wage_base"] * 1.6,
        ))

        G.nodes[district]["housing_price_m2"] = round(new_housing, 0)
        G.nodes[district]["avg_wage"]          = round(new_wage, 0)
        G.nodes[district]["agent_count"]        = current_agents


def print_graph_summary(G: nx.DiGraph):
    from collections import Counter
    print("=" * 72)
    print("ГРАФ СЛОВАКИИ — COMMUTING")
    print("=" * 72)
    print(f"Узлов: {G.number_of_nodes()}  |  Рёбер: {G.number_of_edges()}")
    regions = Counter(G.nodes[d].get("region", "XX") for d in G.nodes)
    print("Районов по регионам:", dict(sorted(regions.items())))

    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    print("\nТоп-10 районов по входящим потокам (in-degree):")
    for node, deg in in_deg[:10]:
        name = node.replace("District of ", "")
        wage = G.nodes[node].get("avg_wage", 0)
        unemp = G.nodes[node].get("unemployment_rate", 0)
        pop  = G.nodes[node].get("real_population", 0)
        print(f"  {name:<28} in={deg:3d}  wage={wage:,.0f}€  unemp={unemp:.1%}  pop={pop:,}")
    print("=" * 72)


if __name__ == "__main__":
    G = build_graph()
    print_graph_summary(G)
