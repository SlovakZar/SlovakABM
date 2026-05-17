"""
graph.py v2 — граф всей Словакии на основе реальных commuting потоков.
(остальной docstring без изменений)
"""

import json
import math
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

HOUSING_ALPHA = 0.03
WAGE_ALPHA    = 0.02

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
    min_flow: int = 0,
) -> nx.DiGraph:
    # ... (загрузка commuting данных без изменений) ...
    commuting_path_obj = Path(commuting_path)
    if not commuting_path_obj.exists():
        commuting_path_obj = Path(__file__).parent / commuting_path
    df_comm = pd.read_csv(commuting_path_obj)

    mask = (
        df_comm['origin_district'].str.startswith('District of') &
        df_comm['destination_district'].str.startswith('District of')
    )
    df_comm = df_comm[mask].copy()

    self_loops = df_comm[df_comm['origin_district'] == df_comm['destination_district']].copy()
    edges_df   = df_comm[df_comm['origin_district'] != df_comm['destination_district']].copy()
    internal_flow = dict(zip(self_loops['origin_district'], self_loops['total_flow']))

    env_path_obj = Path(env_path)
    if not env_path_obj.exists():
        env_path_obj = Path(__file__).parent / env_path
    with open(env_path_obj, encoding='utf-8') as f:
        env = json.load(f)
    locations = env.get("locations", {})

    all_districts = set(edges_df['origin_district']) | set(edges_df['destination_district'])
    for name, data in locations.items():
        if data.get("type") == "district":
            all_districts.add(name)

    max_flow = edges_df['total_flow'].max() if len(edges_df) > 0 else 1.0
    G = nx.DiGraph()

    for district in sorted(all_districts):
        data = locations.get(district)
        if data is None:
            continue

        region_code = data.get("region", "XX")
        avg_wage = data.get("avg_wage", 1200.0)
        housing_data = data.get("housing", {})
        housing_price = housing_data.get("price_m2", 2000.0)

        population = data.get("population", 10000)
        unemployment = data.get("unemployment_rate", 0.05)
        jobs_capacity = max(int(population * (1 - unemployment)), 1)

        # ── Новые атрибуты: инфраструктура и бизнес ──────────────────────────
        infrastructure = data.get("infrastructure", {})
        # Приводим к стандартным полям (если отсутствуют – ставим 0)
        infra_defaults = {
            "polyclinics": 0,
            "hospitals": 0,
            "cinemas": 0,
            "museums": 0,
            "galleries": 0,
        }
        for key in infra_defaults:
            if key not in infrastructure:
                infrastructure[key] = infra_defaults[key]

        business = data.get("business", {})
        business_defaults = {
            "total_companies": 0,
            "foreign_companies": 0,
            "small_companies": 0,
            "medium_companies": 0,
            "large_companies": 0,
        }
        for key in business_defaults:
            if key not in business:
                business[key] = business_defaults[key]

        G.add_node(
            district,
            region=region_code,
            avg_wage_base=float(avg_wage),
            housing_price_base=float(housing_price),
            jobs_capacity=jobs_capacity,
            real_population=int(population),
            internal_flow=float(internal_flow.get(district, 0)),
            avg_wage=float(avg_wage),
            housing_price_m2=float(housing_price),
            agent_count=0,
            # Добавляем инфраструктуру и бизнес
            infrastructure=infrastructure,
            business=business,
        )

    for _, row in edges_df.iterrows():
        src = row['origin_district']
        dst = row['destination_district']
        if src not in G.nodes or dst not in G.nodes:
            continue
        flow = float(row['total_flow'])
        flow_work = float(row.get('flow_work', 0))
        flow_school = float(row.get('flow_school', 0))
        travel_t = float(row.get('travel_time_min', 60))
        weight = flow / max_flow
        G.add_edge(src, dst,
                   travel_time_min=round(travel_t, 2),
                   flow_work=flow_work,
                   flow_school=flow_school,
                   total_flow=flow,
                   flow_weight=round(weight, 4))

    print(f"  Граф: {G.number_of_nodes()} узлов | {G.number_of_edges()} рёбер")
    print(f"  Покрытие self-loops: {len(self_loops)} районов")
    _print_top_edges(G, n=5)
    return G


def _print_top_edges(G: nx.DiGraph, n: int = 5):
    # без изменений
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get('total_flow', 0), reverse=True)
    print(f"  Топ-{n} рёбер по потоку:")
    for src, dst, attr in edges[:n]:
        s = src.replace("District of ", "")
        d = dst.replace("District of ", "")
        print(f"    {s} → {d}: {attr['total_flow']:,.0f} чел | {attr['travel_time_min']:.0f} мин")


def get_neighbors(G: nx.DiGraph, district: str, max_travel_time: float = None) -> list:
    # без изменений
    neighbors = []
    for _, dst, attr in G.out_edges(district, data=True):
        if max_travel_time is not None and attr.get('travel_time_min', 999) > max_travel_time:
            continue
        neighbors.append(dst)
    return neighbors


def get_awareness_set(
    G: nx.DiGraph,
    district: str,
    network_location: bool = False,
    perceived_control: float = 0.5,
    max_candidates: int = 15,
) -> list:
    # без изменений
    time_limit = 30 + 150 * perceived_control
    candidates = set()
    for _, dst, attr in G.out_edges(district, data=True):
        if attr.get('travel_time_min', 999) <= time_limit:
            candidates.add(dst)
    if network_location:
        current_region = G.nodes[district].get('region', 'XX')
        for region_code, center in REGIONAL_CENTERS.items():
            if region_code != current_region and center in G.nodes:
                candidates.discard(district)
                candidates.add(center)
    candidates.discard(district)
    if len(candidates) > max_candidates:
        def edge_weight(dst):
            if G.has_edge(district, dst):
                return G[district][dst].get('flow_weight', 0)
            return 0.0
        candidates = sorted(candidates, key=edge_weight, reverse=True)[:max_candidates]
    else:
        candidates = list(candidates)
    return candidates


def update_graph(G: nx.DiGraph, agent_district_counts: dict, total_agents: int):
    # без изменений
    total_real_pop = sum(G.nodes[d]['real_population'] for d in G.nodes)
    for district in G.nodes:
        attr = G.nodes[district]
        current_agents = agent_district_counts.get(district, 0)
        expected_share = attr['real_population'] / max(total_real_pop, 1)
        actual_share = current_agents / max(total_agents, 1)
        density_ratio = actual_share / max(expected_share, 0.001)
        housing_pressure = density_ratio - 1.0
        new_housing = attr['housing_price_m2'] * (1 + HOUSING_ALPHA * housing_pressure)
        new_housing = float(np.clip(new_housing, attr['housing_price_base'] * 0.55, attr['housing_price_base'] * 2.2))
        neutral_labour = (attr['real_population'] / max(attr['jobs_capacity'], 1)) / 100
        labour_pressure = current_agents / max(attr['jobs_capacity'], 1)
        wage_pressure = (neutral_labour - labour_pressure) / max(neutral_labour, 0.001)
        wage_pressure = float(np.clip(wage_pressure, -0.5, 0.5))
        new_wage = attr['avg_wage'] * (1 + WAGE_ALPHA * wage_pressure)
        new_wage = float(np.clip(new_wage, attr['avg_wage_base'] * 0.65, attr['avg_wage_base'] * 1.6))
        G.nodes[district]['housing_price_m2'] = round(new_housing, 0)
        G.nodes[district]['avg_wage'] = round(new_wage, 0)
        G.nodes[district]['agent_count'] = current_agents


def print_graph_summary(G: nx.DiGraph):
    """Краткая сводка по графу, включая инфраструктуру и бизнес."""
    print("=" * 72)
    print("ГРАФ СЛОВАКИИ — COMMUTING")
    print("=" * 72)
    print(f"Узлов: {G.number_of_nodes()}  |  Рёбер: {G.number_of_edges()}")

    from collections import Counter
    regions = Counter(G.nodes[d].get('region', 'XX') for d in G.nodes)
    print("Районов по регионам:", dict(sorted(regions.items())))

    # Топ-10 по in-degree
    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    print("\nТоп-10 районов по входящим потокам (in-degree):")
    for node, deg in in_deg[:10]:
        name = node.replace("District of ", "")
        wage = G.nodes[node].get('avg_wage', 0)
        pop  = G.nodes[node].get('real_population', 0)
        print(f"  {name:<28} in-edges={deg:3d}  wage={wage:,.0f}€  pop={pop:,}")

    # ── Статистика по инфраструктуре и бизнесу ──────────────────────────────
    print("\nИнфраструктура (средние значения по районам):")
    infra_keys = ['polyclinics', 'hospitals', 'cinemas', 'museums', 'galleries']
    infra_sums = {k: 0 for k in infra_keys}
    for node in G.nodes:
        infra = G.nodes[node].get('infrastructure', {})
        for k in infra_keys:
            infra_sums[k] += infra.get(k, 0)
    n_nodes = G.number_of_nodes()
    for k in infra_keys:
        print(f"  {k:<12}: {infra_sums[k]/n_nodes:.2f} в среднем")

    print("\nБизнес-активность (суммарно по районам):")
    bus_keys = ['total_companies', 'foreign_companies', 'small_companies', 'medium_companies', 'large_companies']
    bus_sums = {k: 0 for k in bus_keys}
    for node in G.nodes:
        bus = G.nodes[node].get('business', {})
        for k in bus_keys:
            bus_sums[k] += bus.get(k, 0)
    for k in bus_keys:
        print(f"  {k:<18}: {bus_sums[k]:,}")

    print("=" * 72)


if __name__ == "__main__":
    G = build_graph()
    print_graph_summary(G)
