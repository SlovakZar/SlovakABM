"""
graph.py v2 — граф всей Словакии на основе реальных commuting потоков.

Источник топологии: commuting_filtered_with_travel.csv
  - origin_district / destination_district: названия районов
  - flow_work / flow_school / total_flow: потоки занятых и студентов
  - travel_time_min: время в пути на машине (OSRM)

Структура графа:
  Узлы: 79 районов Словакии
  Рёбра: направленные (origin → dest) где total_flow > 0
  Self-loops исключаются из рёбер — они хранятся в атрибуте узла как
  внутренний поток (internal_flow) для нормализации.

Атрибуты узлов (инициализируются из environment.json через build_graph):
  avg_wage, avg_wage_base: средняя зарплата по данным переписи
  housing_price_m2, housing_price_base: цена жилья
  jobs_capacity: ёмкость рынка труда (occupations.Total)
  real_population: численность населения
  region: код региона (BA, TT, TN, NR, ZA, BB, PO, KE)
  agent_count: текущее число агентов (обновляется каждый тик)
  internal_flow: число живущих и работающих в том же районе

Атрибуты рёбер:
  travel_time_min: время в пути (минуты)
  flow_work: поток занятых
  flow_school: поток студентов
  total_flow: суммарный поток
  flow_weight: нормированный вес ребра [0,1] (для awareness агента)

Реакция среды (update_graph):
  housing_price реагирует на плотность агентов vs ожидаемую долю
  avg_wage реагирует на соотношение рабочей силы и jobs_capacity
"""

import json
import math
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# ── Параметры реакции среды ───────────────────────────────────────────────────
HOUSING_ALPHA = 0.03   # скорость реакции цен на жильё
WAGE_ALPHA    = 0.02   # скорость реакции зарплат

# Маппинг регионов (kraj) — код → аббревиатура
REGION_CODE = {
    "Region of Bratislava":    "BA",
    "Region of Trnava":        "TT",
    "Region of Trenčín":       "TN",
    "Region of Nitra":         "NR",
    "Region of Žilina":        "ZA",
    "Region of Banská Bystrica": "BB",
    "Region of Prešov":        "PO",
    "Region of Košice":        "KE",
}

# Базовые цены жилья по регионам (€/м², данные Národná banka Slovenska 2023)
REGIONAL_HOUSING = {
    "BA": 3850, "TT": 2100, "TN": 1650, "NR": 1550,
    "ZA": 1700, "BB": 1450, "PO": 1350, "KE": 1800,
}

# Региональные центры — к ним добавляются long-range рёбра
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


def _get_latest(data: dict, year: int = 2024):
    """Берём значение за конкретный год, fallback на ближайший."""
    if not data:
        return None
    if year in data:
        return data[year]
    for y in sorted(data.keys(), reverse=True):
        if data[y] is not None:
            return data[y]
    return None


def _infer_region(district_name: str, locations: dict) -> str:
    """Определяем регион района из environment.json."""
    data = locations.get(district_name, {})
    region_full = data.get("region", "")
    return REGION_CODE.get(region_full, "XX")


def build_graph(
    env_path: str = "environment.json",
    commuting_path: str = "commuting_filtered_with_travel.csv",
    year: int = 2024,
    min_flow: int = 0,       # минимальный поток для включения ребра (уже отфильтровано)
) -> nx.DiGraph:
    """
    Строит направленный граф Словакии из commuting-матрицы.

    Возвращает DiGraph где узлы — районы, рёбра — реальные потоки.
    Атрибуты узлов заполняются из environment.json.
    """
    # ── Загрузка данных ───────────────────────────────────────────────────────
    commuting_path_obj = Path(commuting_path)
    if not commuting_path_obj.exists():
        # Пробуем рядом со скриптом
        commuting_path_obj = Path(__file__).parent / commuting_path
    df_comm = pd.read_csv(commuting_path_obj)

    # Оставляем только рёбра district → district (без Abroad, Not found out и т.д.)
    mask = (
        df_comm['origin_district'].str.startswith('District of') &
        df_comm['destination_district'].str.startswith('District of')
    )
    df_comm = df_comm[mask].copy()

    # Self-loops выделяем отдельно
    self_loops = df_comm[df_comm['origin_district'] == df_comm['destination_district']].copy()
    edges_df   = df_comm[df_comm['origin_district'] != df_comm['destination_district']].copy()

    # Словарь internal_flow по районам
    internal_flow = dict(zip(self_loops['origin_district'], self_loops['total_flow']))

    # ── Загрузка environment.json ─────────────────────────────────────────────
    env_path_obj = Path(env_path)
    if not env_path_obj.exists():
        env_path_obj = Path(__file__).parent / env_path

    with open(env_path_obj, encoding='utf-8') as f:
        env = json.load(f)
    locations = env.get("locations", {})

    # ── Сбор всех районов ─────────────────────────────────────────────────────
    all_districts = set(edges_df['origin_district']) | set(edges_df['destination_district'])
    # Дополняем районами из environment.json (на случай изолированных)
    for name, data in locations.items():
        if data.get("type") == "district":
            all_districts.add(name)

    # ── Нормализация потоков для весов рёбер ─────────────────────────────────
    max_flow = edges_df['total_flow'].max() if len(edges_df) > 0 else 1.0

    # ── Строим граф ───────────────────────────────────────────────────────────
    G = nx.DiGraph()

    for district in sorted(all_districts):
        data = locations.get(district, {})
        region_full = data.get("region", "")
        region_code = REGION_CODE.get(region_full, "XX")

        # Зарплата
        wages = data.get("wages", {})
        avg_wage = _get_latest(wages.get("Total", {}), year) or 0
        if avg_wage == 0 and wages:
            vals = [_get_latest(v, year) for v in wages.values() if _get_latest(v, year)]
            avg_wage = float(np.mean(vals)) if vals else 1200.0

        # Цена жилья: базовая из региональных данных
        housing_base = REGIONAL_HOUSING.get(region_code, 1500)
        # Небольшая вариация по зарплатному коэффициенту
        region_avg_wage = 1573.0
        wage_ratio = avg_wage / region_avg_wage if avg_wage > 0 else 1.0
        housing_price = housing_base * (0.8 + 0.4 * wage_ratio)

        # Занятость
        occ = data.get("occupations", {})
        jobs_capacity = _get_latest(occ.get("Total", {}), year) or 1

        # Население из возрастных групп
        ag = data.get("age_groups", {})
        population = 0
        for age_data in ag.values():
            population += (_get_latest(age_data.get("male", {}), year) or 0)
            population += (_get_latest(age_data.get("female", {}), year) or 0)
        if population == 0:
            population = 10000  # fallback

        G.add_node(
            district,
            region=region_code,
            # Базовые (не изменяются)
            avg_wage_base=float(avg_wage),
            housing_price_base=float(housing_price),
            jobs_capacity=max(int(jobs_capacity), 1),
            real_population=int(population),
            internal_flow=float(internal_flow.get(district, 0)),
            # Динамические (обновляются каждый тик)
            avg_wage=float(avg_wage),
            housing_price_m2=float(housing_price),
            agent_count=0,
        )

    # ── Рёбра из commuting матрицы ────────────────────────────────────────────
    for _, row in edges_df.iterrows():
        src = row['origin_district']
        dst = row['destination_district']
        if src not in G.nodes or dst not in G.nodes:
            continue

        flow       = float(row['total_flow'])
        flow_work  = float(row.get('flow_work', 0))
        flow_school= float(row.get('flow_school', 0))
        travel_t   = float(row.get('travel_time_min', 60))
        weight     = flow / max_flow  # нормированный вес [0,1]

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
    """Выводит топ рёбер по потоку."""
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get('total_flow', 0), reverse=True)
    print(f"  Топ-{n} рёбер по потоку:")
    for src, dst, attr in edges[:n]:
        s = src.replace("District of ", "")
        d = dst.replace("District of ", "")
        print(f"    {s} → {d}: {attr['total_flow']:,.0f} чел | {attr['travel_time_min']:.0f} мин")


def get_neighbors(G: nx.DiGraph, district: str, max_travel_time: float = None) -> list:
    """
    Возвращает список районов куда есть исходящие рёбра от district.
    Опционально фильтрует по max_travel_time (минуты).
    """
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
    """
    Формирует awareness_set агента — список районов которые он реально рассматривает.

    Логика:
      1. Все соседи по commuting-графу (исходящие рёбра)
      2. Фильтрация по perceived_control: чем ниже контроль, тем меньше
         агент рассматривает дальние варианты (порог по travel_time)
      3. Если network_location=True — добавляются региональные центры других краёв
         (слабые связи расширяют горизонт)

    Возвращает список районов (без текущего).
    """
    # Время в пути как функция perceived_control: 0.3 → 45 мин, 0.7 → 120 мин
    time_limit = 30 + 150 * perceived_control  # [30, 180] минут

    candidates = set()

    # Прямые commuting-связи в пределах time_limit
    for _, dst, attr in G.out_edges(district, data=True):
        if attr.get('travel_time_min', 999) <= time_limit:
            candidates.add(dst)

    # Если есть удалённые контакты — добавляем региональные центры
    if network_location:
        current_region = G.nodes[district].get('region', 'XX')
        for region_code, center in REGIONAL_CENTERS.items():
            if region_code != current_region and center in G.nodes:
                candidates.discard(district)
                candidates.add(center)

    # Убираем текущий район
    candidates.discard(district)

    # Ограничиваем размер — сортируем по весу ребра
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
    """
    Обновляет динамические атрибуты графа на основе текущего распределения агентов.

    Жильё: растёт если агентов больше ожидаемого, падает при оттоке.
    Зарплата: реагирует на соотношение рабочей силы и jobs_capacity.
    """
    total_real_pop = sum(G.nodes[d]['real_population'] for d in G.nodes)

    for district in G.nodes:
        attr = G.nodes[district]
        current_agents = agent_district_counts.get(district, 0)

        # Ожидаемая доля агентов пропорционально реальному населению
        expected_share = attr['real_population'] / max(total_real_pop, 1)
        actual_share   = current_agents / max(total_agents, 1)
        density_ratio  = actual_share / max(expected_share, 0.001)

        # Давление на жильё
        housing_pressure = density_ratio - 1.0
        new_housing = attr['housing_price_m2'] * (1 + HOUSING_ALPHA * housing_pressure)
        new_housing = float(np.clip(
            new_housing,
            attr['housing_price_base'] * 0.55,
            attr['housing_price_base'] * 2.2,
        ))

        # Давление на зарплату
        neutral_labour = (attr['real_population'] / max(attr['jobs_capacity'], 1)) / 100
        labour_pressure = current_agents / max(attr['jobs_capacity'], 1)
        wage_pressure = (neutral_labour - labour_pressure) / max(neutral_labour, 0.001)
        wage_pressure = float(np.clip(wage_pressure, -0.5, 0.5))

        new_wage = attr['avg_wage'] * (1 + WAGE_ALPHA * wage_pressure)
        new_wage = float(np.clip(
            new_wage,
            attr['avg_wage_base'] * 0.65,
            attr['avg_wage_base'] * 1.6,
        ))

        G.nodes[district]['housing_price_m2'] = round(new_housing, 0)
        G.nodes[district]['avg_wage']          = round(new_wage, 0)
        G.nodes[district]['agent_count']        = current_agents


def print_graph_summary(G: nx.DiGraph):
    """Краткая сводка по графу."""
    print("=" * 72)
    print("ГРАФ СЛОВАКИИ — COMMUTING")
    print("=" * 72)
    print(f"Узлов: {G.number_of_nodes()}  |  Рёбер: {G.number_of_edges()}")

    # По регионам
    from collections import Counter
    regions = Counter(G.nodes[d].get('region', 'XX') for d in G.nodes)
    print("Районов по регионам:", dict(sorted(regions.items())))

    # Топ-10 по in-degree (принимают больше всего потоков)
    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    print("\nТоп-10 районов по входящим потокам (in-degree):")
    for node, deg in in_deg[:10]:
        name = node.replace("District of ", "")
        wage = G.nodes[node].get('avg_wage', 0)
        pop  = G.nodes[node].get('real_population', 0)
        print(f"  {name:<28} in-edges={deg:3d}  wage={wage:,.0f}€  pop={pop:,}")
    print("=" * 72)


if __name__ == "__main__":
    G = build_graph()
    print_graph_summary(G)
