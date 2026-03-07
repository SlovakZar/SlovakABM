"""
graph.py — граф Трнавского края с динамическими атрибутами узлов.

v1: среда реагирует на распределение агентов:
  - housing_price растёт при притоке агентов
  - wage_level меняется от соотношения worker_density / jobs
"""

import json
import math
import networkx as nx
import numpy as np
from pathlib import Path

DISTRICT_CENTERS = {
    "District of Trnava":              (48.3774, 17.5884),
    "District of Dunajská\xa0Streda":  (47.9959, 17.6169),
    "District of Galanta":             (48.1889, 17.7283),
    "District of Hlohovec":            (48.4317, 17.8003),
    "District of Piešťany":            (48.5880, 17.8328),
    "District of Senica":              (48.6797, 17.3659),
    "District of Skalica":             (48.8479, 17.2264),
}

TRNAVA_DISTRICTS = list(DISTRICT_CENTERS.keys())

# Скорость реакции среды (сглаживание)
HOUSING_ALPHA    = 0.03   # насколько быстро цены реагируют на плотность агентов
WAGE_ALPHA       = 0.02   # насколько быстро зарплаты реагируют на рынок труда


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_latest(data: dict, year: int = 2024):
    if not data: return None
    if year in data: return data[year]
    for y in sorted(data.keys(), reverse=True):
        if data[y] is not None: return data[y]
    return None


def build_graph(env_path: str = "environment.json", year: int = 2024) -> nx.Graph:
    """Строит начальный граф из environment.json."""
    with open(env_path) as f:
        env = json.load(f)

    locations = env["locations"]
    G = nx.Graph()

    for district, (lat, lon) in DISTRICT_CENTERS.items():
        data = locations.get(district, {})

        # Зарплата
        wages = data.get("wages", {})
        avg_wage = _get_latest(wages.get("Total", {}), year) or 0
        if avg_wage == 0 and wages:
            vals = [_get_latest(v, year) for v in wages.values() if _get_latest(v, year)]
            avg_wage = sum(vals) / len(vals) if vals else 1500

        # Цена жилья — используем региональный базис (одинаковый),
        # но инициализируем с небольшим разбросом ±15% по зарплатному коэффициенту
        regional_wage_avg = 1573
        wage_ratio = avg_wage / regional_wage_avg
        housing_base = 2546  # реальная региональная цена €/м²
        housing_price = housing_base * (0.75 + 0.5 * wage_ratio)

        # Занятость: occupations.Total = реальные работники в районе
        occ = data.get("occupations", {})
        workers_employed = _get_latest(occ.get("Total", {}), year) or 0

        # Население из возрастных групп
        ag = data.get("age_groups", {})
        population = 0
        for age_data in ag.values():
            population += (_get_latest(age_data.get("male", {}), year) or 0)
            population += (_get_latest(age_data.get("female", {}), year) or 0)

        # jobs_capacity: оцениваем через occupations.Total
        # (сколько рабочих мест "исторически" поглощает район)
        jobs_capacity = workers_employed  # базовая ёмкость рынка труда

        G.add_node(
            district,
            lat=lat,
            lon=lon,
            # --- статичные базовые значения (не меняются) ---
            avg_wage_base=avg_wage,
            housing_price_base=housing_price,
            jobs_capacity=max(jobs_capacity, 1),
            real_population=population,
            # --- динамические (обновляются каждый тик) ---
            avg_wage=avg_wage,
            housing_price_m2=housing_price,
            agent_count=0,           # будет заполнено после создания агентов
        )

    # Рёбра: расстояния
    districts = TRNAVA_DISTRICTS
    for i in range(len(districts)):
        for j in range(i + 1, len(districts)):
            lat1, lon1 = DISTRICT_CENTERS[districts[i]]
            lat2, lon2 = DISTRICT_CENTERS[districts[j]]
            G.add_edge(districts[i], districts[j],
                       distance_km=round(_haversine_km(lat1, lon1, lat2, lon2), 1))

    return G


def update_graph(G: nx.Graph, agent_district_counts: dict, total_agents: int):
    """
    Обновляет динамические атрибуты графа на основе текущего распределения агентов.

    Логика:
      - housing_price растёт если агентов в районе больше чем базовая доля
      - avg_wage снижается если агентов (рабочей силы) слишком много относительно jobs_capacity
    """
    for district in TRNAVA_DISTRICTS:
        attr = G.nodes[district]
        current_agents = agent_district_counts.get(district, 0)

        # Ожидаемая доля агентов = real_population / total_real_pop
        total_real_pop = sum(G.nodes[d]["real_population"] for d in TRNAVA_DISTRICTS)
        expected_share = attr["real_population"] / max(total_real_pop, 1)
        actual_share   = current_agents / max(total_agents, 1)

        # --- Жильё: давление спроса ---
        # Если доля агентов > ожидаемой — цены растут
        density_ratio = actual_share / max(expected_share, 0.001)
        # Мягкое экспоненциальное давление
        housing_pressure = (density_ratio - 1.0)  # > 0 = перегрузка, < 0 = отток
        new_housing = attr["housing_price_m2"] * (1 + HOUSING_ALPHA * housing_pressure)
        # Ограничиваем диапазон: не менее 60% и не более 200% от базы
        new_housing = float(np.clip(new_housing,
                                     attr["housing_price_base"] * 0.6,
                                     attr["housing_price_base"] * 2.0))

        # --- Зарплата: давление предложения труда ---
        # Агенты работоспособного возраста как прокси рабочей силы
        # (упрощение: все агенты в районе / jobs_capacity)
        labour_pressure = current_agents / max(attr["jobs_capacity"], 1)
        # labour_pressure > 1 → избыток труда → зарплаты вниз
        # labour_pressure < 1 → дефицит → зарплаты вверх
        # Нейтральный уровень: ~5 агентов на реального работника (масштаб 1:110)
        neutral_labour = (attr["real_population"] / max(attr["jobs_capacity"], 1)) / 110
        wage_pressure = (neutral_labour - labour_pressure) / max(neutral_labour, 0.001)
        wage_pressure = float(np.clip(wage_pressure, -0.5, 0.5))

        new_wage = attr["avg_wage"] * (1 + WAGE_ALPHA * wage_pressure)
        new_wage = float(np.clip(new_wage,
                                  attr["avg_wage_base"] * 0.7,
                                  attr["avg_wage_base"] * 1.5))

        G.nodes[district]["housing_price_m2"] = round(new_housing, 0)
        G.nodes[district]["avg_wage"] = round(new_wage, 0)
        G.nodes[district]["agent_count"] = current_agents


def print_graph_summary(G: nx.Graph):
    print("=" * 68)
    print("ГРАФ ТРНАВСКОГО КРАЯ")
    print("=" * 68)
    print(f"Узлов: {G.number_of_nodes()}  |  Рёбер: {G.number_of_edges()}")
    print()
    print(f"{'Район':<20} {'Зарплата':>10} {'Жильё/м²':>10} {'Jobs':>8} {'Нас-е':>9}")
    print("-" * 60)
    for node, attr in sorted(G.nodes(data=True), key=lambda x: -x[1].get("avg_wage", 0)):
        name = node.replace("District of ", "")
        print(f"  {name:<18} {attr['avg_wage']:>9,.0f}€ {attr['housing_price_m2']:>9,.0f}€"
              f" {attr['jobs_capacity']:>8,} {attr['real_population']:>9,}")
    print()
    edges = sorted(G.edges(data=True), key=lambda x: x[2]["distance_km"])
    print("Расстояния (топ-3):")
    for u, v, attr in edges[:3]:
        print(f"  {u.replace('District of ','')} ↔ {v.replace('District of ','')} : {attr['distance_km']} км")


if __name__ == "__main__":
    G = build_graph("environment.json")
    print_graph_summary(G)
