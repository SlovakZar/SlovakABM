"""
graph.py — граф Трнавского края.

Узлы: 7 районов Region of Trnava
Атрибуты узла: avg_wage, housing_price_m2, total_employment, population
Рёбра: географическое расстояние между административными центрами (км)

Расстояния рассчитаны по координатам центров районных городов.
"""

import json
import math
import networkx as nx
from pathlib import Path

# ── координаты административных центров районов (lat, lon) ───────────────────
DISTRICT_CENTERS = {
    "District of Trnava":          (48.3774, 17.5884),
    "District of Dunajská\xa0Streda": (47.9959, 17.6169),  # note: non-breaking space in json key
    "District of Galanta":         (48.1889, 17.7283),
    "District of Hlohovec":        (48.4317, 17.8003),
    "District of Piešťany":        (48.5880, 17.8328),
    "District of Senica":          (48.6797, 17.3659),
    "District of Skalica":         (48.8479, 17.2264),
}

# Alias — в json ключ содержит \xa0 (неразрывный пробел)
DISTRICT_ALIASES = {
    "District of Dunajská Streda": "District of Dunajská\xa0Streda",
}

TRNAVA_DISTRICTS = list(DISTRICT_CENTERS.keys())


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Расстояние между двумя точками на сфере (км)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_latest(data: dict, year: int = 2024):
    if not data:
        return None
    if year in data:
        return data[year]
    for y in sorted(data.keys(), reverse=True):
        if data[y] is not None:
            return data[y]
    return None


def build_graph(env_path: str = "environment.json", year: int = 2024) -> nx.Graph:
    """
    Строит граф Трнавского края из environment.json.
    Возвращает nx.Graph с атрибутами на узлах и расстояниями на рёбрах.
    """
    with open(env_path) as f:
        env = json.load(f)

    locations = env["locations"]
    regions = env.get("regions", {})
    trnava_region = regions.get("Region of Trnava", {})

    G = nx.Graph()

    # ── добавляем узлы ────────────────────────────────────────────────────────
    for district, (lat, lon) in DISTRICT_CENTERS.items():
        # Ищем данные — учитываем alias
        data = locations.get(district) or locations.get(DISTRICT_ALIASES.get(district, district))
        if data is None:
            print(f"  [warn] No data for {district}")
            data = {}

        # Зарплата: Total по всем секторам
        wages = data.get("wages", {})
        avg_wage = _get_latest(wages.get("Total", {}), year) or 0

        # Если Total пустой — берём среднее по секторам
        if avg_wage == 0 and wages:
            vals = [_get_latest(v, year) for v in wages.values() if _get_latest(v, year)]
            avg_wage = sum(vals) / len(vals) if vals else 0

        # Цена жилья за м²
        # Housing данные downscaled с регионального уровня — одинаковы для всех районов.
        # Используем прокси: цены коррелируют с зарплатами (богаче → дороже).
        # Берём региональную базу и масштабируем через wage_ratio.
        housing = data.get("housing", {})
        housing_base = _get_latest(
            housing.get("Average price for buying an apartment per m2", {}), year
        ) or 2500  # дефолт для Трнавского края

        # Корректируем по зарплате: wage_ratio относительно региональной средней
        regional_wage_avg = 1580  # средняя по региону из данных
        wage_ratio = avg_wage / regional_wage_avg if avg_wage > 0 else 1.0
        housing_price = housing_base * (0.6 + 0.8 * wage_ratio)  # диапазон ±40% от базы

        # Занятость на душу населения (нормализованная)
        # Используем кол-во рабочих мест / население для межрайонного сравнения
        labour = data.get("labour", {})
        employment_abs = _get_latest(
            labour.get("Total employment / jobs (work place based)", {}), year
        ) or 0

        # Население: сумма по возрастным группам
        ag = data.get("age_groups", {})
        population = 0
        for age_data in ag.values():
            m = _get_latest(age_data.get("male", {}), year) or 0
            f_ = _get_latest(age_data.get("female", {}), year) or 0
            population += m + f_

        # Занятость на душу: jobs/population (чем выше → больше работы)
        employment_per_capita = (employment_abs / population) if population > 0 else 0.4

        G.add_node(
            district,
            lat=lat,
            lon=lon,
            avg_wage=avg_wage,
            housing_price_m2=round(housing_price, 0),
            employment=employment_abs,
            employment_per_capita=employment_per_capita,
            population=population,
        )

    # ── добавляем рёбра: расстояние между всеми парами ───────────────────────
    districts = list(DISTRICT_CENTERS.keys())
    for i in range(len(districts)):
        for j in range(i + 1, len(districts)):
            d_i = districts[i]
            d_j = districts[j]
            lat1, lon1 = DISTRICT_CENTERS[d_i]
            lat2, lon2 = DISTRICT_CENTERS[d_j]
            dist_km = _haversine_km(lat1, lon1, lat2, lon2)
            G.add_edge(d_i, d_j, distance_km=round(dist_km, 1))

    return G


def print_graph_summary(G: nx.Graph):
    print("=" * 60)
    print("ГРАФ ТРНАВСКОГО КРАЯ")
    print("=" * 60)
    print(f"Узлов: {G.number_of_nodes()}  |  Рёбер: {G.number_of_edges()}")
    print()
    print(f"{'Район':<35} {'Зарплата':>10} {'Жильё/м²':>10} {'Нас-е':>8}")
    print("-" * 65)
    for node, attr in sorted(G.nodes(data=True), key=lambda x: -x[1].get("avg_wage", 0)):
        name = node.replace("District of ", "")
        print(f"  {name:<33} {attr['avg_wage']:>9,.0f}€ {attr['housing_price_m2']:>9,.0f}€ {attr['population']:>8,}")
    print()
    print("Расстояния (топ-3 ближайших пар):")
    edges = sorted(G.edges(data=True), key=lambda x: x[2]["distance_km"])
    for u, v, attr in edges[:3]:
        print(f"  {u.replace('District of ','')} ↔ {v.replace('District of ','')} : {attr['distance_km']} км")


if __name__ == "__main__":
    G = build_graph("environment.json")
    print_graph_summary(G)
