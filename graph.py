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
  infrastructure_score          — уровень инфраструктуры [0,1]
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

# ── Чувствительность рынка жилья по регионам ─────────────────────────────────
# BA (Братислава) = 1.75 — высокая конкуренция за жильё
# KE (Кошице)     = 1.50 — второй город
# TT, NR, ZA      = 1.05–1.10 — региональные центры
# TN, BB, PO      = 1.00 — стандартная чувствительность
REGION_HOUSING_SENSITIVITY = {
    "BA": 1.75,
    "KE": 1.50,
    "TT": 1.10,
    "NR": 1.05,
    "ZA": 1.05,
    "TN": 1.00,
    "BB": 1.00,
    "PO": 1.00,
}

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

# ── Вспомогательная функция для расчёта infrastructure_score ─────────────────
def _compute_infrastructure_score(infra_data: dict) -> float:
    """
    Вычисляет score инфраструктуры района в диапазоне [0, 1] на основе
    количества медицинских, культурных и социальных учреждений.
    """
    if not infra_data:
        return 0.5  # значение по умолчанию

    polyclinics = infra_data.get("polyclinics", 0)
    hospitals   = infra_data.get("hospitals", 0)
    cinemas     = infra_data.get("cinemas", 0)
    museums     = infra_data.get("museums", 0)
    galleries   = infra_data.get("galleries", 0)

    # Веса: больницы и поликлиники важнее, культура — дополнительный бонус
    raw = (polyclinics * 0.2 +
           hospitals   * 0.5 +
           cinemas     * 0.1 +
           museums     * 0.1 +
           galleries   * 0.1)

    # Нормализация: максимальное raw по данным ~7.4 (Bratislava I) → делим на 5,
    # чтобы получить score ~0.8-1.0 для лидеров.
    score = min(1.0, raw / 5.0)
    return round(score, 3)


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
        housing_data  = loc.get("housing", {}) or {}
        housing_m2    = housing_data.get("price_m2") or 1500.0
        owner_share   = housing_data.get("owner_share") or 0.65
        total_dwell   = housing_data.get("total_dwellings") or 0
        vacant_dwell  = housing_data.get("vacant_dwellings") or 0

        # Чувствительность рынка жилья: из данных локации или fallback по региону
        housing_mkt_sens = housing_data.get("housing_market_sensitivity")
        if housing_mkt_sens is None:
            housing_mkt_sens = REGION_HOUSING_SENSITIVITY.get(region, 1.00)

        # Вычисляем infrastructure_score на основе блока "infrastructure"
        infra_data = loc.get("infrastructure", {})
        infrastructure_score = _compute_infrastructure_score(infra_data)

        # jobs_capacity — сумма занятых по всем отраслям из environment
        salary_by_ind = loc.get("salary_by_industry", {})
        business_data = loc.get("business", {})
        # Если нет детальных данных — используем population * занятость ~45%
        jobs_capacity = max(1, int(population * (1 - unemployment_rate) * 0.45))

        # ── Отраслевая ёмкость: распределяем jobs_capacity по отраслям ────
        # Вес отрасли = её зарплата / сумма зарплат → более высокооплачиваемые
        # отрасли имеют пропорционально большую долю рынка труда.
        total_salary = sum(salary_by_ind.values()) if salary_by_ind else 1.0
        industry_capacity = {}
        if salary_by_ind:
            for ind, sal in salary_by_ind.items():
                share = sal / max(total_salary, 1.0)
                industry_capacity[ind] = max(1, int(jobs_capacity * share))
        else:
            industry_capacity = {"Other": jobs_capacity}

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
            infrastructure_score=infrastructure_score,
            salary_by_industry=salary_by_ind,             # отраслевые зарплаты
            industry_capacity=industry_capacity,          # отраслевая ёмкость (базовая)
            business=business_data,                       # бизнес-статистика
            # Жильё (из environment.json)
            total_dwellings=int(total_dwell),             # общее число жилищ
            vacant_dwellings=int(vacant_dwell),           # свободное жильё
            housing_market_sensitivity=float(housing_mkt_sens),  # чувствительность рынка
            # Динамические (обновляются каждый тик)
            avg_wage=float(avg_wage),
            housing_price_m2=float(housing_m2),
            agent_count=0,
            industry_pressure={},                         # давление по отраслям
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
    info_quality: float = 0.5,
    max_candidates: int = 15,
    mode: str = "work",  # "work" | "residence" | "satellite"
) -> list:
    """
    Формирует awareness_set — районы которые агент реально рассматривает.

      1. Соседи по commuting-графу в пределах time_limit
         time_limit растёт с perceived_control и info_quality.
      2. Если network_location=True — региональные центры других краёв
      3. Кандидаты сортируются по total_flow и обрезаются до max_candidates.

    mode:
      "work" — out-edges из residence (куда ездят работать)
      "residence" — out-edges из residence (куда можно переехать)
      "satellite" — in-edges в dst_work (откуда ездят работать — спутники)
    """
    # time_limit: perceived_control задаёт базовый радиус, info_quality расширяет
    time_limit = 30.0 + 150.0 * perceived_control * (0.7 + 0.6 * info_quality)

    candidates = set()

    if mode == "satellite":
        # Входящие потоки: кто ездит работать в этот район
        for src, _, attr in G.in_edges(district, data=True):
            if attr.get("travel_time_min", 999) <= time_limit:
                candidates.add(src)
    else:
        # Исходящие потоки: куда ездят из этого района
        for _, dst, attr in G.out_edges(district, data=True):
            if attr.get("travel_time_min", 999) <= time_limit:
                candidates.add(dst)

    if network_location:
        current_region = G.nodes[district].get("region", "XX")
        for rcode, center in REGIONAL_CENTERS.items():
            if rcode != current_region and center in G.nodes:
                candidates.add(center)

    candidates.discard(district)

    # Сортировка по total_flow (убывание) и обрезка
    def _flow_weight(d):
        if G.has_edge(district, d):
            return G[district][d].get("total_flow", 0)
        elif G.has_edge(d, district):
            return G[d][district].get("total_flow", 0)
        return 0

    sorted_candidates = sorted(candidates, key=_flow_weight, reverse=True)
    if len(sorted_candidates) > max_candidates:
        sorted_candidates = sorted_candidates[:max_candidates]

    return sorted_candidates


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


def update_industry_pressure(G: nx.DiGraph, df):
    """
    Вычисляет отраслевое давление (industry_pressure) для каждого узла графа.

    v3: Использует industry_jobs (occupied+vacant) если доступно,
    иначе fallback на industry_capacity.

    industry_pressure[district][industry] = занятые_агенты / (occupied + vacant).
    Значение > 1.0 означает перегрузку отрасли в районе.
    Вызывается каждый тик из engine.tick().
    """
    # Подсчитываем занятых агентов по (workplace_district, industry)
    employed = df[df["is_employed"]]
    wp_ind_counts = (employed
                     .groupby(["workplace_district", "industry"])["id"]
                     .count()
                     .to_dict())

    for district in G.nodes:
        # v3: предпочитаем industry_jobs (occupied+vacant)
        ind_jobs = G.nodes[district].get("industry_jobs", {})
        if ind_jobs:
            pressure = {}
            for ind, jobs in ind_jobs.items():
                cnt = wp_ind_counts.get((district, ind), 0)
                total_cap = jobs["occupied"] + jobs["vacant"]
                pressure[ind] = round(cnt / max(total_cap, 1), 3)
            G.nodes[district]["industry_pressure"] = pressure
        else:
            # Fallback: старый industry_capacity
            cap = G.nodes[district].get("industry_capacity", {})
            pressure = {}
            for ind, capacity in cap.items():
                cnt = wp_ind_counts.get((district, ind), 0)
                pressure[ind] = round(cnt / max(capacity, 1), 3)
            G.nodes[district]["industry_pressure"] = pressure


def sync_industry_jobs_to_graph(G: nx.DiGraph, industry_jobs: dict, jobs_capacity: dict):
    """
    v3: Синхронизирует INDUSTRY_JOBS_CAPACITY и JOBS_CAPACITY в узлы графа.

    Вызывается из run.py после create_agents().
    """
    for district in G.nodes:
        if district in industry_jobs:
            G.nodes[district]["industry_jobs"] = {
                ind: {"occupied": v["occupied"], "vacant": v["vacant"]}
                for ind, v in industry_jobs[district].items()
            }
        if district in jobs_capacity:
            G.nodes[district]["jobs_capacity"] = jobs_capacity[district]


def print_graph_summary(G: nx.DiGraph):
    from collections import Counter
    print("=" * 72)
    print("ГРАФ СЛОВАКИИ — COMMUTING")
    print("=" * 72)
    print(f"Узлов: {G.number_of_nodes()}  |  Рёбер: {G.number_of_edges()}")
    regions = Counter(G.nodes[d].get("region", "XX") for d in G.nodes)
    print("Районов по регионам:", dict(sorted(regions.items())))

    # Дополнительно: средний infrastructure_score
    infra_scores = [G.nodes[d].get("infrastructure_score", 0.5) for d in G.nodes]
    print(f"Средний infrastructure_score: {np.mean(infra_scores):.3f} (min={np.min(infra_scores):.3f}, max={np.max(infra_scores):.3f})")

    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    print("\nТоп-10 районов по входящим потокам (in-degree):")
    for node, deg in in_deg[:10]:
        name = node.replace("District of ", "")
        wage = G.nodes[node].get("avg_wage", 0)
        unemp = G.nodes[node].get("unemployment_rate", 0)
        pop  = G.nodes[node].get("real_population", 0)
        infra = G.nodes[node].get("infrastructure_score", 0)
        print(f"  {name:<28} in={deg:3d}  wage={wage:,.0f}€  unemp={unemp:.1%}  pop={pop:,}  infra={infra:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    G = build_graph()
    print_graph_summary(G)
