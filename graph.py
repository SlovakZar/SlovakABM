"""
graph.py v3 — Slovakia graph based on real commuting flows.

Sources:
  environment.json                — node attributes (SODB 2021 / NBS / ŠÚ SR)
  commuting_filtered_with_travel.csv — flow matrix + travel time (OSRM)

Graph structure:
  Nodes: 79 districts of Slovakia
  Edges: directed (origin → dest) based on real commuting flows

Node attributes:
  avg_wage, avg_wage_base       — average wage (€/month)
  housing_price_m2, ..._base    — housing price (€/m²)
  unemployment_rate             — unemployment rate [0,1]
  jobs_capacity                 — total employment by district (sum of WC)
  real_population               — population count
  owner_share                   — homeowner share [0,1]
  region                        — region code (BA, TT, TN, NR, ZA, BB, PO, KE)
  infrastructure_score          — infrastructure level [0,1]
  agent_count                   — current agent count (updated each tick)

Edge attributes:
  travel_time_min  — travel time (min)
  flow_work        — employed flow
  total_flow       — total flow
  flow_weight      — normalized edge weight [0,1]

Environment response (update_graph):
  housing_price responds to agent density vs expected share
  avg_wage responds to agent-to-jobs_capacity ratio
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# ── Environment response parameters ───────────────────────────────────────────────────
HOUSING_ALPHA = 0.03
WAGE_ALPHA    = 0.02

# ── Dynamic housing parameters ─────────────────────────────────────────────
AGENT_HOUSING_FOOTPRINT = 1.1   # один агент занимает ~1.1 условных квартир
HOUSING_REMAINING_FLOOR = 1.5   # пол для remaining (порог остановки роста цены)

# ── v5: Market capacity from companies ────────────────────────────────────────────
SIZE_EMPLOYEES = {"small": 25, "medium": 130, "large": 400}


def recompute_industry_jobs(G: nx.DiGraph, district: str, industry_shares: dict) -> None:
    """
    v5: Recalculates industry_jobs[district] based on current company count.

    capacity = Σ(size × employees_per_size)
    By industry: capacity_ind = capacity × industry_share
    vacant = max(0, capacity_ind − occupied)

    WARNING: business data is stored in real values (на ~5.4M жителей),
    поэтому ёмкость scaled через G.graph["agent_scale"].
    Если scale не задан (нет in graph), используется 1.0 (без масштабирования).

    Called after change_company_count() and during initialization.
    """
    biz = G.nodes[district].get("business", {})
    if not biz:
        return

    scale = G.graph.get("agent_scale", 1.0)

    total_cap = (biz.get("small_companies", 0) * SIZE_EMPLOYEES["small"] +
                 biz.get("medium_companies", 0) * SIZE_EMPLOYEES["medium"] +
                 biz.get("large_companies", 0) * SIZE_EMPLOYEES["large"])
    if total_cap <= 0:
        return

    total_cap = max(1, int(total_cap * scale))

    ind_jobs = G.nodes[district].get("industry_jobs", {})
    if not ind_jobs or not industry_shares:
        return

    total_share = sum(industry_shares.values()) or 1.0
    for ind, share in industry_shares.items():
        if ind not in ind_jobs:
            continue
        norm_share = share / total_share
        cap_ind = max(1, int(total_cap * norm_share))
        occ = ind_jobs[ind].get("occupied", 0)
        ind_jobs[ind]["vacant"] = max(0, cap_ind - occ)
        ind_jobs[ind]["capacity"] = cap_ind

    G.nodes[district]["jobs_capacity"] = max(1, total_cap)


def change_company_count(G: nx.DiGraph, district: str, size: str, delta: int,
                         industry_shares: dict) -> int:
    """
    v5: +1 или −1 компанию размера size. Пересчитывает ёмкость.

    Returns approximate number of affected jobs.
    """
    if district not in G.nodes:
        return 0
    biz = G.nodes[district].get("business")
    if not biz:
        return 0

    key = f"{size}_companies"
    old_val = biz.get(key, 0)
    biz[key] = max(0, old_val + delta)

    n_jobs = SIZE_EMPLOYEES.get(size, 25)
    recompute_industry_jobs(G, district, industry_shares)
    return n_jobs

# ── Infrastructure spillover ─────────────────────────────────────────────────
SPILLOVER_WEIGHT = 0.25         # вес бонуса от соседей (0 = выключен)
SPILLOVER_TIME_MAX = 90.0       # макс. travel time (min), после которого spillover = 0

# ── Housing market sensitivity by region ─────────────────────────────────
# BA (Bratislava) = 1.30 — increased housing competition
# All others   = 1.00 — standard sensitivity
REGION_HOUSING_SENSITIVITY = {
    "BA": 1.30,
    "KE": 1.00,
    "TT": 1.00,
    "NR": 1.00,
    "ZA": 1.00,
    "TN": 1.00,
    "BB": 1.00,
    "PO": 1.00,
}

# Regional centers for long-range edges in awareness_set
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

# ── Helper functions for calculating infrastructure_score ─────────────────

def _compute_infrastructure_raw(infra_data: dict) -> float:
    """
    "Raw" linear count — WITHOUT diminishing returns.
    Used for spillover: neighbor with 5 hospitals gives a larger bonus,
    чем сосед с 2 больницами.
    """
    if not infra_data:
        return 0.0

    return (infra_data.get("polyclinics", 0) * 0.2 +
            infra_data.get("hospitals", 0)   * 0.5 +
            infra_data.get("cinemas", 0)     * 0.1 +
            infra_data.get("museums", 0)     * 0.1 +
            infra_data.get("galleries", 0)   * 0.1)


def _compute_infrastructure_score(infra_data: dict) -> float:
    """
    Calculates district score [0, 1] with diminishing returns для больниц и поликлиник.
    First 2 facilities — full weight, each next — sqrt(k).
    For spillover _compute_infrastructure_raw is used (без diminishing).
    """
    if not infra_data:
        return 0.5  # default value

    # diminishing returns: count → effective_count (first 2 — linear, beyond — sqrt)
    def _dim(count: int) -> float:
        if count <= 2:
            return float(count)
        return 2.0 + (count - 2) ** 0.5

    poly_eff = _dim(infra_data.get("polyclinics", 0))
    hosp_eff = _dim(infra_data.get("hospitals", 0))

    raw = (poly_eff * 0.2 +
           hosp_eff * 0.5 +
           infra_data.get("cinemas", 0)   * 0.1 +
           infra_data.get("museums", 0)   * 0.1 +
           infra_data.get("galleries", 0) * 0.1)

    # Normalization: max raw in data ~7.4 (Bratislava I) → divide by 5,
    # to get score ~0.8-1.0 for leaders.
    score = min(1.0, raw / 5.0)
    return round(score, 3)


def _get_min_travel_between(G: nx.DiGraph, a: str, b: str) -> float:
    """Minimum travel_time between districts a and b in any direction."""
    t = 999.0
    if G.has_edge(a, b):
        t = min(t, G.edges[a, b].get("travel_time_min", 999.0))
    if G.has_edge(b, a):
        t = min(t, G.edges[b, a].get("travel_time_min", 999.0))
    return t


def get_effective_infrastructure(G: nx.DiGraph, district: str) -> float:
    """
    District infrastructure score with spillover from neighbors.

    Neighbors — все районы, с которыми есть commuting-связь (входящая или исходящая).
    Bonus = SPILLOVER_WEIGHT × (neighbor_raw − own_raw) × time_factor,
    only if neighbor is "stronger" in raw infrastructure.
    """
    loc = G.nodes[district]
    infra_data = loc.get("infrastructure", {})
    own_raw = _compute_infrastructure_raw(infra_data)
    own_score = loc.get("infrastructure_score", 0.5)

    # Collect neighbors from commuting edges (входящие + исходящие)
    neighbors: set[str] = set()
    for src, _ in G.in_edges(district):
        neighbors.add(src)
    for _, dst in G.out_edges(district):
        neighbors.add(dst)
    neighbors.discard(district)

    if not neighbors:
        return own_score

    spillover_bonus = 0.0
    for nb in neighbors:
        nb_data = G.nodes[nb].get("infrastructure", {})
        nb_raw = _compute_infrastructure_raw(nb_data)

        # Bonus only if neighbor is objectively "stronger" по сырой инфраструктуре
        if nb_raw <= own_raw:
            continue

        raw_diff = nb_raw - own_raw

        # Actualор времени: чем быстрее доехать, тем сильнее spillover
        travel = _get_min_travel_between(G, district, nb)
        if travel >= SPILLOVER_TIME_MAX:
            continue
        time_factor = 1.0 - travel / SPILLOVER_TIME_MAX

        spillover_bonus += SPILLOVER_WEIGHT * raw_diff * time_factor

    return min(1.0, own_score + spillover_bonus)


def build_graph(
    env_path: str = "data/environment.json",
    commuting_path: str = "data/commuting_filtered_with_travel.csv",
) -> nx.DiGraph:
    """
    Builds directed Slovakia graph from commuting matrix.
    Node attributes загружаются из environment.json (создан build_environment.py).
    """
    # ── Loading environment.json ─────────────────────────────────────────────
    env_path_obj = Path(env_path)
    if not env_path_obj.exists():
        env_path_obj = Path(__file__).parent / env_path
    with open(env_path_obj, encoding="utf-8") as f:
        env = json.load(f)
    locations = env.get("locations", {})

    # ── Loading commuting matrix ────────────────────────────────────────────
    comm_path_obj = Path(commuting_path)
    if not comm_path_obj.exists():
        comm_path_obj = Path(__file__).parent / commuting_path
    df_comm = pd.read_csv(comm_path_obj)

    # Only district → district edges
    mask = (
        df_comm["origin_district"].str.startswith("District of") &
        df_comm["destination_district"].str.startswith("District of")
    )
    df_comm = df_comm[mask].copy()

    # Self-loops → internal flow attribute
    self_loops = df_comm[df_comm["origin_district"] == df_comm["destination_district"]]
    edges_df   = df_comm[df_comm["origin_district"] != df_comm["destination_district"]]
    internal_flow = dict(zip(self_loops["origin_district"], self_loops["total_flow"]))

    max_flow = edges_df["total_flow"].max() if len(edges_df) > 0 else 1.0

    # ── Collect all districts ─────────────────────────────────────────────────────
    all_districts = (set(edges_df["origin_district"]) |
                     set(edges_df["destination_district"]) |
                     set(locations.keys()))

    # ── Build graph ───────────────────────────────────────────────────────────
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

        # Housing market sensitivity: from location data or region fallback
        housing_mkt_sens = housing_data.get("housing_market_sensitivity")
        if housing_mkt_sens is None:
            housing_mkt_sens = REGION_HOUSING_SENSITIVITY.get(region, 1.00)

        # Compute infrastructure_score based on "infrastructure" block
        infra_data = loc.get("infrastructure", {})
        infrastructure_score = _compute_infrastructure_score(infra_data)

        # jobs_capacity — сумма занятых по всем отраслям из environment
        salary_by_ind = loc.get("salary_by_industry", {})
        business_data = loc.get("business", {})
        # Если нет детальных данных — используем population * occupiedсть ~45%
        jobs_capacity = max(1, int(population * (1 - unemployment_rate) * 0.45))

        # ── Industry capacity: distribute jobs_capacity across industries ────
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

        # ── Initialize industry_pressure = 1.0 for each industry ──
        initial_industry_pressure = {ind: 1.0 for ind in industry_capacity.keys()}

        G.add_node(
            district,
            region=region,
            real_population=int(population),
            unemployment_rate=float(unemployment_rate),
            owner_share=float(owner_share),
            jobs_capacity=jobs_capacity,
            internal_flow=float(internal_flow.get(district, 0)),
            # Base (immutable)
            avg_wage_base=float(avg_wage),
            housing_price_base=float(housing_m2),
            infrastructure_score=infrastructure_score,
            salary_by_industry=salary_by_ind,             # отраслевые wage
            industry_capacity=industry_capacity,          # industry capacity (базовая)
            business=business_data,                       # business statistics
            # Housing (from environment.json)
            total_dwellings=int(total_dwell),             # total dwellings
            vacant_dwellings=int(vacant_dwell),           # vacantе жильё
            housing_market_sensitivity=float(housing_mkt_sens),  # market sensitivity
            # Dynamic (updated each tick)
            avg_wage=float(avg_wage),
            housing_price_m2=float(housing_m2),
            effective_housing_price_m2=float(housing_m2),  # effective price (accounting for remaining)
            housing_remaining=float(max(1.0, vacant_dwell)),  # housing remaining (иниц. в init_housing_remaining)
            agent_count=0,
            industry_pressure=initial_industry_pressure,  # v4: начальное давление = 1.0 (накопительная система)
        )

    # ── Edges ────────────────────────────────────────────────────────────────
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

    print(f"  Graph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
    _print_top_edges(G)
    return G


def _print_top_edges(G: nx.DiGraph, n: int = 5):
    edges = sorted(G.edges(data=True),
                   key=lambda e: e[2].get("total_flow", 0), reverse=True)
    print(f"  Топ-{n} edges по потоку:")
    for src, dst, attr in edges[:n]:
        s = src.replace("District of ", "")
        d = dst.replace("District of ", "")
        print(f"    {s} → {d}: {attr['total_flow']:,.0f} people | {attr['travel_time_min']:.0f} min")


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
    Forms awareness_set — districts the agent actually considers.

      1. Neighbors по commuting-графу в пределах time_limit
         time_limit grows with perceived_control and info_quality.
      2. If network_location=True — regional centers of other regions
      3. Candidates sorted by total_flow and truncated to max_candidates.

    mode:
      "work" — out-edges from residence (where they commute to work)
      "residence" — out-edges from residence (where they can move to)
      "satellite" — in-edges to dst_work (where people commute from — satellites)
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
        # Outgoing flows: where people commute from this district
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
    Updates dynamic node attributes each tick.

    Housing: растёт при притоке agents, падает при оттоке.
    Wage: responds to labor force to jobs_capacity ratio.
    """
    total_real_pop = sum(G.nodes[d]["real_population"] for d in G.nodes)

    for district in G.nodes:
        attr = G.nodes[district]
        current_agents = agent_district_counts.get(district, 0)

        expected_share = attr["real_population"] / max(total_real_pop, 1)
        actual_share   = current_agents / max(total_agents, 1)
        density_ratio  = actual_share / max(expected_share, 0.001)

        # Housing
        housing_pressure = density_ratio - 1.0
        new_housing = attr["housing_price_m2"] * (1 + HOUSING_ALPHA * housing_pressure)
        new_housing = float(np.clip(
            new_housing,
            attr["housing_price_base"] * 0.55,
            attr["housing_price_base"] * 2.5,
        ))

        # Wage
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

        # ── Effective housing price (accounting for housing_remaining) ──────────
        remaining = attr.get("housing_remaining", HOUSING_REMAINING_FLOOR)
        sensitivity = attr.get("housing_market_sensitivity", 1.0)
        delta = new_housing * (AGENT_HOUSING_FOOTPRINT / max(remaining, HOUSING_REMAINING_FLOOR)) * sensitivity
        G.nodes[district]["effective_housing_price_m2"] = round(new_housing + delta, 0)


def init_housing_remaining(G: nx.DiGraph, n_agents: int) -> None:
    """
    Initializes housing_remaining in graph nodes.

    Для каждого района:
      scaled_vacant = vacant_dwellings × (n_agents / total_population)
    где n_agents / total_population — масштабный коэффициент симуляции.

    Также сразу вычисляет effective_housing_price_m2.
    """
    total_pop = sum(
        G.nodes[d].get("real_population", 1)
        for d in G.nodes
    )
    if total_pop == 0:
        return

    scale = n_agents / total_pop

    for district in G.nodes:
        attr = G.nodes[district]
        vacant = attr.get("vacant_dwellings", 0)
        remaining = max(HOUSING_REMAINING_FLOOR, vacant * scale)
        G.nodes[district]["housing_remaining"] = float(remaining)

        # Immediately compute effective price
        base_price = attr.get("housing_price_m2", 1800.0)
        sensitivity = attr.get("housing_market_sensitivity", 1.0)
        delta = base_price * (AGENT_HOUSING_FOOTPRINT / remaining) * sensitivity
        G.nodes[district]["effective_housing_price_m2"] = round(base_price + delta, 0)

    # Diagnostics
    total_scaled = sum(G.nodes[d]["housing_remaining"] for d in G.nodes)
    print(f"  HOUSING_REMAINING (граф): {G.number_of_nodes()} nodes")
    print(f"    total_scaled_vacant={total_scaled:,.1f}  "
          f"scale={scale:.6f}  n_agents={n_agents:,}")
    ba1 = "District of Bratislava I"
    if ba1 in G.nodes:
        print(f"    {ba1}: remaining={G.nodes[ba1]['housing_remaining']:.1f}  "
              f"sensitivity={G.nodes[ba1].get('housing_market_sensitivity', 1.0)}  "
              f"effective_price={G.nodes[ba1]['effective_housing_price_m2']:.0f}")


def get_effective_housing_price(G: nx.DiGraph, district: str) -> float:
    """
    Returns effective housing price (€/m²) for district from graph.

    Value is already precomputed in update_graph() each tick,
    here we just read from node attributes.
    """
    if district in G.nodes:
        return float(G.nodes[district].get("effective_housing_price_m2", 1800.0))
    return 1800.0


def update_industry_pressure(G: nx.DiGraph, df=None):
    """
    v4: Initializes industry_pressure only if not yet initialized.
    
    Now this is an accumulative pressure system (не пересчет каждый тик):
      - Начальное: pressure[industry] = 1.0
      - When a position is filled: pressure += 1 / max(vacant, 1)
      - When a position is freed: pressure -= 1 / max(old_vacant, 1)
    
    This function is called only during graph initialization и не пересчитывает
    давление каждый тик. Pressure updates happen through update_industry_pressure_delta()
    при событиях COMMUTE, JOB_CHANGE, AGENT_MOVED.
    """
    # Проверяем, инициализировано ли давление
    for district in G.nodes:
        pressure = G.nodes[district].get("industry_pressure", {})
        if not pressure:
            # Initialize for the first time
            capacity = G.nodes[district].get("industry_capacity", {})
            initial_pressure = {ind: 1.0 for ind in capacity.keys()}
            G.nodes[district]["industry_pressure"] = initial_pressure


def initialize_industry_pressure_from_agents(G: nx.DiGraph, df: pd.DataFrame) -> None:
    """
    v5: Initializes industry_pressure as ratio: agents / capacity.

    Called once after agent creation and sync_industry_jobs_to_graph.
    pressure > 1.0 → market is oversaturated.
    """
    employed = df[df["is_employed"]]
    wp_ind_counts = (employed
                     .groupby(["workplace_district", "industry"])["id"]
                     .count()
                     .to_dict())

    for district in G.nodes:
        ind_jobs = G.nodes[district].get("industry_jobs", {})
        pressure_dict = G.nodes[district].get("industry_pressure", {})

        if ind_jobs and pressure_dict:
            for industry in ind_jobs.keys():
                cnt = wp_ind_counts.get((district, industry), 0)
                cap = ind_jobs[industry].get("capacity",
                       ind_jobs[industry].get("occupied", 1) + ind_jobs[industry].get("vacant", 0))
                pressure_dict[industry] = cnt / max(cap, 1)


def update_industry_pressure_delta(G: nx.DiGraph, district: str, industry: str, 
                                    delta: float) -> None:
    """
    v5: Updates industry_pressure by ratio system.

    При заполнении: delta = +1 / max(capacity, 1)
    При освобождении: delta = -1 / max(capacity, 1)

    capacity = occupied + vacant из industry_jobs.

    Вызывается из engine.py при событиях COMMUTE/JOB_CHANGE/AGENT_MOVED.
    """
    if district not in G.nodes:
        return

    pressure_dict = G.nodes[district].get("industry_pressure", {})
    if not pressure_dict:
        capacity = G.nodes[district].get("industry_capacity", {})
        pressure_dict = {ind: 0.0 for ind in capacity.keys()}
        G.nodes[district]["industry_pressure"] = pressure_dict

    if industry in pressure_dict:
        new_pressure = pressure_dict[industry] + delta
        pressure_dict[industry] = max(0.0, new_pressure)
    else:
        pressure_dict[industry] = max(0.0, delta)


def sync_industry_jobs_to_graph(G: nx.DiGraph, industry_jobs: dict, jobs_capacity: dict,
                                 n_agents: int = 0):
    """
    v3/v5: Syncs INDUSTRY_JOBS_CAPACITY and JOBS_CAPACITY into graph nodes.

    Also sets G.graph["agent_scale"] = n_agents / real_population
    для масштабирования business-ёмкости (recompute_industry_jobs).

    Called from run.py after create_agents().
    """
    for district in G.nodes:
        if district in industry_jobs:
            G.nodes[district]["industry_jobs"] = {
                ind: {"occupied": v["occupied"], "vacant": v["vacant"]}
                for ind, v in industry_jobs[district].items()
            }
        if district in jobs_capacity:
            G.nodes[district]["jobs_capacity"] = jobs_capacity[district]

    # Set scale factor for business data
    if n_agents > 0:
        total_real_pop = sum(
            G.nodes[d].get("real_population", 0) for d in G.nodes
        )
        G.graph["agent_scale"] = n_agents / max(total_real_pop, 1)


def print_graph_summary(G: nx.DiGraph):
    from collections import Counter
    print("=" * 72)
    print("SLOVAKIA GRAPH — COMMUTING")
    print("=" * 72)
    print(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
    regions = Counter(G.nodes[d].get("region", "XX") for d in G.nodes)
    print("Districtов по регионам:", dict(sorted(regions.items())))

    # Additionally: средний infrastructure_score
    infra_scores = [G.nodes[d].get("infrastructure_score", 0.5) for d in G.nodes]
    print(f"Average infrastructure_score: {np.mean(infra_scores):.3f} (min={np.min(infra_scores):.3f}, max={np.max(infra_scores):.3f})")

    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    print("\nTop-10 districts by incoming flows (in-degree):")
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
