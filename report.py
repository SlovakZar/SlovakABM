"""
Looking at vacancies in Slovakia on Data Analysts and realizing I write code with Copilot and LLMs myself. I am a fraud and cannot claim Python skills as my own for example.

report.py v7 — Prioritized console report with heatmap.

v7 (regional heatmap):
  - _district_heatmap: visual graph-map with color indication Δ population.
  - demographic_portrait: region table by ticks, top-10 directions
    moves and new commutes (detail=True).
  - migration_summary: economic indicator trends by year (bars).
  - compare_snapshots: regional balance with Δ unemployment and wage.
  - agent_behavior_audit: only in full mode (detail=True).
  - Resilience to old data: all fields checked via .get().
"""

import math
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from collections import Counter, defaultdict
import networkx as nx

from agents import DISTRICT_TO_REGION_CODE



AGE_BINS   = [0, 18, 30, 45, 60, 75, 150]
AGE_LABELS = ["0–17", "18–29", "30–44", "45–59", "60–74", "75+"]

REGION_NAMES = {
    "BA": "Bratislavský",
    "TT": "Trnavský",
    "TN": "Trenčínský",
    "NR": "Nitrianský",
    "ZA": "Žilinský",
    "BB": "Banskobystrický",
    "PO": "Prešovský",
    "KE": "Košický",
}

STATUS_LABELS = {
    "stay":       "Live and work at home",
    "commute":    "Commuters",
    "unemployed": "Unemployed",
    "student":    "Students",
}

# ── Helpers ───────────────────────────────────────────────────────────

def _pct(part, total) -> str:
    if total == 0: return "  0.0%"
    return f"{part/total*100:5.1f}%"


def _bar(value, max_value, width=18) -> str:
    if max_value == 0: return " " * width
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


def _append_dynamic_vars_section(lines: list, df: pd.DataFrame) -> None:
    """v2: Adds dynamic variables section signal system."""
    dyn_vars = [
        ("econ_penalty",             "Econ Penalty",           "penalty to D_econ"),
        ("infra_bonus",              "Infra Bonus",            "infrastructure bonus"),
        ("inertia_mobility_penalty", "Inertia Mobility Penalty", "inertia penalty from neighbors"),
        ("jobloss_econ_gap_bonus",   "Jobloss Econ Gap Bonus", "ramp-bonus to econ_gap from LOST_JOB"),
        ("migration_pressure",       "Migration Pressure",     "accumulated migration pressure"),
    ]

    available = []
    for col, label, desc in dyn_vars:
        if col in df.columns:
            available.append((col, label, desc))

    if not available:
        return

    lines.append(_section("DYNAMIC VARIABLES OF THE SIGNAL SYSTEM v2"))
    lines.append(f"  {'Variable':<28} {'Mean':>10}  {'Median':>10}  {'Q75':>10}  {'Max':>10}  {'Share>0':>8}")
    lines.append("  " + _hline(82))

    for col, label, desc in available:
        vals = df[col].dropna()
        if len(vals) == 0:
            lines.append(f"  {label:<28} {'—':>10}  {'—':>10}  {'—':>10}  {'—':>10}  {'—':>8}")
            continue
        m = vals.mean()
        med = vals.median()
        q75 = vals.quantile(0.75)
        vmax = vals.max()
        pos_share = (vals > 0.001).mean()
        lines.append(f"  {label:<28} {m:>10.4f}  {med:>10.4f}  {q75:>10.4f}  {vmax:>10.4f}  {pos_share:>7.1%}")

    # Additional: means by employment status for econ_penalty and jobloss_econ_gap_bonus
    if "status" in df.columns:
        statuses = ["stay", "commute", "unemployed"]
        sl = {"stay": "Stay    ", "commute": "Commute ", "unemployed": "Unemp   "}
        key_vars = [c for c, _, _ in available if c in ("econ_penalty", "jobloss_econ_gap_bonus")]
        if key_vars:
            lines.append(f"\n  ── By employment status ──")
            lines.append(f"  {'Status':<10} " + " ".join(f"{col:>12}" for col in key_vars))
            lines.append("  " + _hline(14 + len(key_vars) * 13))
            for st in statuses:
                sub = df[df["status"] == st]
                if sub.empty:
                    continue
                row = f"  {sl.get(st, st):<10}"
                for col in key_vars:
                    row += f" {sub[col].mean():>11.4f}"
                lines.append(row)


def _hline(width=78, char="─") -> str:
    return char * width


def _section(title: str, width=78) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions v6: regions by tick, top-10 moves/commutes
# ══════════════════════════════════════════════════════════════════════════════


def _build_region_time_series(tick_stats: list, G) -> str:
    """
    Builds table of all districts (okresy) by ticks: population, Δ.

    district_counts from tick_stats for each graph node.
    """
    if not tick_stats or G is None:
        return ""

    n_ticks = len(tick_stats)
    tick_nums = list(range(1, n_ticks + 1))
    all_districts = sorted(G.nodes)

    # Collect population by district for each tick
    district_series: dict[str, list[int]] = {}
    for d in all_districts:
        district_series[d] = []
    for s in tick_stats:
        dc = s.get("district_counts", {})
        for d in all_districts:
            district_series[d].append(dc.get(d, 0))

    lines = []
    lines.append(_section(f"ALL DISTRICTS BY TICK: POPULATION ({len(all_districts)} districts)"))
    lines.append(f"  Ticks: {n_ticks}")
    lines.append("")

    # Header
    header = f"  {'District':<30}"
    for tn in tick_nums:
        header += f" {'T' + str(tn):>6}"
    header += f" {'Δ':>8}"
    lines.append(header)
    delta_w = 8
    line_w = 32 + n_ticks * 7 + delta_w
    lines.append("  " + _hline(line_w, "─"))

    for d in all_districts:
        name = d.replace("District of ", "")[:28]
        vals = district_series[d]
        row = f"  {name:<30}"
        for v in vals:
            row += f" {v:>6,}"
        delta = vals[-1] - vals[0] if len(vals) >= 2 else 0
        sign = "+" if delta >= 0 else ""
        row += f"  {sign}{delta:>6,}"
        lines.append(row)

    # TOTAL
    lines.append("  " + _hline(line_w, "─"))
    total_row = f"  {'TOTAL':<30}"
    for i in range(n_ticks):
        t = sum(district_series[d][i] for d in all_districts)
        total_row += f" {t:>6,}"
    lines.append(total_row)

    return "\n".join(lines)


def _top_move_routes(all_action_log: list, top_n: int = 10) -> str:
    """
    Top-N physical move directions (move / satellite_move):
    origin (departure) and destination (arrival).
    """
    if not all_action_log:
        return ""

    origins = Counter()      # where they left from
    destinations = Counter()  # where they arrived

    for entry in all_action_log:
        decision = entry.get("decision", "")
        if decision in ("move", "satellite_move"):
            prev = str(entry.get("prev_residence", "")).replace("District of ", "")
            new_r = str(entry.get("new_residence", "")).replace("District of ", "")
            if prev:
                origins[prev] += 1
            if new_r:
                destinations[new_r] += 1

    if not origins and not destinations:
        return ""

    lines = []
    lines.append(_section(f"TOP-{top_n} RELOCATION DIRECTIONS (MOVE / SATELLITE_MOVE)"))
    lines.append(f"  {'Origin (departed)':<28} {'Count':>8}    "
                 f"{'Destination (arrived)':<28} {'Count':>8}")
    lines.append("  " + _hline(76, "─"))

    top_origins = origins.most_common(top_n)
    top_dests = destinations.most_common(top_n)
    max_len = max(len(top_origins), len(top_dests))

    for i in range(max_len):
        left = f"  {top_origins[i][0]:<28} {top_origins[i][1]:>8,}" if i < len(top_origins) else f"  {'':<28} {'':>8}"
        right = f"  {top_dests[i][0]:<28} {top_dests[i][1]:>8,}" if i < len(top_dests) else ""
        lines.append(f"{left}    {right}")

    return "\n".join(lines)


def _top_commute_routes(all_action_log: list, top_n: int = 10) -> str:
    """
    Top-N new commute directions (where they live → where they work).
    Counts pairs (prev_residence, new_workplace) for commute decisions.
    """
    if not all_action_log:
        return ""

    routes = Counter()  # (residence, workplace)

    for entry in all_action_log:
        if entry.get("decision") == "commute":
            prev_res = str(entry.get("prev_residence", "")).replace("District of ", "")
            new_wp = str(entry.get("new_workplace", "")).replace("District of ", "")
            if prev_res and new_wp:
                routes[(prev_res, new_wp)] += 1

    if not routes:
        return ""

    lines = []
    lines.append(_section(f"TOP-{top_n} NEW COMMUTE DIRECTIONS"))
    lines.append(f"  {'Lives in district':<26} →  {'Works in district':<26} | {'Agents':>7}")
    lines.append("  " + _hline(68, "─"))

    for (res, wp), count in routes.most_common(top_n):
        lines.append(f"  {res:<26} →  {wp:<26} | {count:>7,}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# v7: Regional heatmap (matplotlib + networkx)
# ══════════════════════════════════════════════════════════════════════════════


def _district_heatmap(
    tick_stats: list,
    G,
    snapshots: Optional[dict] = None,
    output_path: str = "heatmap.png",
) -> str:
    """
    Draws district heatmap graph 79 districts by real coordinates:
      - Node color: green = population growth, red = decline
      - Node size: district population
      - Edge thickness: commuting flow intensity
      - Positions: from valid_districts_coords.csv (lon/lat), center — Bratislava I (0,0)

    Saves PNG, returns Markdown string for embedding in report.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return "\n  [Heatmap unavailable — matplotlib not installed]"

    if not tick_stats or G is None:
        return ""

    # 1. Collect population at first and last tick
    first_dc = tick_stats[0].get("district_counts", {})
    last_dc  = tick_stats[-1].get("district_counts", {})

    # If snapshots[0] exists, use it for tick 0
    if snapshots and 0 in snapshots:
        t0_counts = snapshots[0].groupby("district")["id"].count().to_dict()
        for d, c in t0_counts.items():
            first_dc[d] = c

    districts = sorted(G.nodes)
    deltas = []
    pops = []
    labels_list = []
    for d in districts:
        p0 = first_dc.get(d, 0)
        p1 = last_dc.get(d, 0)
        deltas.append(p1 - p0)
        pops.append(p1)
        labels_list.append(d.replace("District of ", "")[:14])

    delta_range = max(abs(d) for d in deltas) if deltas else 0
    if delta_range == 0:
        return "\n  [No population change — heatmap not generated]"

    # 2. Node positions from valid_districts_coords.csv
    #    lon → x, lat → y, center on Bratislava I
    import os
    _coords_path = os.path.join(os.path.dirname(__file__), "valid_districts_coords.csv")
    pos = {}
    if os.path.exists(_coords_path):
        df_coords = pd.read_csv(_coords_path)
        coord_map = {}
        for _, row in df_coords.iterrows():
            key = row["district"].replace(" - ", "-")  # "Košice - okolie" → "Košice-okolie"
            coord_map[key] = (float(row["lon"]), float(row["lat"]))
        for d in districts:
            if d in coord_map:
                pos[d] = coord_map[d]
        # Center on Bratislava I: subtract its coordinates
        center_node = "District of Bratislava I"
        if center_node in pos:
            cx, cy = pos[center_node]
            for node in pos:
                pos[node] = (pos[node][0] - cx, pos[node][1] - cy)
    if not pos:
        # fallback: if no CSV, simple circular layout
        pos = nx.circular_layout(G)

    # ── Slovakia aspect ratio correction ──────────────────────────────────────
    # Slovakia is elongated W→E: span lon ~5.3°, lat ~1.6°.
    # At latitude ~48.15° a degree of longitude is shorter than a degree of latitude by cos(48.15°) ≈ 0.667 times.
    # Compress x (lon) by cos(center_lat), to get approximately equal
    # physical distances along axes.
    center_lat_deg = 48.15  # mean latitude of Slovakia
    lat_correction = math.cos(math.radians(center_lat_deg))
    for node in pos:
        x, y = pos[node]
        pos[node] = (x * lat_correction, y)

    # Compute proportions for figsize
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_range = max(xs) - min(xs) if xs else 1
    y_range = max(ys) - min(ys) if ys else 1
    # Add small margin at edges (15%)
    pad = 0.15
    x_range *= (1 + pad)
    y_range *= (1 + pad)

    # 3. Colors: red(-) → white(0) → green(+)
    vmin_d = min(deltas)
    vmax_d = max(deltas)
    cmap = plt.cm.RdYlGn

    # 4. Draw — figure size by Slovakia proportions
    base_width = 20
    aspect = x_range / max(y_range, 0.01)
    fig_h = base_width / aspect
    fig, ax = plt.subplots(1, 1, figsize=(base_width, fig_h))
    ax.set_aspect("equal")

    # Edges with transparency by flow
    edge_weights = []
    for u, v, data in G.edges(data=True):
        edge_weights.append(data.get("flow_work", 1))
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[max(w / max_w * 4, 0.1) for w in edge_weights],
        alpha=0.12, edge_color="gray",
    )

    # Nodes: size by population, color by delta
    node_sizes = [max(p / max(pops) * 1200, 30) for p in pops] if max(pops) > 0 else 30
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=deltas,
        cmap=cmap,
        vmin=vmin_d,
        vmax=vmax_d,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )

    # Labels (short names)
    labels_dict = dict(zip(districts, labels_list))
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=5, ax=ax)

    n_ticks = len(tick_stats)
    ax.set_title(
        f"Δ population by districts of Slovakia (ticks 1–{n_ticks})\n"
        f"Green = growth, Red = decline | Node size = population",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")

    # Colorbar
    norm = mcolors.Normalize(vmin=vmin_d, vmax=vmax_d)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(deltas)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Δ population (people.)", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return (
        f"\n  🗺 Heatmap: `{output_path}`\n"
        f"  ![District Heatmap]({output_path})\n"
    )


# ══════════════════════════════════════════════════════════════════════════════

def industry_jobs_snapshot(G, df=None, top_n: int = 12) -> str:
    """
    v3: Industry jobs snapshot — occupied and vacant positions by industry.

    Shows for top-N districts (by total capacity) breakdown:
      occupied — filled positions (agents with workplace=district)
      vacant   — open vacancies
      pressure — occupied / (occupied + vacant)

    If df is passed, occupied is counted from actual agents.
    """
    lines = []
    lines.append(_section("INDUSTRY JOBS: OCCUPIED AND VACANT POSITIONS BY INDUSTRY (v3)"))

    if G is None:
        lines.append("  [Graph not provided]")
        return "\n".join(lines)

    # Collect statistics by district
    district_stats = []
    for district in G.nodes:
        ind_jobs = G.nodes[district].get("industry_jobs", {})
        if not ind_jobs:
            continue
        total_occ = sum(v["occupied"] for v in ind_jobs.values())
        total_vac = sum(v["vacant"] for v in ind_jobs.values())
        total_cap = total_occ + total_vac
        if total_cap == 0:
            continue
        # Actual number of employed agents (if df passed)
        actual_occ = total_occ
        if df is not None:
            actual_occ = int(
                (df["workplace_district"] == district).sum()
                if "workplace_district" in df.columns
                else total_occ
            )
        district_stats.append((
            district, total_occ, total_vac, total_cap,
            actual_occ / max(total_cap, 1)
        ))

    if not district_stats:
        lines.append("  [No data industry_jobs in graph]")
        return "\n".join(lines)

    # Sort by total capacity
    district_stats.sort(key=lambda x: -x[3])
    district_stats = district_stats[:top_n]

    lines.append(f"  {'District':<30} {'Occupied':>10} {'Vacant':>10} "
                 f"{'Total':>10} {'Pressure':>9}")
    lines.append("  " + _hline(75))

    for d, occ, vac, cap, press in district_stats:
        name = d.replace("District of ", "")[:28]
        lines.append(f"  {name:<30} {occ:>10,} {vac:>10,} {cap:>10,} {press:>8.3f}")

    # ── Detailed by industry for top-3 districts ──────────────────────────
    lines.append(f"\n  DETAILED BY INDUSTRY (top-3 districts):")
    for d, _, _, _, _ in district_stats[:3]:
        name = d.replace("District of ", "")
        ind_jobs = G.nodes[d].get("industry_jobs", {})
        if not ind_jobs:
            continue

        # Count actual employed agents by industry
        actual_by_ind = {}
        if df is not None and "workplace_district" in df.columns and "industry" in df.columns:
            sub = df[df["workplace_district"] == d]
            actual_by_ind = sub.groupby("industry")["id"].count().to_dict()

        lines.append(f"\n  ── {name} ──")
        lines.append(f"  {'Industry':<45} {'Occ':>8} {'Vac':>8} "
                     f"{'Total':>8} {'Actual':>8} {'Press':>7}")
        lines.append("  " + _hline(90))

        # Sort industries by total capacity
        sorted_inds = sorted(
            ind_jobs.items(),
            key=lambda x: -(x[1]["occupied"] + x[1]["vacant"])
        )
        for ind, jobs in sorted_inds[:8]:
            occ = jobs["occupied"]
            vac = jobs["vacant"]
            cap = occ + vac
            actual = actual_by_ind.get(ind, occ)
            press = actual / max(cap, 1)
            ind_short = str(ind)[:43]
            lines.append(f"  {ind_short:<45} {occ:>8,} {vac:>8,} "
                         f"{cap:>8,} {actual:>8,} {press:>6.3f}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7b. RANDOM REGION INDUSTRY JOBS — random regions and industries
# ══════════════════════════════════════════════════════════════════════════════

# All possible industries (for random selection)
ALL_INDUSTRIES = [
    "Accommodation and food service activities",
    "Administrative and support service activities",
    "Agriculture, forestry and fishing",
    "Construction",
    "Human health and social work activities",
    "Information and communication",
    "Manufacturing total",
    "Other",
    "Professional, scientific and technical activities",
    "Public administration and defence",
    "Transportation and storage",
    "Water supply; sewerage, waste management and remediation activities",
    "Wholesale and retail trade; repair of motor vehicles and motorcycles",
]


def _random_region_industry_jobs(
    G,
    df: Optional[pd.DataFrame] = None,
    n_regions: int = 3,
    n_industries: int = 3,
    seed: int = 42,
) -> str:
    """
    Selects n_regions random regions and n_industries random industries
    for each, shows occupied and vacant positions.

    Data aggregated by district (okres) → region (kraj) via
    DISTRICT_TO_REGION_CODE.

    If df is passed, occupied is counted from actual agents in region.
    """
    lines = []
    lines.append(_section(
        f"RANDOM REGIONS AND INDUSTRIES: OCCUPIED AND VACANT POSITIONS "
        f"({n_regions} regions × {n_industries} industries)"
    ))

    if G is None:
        lines.append("  [Graph not provided]")
        return "\n".join(lines)

    rng = np.random.default_rng(seed)

    # 1. Aggregate industry_jobs by region
    region_data: dict[str, dict[str, dict[str, int]]] = {}
    # {region: {industry: {"occupied": int, "vacant": int}}}

    for district in G.nodes:
        ind_jobs = G.nodes[district].get("industry_jobs", {})
        if not ind_jobs:
            continue
        region_code = DISTRICT_TO_REGION_CODE.get(district, "XX")
        if region_code not in region_data:
            region_data[region_code] = {}
        for industry, jobs in ind_jobs.items():
            if industry not in region_data[region_code]:
                region_data[region_code][industry] = {"occupied": 0, "vacant": 0}
            region_data[region_code][industry]["occupied"] += jobs.get("occupied", 0)
            region_data[region_code][industry]["vacant"]   += jobs.get("vacant", 0)

    if not region_data:
        lines.append("  [No data industry_jobs in graph]")
        return "\n".join(lines)

    # 2. Actual employed by region from df (if passed)
    actual_by_region_industry: dict[str, dict[str, int]] = {}
    if df is not None and "workplace_district" in df.columns and "industry" in df.columns:
        for _, agent in df.iterrows():
            wp = agent.get("workplace_district", "")
            ind = str(agent.get("industry", ""))
            reg = DISTRICT_TO_REGION_CODE.get(wp, "XX")
            if reg not in actual_by_region_industry:
                actual_by_region_industry[reg] = {}
            if ind not in actual_by_region_industry[reg]:
                actual_by_region_industry[reg][ind] = 0
            actual_by_region_industry[reg][ind] += 1

    # 3. Select random regions
    all_region_codes = sorted(region_data.keys())
    if len(all_region_codes) < n_regions:
        selected_regions = all_region_codes
    else:
        selected_regions = list(rng.choice(all_region_codes, n_regions, replace=False))

    # 4. For each region select random industries and display
    for region_code in selected_regions:
        region_name = REGION_NAMES.get(region_code, region_code)
        industries = region_data[region_code]

        # All industries in this region
        all_inds = sorted(industries.keys())
        n_sel = min(n_industries, len(all_inds))

        if n_sel < n_industries:
            selected_inds = all_inds  # fewer industries than requested
        else:
            selected_inds = list(rng.choice(all_inds, n_sel, replace=False))

        # Regional summary data
        total_occ = sum(v["occupied"] for v in industries.values())
        total_vac = sum(v["vacant"] for v in industries.values())
        total_cap = total_occ + total_vac

        lines.append(f"\n  ── {region_name} (code: {region_code}) ──")
        lines.append(f"  Regional total: {total_occ:>8,} occupied, "
                     f"{total_vac:>8,} vacant, {total_cap:>8,} total")
        lines.append("")
        lines.append(f"  {'Industry':<55} {'Occupied':>10} {'Vacant':>10} "
                     f"{'Total':>10} {'Actual':>10} {'Free%':>10}")
        lines.append("  " + _hline(110))

        for ind in selected_inds:
            jobs = industries[ind]
            occ = jobs["occupied"]
            vac = jobs["vacant"]
            cap = occ + vac
            actual = actual_by_region_industry.get(region_code, {}).get(ind, occ)
            free_pct = vac / max(cap, 1) * 100
            ind_short = str(ind)[:53]
            lines.append(f"  {ind_short:<55} {occ:>10,} {vac:>10,} "
                         f"{cap:>10,} {actual:>10,} {free_pct:>9.1f}%")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEMOGRAPHIC PORTRAIT — main portrait with metric prioritization
# ══════════════════════════════════════════════════════════════════════════════

def demographic_portrait(
    df: pd.DataFrame,
    label: str = "Snapshot",
    tick_num: Optional[int] = None,
    exclude_students: bool = True,
    detail: bool = False,
) -> str:
    """
    Demographic portrait with metric prioritization.

    Critical (always):
      - Number of agents (excluding students), unemployment rate
      - Average wage (mean, median)
      - Regional table: population, average wage, unemployment rate

    Important (always):
      - Satisfaction across 4 domains
      - Gaps by domain broken down by employment status
      - Top-5 industries among employed and unemployed

    Additional (detail=True):
      - Agent type distribution
      - Psychological parameters by type
      - Top-10 commuting routes

    v7:
      - tick_stats, all_action_log, G — no longer used directly
        (top-10 and regional table only in final summary_report)
    """
    lines = []
    total_all = len(df)
    n_students = int((df["status"] == "student").sum()) if "status" in df.columns else 0

    # Student filter
    if exclude_students and "status" in df.columns:
        work_df = df[df["status"] != "student"].copy()
    else:
        work_df = df.copy()
    total = len(work_df)

    # ── Header ────────────────────────────────────────────────────────────────
    header = "=" * 78
    title  = f"  DEMOGRAPHIC PORTRAIT — {label}"
    if tick_num is not None:
        yr = tick_num // 12
        mo = tick_num % 12 or 12
        title += f"  [tick {tick_num} / year {yr} month {mo}]"
    lines += [header, title, header]

    lines.append(f"\n  Agents total: {total_all:,}")
    if exclude_students and n_students > 0:
        lines.append(f"    of which students: {n_students:,} (excluded from analysis)")
    lines.append(f"  Analyzed population (excluding students): {total:,}\n")

    # ═══ CRITICAL METRICS ════════════════════════════════════════════════
    lines.append(_section("CRITICAL METRICS"))

    # Unemployment
    if "status" in work_df.columns:
        n_unemp = int((work_df["status"] == "unemployed").sum())
        unemp_rate = n_unemp / total if total > 0 else 0
        n_commuters = int((work_df["status"] == "commute").sum())
        n_stay = int((work_df["status"] == "stay").sum())
        lines.append(f"  Unemployed:        {n_unemp:>8,}  ({unemp_rate:.1%})")
        lines.append(f"  Employed locally:   {n_stay:>8,}  ({n_stay/total*100 if total else 0:.1f}%)")
        lines.append(f"  Commuters:          {n_commuters:>8,}  ({n_commuters/total*100 if total else 0:.1f}%)")
        bar_max = max(n_stay, n_commuters, n_unemp, 1)
        lines.append(f"  {'Statuses:':<20}  {_bar(n_stay, bar_max)} stay")
        lines.append(f"  {'':20}  {_bar(n_commuters, bar_max)} commute")
        lines.append(f"  {'':20}  {_bar(n_unemp, bar_max)} unemployed")

    # Wage
    if "wage" in work_df.columns:
        employed_w = work_df[work_df["wage"] > 0]
        if not employed_w.empty:
            avg_w = employed_w["wage"].mean()
            med_w = employed_w["wage"].median()
            q25 = employed_w["wage"].quantile(0.25)
            q75 = employed_w["wage"].quantile(0.75)
            lines.append(f"\n  Average wage (employed):  {avg_w:>10,.0f} €")
            lines.append(f"  Median wage:         {med_w:>10,.0f} €")
            lines.append(f"  Q25–Q75:                    {q25:>10,.0f} – {q75:,.0f} €")
        else:
            lines.append("\n  [No wage data for employed]")

    # ═══ REGIONAL TABLE ═══════════════════════════════════════════════
    lines.append(_section("REGIONS: POPULATION, WAGE, UNEMPLOYMENT"))

    if "region" in work_df.columns:
        lines.append(f"  {'Region':<22} {'Population':>8}  {'Share':>6}  "
                     f"{'Avg.wage':>11}  {'Unemp.':>8}  {'Count':>6}")
        lines.append("  " + _hline(72))

        region_stats = []
        for code, name in REGION_NAMES.items():
            reg = work_df[work_df["region"] == code]
            pop = len(reg)
            if pop == 0:
                continue
            share = pop / total if total else 0
            avg_w = reg[reg["wage"] > 0]["wage"].mean() if "wage" in reg.columns else 0
            n_unemp_r = int((reg["status"] == "unemployed").sum()) if "status" in reg.columns else 0
            unemp_r = n_unemp_r / pop if pop else 0
            region_stats.append((name, pop, share, avg_w, unemp_r, n_unemp_r))

        # Sort by population
        region_stats.sort(key=lambda x: x[1], reverse=True)

        max_pop = max(r[1] for r in region_stats) if region_stats else 1
        for name, pop, share, avg_w, unemp_r, n_unemp_r in region_stats:
            lines.append(f"  {name:<22} {pop:>8,}  {share:>5.1%}  "
                         f"{avg_w:>9,.0f} €  {unemp_r:>7.1%}  {n_unemp_r:>6,}")
    else:
        lines.append("  [Region column missing]")

    # ═══ IMPORTANT METRICS ══════════════════════════════════════════════════════
    lines.append(_section("IMPORTANT METRICS"))

    # Top-5 industries
    employed = work_df[work_df["is_employed"] == True] if "is_employed" in work_df.columns else work_df[work_df["status"].isin(["stay", "commute"])]
    if not employed.empty and "industry" in employed.columns:
        lines.append(_section("TOP-5 INDUSTRIES AMONG EMPLOYED"))
        lines.append(f"  {'Industry':<30} {'Agents':>8}  {'Avg.wage':>12}")
        lines.append("  " + _hline(54))
        top_ind = employed["industry"].value_counts().head(5)
        for ind, count in top_ind.items():
            avg_w = employed[employed["industry"] == ind]["wage"].mean()
            lines.append(f"  {str(ind)[:30]:<30} {count:>8,}  {avg_w:>10,.0f} €")

    unemployed = work_df[work_df["status"] == "unemployed"] if "status" in work_df.columns else pd.DataFrame()
    if not unemployed.empty and "industry" in unemployed.columns:
        lines.append(_section("TOP-5 INDUSTRIES AMONG UNEMPLOYED"))
        lines.append(f"  {'Industry':<30} {'Agents':>8}")
        lines.append("  " + _hline(42))
        top_unemp = unemployed["industry"].value_counts().head(5)
        for ind, count in top_unemp.items():
            lines.append(f"  {str(ind)[:30]:<30} {count:>8,}")

    # ═══ v2: DYNAMIC VARIABLES OF THE SIGNAL SYSTEM ═══════════════════
    _append_dynamic_vars_section(lines, work_df)

    # ═══ ADDITIONAL METRICS (detail=True) ═══════════════════════════════
    if detail:
        lines.append(_section("ADDITIONAL METRICS (detail=True)"))

        # Psychological parameters
        psych_params = [
            ("inertia",                 "Inertia"),
            ("perceived_control",       "Perceived Control"),
            ("econ_perceived_control",  "Econ PC"),
            ("job_flexibility",         "Job Flexibility"),
            ("shock_sensitivity",       "Shock Sensitivity"),
            ("info_quality",            "Info Quality"),
            ("commuter_threshold",      "Commuter Thr."),
            ("internal_mig_thr",        "Internal Mig Thr."),
        ]
        available_psych = [(c, l) for c, l in psych_params if c in work_df.columns]
        if available_psych:
            lines.append(_section("PSYCHOLOGICAL PARAMETERS (GENERAL)"))
            for col, label_p in available_psych:
                m = work_df[col].mean()
                lines.append(f"  {label_p:<24} {m:.4f}")

        # Top-10 commuting routes
        if "residence_district" in work_df.columns and "workplace_district" in work_df.columns:
            lines.append(_section("TOP-10 COMMUTING DIRECTIONS"))
            lines.append(f"  {'Live in district':<22} →  {'Work in district':<22} | {'Agents':>6}")
            lines.append("  " + _hline(60))
            commuters_df = work_df[work_df["residence_district"] != work_df["workplace_district"]]
            if not commuters_df.empty:
                top_commutes = (commuters_df.groupby(["residence_district", "workplace_district"])
                                .size().sort_values(ascending=False).head(10))
                for (res, work), count in top_commutes.items():
                    r_name = str(res).replace("District of ", "")[:20]
                    w_name = str(work).replace("District of ", "")[:20]
                    lines.append(f"  {r_name:<22} →  {w_name:<22} | {count:>6,}")
            else:
                lines.append("  [No commuter connections between districts found]")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 2. AGENT BEHAVIOR AUDIT — deep diagnostics (detail=True only)
# ══════════════════════════════════════════════════════════════════════════════

def agent_behavior_audit(action_log: Optional[List[dict]], sample_size: int = 30) -> str:
    """
    Behavioral audit based on action_log from FFT-pipeline.

    Shown only in full report (detail=True / mode='full').
    """
    lines = [
        "\n" + "═" * 100,
        f"  BEHAVIORAL AUDIT OF AGENTS (Slice up to {sample_size} random decisions from log)",
        "═" * 100,
    ]

    if not action_log:
        lines.append("  [No behavioral audit data — action_log is empty]")
        lines.append("═" * 100)
        return "\n".join(lines)

    rng = np.random.default_rng(42)
    n_sample = min(sample_size, len(action_log))
    indices = rng.choice(len(action_log), n_sample, replace=False)
    sampled = [action_log[i] for i in indices]

    lines.append(
        f"  {'ID':<6} | {'Domain':<10} | {'Decision':<9} | "
        f"{'Housing before→after':<28} | {'Job before→after':<28} | "
        f"{'Wage':>8} | {'Industry':<18} | {'Raise':>8}"
    )
    lines.append("  " + "─" * 97)

    for ag in sampled:
        ag_id       = str(ag.get("id", "?"))[:6]
        act_domain  = str(ag.get("activation_domain", "?"))[:10]
        decision    = str(ag.get("decision", "?"))[:9]

        prev_res    = str(ag.get("prev_residence", "?")).replace("District of ", "")[:12]
        new_res     = str(ag.get("new_residence", "?")).replace("District of ", "")[:12]
        prev_wp     = str(ag.get("prev_workplace", "?")).replace("District of ", "")[:12]
        new_wp      = str(ag.get("new_workplace", "?")).replace("District of ", "")[:12]

        res_flow    = f"{prev_res}→{new_res}"
        work_flow   = f"{prev_wp}→{new_wp}"

        wage_val    = ag.get("wage", 0)
        wage_str    = f"{wage_val:,.0f}€" if wage_val else "—"
        industry    = str(ag.get("industry", "—"))[:18]
        desired     = ag.get("desired_raise", 0)
        desired_str = f"{desired:.1%}" if desired else "—"

        lines.append(
            f"  {ag_id:<6} | {act_domain:<10} | {decision:<9} | "
            f"{res_flow:<28} | {work_flow:<28} | "
            f"{wage_str:>8} | {industry:<18} | {desired_str:>8}"
        )

    lines.append("  " + "─" * 97)

    # Sample summary
    domains = [str(a.get("activation_domain", "?")) for a in sampled]
    dom_counts = pd.Series(domains).value_counts()
    dom_str = ", ".join([f"{k}: {v}" for k, v in dom_counts.items()])
    lines.append(f"  Activations by domain: {dom_str}")

    decisions = [str(a.get("decision", "?")) for a in sampled]
    dec_counts = pd.Series(decisions).value_counts()
    dec_str = ", ".join([f"{k}: {v}" for k, v in dec_counts.items()])
    lines.append(f"  Decisions: {dec_str}")

    raises = [a.get("desired_raise", 0) for a in sampled
              if a.get("desired_raise", 0) > 0]
    if raises:
        avg_raise = sum(raises) / len(raises)
        lines.append(f"  Average desired raise (among job seekers): {avg_raise:.1%}")
    else:
        lines.append("  Average desired raise: — (no data)")

    lines.append("═" * 100)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MIGRATION SUMMARY — dynamics summary with economic trends
# ══════════════════════════════════════════════════════════════════════════════

def migration_summary(tick_stats: list) -> str:
    """
    Dynamics summary: moves, activations + economic trends by year.

    Yearly bars for:
      - avg_wage (average wage)
      - avg_dissat (average dissatisfaction)
      - n_unemployed (unemployed count)
      - jobs_pressure_max (max labor market pressure)
      - moves (relocations)
    """
    if not tick_stats:
        return "No data."

    total_moves       = sum(s.get("moves", 0) for s in tick_stats)
    total_commutes    = sum(s.get("commutes", 0) for s in tick_stats)
    total_adapts      = sum(s.get("adapts", 0) for s in tick_stats)
    total_satellite   = sum(s.get("satellite_moves", 0) for s in tick_stats)
    total_econ_moves  = sum(s.get("econ_driven_moves", 0) for s in tick_stats)
    total_place_moves = sum(s.get("place_driven_moves", 0) for s in tick_stats)
    total_econ_act    = sum(s.get("econ_activated", 0) for s in tick_stats)
    total_place_act   = sum(s.get("place_activated", 0) for s in tick_stats)
    avg_moves = total_moves / len(tick_stats) if tick_stats else 0

    lines = [
        "\n" + "=" * 78,
        "  DYNAMICS SUMMARY AND ECONOMIC TRENDS",
        "=" * 78,
        f"  Activations — economic:     {total_econ_act:>8,}",
        f"  Activations — place:         {total_place_act:>8,}",
        f"  Physical moves (Move):     {total_moves:>8,}",
        f"    due to economy:               {total_econ_moves:>8,}",
        f"    due to place:                   {total_place_moves:>8,}",
        f"    to satellites:                    {total_satellite:>8,}",
        f"  Commute decisions (Commute):   {total_commutes:>8,}",
        f"  Forced adaptations (Adapt):   {total_adapts:>8,}",
        f"  Average migration intensity: {avg_moves:.1f} moves/tick",
    ]

    # ── Yearly trends: aggregation ────────────────────────────────────────────
    yearly_total    = {}
    yearly_econ     = {}
    yearly_place    = {}
    yearly_unemp    = {}

    for s in tick_stats:
        year = (s["tick"] - 1) // 12 + 1
        yearly_total.setdefault(year, []).append(s.get("moves", 0))
        yearly_econ.setdefault(year, []).append(s.get("econ_driven_moves", 0))
        yearly_place.setdefault(year, []).append(s.get("place_driven_moves", 0))
        yearly_unemp.setdefault(year, []).append(s.get("n_unemployed", 0))

    years = sorted(yearly_total.keys())

    # ── Move trend (with breakdown econ/place) ────────────────────────────
    lines.append(_section("MOVE TREND BY YEAR"))
    has_econ  = any(sum(v) > 0 for v in yearly_econ.values())
    has_place = any(sum(v) > 0 for v in yearly_place.values())
    max_moves = max(sum(v) for v in yearly_total.values()) if yearly_total else 1

    if has_econ or has_place:
        lines.append(f"  {'Year':<6} {'Total':>8}  {'Econ':>8}  {'Place':>8}  {'':22}")
        lines.append("  " + _hline(58))
        for year in years:
            t = sum(yearly_total[year])
            e = sum(yearly_econ[year])
            p = sum(yearly_place[year])
            bar = _bar(t, max_moves, width=22)
            lines.append(f"  {year:<6} {t:>8,}  {e:>8,}  {p:>8,}  {bar}")
    else:
        for year in years:
            t = sum(yearly_total[year])
            bar = _bar(t, max_moves, width=22)
            lines.append(f"  Year {year:2d}  {t:>6,} move acts  {bar}")

    # ── Unemployment trend ───────────────────────────────────────────────────
    lines.append(_section("UNEMPLOYMENT TREND BY YEAR"))
    unemp_vals = [sum(yearly_unemp[y]) / len(yearly_unemp[y]) if yearly_unemp[y] else 0 for y in years]
    unemp_max = max(unemp_vals) if unemp_vals else 1
    lines.append(f"  {'Year':<6} {'Unemployed':>12}  {'':30}")
    lines.append("  " + _hline(52))
    for year, u_avg in zip(years, unemp_vals):
        bar = _bar(u_avg, unemp_max, width=30) if unemp_max > 0 else " " * 30
        lines.append(f"  {year:<6} {u_avg:>10,.0f}  {bar}")

    lines.append("=" * 78)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPARE SNAPSHOTS — interregional balance + migration
# ══════════════════════════════════════════════════════════════════════════════

def compare_snapshots(
    snapshots: dict,
    tick_stats: list,
    all_action_log: Optional[List[dict]] = None,
    detail: bool = False,
) -> str:
    """
    Interregional population balance with economic deltas.

    For each snapshot shows:
      - Key figures (population, unemployment, average wage)
      - Regional table with Δ population, unemployment, wage

    Then migration_summary with trends.
    With detail=True — agent_behavior_audit.
    """
    ticks = sorted(snapshots.keys())
    lines = [
        "\n" + "=" * 78,
        "  INTERREGIONAL POPULATION BALANCE (Regions of Slovakia)",
        "=" * 78,
    ]

    # ── Summary table by ticks ────────────────────────────────────────────
    first_tick = ticks[0]
    last_tick  = ticks[-1]

    if "region" in snapshots[first_tick].columns:
        all_regions = sorted(snapshots[first_tick]["region"].unique())
    else:
        all_regions = []

    # Header with three metrics per tick: population, unemployment, wage
    header = f"  {'Region':<22}"
    for t in ticks:
        lbl = "Start" if t == 0 else f"Tick {t}"
        header += f"  {lbl:>9}"
    header += f"  {'Δ pop.':>8}  {'Δ unemp.':>10}  {'Δ wage':>10}"
    lines.append(header)
    lines.append("  " + _hline(22 + len(ticks) * 11 + 34))

    for region in all_regions:
        name = REGION_NAMES.get(region, region)
        row = f"  {name:<22}"
        counts = []
        unemp_rates = []
        avg_wages = []
        for t in ticks:
            snap = snapshots[t]
            reg_df = snap[snap["region"] == region]
            count = len(reg_df)
            counts.append(count)
            row += f"  {count:>9,}"

            # Unemployment
            if "status" in reg_df.columns:
                unemp_r = (reg_df["status"] == "unemployed").mean()
            else:
                unemp_r = 0
            unemp_rates.append(unemp_r)

            # Wage
            if "wage" in reg_df.columns:
                w = reg_df[reg_df["wage"] > 0]["wage"].mean()
            else:
                w = 0
            avg_wages.append(w)

        # Δ population
        delta_pop = counts[-1] - counts[0]
        sign_pop  = "+" if delta_pop >= 0 else ""
        # Δ unemployment (in p.p.)
        delta_unemp = (unemp_rates[-1] - unemp_rates[0]) * 100
        sign_unemp  = "+" if delta_unemp >= 0 else ""
        # Δ wage
        delta_wage = avg_wages[-1] - avg_wages[0]
        sign_wage  = "+" if delta_wage >= 0 else ""

        row += f"  {sign_pop}{delta_pop:>7,}  {sign_unemp}{delta_unemp:>8.1f}pp  {sign_wage}{delta_wage:>8,.0f}€"
        lines.append(row)

    # ── TOTAL row ────────────────────────────────────────────────────────
    first_df = snapshots[first_tick]
    last_df  = snapshots[last_tick]
    total_first = len(first_df)
    total_last  = len(last_df)
    u_first = (first_df["status"] == "unemployed").mean() if "status" in first_df.columns else 0
    u_last  = (last_df["status"] == "unemployed").mean() if "status" in last_df.columns else 0
    w_first = first_df[first_df["wage"] > 0]["wage"].mean() if "wage" in first_df.columns else 0
    w_last  = last_df[last_df["wage"] > 0]["wage"].mean() if "wage" in last_df.columns else 0

    lines.append("  " + _hline(22 + len(ticks) * 11 + 34))
    row_total = f"  {'TOTAL':<22}"
    for t in ticks:
        row_total += f"  {len(snapshots[t]):>9,}"
    d_pop = total_last - total_first
    d_unemp = (u_last - u_first) * 100
    d_wage = w_last - w_first
    sp = "+" if d_pop >= 0 else ""
    su = "+" if d_unemp >= 0 else ""
    sw = "+" if d_wage >= 0 else ""
    row_total += f"  {sp}{d_pop:>7,}  {su}{d_unemp:>8.1f}pp  {sw}{d_wage:>8,.0f}€"
    lines.append(row_total)

    lines.append("=" * 78)

    # ── Migration summary with trends ──────────────────────────────────────
    lines.append(migration_summary(tick_stats))

    # ── Behavioral audit (detail only) ─────────────────────────────────
    if detail:
        lines.append(agent_behavior_audit(all_action_log, sample_size=30))

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 5. AGENT PARAMETERS TABLE — agent parameters matrix with dynamics
# ══════════════════════════════════════════════════════════════════════════════

# Constants (synchronized with engine.py)
_NATIONAL_AVG_WAGE = 1614.0
_EDU_MAP = {"low": 0.25, "medium": 0.55, "high": 0.85}
_HOUSING_BUDGET = 0.35


def _compute_capabilities(agent_row) -> float:
    """capabilities = (income_index + education_index + weak_ties) / 3."""
    wage = float(agent_row.get("wage", 0))
    income_index = min(wage / (1.5 * _NATIONAL_AVG_WAGE), 1.0)
    edu = str(agent_row.get("education", "medium"))
    education_index = _EDU_MAP.get(edu, 0.55)
    weak_ties = float(agent_row.get("weak_ties_utility", 0.0))
    return round((income_index + education_index + weak_ties) / 3.0, 4)


def _compute_dynamic_inertia(agent_row) -> float:
    """v3: Dynamic inertia of barrier 2 = inertia × max(0.15, 1 − social_boost)."""
    inertia = float(agent_row.get("inertia", 0.5))
    social_boost = float(agent_row.get("social_boost", 0.0))
    return round(inertia * max(0.15, 1.0 - social_boost), 4)


def _compute_dynamic_threshold_stage1(agent_row) -> float:
    """v3: Dynamic threshold of barrier 1 = (internal_mig_thr + inertia_mob_penalty) × max(0.15, 1 − signal_reduction)."""
    internal_thr = float(agent_row.get("internal_mig_thr", 0.5))
    inertia_mob_pen = float(agent_row.get("inertia_mobility_penalty", 0.0))
    signal_red = float(agent_row.get("signal_reduction", 0.0))
    return round((internal_thr + inertia_mob_pen) * max(0.15, 1.0 - signal_red), 4)


def _industry_wage_in_district_report(G, district: str, industry: str) -> float:
    """Industry wage in graph node; fallback → avg_wage → NATIONAL_AVG_WAGE."""
    if G is None:
        return _NATIONAL_AVG_WAGE
    attr = G.nodes.get(district, {})
    sal = attr.get("salary_by_industry", {})
    if sal:
        return float(sal.get(industry, attr.get("avg_wage", _NATIONAL_AVG_WAGE)))
    return float(attr.get("avg_wage", _NATIONAL_AVG_WAGE))


def _compute_d_components(agent_row, G=None) -> dict:
    """
    Computes D_instant v2 components (mirrors engine._compute_d_instant).

    Returns dict:
      D_econ, wage_pressure, D_place, place_reality, affordability, D_instant
    """
    wage = float(agent_row.get("wage", 0))
    industry = str(agent_row.get("industry", "Other"))
    workplace = str(agent_row.get("workplace_district", ""))
    residence = str(agent_row.get("district", ""))
    econ_gap = float(agent_row.get("econ_gap", 0.5))
    job_flex = float(agent_row.get("job_flexibility", 0.5))
    housing = float(agent_row.get("housing_price_m2", 1800.0))
    domain_future_place = float(agent_row.get("domain_future_place", 0.3))
    w_econ = float(agent_row.get("w_economic", 0.3))
    w_future = float(agent_row.get("w_future", 0.3))
    econ_penalty = float(agent_row.get("econ_penalty", 0.0))
    infra_bonus = float(agent_row.get("infra_bonus", 0.0))

    # wage_pressure: how much the agent's wage lags behind the industry average in the workplace district
    industry_avg_wp = _industry_wage_in_district_report(G, workplace, industry)
    if wage > 0 and industry_avg_wp > 0:
        wage_pressure = industry_avg_wp / wage
    else:
        wage_pressure = 1.0  # unemployed → maximum pressure

    # v3: econ_penalty — direct addition to D_econ (not smoothed by formula)
    D_econ = w_econ * wage_pressure * (econ_gap / max(job_flex, 0.01)) + econ_penalty

    # place_reality: housing and infrastructure quality (0–1) — v2
    monthly_cost = housing * 50 * 0.004
    burden = monthly_cost / max(wage, 1.0)
    affordability = max(0.0, 1.0 - burden / _HOUSING_BUDGET)

    # infrastructure_score — from graph if available, else 0.5
    if G is not None:
        infra = float(G.nodes.get(residence, {}).get("infrastructure_score", 0.5))
    else:
        infra = 0.5

    # v2: infra_component = 0.3 * (1 - infra + infra_bonus)
    infra_component = 0.3 * (1.0 - infra + infra_bonus)
    place_reality = 0.7 * affordability + infra_component

    gap = max(0.0, domain_future_place - place_reality)
    place_ratio = domain_future_place / max(place_reality, 0.001)
    amplifier = max(1.0, place_ratio)
    place_penalty = float(agent_row.get("place_deficit_penalty", 0.0))
    D_place = w_future * gap * amplifier * (1.0 + place_penalty)
    D_instant = float(np.clip(D_econ + D_place, 0.0, 1.0))

    return {
        "D_econ":           round(D_econ, 4),
        "wage_pressure":    round(wage_pressure, 4),
        "D_place":          round(D_place, 4),
        "place_reality":    round(place_reality, 4),
        "affordability":    round(affordability, 4),
        "D_instant":        round(D_instant, 4),
    }


def _fmt_arrow(v0, v6, fmt_spec=".3f", width=10) -> str:
    """Formats tick0→tick6 value with alignment."""
    s0 = f"{v0:{fmt_spec}}"
    s6 = f"{v6:{fmt_spec}}"
    arrow = f"{s0}→{s6}"
    return f"{arrow:>{width}}"


def _fmt_val(v, fmt_spec=".3f", width=10) -> str:
    """Formats single value with alignment."""
    return f"{v:{fmt_spec}}".rjust(width)


def _fmt_bool_arrow(v0, v6, width=8) -> str:
    """Formats boolean value ✓/✗."""
    s0 = "✓" if v0 else "✗"
    s6 = "✓" if v6 else "✗"
    arrow = f"{s0}→{s6}"
    return f"{arrow:>{width}}"


def agent_parameters_table(
    snapshots: dict,
    G=None,
    n_show: int = 20,
    tick_a: int = 0,
    tick_b: int = 6,
    seed: int = 42,
) -> str:
    """
    Agent parameters matrix: aspirations, D_econ/D_place and their components,
    capabilities, inertia, dynamic_inertia, TPB, threshold, signal_reduction.

    Compares tick_a (start) and tick_b (after N ticks) for the same agents.
    Shows change dynamics in «value_0 → value_N» format.

    Parameters:
      snapshots — dictionary {tick: DataFrame}
      G         — graph (networkx.DiGraph) for computing industry wages and infrastructure
      n_show    — how many agents to show (default 20)
      tick_a    — start tick (default 0)
      tick_b    — end tick (default 6)
      seed      — for reproducible sampling
    """
    lines = []

    # Check snapshot availability
    if tick_a not in snapshots:
        return f"\n  [agent_parameters_table] Snapshot for tick {tick_a} missing."
    if tick_b not in snapshots:
        return f"\n  [agent_parameters_table] Snapshot for tick {tick_b} missing."

    df_a = snapshots[tick_a]
    df_b = snapshots[tick_b]

    # Find common agents by ID
    ids_a = set(df_a["id"].values)
    ids_b = set(df_b["id"].values)
    common_ids = sorted(ids_a & ids_b)

    if len(common_ids) == 0:
        return "\n  [agent_parameters_table] No common agents between snapshots."

    # Agent sampling
    rng = np.random.default_rng(seed)
    n_sample = min(n_show, len(common_ids))
    sampled_ids = sorted(rng.choice(list(common_ids), n_sample, replace=False))

    # Prepare sorted samples
    sampled_df_a = df_a[df_a["id"].isin(sampled_ids)].sort_values("id").reset_index(drop=True)
    sampled_df_b = df_b[df_b["id"].isin(sampled_ids)].sort_values("id").reset_index(drop=True)

    # ── Header ────────────────────────────────────────────────────────
    header = "═" * 160
    title = f"  AGENT PARAMETERS MATRIX: Dynamics Tick {tick_a} → Tick {tick_b} (n={n_sample})"
    lines += [header, title, header]

    # ── Legend ──────────────────────────────────────────────────────────
    lines.append("  ═══ BARRIER 1 — Migration Potential (Aspirations × Capabilities vs Dynamic Inertia) ═══")
    lines.append("  Aspirations — EWMA accumulation of D_instant. Start=0 (cold), at tick 1 = D_instant.")
    lines.append("  D_econ      — economic dissatisfaction: w_econ × wage_pressure × econ_gap × (1−job_flex)")
    lines.append("  wage_pr     — wage_pressure: wage lag behind industry average in workplace district (0–1)")
    lines.append("  D_place     — housing dissatisfaction: w_future × gap × (dfp/pr) × (1+penalty)")
    lines.append("  place_r     — place_reality: 0.6×affordability + 0.4×infrastructure_score")
    lines.append("  PlacePen    — place_deficit_penalty (accumulated penalty)")
    lines.append("  EPen/IBonus — v2: econ_penalty / infra_bonus (dynamic signal variables)")
    lines.append("  InMobPen    — v2: inertia_mobility_penalty (inertia penalty from neighbors moving)")
    lines.append("  JlBonus     — v2: jobloss_econ_gap_bonus (ramp-bonus from LOST_JOB)")
    lines.append("  Capab.      — capabilities: (income_index + education_index + weak_ties) / 3")
    lines.append("  Inertia     — base agent inertia")
    lines.append("  ═══ BARRIER 1: Potential vs Dynamic Threshold ═══")
    lines.append("  DynThr1     — dynamic threshold: (internal_mig_thr + InMobPen) × max(0.15, 1 − signal_reduction)")
    lines.append("  Thr_mig     — internal_mig_threshold (base barrier 1 threshold)")
    lines.append("  SignRed     — signal_reduction (accumulated signal effect lowering the threshold)")
    lines.append("  ═══ BARRIER 2: D_perceived vs Dynamic Inertia ═══")
    lines.append("  D_perc      — D_perceived = D_instant × Attribution × SocialCalibration")
    lines.append("  Attrib      — Attribution = PC × (1 − helplessness)")
    lines.append("  Help        — helplessness = clip(1 − PC − weak_ties × 0.3, 0, 1)")
    lines.append("  SocCal      — SocialCalibration = 1 + net_signal_susc × soc_calibration_signal")
    lines.append("  DynInert    — dynamic inertia S2 = inertia × max(0.15, 1 − social_boost)")
    lines.append("  MigrPress   — v4: accumulated migration pressure (0–2)")
    lines.append("  TPB         — activity flag / intention delay counter")
    lines.append("  IntState    — intention_state (none | seeking_work | seeking_residence)")

    # ── Table header ────────────────────────────────────────────────────
    lines.append("")
    lines.append(
        f"  {'ID':>5} {'Type':<11} {'Status':<17} "
        f"{'Aspirations':>13} {'D_econ':>10} {'wage_pr':>8} {'D_place':>10} {'place_r':>8} {'PlacePen':>9} "
        f"{'EPen':>8} {'IBonus':>8} {'InMobPen':>9} {'JlBonus':>8} {'SocCalSig':>9} "
        f"{'Capab.':>10} {'Inertia':>13} {'DynThr1':>13} "
        f"{'D_perc':>10} {'Attrib':>8} {'Help':>8} {'SocCal':>8} {'DynInert':>13} {'MigrPress':>10} "
        f"{'TPB(act/del)':>13} {'Thr_mig':>8} {'SignRed':>13} {'IntState':<18}"
    )
    lines.append("  " + "─" * 210)

    # ── Agent rows ──────────────────────────────────────────────────
    for i in range(len(sampled_df_a)):
        ra = sampled_df_a.iloc[i]
        rb = sampled_df_b.iloc[i]
        agent_id = int(ra["id"])

        def _get(row, col, default=""):
            try:
                return row[col]
            except (KeyError, TypeError):
                return default

        # Base fields
        status_a   = str(_get(ra, "status", "?"))
        status_b   = str(_get(rb, "status", "?"))

        # ── Barrier 1: Aspirations and D-components ─────────────────────────
        aspirations_a = float(_get(ra, "aspirations", 0))
        aspirations_b = float(_get(rb, "aspirations", 0))

        # D-components: compute for both ticks
        d_a = _compute_d_components(ra, G)
        d_b = _compute_d_components(rb, G)

        capabilities_a = _compute_capabilities(ra)
        capabilities_b = _compute_capabilities(rb)

        inertia_a = float(_get(ra, "inertia", 0))
        inertia_b = float(_get(rb, "inertia", 0))

        # v3: Barrier 1 — dynamic threshold
        dyn_thr1_a = _compute_dynamic_threshold_stage1(ra)
        dyn_thr1_b = _compute_dynamic_threshold_stage1(rb)

        # v3: Barrier 2 — D_perceived model
        pc_a = float(_get(ra, "perceived_control", 0.5))
        pc_b = float(_get(rb, "perceived_control", 0.5))
        wt_a = float(_get(ra, "weak_ties_utility", 0.0))
        wt_b = float(_get(rb, "weak_ties_utility", 0.0))
        nss_a = float(_get(ra, "net_signal_susc", 0.5))
        nss_b = float(_get(rb, "net_signal_susc", 0.5))
        scs_a = float(_get(ra, "soc_calibration_signal", 0.0))
        scs_b = float(_get(rb, "soc_calibration_signal", 0.0))
        sb_a  = float(_get(ra, "social_boost", 0.0))
        sb_b  = float(_get(rb, "social_boost", 0.0))

        # helplessness = clip(1 − PC − weak_ties × 0.3, 0, 1)
        help_a = float(np.clip(1.0 - pc_a - wt_a * 0.3, 0.0, 1.0))
        help_b = float(np.clip(1.0 - pc_b - wt_b * 0.3, 0.0, 1.0))
        # Attribution = PC × (1 − helplessness)
        attr_a = pc_a * (1.0 - help_a)
        attr_b = pc_b * (1.0 - help_b)
        # SocialCalibration = 1 + net_signal_susc × soc_calibration_signal
        soccal_a = 1.0 + nss_a * scs_a
        soccal_b = 1.0 + nss_b * scs_b
        # D_perceived = D_instant × Attribution × SocialCalibration
        D_perc_a = d_a["D_instant"] * attr_a * soccal_a
        D_perc_b = d_b["D_instant"] * attr_b * soccal_b

        dyn_inertia_a = _compute_dynamic_inertia(ra)
        dyn_inertia_b = _compute_dynamic_inertia(rb)

        # ── Barrier 2: TPB ────────────────────────────────────────────────
        tpb_active_a = bool(_get(ra, "tpb_active", False))
        tpb_active_b = bool(_get(rb, "tpb_active", False))
        tpb_delay_a  = int(_get(ra, "intention_delay", 0))
        tpb_delay_b  = int(_get(rb, "intention_delay", 0))

        thr_mig_a = float(_get(ra, "internal_mig_thr", 0))
        thr_mig_b = float(_get(rb, "internal_mig_thr", 0))

        sign_red_a = float(_get(ra, "signal_reduction", 0))
        sign_red_b = float(_get(rb, "signal_reduction", 0))

        int_state_a = str(_get(ra, "intention_state", "none"))
        int_state_b = str(_get(rb, "intention_state", "none"))

        # Formatting
        id_str      = f"{agent_id:>5}"
        status_str  = f"{status_a}→{status_b}"
        status_str  = f"{status_str:<17}"

        aspir_str   = _fmt_arrow(aspirations_a, aspirations_b, ".3f", 13)
        d_econ_str  = _fmt_arrow(d_a["D_econ"], d_b["D_econ"], ".3f", 10)
        wp_str      = _fmt_arrow(d_a["wage_pressure"], d_b["wage_pressure"], ".3f", 8)
        d_place_str = _fmt_arrow(d_a["D_place"], d_b["D_place"], ".3f", 10)
        pr_str      = _fmt_arrow(d_a["place_reality"], d_b["place_reality"], ".3f", 8)
        place_pen_a = float(_get(ra, "place_deficit_penalty", 0.0))
        place_pen_b = float(_get(rb, "place_deficit_penalty", 0.0))
        pp_str      = _fmt_arrow(place_pen_a, place_pen_b, ".2f", 9)

        # v2: dynamic variables
        ep_a  = float(_get(ra, "econ_penalty", 0.0))
        ep_b  = float(_get(rb, "econ_penalty", 0.0))
        ib_a  = float(_get(ra, "infra_bonus", 0.0))
        ib_b  = float(_get(rb, "infra_bonus", 0.0))
        imp_a = float(_get(ra, "inertia_mobility_penalty", 0.0))
        imp_b = float(_get(rb, "inertia_mobility_penalty", 0.0))
        jlb_a = float(_get(ra, "jobloss_econ_gap_bonus", 0.0))
        jlb_b = float(_get(rb, "jobloss_econ_gap_bonus", 0.0))
        ep_str  = _fmt_arrow(ep_a, ep_b, ".3f", 8)
        ib_str  = _fmt_arrow(ib_a, ib_b, ".3f", 8)
        imp_str = _fmt_arrow(imp_a, imp_b, ".3f", 9)
        jlb_str = _fmt_arrow(jlb_a, jlb_b, ".3f", 8)
        scs_str = _fmt_arrow(scs_a, scs_b, ".3f", 9)

        capab_str   = _fmt_arrow(capabilities_a, capabilities_b, ".3f", 10)
        inertia_str = _fmt_arrow(inertia_a, inertia_b, ".3f", 13)
        dyn_thr1_str = _fmt_arrow(dyn_thr1_a, dyn_thr1_b, ".3f", 13)

        D_perc_str  = _fmt_arrow(D_perc_a, D_perc_b, ".3f", 10)
        attr_str    = _fmt_arrow(attr_a, attr_b, ".3f", 8)
        help_str    = _fmt_arrow(help_a, help_b, ".3f", 8)
        soccal_str  = _fmt_arrow(soccal_a, soccal_b, ".3f", 8)
        dyn_str     = _fmt_arrow(dyn_inertia_a, dyn_inertia_b, ".3f", 13)

        # v4: migration_pressure
        migr_a = float(_get(ra, "migration_pressure", 0.0))
        migr_b = float(_get(rb, "migration_pressure", 0.0))
        migr_str = _fmt_arrow(migr_a, migr_b, ".3f", 10)

        tpb_str = f"{_fmt_bool_arrow(tpb_active_a, tpb_active_b, 6)} {tpb_delay_a}→{tpb_delay_b}"
        tpb_str = f"{tpb_str:>13}"

        thr_str     = _fmt_arrow(thr_mig_a, thr_mig_b, ".3f", 8)
        sign_str    = _fmt_arrow(sign_red_a, sign_red_b, ".3f", 13)
        int_state_s = f"{int_state_a}→{int_state_b}"
        int_state_s = f"{int_state_s:<18}"

        lines.append(
            f"  {id_str} {'':<11} {status_str} "
            f"{aspir_str} {d_econ_str} {wp_str} {d_place_str} {pr_str} {pp_str} "
            f"{ep_str} {ib_str} {imp_str} {jlb_str} {scs_str} "
            f"{capab_str} {inertia_str} {dyn_thr1_str} "
            f"{D_perc_str} {attr_str} {help_str} {soccal_str} {dyn_str} {migr_str} "
            f"{tpb_str} {thr_str} {sign_str} {int_state_s}"
        )

    # ── Sample summary statistics ────────────────────────────────────
    lines.append("  " + "─" * 210)

    def _col_mean(df_sub, col):
        if col not in df_sub.columns:
            return 0.0
        return float(df_sub[col].mean())

    lines.append("  SAMPLE SUMMARY (means):")
    lines.append(
        f"  {'':>5} {'':11} {'':17} "
        f"{'Aspirations':>13} {'D_econ':>10} {'wage_pr':>8} {'D_place':>10} {'place_r':>8} "
        f"{'EPen':>8} {'IBonus':>8} {'InMobPen':>9} {'JlBonus':>8} "
        f"{'Capab.':>10} {'Inertia':>13} {'DynInert':>13} {'MigrPress':>10} "
        f"{'TPB(act/del)':>13} {'Thr_mig':>8} {'SignRed':>13} {'IntState':<18}"
    )

    # Barrier 1 — means
    m_asp_a = _col_mean(sampled_df_a, "aspirations")
    m_asp_b = _col_mean(sampled_df_b, "aspirations")
    m_in_a  = _col_mean(sampled_df_a, "inertia")
    m_in_b  = _col_mean(sampled_df_b, "inertia")
    m_sr_a  = _col_mean(sampled_df_a, "signal_reduction")
    m_sr_b  = _col_mean(sampled_df_b, "signal_reduction")
    m_th_a  = _col_mean(sampled_df_a, "internal_mig_thr")
    m_th_b  = _col_mean(sampled_df_b, "internal_mig_thr")
    m_tpb_a = _col_mean(sampled_df_a, "tpb_active")
    m_tpb_b = _col_mean(sampled_df_b, "tpb_active")
    m_del_a = _col_mean(sampled_df_a, "intention_delay")
    m_del_b = _col_mean(sampled_df_b, "intention_delay")

    # D-components: sample means
    d_vals_a = [_compute_d_components(sampled_df_a.iloc[i], G) for i in range(len(sampled_df_a))]
    d_vals_b = [_compute_d_components(sampled_df_b.iloc[i], G) for i in range(len(sampled_df_b))]
    m_de_a  = np.mean([d["D_econ"] for d in d_vals_a])
    m_de_b  = np.mean([d["D_econ"] for d in d_vals_b])
    m_wp_a  = np.mean([d["wage_pressure"] for d in d_vals_a])
    m_wp_b  = np.mean([d["wage_pressure"] for d in d_vals_b])
    m_dp_a  = np.mean([d["D_place"] for d in d_vals_a])
    m_dp_b  = np.mean([d["D_place"] for d in d_vals_b])
    m_pr_a  = np.mean([d["place_reality"] for d in d_vals_a])
    m_pr_b  = np.mean([d["place_reality"] for d in d_vals_b])

    caps_a_vals = [_compute_capabilities(sampled_df_a.iloc[i]) for i in range(len(sampled_df_a))]
    caps_b_vals = [_compute_capabilities(sampled_df_b.iloc[i]) for i in range(len(sampled_df_b))]
    m_cap_a = np.mean(caps_a_vals) if caps_a_vals else 0.0
    m_cap_b = np.mean(caps_b_vals) if caps_b_vals else 0.0

    dyn_a_vals = [_compute_dynamic_inertia(sampled_df_a.iloc[i]) for i in range(len(sampled_df_a))]
    dyn_b_vals = [_compute_dynamic_inertia(sampled_df_b.iloc[i]) for i in range(len(sampled_df_b))]
    m_dyn_a = np.mean(dyn_a_vals) if dyn_a_vals else 0.0
    m_dyn_b = np.mean(dyn_b_vals) if dyn_b_vals else 0.0

    # v2: dynamic variable means
    m_ep_a  = _col_mean(sampled_df_a, "econ_penalty")
    m_ep_b  = _col_mean(sampled_df_b, "econ_penalty")
    m_ib_a  = _col_mean(sampled_df_a, "infra_bonus")
    m_ib_b  = _col_mean(sampled_df_b, "infra_bonus")
    m_imp_a = _col_mean(sampled_df_a, "inertia_mobility_penalty")
    m_imp_b = _col_mean(sampled_df_b, "inertia_mobility_penalty")
    m_jlb_a = _col_mean(sampled_df_a, "jobloss_econ_gap_bonus")
    m_jlb_b = _col_mean(sampled_df_b, "jobloss_econ_gap_bonus")

    # v4: migration_pressure
    m_migr_a = _col_mean(sampled_df_a, "migration_pressure")
    m_migr_b = _col_mean(sampled_df_b, "migration_pressure")

    # Format summary row
    m_asp_str = _fmt_arrow(m_asp_a, m_asp_b, ".3f", 13)
    m_de_str  = _fmt_arrow(m_de_a, m_de_b, ".3f", 10)
    m_wp_str  = _fmt_arrow(m_wp_a, m_wp_b, ".3f", 8)
    m_dp_str  = _fmt_arrow(m_dp_a, m_dp_b, ".3f", 10)
    m_pr_str  = _fmt_arrow(m_pr_a, m_pr_b, ".3f", 8)
    m_ep_s    = _fmt_arrow(m_ep_a, m_ep_b, ".3f", 8)
    m_ib_s    = _fmt_arrow(m_ib_a, m_ib_b, ".3f", 8)
    m_imp_s   = _fmt_arrow(m_imp_a, m_imp_b, ".3f", 9)
    m_jlb_s   = _fmt_arrow(m_jlb_a, m_jlb_b, ".3f", 8)
    m_cap_str = _fmt_arrow(m_cap_a, m_cap_b, ".3f", 10)
    m_in_str  = _fmt_arrow(m_in_a, m_in_b, ".3f", 13)
    m_dyn_s   = _fmt_arrow(m_dyn_a, m_dyn_b, ".3f", 13)
    m_migr_s  = _fmt_arrow(m_migr_a, m_migr_b, ".3f", 10)

    m_tpb_s = f"{_fmt_bool_arrow(m_tpb_a > 0.5, m_tpb_b > 0.5, 6)} {m_del_a:.1f}→{m_del_b:.1f}"
    m_tpb_s = f"{m_tpb_s:>13}"

    m_th_str = _fmt_arrow(m_th_a, m_th_b, ".3f", 8)
    m_sr_str = _fmt_arrow(m_sr_a, m_sr_b, ".3f", 13)

    # Statuses
    st_a_counts = sampled_df_a["status"].value_counts().to_dict() if "status" in sampled_df_a.columns else {}
    st_b_counts = sampled_df_b["status"].value_counts().to_dict() if "status" in sampled_df_b.columns else {}
    st_a_top = max(st_a_counts, key=st_a_counts.get) if st_a_counts else "?"
    st_b_top = max(st_b_counts, key=st_b_counts.get) if st_b_counts else "?"
    m_st_str = f"{'MEAN':>5} {'—':11} {st_a_top+'→'+st_b_top:<17}"

    # Intention states
    is_a_counts = sampled_df_a["intention_state"].value_counts().to_dict() if "intention_state" in sampled_df_a.columns else {}
    is_b_counts = sampled_df_b["intention_state"].value_counts().to_dict() if "intention_state" in sampled_df_b.columns else {}
    is_a_top = max(is_a_counts, key=is_a_counts.get) if is_a_counts else "none"
    is_b_top = max(is_b_counts, key=is_b_counts.get) if is_b_counts else "none"
    m_is_str = f"{is_a_top}→{is_b_top}"

    lines.append(
        f"  {m_st_str} "
        f"{m_asp_str} {m_de_str} {m_wp_str} {m_dp_str} {m_pr_str} "
        f"{m_ep_s} {m_ib_s} {m_imp_s} {m_jlb_s} "
        f"{m_cap_str} {m_in_str} {m_dyn_s} {m_migr_s} "
        f"{m_tpb_s} {m_th_str} {m_sr_str} {m_is_str}"
    )

    # ── Dynamics analysis ─────────────────────────────────────────────────
    lines.append("")
    lines.append("  DYNAMICS ANALYSIS:")

    n_asp_up = int(
        (sampled_df_b["aspirations"].values > sampled_df_a["aspirations"].values).sum()
    ) if "aspirations" in sampled_df_a.columns else 0

    tpb_a_arr = sampled_df_a["tpb_active"].values.astype(bool) if "tpb_active" in sampled_df_a.columns else np.zeros(n_sample, dtype=bool)
    tpb_b_arr = sampled_df_b["tpb_active"].values.astype(bool) if "tpb_active" in sampled_df_b.columns else np.zeros(n_sample, dtype=bool)
    n_tpb_new = int((~tpb_a_arr & tpb_b_arr).sum())

    is_a_arr = sampled_df_a["intention_state"].values if "intention_state" in sampled_df_a.columns else np.full(n_sample, "none")
    is_b_arr = sampled_df_b["intention_state"].values if "intention_state" in sampled_df_b.columns else np.full(n_sample, "none")
    n_state_changed = int((is_a_arr != is_b_arr).sum())

    st_a_arr = sampled_df_a["status"].values if "status" in sampled_df_a.columns else np.full(n_sample, "?")
    st_b_arr = sampled_df_b["status"].values if "status" in sampled_df_b.columns else np.full(n_sample, "?")
    n_status_changed = int((st_a_arr != st_b_arr).sum())

    # D-component dynamics
    d_inst_a = np.array([d["D_instant"] for d in d_vals_a])
    d_inst_b = np.array([d["D_instant"] for d in d_vals_b])
    n_d_up = int((d_inst_b > d_inst_a).sum())

    lines.append(f"  Agents with increased aspirations:       {n_asp_up}/{n_sample}")
    lines.append(f"  Agents with increased D_instant:         {n_d_up}/{n_sample}")
    lines.append(f"  New TPB activations:                     {n_tpb_new}/{n_sample}")
    lines.append(f"  Changed intention_state:                 {n_state_changed}/{n_sample}")
    lines.append(f"  Changed employment status:               {n_status_changed}/{n_sample}")

    # Mean D-components
    lines.append(f"  Mean D_econ (tick {tick_b}):                     {m_de_b:.4f}")
    lines.append(f"  Mean wage_pressure (tick {tick_b}):              {m_wp_b:.4f}")
    lines.append(f"  Mean D_place (tick {tick_b}):                    {m_dp_b:.4f}")
    lines.append(f"  Mean place_reality (tick {tick_b}):              {m_pr_b:.4f}")

    # v2: dynamic variables
    lines.append(f"  Mean econ_penalty (tick {tick_b}):              {m_ep_b:.4f}")
    lines.append(f"  Mean infra_bonus (tick {tick_b}):               {m_ib_b:.4f}")
    lines.append(f"  Mean inertia_mobility_penalty (tick {tick_b}):  {m_imp_b:.4f}")
    lines.append(f"  Mean jobloss_econ_gap_bonus (tick {tick_b}):    {m_jlb_b:.4f}")

    lines.append("═" * 160)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MASTER DISTRICT TABLE — master table for 79 districts (okresy)
# ══════════════════════════════════════════════════════════════════════════════

def master_district_table(
    tick_stats: list,
    G,
    all_action_log: Optional[List[dict]] = None,
    snapshots: Optional[dict] = None,
) -> str:
    """
    Master table for 79 districts of Slovakia (okresy, not regions/kraje).

    Rows: 79 districts (from DISTRICT_TO_REGION_CODE in agents.py).
    Columns: for each tick — agent count per district and housing price.
    Final columns:
      - Δ econ  — inflow by economic motivation over entire simulation
      - Δ place — inflow by place motivation over entire simulation

    Uses:
      - tick_stats[i][\"district_counts\"] for agent counts by tick
      - G.nodes[district][\"housing_price_m2\"] for housing price
      - all_action_log for counting motivated moves
      - snapshots[0] for tick 0 data (if available)
    """
    # All districts from graph (79 districts)
    all_districts = sorted(G.nodes)

    # Prepare tick lists
    n_ticks = len(tick_stats)
    tick_nums = list(range(1, n_ticks + 1))

    # ── Agents at tick 0 (from snapshots) ──────────────────────────────────
    t0_counts = {}
    if snapshots and 0 in snapshots:
        t0_counts = snapshots[0].groupby("district")["id"].count().to_dict()

    # ── Agents at each tick (from tick_stats) ────────────────────────────
    tick_counts = {}  # {district: [count_t1, count_t2, ...]}
    for district in all_districts:
        tick_counts[district] = []
    for s in tick_stats:
        dc = s.get("district_counts", {})
        for district in all_districts:
            tick_counts[district].append(dc.get(district, 0))

    # ── Housing price: effective price from graph ──
    # G.nodes[district]["effective_housing_price_m2"] is precomputed
    # in update_graph() each tick accounting for housing_remaining and sensitivity.

    # Effective price at each tick (read from graph, which is updated each tick)
    # But we do not have a per-tick graph in the report — use housing_remaining from tick_stats
    # and the formula from graph.py to reproduce it.
    _AGENT_FOOTPRINT = 1.1  # AGENT_HOUSING_FOOTPRINT from graph.py
    _REMAINING_FLOOR = 1.5  # HOUSING_REMAINING_FLOOR from graph.py

    # Base price and sensitivity from graph
    base_prices = {}
    sensitivities = {}
    for district in all_districts:
        base_prices[district] = G.nodes[district].get("housing_price_m2", 1800.0)
        sensitivities[district] = G.nodes[district].get("housing_market_sensitivity", 1.0)

    # Effective price at each tick (using housing_remaining from tick_stats)
    housing_by_tick = {}  # {district: [price_t0, price_t1, ...]}
    for district in all_districts:
        housing_by_tick[district] = []

    # Tick 0: no housing_remaining data → use base price
    if snapshots and 0 in snapshots:
        for district in all_districts:
            housing_by_tick[district].append(base_prices[district])

    # Ticks 1..N: from tick_stats district_housing_remaining
    for s in tick_stats:
        hr = s.get("district_housing_remaining", {})
        for district in all_districts:
            remaining = hr.get(district, _REMAINING_FLOOR)
            bp = base_prices[district]
            sens = sensitivities[district]
            # Use the same formula with floor as in graph.py update_graph
            delta = bp * (_AGENT_FOOTPRINT / max(remaining, _REMAINING_FLOOR)) * sens
            effective = bp + delta
            housing_by_tick[district].append(effective)

    # ── Count moves by motivation from all_action_log ──────────────────
    econ_inflow = {d: 0 for d in all_districts}
    place_inflow = {d: 0 for d in all_districts}
    if all_action_log:
        for entry in all_action_log:
            decision = entry.get("decision", "")
            if decision in ("move", "satellite_move"):
                new_res = entry.get("new_residence", "")
                domain = entry.get("activation_domain", "")
                if new_res in econ_inflow:
                    if domain == "economic":
                        econ_inflow[new_res] += 1
                    elif domain == "place":
                        place_inflow[new_res] += 1

    # ── Build table ─────────────────────────────────────────────
    lines = []
    lines.append(_section("MASTER TABLE FOR 79 DISTRICTS (OKRESY)"))

    # Explanation
    has_t0 = (snapshots and 0 in snapshots)
    lines.append(f"  Total districts: {len(all_districts)} | Ticks: {n_ticks}")
    lines.append(f"  Table 1: agent count per district at each tick")
    lines.append(f"  Table 2: effective housing price (€/m²) accounting for remaining units")
    lines.append(f"  Δ econ / Δ place: total inflow by motivation over entire simulation")
    lines.append("")

    # ── SUBTABLE 1: Agent count ─────────────────────────────────
    lines.append(_section("AGENTS PER DISTRICT BY TICK"))
    header1 = f"  {'District':<30}"
    if has_t0:
        header1 += f" {'T0':>6}"
    for tn in tick_nums:
        header1 += f" {'T' + str(tn):>6}"
    header1 += f" {'Δ econ':>8} {'Δ place':>8}"
    lines.append(header1)

    col_count = (1 if has_t0 else 0) + n_ticks
    line_w1 = 32 + col_count * 7 + 9 + 9
    lines.append("  " + _hline(line_w1, "─"))

    for district in all_districts:
        name = district.replace("District of ", "")[:28]
        row = f"  {name:<30}"
        if has_t0:
            row += f" {t0_counts.get(district, 0):>6,}"
        for count in tick_counts[district]:
            row += f" {count:>6,}"
        row += f" {econ_inflow[district]:>8,} {place_inflow[district]:>8,}"
        lines.append(row)

    # TOTAL for agents
    lines.append("  " + _hline(line_w1, "─"))
    total_row1 = f"  {'TOTAL':<30}"
    if has_t0:
        total_row1 += f" {sum(t0_counts.values()):>6,}"
    for i in range(n_ticks):
        t_total = sum(tick_counts[d][i] for d in all_districts)
        total_row1 += f" {t_total:>6,}"
    total_econ = sum(econ_inflow.values())
    total_place = sum(place_inflow.values())
    total_row1 += f" {total_econ:>8,} {total_place:>8,}"
    lines.append(total_row1)

    # ── Subtable 2: Effective housing price ────────────────────────
    lines.append("")
    lines.append(_section("EFFECTIVE HOUSING PRICE BY TICK (€/m²)"))
    lines.append(f"  Formula: base_price × (1 + 1.1 / remaining_units × sensitivity)")
    lines.append(f"  Fewer remaining units → higher effective price (competition).")
    lines.append("")

    header2 = f"  {'District':<30}"
    if has_t0:
        header2 += f" {'T0':>9}"
    for tn in tick_nums:
        header2 += f" {'T' + str(tn):>9}"
    lines.append(header2)

    line_w2 = 32 + col_count * 10
    lines.append("  " + _hline(line_w2, "─"))

    for district in all_districts:
        name = district.replace("District of ", "")[:28]
        row = f"  {name:<30}"
        for hp in housing_by_tick[district]:
            row += f" {hp:>8,.0f}€"
        lines.append(row)

    # TOTAL for housing (average)
    lines.append("  " + _hline(line_w2, "─"))
    total_row2 = f"  {'AVERAGE':<30}"
    for tick_idx in range(len(tick_nums) + (1 if has_t0 else 0)):
        avg = sum(housing_by_tick[d][tick_idx] for d in all_districts) / max(len(all_districts), 1)
        total_row2 += f" {avg:>8,.0f}€"
    lines.append(total_row2)

    # ── Move statistics ──────────────────────────────────────────
    lines.append("")
    lines.append(f"  Total economic-motivated inflow: {total_econ:,}")
    lines.append(f"  Total place-motivated inflow:    {total_place:,}")

    # Top districts by inflow
    if total_econ > 0:
        top_econ = sorted(econ_inflow.items(), key=lambda x: -x[1])[:5]
        lines.append(f"  Top-5 districts by econ-inflow: "
                     + ", ".join(f"{d.replace('District of ', '')}({c})" for d, c in top_econ if c > 0))
    if total_place > 0:
        top_place = sorted(place_inflow.items(), key=lambda x: -x[1])[:5]
        lines.append(f"  Top-5 districts by place-inflow: "
                     + ", ".join(f"{d.replace('District of ', '')}({c})" for d, c in top_place if c > 0))

    lines.append("=" * 78)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY REPORT — wrapper for full report
# ══════════════════════════════════════════════════════════════════════════════

# ── Report sections (flags for Colab checkboxes) ──────────────────────────────
DEFAULT_SECTIONS = {
    "agent_params":     True,   # Agent parameters matrix (tick0→tick6)
    "demographic":      True,   # Demographic portrait of final state
    "migration_trends": True,   # Dynamics summary + yearly trends
    "region_balance":   True,   # Interregional population balance
    "master_table":     True,   # Master table 79 districts + housing
    "top_routes":       True,   # Top-10 moves and commutes
    "heatmap":          True,   # Heatmap
    "random_industry":  True,   # Random regions and industries
    "behavior_audit":   False,  # Behavioral audit (detail only)
}

def summary_report(
    df_final: pd.DataFrame,
    tick_stats: list,
    all_action_log: Optional[List[dict]] = None,
    snapshots: Optional[dict] = None,
    detail: bool = False,
    G=None,
    sections: Optional[dict] = None,
) -> str:
    """
    Final report with per-section control via `sections`.

    sections — dict {section_name: bool}, with keys from DEFAULT_SECTIONS.
    If None — DEFAULT_SECTIONS are used.
    """
    if sections is None:
        sections = DEFAULT_SECTIONS
    parts = []

    # 0. AGENT PARAMETERS MATRIX
    if sections.get("agent_params", True) and snapshots and 0 in snapshots and 6 in snapshots:
        parts.append(agent_parameters_table(snapshots, G=G, n_show=20, tick_a=0, tick_b=6))

    # 1. Demographic portrait of final state
    if sections.get("demographic", True):
        parts.append(demographic_portrait(
            df_final,
            label="FINAL",
            tick_num=None,
            exclude_students=True,
            detail=detail,
        ))

    # 2. Dynamics summary
    if sections.get("migration_trends", True):
        parts.append(migration_summary(tick_stats))

    # 3. Interregional balance
    if sections.get("region_balance", True) and snapshots and len(snapshots) >= 2:
        parts.append(compare_snapshots(snapshots, tick_stats, all_action_log, detail=detail))
    elif detail and all_action_log and sections.get("behavior_audit", False):
        parts.append(agent_behavior_audit(all_action_log, sample_size=30))

    # 4. Master table for 79 districts
    if sections.get("master_table", True) and G is not None and tick_stats:
        parts.append(master_district_table(tick_stats, G, all_action_log, snapshots=snapshots))

    # 5. RANDOM REGIONS AND INDUSTRIES
    if sections.get("random_industry", True) and G is not None:
        parts.append(_random_region_industry_jobs(G, df=df_final, n_regions=3, n_industries=3))

    # 6. Top-10 moves and commutes
    if sections.get("top_routes", True) and all_action_log is not None:
        top_moves = _top_move_routes(all_action_log, top_n=10)
        if top_moves:
            parts.append(top_moves)
        if detail:
            top_comm = _top_commute_routes(all_action_log, top_n=10)
            if top_comm:
                parts.append(top_comm)

    # 7. Heatmap
    if sections.get("heatmap", True) and G is not None and tick_stats:
        heatmap_block = _district_heatmap(tick_stats, G, snapshots=snapshots)
        if heatmap_block:
            parts.append(heatmap_block)

    return "\n\n".join(parts)

