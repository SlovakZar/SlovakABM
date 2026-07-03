"""
run.py v2 — entry point for running the simulation (all Slovakia, 70k agents).

Usage:
  python run.py                            # default: 70000 agents, 60 ticks
  python run.py --agents 10000 --ticks 24  # quick test
  python run.py --output report.txt

In Google Colab:
  from run import run
  df_final, snapshots, stats = run(n_agents=70000, n_ticks=60)
"""

import argparse
import sys
import time
import json
from pathlib import Path

SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(SIM_DIR))

from graph   import build_graph, print_graph_summary, sync_industry_jobs_to_graph, initialize_industry_pressure_from_agents
from agents  import create_agents
import agents as _ag
from engine  import run_simulation
from signals import EventBus, create_default_dispatcher
from scenario import Scenario
from report  import summary_report, agent_parameters_table, industry_jobs_snapshot


def run(
    env_path:       str  = "data/environment.json",
    commuting_path: str  = "data/commuting_filtered_with_travel.csv",
    agent_dist_path: str = "data/agent_init_distributions.json",
    n_agents:       int  = 70000,
    n_ticks:        int  = 60,
    seed:           int  = 42,
    output_file:    str  = None,
    verbose:        bool = True,
    detail:         bool = False,
    sections:       dict = None,
) -> tuple:
    t0 = time.time()

    if verbose:
        print("\n[1/4] Build Slovakia graph from commuting matrix...")
    G = build_graph(env_path, commuting_path)
    if verbose:
        print_graph_summary(G)

    if verbose:
        print(f"\n[2/4] Create agents (n={n_agents:,}, seed={seed})...")
    df = create_agents(agent_dist_path, n_agents=n_agents, seed=seed, commuting_path=commuting_path, G=G)

    # v3: Sync industry_jobs (occupied+vacant) and jobs_capacity into graph nodes.
    # INDUSTRY_JOBS_CAPACITY filled in create_agents → _init_industry_jobs.
    sync_industry_jobs_to_graph(G, _ag.INDUSTRY_JOBS_CAPACITY, _ag.JOBS_CAPACITY, n_agents=n_agents)
    
    # v4: Initialize industry_pressure based on initial agent distribution
    initialize_industry_pressure_from_agents(G, df)

    # Load init_dists for graduation (graduate industry)
    dist_path = Path(agent_dist_path)
    if not dist_path.exists():
        dist_path = SIM_DIR / agent_dist_path
    with open(dist_path, encoding="utf-8") as f:
        init_dists = json.load(f).get("districts", {})

    snapshot_ticks = [0, 6, n_ticks // 4, n_ticks // 2, n_ticks]

    if verbose:
        print(f"\n[3/4] Launch simulation ({n_ticks} ticks = {n_ticks//12} years {n_ticks%12} mo)...")

    # Create event bus
    bus = EventBus(dispatcher=create_default_dispatcher())

    # Load scenario (if file exists)
    scenario = Scenario.from_json("scenario.json")

    df_final, snapshots, tick_stats, all_action_log = run_simulation(
        df, G,
        n_ticks=n_ticks,
        snapshot_ticks=snapshot_ticks,
        seed=seed,
        verbose=verbose,
        jobs_capacity=_ag.JOBS_CAPACITY,
        init_dists=init_dists,
        bus=bus,
        scenario=scenario,
    )

    if verbose:
        print(f"\n[4/4] Generate report...")

    # Assemble final report
    report_parts = []

    # ═══ AGENT PARAMETERS MATRIX: Tick 0 → Tick 6 ═══
    if sections is None or sections.get("agent_params", True):
        agent_table = agent_parameters_table(snapshots, G=G, n_show=20, tick_a=0, tick_b=6, seed=seed)
        report_parts.append(agent_table)

    # ═══ v3: INDUSTRY JOBS SNAPSHOT (occupied/vacant) ═══
    jobs_snap = industry_jobs_snapshot(G, df=df_final)
    report_parts.append(jobs_snap)

    # ═══ Final summary (demographics + trends + tables + map) ═══
    final_summary = summary_report(df_final, tick_stats, all_action_log, snapshots, detail=detail, G=G, sections=sections)
    report_parts.append(final_summary)

    full_report = "\n\n".join(report_parts)
    elapsed = time.time() - t0
    full_report += f"\n\n⏱  {elapsed:.1f} sec | {n_ticks} ticks | {n_agents:,} agents"
    if detail:
        full_report += " | mode: full (detail=True)"
    else:
        full_report += " | mode: summary"

    print(full_report)

    if output_file:
        Path(output_file).write_text(full_report, encoding="utf-8")
        print(f"\n  Report saved: {output_file}")

    return df_final, snapshots, tick_stats, all_action_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABM Миграция — Словакия")
    parser.add_argument("--env",       default="data/environment.json")
    parser.add_argument("--commuting", default="data/commuting_filtered_with_travel.csv")
    parser.add_argument("--agent_dist", default="data/agent_init_distributions.json")
    parser.add_argument("--agents",    type=int, default=70000)
    parser.add_argument("--ticks",     type=int, default=60)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--quiet",     action="store_true")
    parser.add_argument("--detail",    action="store_true", help="Full report with audit и доп. метриками")
    args = parser.parse_args()

    run(
        env_path=args.env,
        commuting_path=args.commuting,
        agent_dist_path=args.agent_dist,
        n_agents=args.agents,
        n_ticks=args.ticks,
        seed=args.seed,
        output_file=args.output,
        verbose=not args.quiet,
        detail=args.detail,
    )
