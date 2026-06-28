"""
grid_runner.py — Сеточные прогоны по 6 ключевым параметрам с таблицей населения по 79 районам.

Использование:
  python grid_runner.py --agents 5000 --ticks 36 --output grid_results

На выходе:
  grid_results.csv          — метрики по прогонам (одна строка = один прогон)
  grid_results_districts.csv — население по 79 районам: тик0, тикN, дельта (для каждого прогона)
"""

import argparse
import json
import sys
import time
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Dict, List

SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(SIM_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Загрузка сетки параметров
# ═══════════════════════════════════════════════════════════════════════════════

def load_grid_spec(spec_path: str = "grid_parameters.json") -> dict:
    p = Path(spec_path)
    if not p.exists():
        p = SIM_DIR / spec_path
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_grid_plan(spec: dict) -> List[Dict[str, float]]:
    """
    Строит список словарей параметров: полный факторный план.
    """
    fixed = spec["fixed"]
    core = spec["grid"]["core"]

    plans = []

    core_keys = list(core.keys())
    core_values = [core[k] for k in core_keys]

    for combo in product(*core_values):
        plan = dict(fixed)
        for k, v in zip(core_keys, combo):
            plan[k] = v
        plans.append(plan)

    return plans


def load_scenarios(scenarios_path: str) -> list:
    """Загружает список ручных сценариев из JSON."""
    p = Path(scenarios_path)
    if not p.exists():
        p = SIM_DIR / scenarios_path
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_scenario_plans(scenarios: list, fixed: dict) -> List[Dict[str, float]]:
    """
    Строит планы из списка сценариев: каждый сценарий = fixed + специфичные параметры.
    """
    plans = []
    for sc in scenarios:
        plan = dict(fixed)
        plan.update(sc)
        plans.append(plan)
    return plans


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Подмена параметров (переиспользуем ParamPatcher из lhs_runner)
# ═══════════════════════════════════════════════════════════════════════════════

from lhs_runner import ParamPatcher, create_patched_dispatcher


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Таблица по 79 районам
# ═══════════════════════════════════════════════════════════════════════════════


def build_district_table(
    snapshots: dict,
    run_id: int,
    run_label: str,
    variant_params: Dict[str, float],
    all_action_log: List[dict],
    district_col: str = "district",
) -> pd.DataFrame:
    """
    Строит таблицу: один район = одна строка.
    """
    ticks = sorted(snapshots.keys())
    if len(ticks) < 2:
        return pd.DataFrame()

    t0, tN = ticks[0], ticks[-1]
    df0 = snapshots[t0]
    dfN = snapshots[tN]

    pop0 = df0[district_col].value_counts()
    popN = dfN[district_col].value_counts()

    moved_in = {}
    moved_out = {}
    for a in all_action_log:
        if a.get("decision") in ("move", "satellite_move"):
            src = a.get("old_residence", a.get("source_district", ""))
            dst = a.get("new_residence", a.get("target_district", ""))
            if src:
                moved_out[src] = moved_out.get(src, 0) + 1
            if dst:
                moved_in[dst] = moved_in.get(dst, 0) + 1

    # Только параметры, не начинающиеся с _
    clean_params = {k: v for k, v in variant_params.items() if not k.startswith("_")}

    all_districts = sorted(set(pop0.index) | set(popN.index))
    rows = []
    for d in all_districts:
        p0 = pop0.get(d, 0)
        pN = popN.get(d, 0)
        delta_abs = pN - p0
        delta_pct = round(delta_abs / max(p0, 1) * 100, 2)

        rows.append({
            "district": d,
            "pop_tick0": p0,
            f"pop_tick{tN}": pN,
            "delta_abs": delta_abs,
            "delta_pct": delta_pct,
            "n_moved_in": moved_in.get(d, 0),
            "n_moved_out": moved_out.get(d, 0),
            "net_flow": moved_in.get(d, 0) - moved_out.get(d, 0),
            "run_id": run_id,
            "run_label": run_label,
            **{f"p_{k}": v for k, v in clean_params.items()},
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Сбор метрик (аналог collect_metrics из lhs_runner)
# ═══════════════════════════════════════════════════════════════════════════════

from lhs_runner import collect_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Worker для параллельных прогонов (picklable, module-level)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_single_plan(args: tuple) -> tuple:
    """Запускает один прогон. Вызывается в дочернем процессе."""
    (run_idx, plan, fixed, n_agents, seed, n_ticks,
     sim_dir, env_path, comm_path, agent_dist_path,
     agent_params_path, dist_path_str) = args

    import sys
    sys.path.insert(0, sim_dir)

    from graph import build_graph, sync_industry_jobs_to_graph, initialize_industry_pressure_from_agents
    from agents import create_agents, JOBS_CAPACITY, INDUSTRY_JOBS_CAPACITY
    from engine import run_simulation
    from signals import EventBus
    from lhs_runner import ParamPatcher, create_patched_dispatcher, collect_metrics
    import json

    # Каждый worker строит свой граф (fork → copy-on-write)
    G = build_graph(env_path, comm_path)

    with open(dist_path_str) as f:
        init_dists = json.load(f).get("districts", {})

    patcher = ParamPatcher(plan)
    signal_params = patcher.apply()

    df = create_agents(agent_dist_path, agent_params_path, comm_path,
                       n_agents=n_agents, seed=seed)

    sync_industry_jobs_to_graph(G, INDUSTRY_JOBS_CAPACITY, JOBS_CAPACITY)
    initialize_industry_pressure_from_agents(G, df)

    dispatcher = create_patched_dispatcher(signal_params)
    bus = EventBus(dispatcher=dispatcher)

    snapshot_ticks = sorted(set([0] + list(range(6, n_ticks, 6)) + [n_ticks]))
    df_final, snapshots, tick_stats, all_action_log = run_simulation(
        df, G, n_ticks=n_ticks, snapshot_ticks=snapshot_ticks,
        seed=seed, verbose=False, jobs_capacity=JOBS_CAPACITY,
        init_dists=init_dists, bus=bus, scenario=None,
    )

    metrics = collect_metrics(df_final, snapshots, tick_stats, all_action_log)

    label = (f"inmob{plan['inertia_mobility_penalty_move']}_"
             f"sbm{plan['social_boost_move']}_"
             f"bamin{plan['base_appetite_min']}_"
             f"mwc{plan['max_work_candidates']}")

    variant = {k: v for k, v in plan.items()
               if not k.startswith("_") and (k not in fixed or plan[k] != fixed.get(k))}

    row = {"run_id": run_idx, "run_label": label,
           **{f"v_{k}": v for k, v in variant.items()}, **metrics}

    from grid_runner import build_district_table
    dist_table = build_district_table(snapshots, run_idx, label, variant, all_action_log)

    patcher.restore()

    return row, dist_table


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Главный цикл
# ═══════════════════════════════════════════════════════════════════════════════

def grid_run(
    n_agents: int = 5000,
    n_ticks: int = 36,
    seed: int = 42,
    spec_path: str = "grid_parameters.json",
    scenarios_path: str = None,
    output_prefix: str = "grid_results",
    parallel: bool = False,
    verbose: bool = True,
):
    from graph import build_graph
    from agents import create_agents, JOBS_CAPACITY, INDUSTRY_JOBS_CAPACITY
    from engine import run_simulation
    from signals import EventBus
    from graph import sync_industry_jobs_to_graph, initialize_industry_pressure_from_agents

    t_start = time.time()

    # Загружаем план
    if scenarios_path:
        spec = load_grid_spec(spec_path)
        scenarios = load_scenarios(scenarios_path)
        plans = build_scenario_plans(scenarios, spec["fixed"])
        n_runs = len(plans)
        fixed = spec["fixed"]
        mode = "сценариев"
    else:
        spec = load_grid_spec(spec_path)
        plans = build_grid_plan(spec)
        n_runs = len(plans)
        fixed = spec["fixed"]
        mode = "сетка"

    n_workers = 12 if parallel else 1
    if verbose:
        print(f"Режим: {mode}  |  Прогонов: {n_runs}  |  "
              f"Воркеров: {n_workers}  |  "
              f"Агентов: {n_agents:,}  |  Тиков: {n_ticks}  |  Seed: {seed}\n")

    if not parallel:
        # ── Последовательный режим ──────────────────────────────────────
        if verbose:
            print("Строим граф Словакии...")
        G = build_graph(
            str(SIM_DIR / "environment.json"),
            str(SIM_DIR / "commuting_filtered_with_travel.csv"),
        )

        dist_path = SIM_DIR / "agent_init_distributions.json"
        with open(dist_path, encoding="utf-8") as f:
            init_dists = json.load(f).get("districts", {})

        all_metrics = []
        all_district_rows = []

        for run_idx, plan in enumerate(plans):
            t_run = time.time()

            variant = {k: v for k, v in plan.items()
                       if not k.startswith("_") and (k not in fixed or plan[k] != fixed.get(k))}

            patcher = ParamPatcher(plan)
            signal_params = patcher.apply()

            df = create_agents(
                str(SIM_DIR / "agent_init_distributions.json"),
                str(SIM_DIR / "agent_params_from_survey.json"),
                str(SIM_DIR / "commuting_filtered_with_travel.csv"),
                n_agents=n_agents, seed=seed,
            )

            sync_industry_jobs_to_graph(G, INDUSTRY_JOBS_CAPACITY, JOBS_CAPACITY)
            initialize_industry_pressure_from_agents(G, df)

            dispatcher = create_patched_dispatcher(signal_params)
            bus = EventBus(dispatcher=dispatcher)

            snapshot_ticks = sorted(set([0] + list(range(6, n_ticks, 6)) + [n_ticks]))
            df_final, snapshots, tick_stats, all_action_log = run_simulation(
                df, G, n_ticks=n_ticks, snapshot_ticks=snapshot_ticks,
                seed=seed, verbose=False, jobs_capacity=JOBS_CAPACITY,
                init_dists=init_dists, bus=bus, scenario=None,
            )

            metrics = collect_metrics(df_final, snapshots, tick_stats, all_action_log)

            label = (f"inmob{plan['inertia_mobility_penalty_move']}_"
                     f"sbm{plan['social_boost_move']}_"
                     f"bamin{plan['base_appetite_min']}_"
                     f"mwc{plan['max_work_candidates']}")

            row = {"run_id": run_idx, "run_label": label,
                   **{f"v_{k}": v for k, v in variant.items()}, **metrics}
            all_metrics.append(row)

            dist_table = build_district_table(snapshots, run_idx, label, variant, all_action_log)
            all_district_rows.append(dist_table)

            patcher.restore()

            elapsed = time.time() - t_run
            if verbose:
                moves = metrics.get("n_moved_economic", 0) + metrics.get("n_moved_place", 0)
                commutes = metrics.get("n_commute_started", 0)
                print(f"  [{run_idx+1:3d}/{n_runs}] {label:<65} "
                      f"moves={moves:5d}  commutes={commutes:4d}  "
                      f"time={elapsed:.1f}s")

    else:
        # ── Параллельный режим (4 воркера) ──────────────────────────────
        import concurrent.futures
        import multiprocessing as mp

        sim_dir_str = str(SIM_DIR)
        env_path = str(SIM_DIR / "environment.json")
        comm_path = str(SIM_DIR / "commuting_filtered_with_travel.csv")
        agent_dist_path = str(SIM_DIR / "agent_init_distributions.json")
        agent_params_path = str(SIM_DIR / "agent_params_from_survey.json")
        dist_path_str = str(SIM_DIR / "agent_init_distributions.json")

        args_list = [
            (run_idx, plan, fixed, n_agents, seed, n_ticks,
             sim_dir_str, env_path, comm_path,
             agent_dist_path, agent_params_path, dist_path_str)
            for run_idx, plan in enumerate(plans)
        ]

        all_metrics = [None] * n_runs
        all_district_rows = [None] * n_runs

        with mp.get_context("fork").Pool(processes=12) as pool:
            results = pool.imap_unordered(_run_single_plan, args_list)
            done = 0
            for row, dist_table in results:
                all_metrics[row["run_id"]] = row
                all_district_rows[row["run_id"]] = dist_table
                done += 1
                if verbose:
                    print(f"  [{done:3d}/{n_runs}] {row['run_label']:<65} "
                          f"moves={row.get('n_moved_economic', 0) + row.get('n_moved_place', 0):5d}  "
                          f"commutes={row.get('n_commute_started', 0):4d}")

        # Восстанавливаем порядок
        all_metrics = [m for m in all_metrics if m is not None]
        all_district_rows = [d for d in all_district_rows if d is not None]

    # ── Сохраняем ──────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)

    # Переставляем колонки: run_*, v_*, затем метрики
    meta_cols = [c for c in metrics_df.columns if c.startswith(("run_", "v_"))]
    metric_cols = [c for c in metrics_df.columns if c not in meta_cols]
    metrics_df = metrics_df[meta_cols + metric_cols]

    districts_df = pd.concat(all_district_rows, ignore_index=True)

    metrics_path = f"{output_prefix}.csv"
    districts_path = f"{output_prefix}_districts.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    districts_df.to_csv(districts_path, index=False, encoding="utf-8")

    total_elapsed = time.time() - t_start
    if verbose:
        print(f"\n{'='*60}")
        print(f"Сетка завершена: {n_runs} прогонов за {total_elapsed:.0f} сек "
              f"({total_elapsed/n_runs:.1f} с/прогон)")
        print(f"Метрики:          {metrics_path}  ({len(metrics_df)} строк)")
        print(f"Таблица районов:  {districts_path}  ({len(districts_df)} строк)")

    return metrics_df, districts_df


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сеточные прогоны SlovakABM по ключевым параметрам")
    parser.add_argument("--agents", dest="n_agents", type=int, default=5000, help="Агентов на прогон")
    parser.add_argument("--ticks", type=int, default=36, help="Тиков на прогон")
    parser.add_argument("--seed", type=int, default=42, help="Базовый seed")
    parser.add_argument("--output", default="grid_results", help="Префикс выходных файлов")
    parser.add_argument("--spec", default="grid_parameters.json", help="JSON-спецификация сетки")
    parser.add_argument("--scenarios", default=None, help="Файл со списком ручных сценариев")
    parser.add_argument("--parallel", action="store_true", help="Параллельный режим (12 воркеров)")
    args = parser.parse_args()

    grid_run(
        n_agents=args.n_agents,
        n_ticks=args.ticks,
        seed=args.seed,
        spec_path=args.spec,
        scenarios_path=args.scenarios,
        output_prefix=args.output,
        parallel=args.parallel,
    )
