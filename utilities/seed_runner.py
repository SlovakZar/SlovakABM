"""
seed_runner.py — Многопрогонный анализ с разными сидами для параметров по умолчанию.

Использование:
  python seed_runner.py --agents 5000 --ticks 36 --runs 30 --output seed_results

Используются параметры из engine.py / signals.py (значения по умолчанию).

На выходе:
  seed_results.csv          — метрики по прогонам (одна строка = один прогон)
  seed_results_districts.csv — население по 79 районам (для каждого прогона)
"""

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

SIM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SIM_DIR))

from lhs_runner import ParamPatcher, create_patched_dispatcher, collect_metrics
from graph import build_graph, sync_industry_jobs_to_graph, initialize_industry_pressure_from_agents
from agents import create_agents, JOBS_CAPACITY, INDUSTRY_JOBS_CAPACITY
from engine import run_simulation
from signals import EventBus
from grid_runner import build_district_table


# ═══════════════════════════════════════════════════════════════════════════════
# Кандидат
# ═══════════════════════════════════════════════════════════════════════════════

CANDIDATE = {}

CANDIDATE_LABEL = "default_params"

# Фиксированные параметры (из grid_parameters.json, на случай если нужны для патча)
FIXED_PARAMS = {
    "aspirations_alpha": 0.15,
    "signal_decay": 0.70,
    "gap_adapt_lambda": 0.05,
    "sat_smoothing": 0.88,
    "social_boost_commute": 0.02,
    "social_boost_new_employer": 0.05,
    "unemployed_signal": 0.35,
    "neighbor_signal_coef": 0.04,
    "inertia_loss_jobloss": -0.25,
    "econ_gap_jobloss": 0.25,
    "infra_bonus_delta": 0.05,
    "aspirations_closed_employer": 0.08,
    "hub_weak_ties_bonus": 0.005,
    "move_weak_ties_penalty": -0.10,
    "migration_pressure_p_min": 0.03,
    "migration_pressure_p_max": 0.80,
    "migration_pressure_divisor": 0.12,
    "pc_d_perceived_modifier": 2.0,
    "migration_cooldown_ticks": 9,
    "sb_move_total_ticks": 6,
    "social_boost_decay": 0.60,
    "decay_social_boost_move": 0.005,
    "decay_inertia_mobility": 0.01,
    "decay_econ_penalty": 0.01,
    "jobloss_ramp_step": 0.05,
    "max_jobs_pressure": 1.20,
    "move_stress_factor": 0.80,
    "place_deficit_penalty_move": 0.03,
    "housing_budget_ratio": 0.35,
    "adapt_flex_threshold": 0.65,
    "adapt_sat_boost": 0.06,
    "min_desired_raise": 0.05,
    "unemployed_wage_floor": 0.70,
    "unemployed_wage_ceil": 0.20,
    "commuter_gate_ref": 0.50,
    "job_flex_gate_ref": 0.50,
    "housing_alpha": 0.03,
    "wage_alpha": 0.04,
    "national_avg_wage": 1614.0,
    "spillover_weight": 0.20,
    "agent_housing_footprint": 1.0,
    "housing_remaining_floor": 3.0,
    "rogers_castro_a1": 0.09,
    "rogers_castro_mu1": 22.0,
    "rogers_castro_alpha1": 0.10,
    "rogers_castro_a2": 0.01,
    "rogers_castro_mu2": 65.0,
    "rogers_castro_alpha2": 0.07,
    "rogers_castro_c": 0.005,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Worker для параллельных прогонов
# ═══════════════════════════════════════════════════════════════════════════════

def _run_single_seed(args: tuple) -> tuple:
    """Запускает один прогон с заданным сидом. Вызывается в дочернем процессе."""
    (run_idx, seed_val, n_agents, n_ticks,
     sim_dir, env_path, comm_path,
     agent_dist_path, agent_params_path, dist_path_str) = args

    import sys
    sys.path.insert(0, sim_dir)

    from graph import build_graph, sync_industry_jobs_to_graph, initialize_industry_pressure_from_agents
    from agents import create_agents, JOBS_CAPACITY, INDUSTRY_JOBS_CAPACITY
    from engine import run_simulation
    from signals import EventBus
    from lhs_runner import ParamPatcher, create_patched_dispatcher, collect_metrics
    from grid_runner import build_district_table
    import json

    # Каждый worker строит свой граф (fork → copy-on-write)
    G = build_graph(env_path, comm_path)

    with open(dist_path_str) as f:
        init_dists = json.load(f).get("districts", {})

    # Полный словарь параметров: fixed + candidate
    plan = dict(FIXED_PARAMS)
    plan.update(CANDIDATE)

    patcher = ParamPatcher(plan)
    signal_params = patcher.apply()

    df = create_agents(agent_dist_path, agent_params_path, comm_path,
                       n_agents=n_agents, seed=seed_val)

    sync_industry_jobs_to_graph(G, INDUSTRY_JOBS_CAPACITY, JOBS_CAPACITY)
    initialize_industry_pressure_from_agents(G, df)

    dispatcher = create_patched_dispatcher(signal_params)
    bus = EventBus(dispatcher=dispatcher)

    snapshot_ticks = sorted(set([0] + list(range(6, n_ticks, 6)) + [n_ticks]))
    df_final, snapshots, tick_stats, all_action_log = run_simulation(
        df, G, n_ticks=n_ticks, snapshot_ticks=snapshot_ticks,
        seed=seed_val, verbose=False, jobs_capacity=JOBS_CAPACITY,
        init_dists=init_dists, bus=bus, scenario=None,
    )

    metrics = collect_metrics(df_final, snapshots, tick_stats, all_action_log)

    label = CANDIDATE_LABEL

    variant = dict(CANDIDATE)

    row = {"run_id": run_idx, "seed": seed_val, "run_label": label,
           **{f"v_{k}": v for k, v in variant.items()}, **metrics}

    dist_table = build_district_table(snapshots, run_idx, label, variant, all_action_log)
    # Добавляем seed в таблицу районов
    if len(dist_table) > 0:
        dist_table["seed"] = seed_val

    patcher.restore()

    return row, dist_table


# ═══════════════════════════════════════════════════════════════════════════════
# Главный запуск
# ═══════════════════════════════════════════════════════════════════════════════

def seed_run(
    n_agents: int = 5000,
    n_ticks: int = 36,
    n_runs: int = 30,
    start_seed: int = 0,
    output_prefix: str = "seed_results",
    parallel: bool = False,
    verbose: bool = True,
):
    t_start = time.time()

    seeds = list(range(start_seed, start_seed + n_runs))

    if verbose:
        print(f"Кандидат: {CANDIDATE_LABEL}")
        if CANDIDATE:
            print(f"Параметры: inertia_mobility_penalty_move={CANDIDATE.get('inertia_mobility_penalty_move', 'default')}, "
                  f"social_boost_move={CANDIDATE.get('social_boost_move', 'default')}, "
                  f"base_appetite_min={CANDIDATE.get('base_appetite_min', 'default')}, "
                  f"max_work_candidates={CANDIDATE.get('max_work_candidates', 'default')}")
        print(f"Прогонов: {n_runs}  |  Сиды: {seeds[0]}…{seeds[-1]}  |  "
              f"Агентов: {n_agents:,}  |  Тиков: {n_ticks}\n")

    if not parallel:
        # ── Последовательный режим ──────────────────────────────────────
        if verbose:
            print("Строим граф Словакии...")
        G = build_graph(
            str(SIM_DIR / "data" / "environment.json"),
            str(SIM_DIR / "data" / "commuting_filtered_with_travel.csv"),
        )

        dist_path = SIM_DIR / "data" / "agent_init_distributions.json"
        with open(dist_path, encoding="utf-8") as f:
            init_dists = json.load(f).get("districts", {})

        all_metrics = []
        all_district_rows = []

        for run_idx, seed_val in enumerate(seeds):
            t_run = time.time()

            plan = dict(FIXED_PARAMS)
            plan.update(CANDIDATE)

            patcher = ParamPatcher(plan)
            signal_params = patcher.apply()

            df = create_agents(
                str(SIM_DIR / "data" / "agent_init_distributions.json"),
                str(SIM_DIR / "data" / "agent_params_from_survey.json"),
                str(SIM_DIR / "data" / "commuting_filtered_with_travel.csv"),
                n_agents=n_agents, seed=seed_val,
            )

            sync_industry_jobs_to_graph(G, INDUSTRY_JOBS_CAPACITY, JOBS_CAPACITY)
            initialize_industry_pressure_from_agents(G, df)

            dispatcher = create_patched_dispatcher(signal_params)
            bus = EventBus(dispatcher=dispatcher)

            snapshot_ticks = sorted(set([0] + list(range(6, n_ticks, 6)) + [n_ticks]))
            df_final, snapshots, tick_stats, all_action_log = run_simulation(
                df, G, n_ticks=n_ticks, snapshot_ticks=snapshot_ticks,
                seed=seed_val, verbose=False, jobs_capacity=JOBS_CAPACITY,
                init_dists=init_dists, bus=bus, scenario=None,
            )

            metrics = collect_metrics(df_final, snapshots, tick_stats, all_action_log)

            label = CANDIDATE_LABEL
            variant = dict(CANDIDATE)

            row = {"run_id": run_idx, "seed": seed_val, "run_label": label,
                   **{f"v_{k}": v for k, v in variant.items()}, **metrics}
            all_metrics.append(row)

            dist_table = build_district_table(snapshots, run_idx, label, variant, all_action_log)
            if len(dist_table) > 0:
                dist_table["seed"] = seed_val
            all_district_rows.append(dist_table)

            patcher.restore()

            elapsed = time.time() - t_run
            if verbose:
                moves = metrics.get("n_moved_economic", 0) + metrics.get("n_moved_place", 0)
                commutes = metrics.get("n_commute_started", 0)
                unemp = metrics.get("unemployment_rate", 0)
                print(f"  [{run_idx+1:3d}/{n_runs}] seed={seed_val:3d}  "
                      f"moves={moves:5d}  commutes={commutes:4d}  "
                      f"unemp={unemp:.3f}  time={elapsed:.1f}s")

    else:
        # ── Параллельный режим ──────────────────────────────────────────
        import concurrent.futures
        import multiprocessing as mp

        sim_dir_str = str(SIM_DIR)
        env_path = str(SIM_DIR / "data" / "environment.json")
        comm_path = str(SIM_DIR / "data" / "commuting_filtered_with_travel.csv")
        agent_dist_path = str(SIM_DIR / "data" / "agent_init_distributions.json")
        agent_params_path = str(SIM_DIR / "data" / "agent_params_from_survey.json")
        dist_path_str = str(SIM_DIR / "data" / "agent_init_distributions.json")

        args_list = [
            (run_idx, seed_val, n_agents, n_ticks,
             sim_dir_str, env_path, comm_path,
             agent_dist_path, agent_params_path, dist_path_str)
            for run_idx, seed_val in enumerate(seeds)
        ]

        all_metrics = [None] * n_runs
        all_district_rows = [None] * n_runs

        with mp.get_context("fork").Pool(processes=10) as pool:
            results = pool.imap_unordered(_run_single_seed, args_list)
            done = 0
            for row, dist_table in results:
                all_metrics[row["run_id"]] = row
                all_district_rows[row["run_id"]] = dist_table
                done += 1
                if verbose:
                    moves = row.get("n_moved_economic", 0) + row.get("n_moved_place", 0)
                    commutes = row.get("n_commute_started", 0)
                    unemp = row.get("unemployment_rate", 0)
                    print(f"  [{done:3d}/{n_runs}] seed={row.get('seed', '?'):3d}  "
                          f"moves={moves:5d}  commutes={commutes:4d}  "
                          f"unemp={unemp:.3f}")

        all_metrics = [m for m in all_metrics if m is not None]
        all_district_rows = [d for d in all_district_rows if d is not None]

    # ── Сохраняем ──────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)

    # Переставляем колонки: run_*, seed, v_*, затем метрики
    meta_cols = [c for c in metrics_df.columns if c.startswith(("run_", "v_")) or c == "seed"]
    metric_cols = [c for c in metrics_df.columns if c not in meta_cols]
    metrics_df = metrics_df[meta_cols + metric_cols]

    districts_df = pd.concat(all_district_rows, ignore_index=True)

    metrics_path = f"{output_prefix}.csv"
    districts_path = f"{output_prefix}_districts.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    districts_df.to_csv(districts_path, index=False, encoding="utf-8")

    # ── Итоговая статистика по 30 прогонам ────────────────────────────────
    if verbose:
        total_elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"Кандидат: {CANDIDATE_LABEL}")
        print(f"Прогонов: {n_runs}  |  Сиды: {seeds[0]}…{seeds[-1]}  |  "
              f"Время: {total_elapsed:.0f}с  ({total_elapsed/n_runs:.1f} с/прогон)")
        print(f"{'='*60}")

        # Средние по ключевым метрикам
        key_metrics = ["unemployment_rate", "commuter_share",
                       "n_moved_economic", "n_moved_place", "n_commute_started",
                       "avg_wage_employed", "avg_migration_pressure",
                       "avg_inertia", "avg_dissatisfaction_weighted"]
        print(f"\n{'Метрика':<35} {'Среднее':>10} {'Стд':>10} {'Мин':>10} {'Макс':>10}")
        print("-" * 75)
        for m in key_metrics:
            if m in metrics_df.columns:
                vals = metrics_df[m]
                print(f"{m:<35} {vals.mean():>10.3f} {vals.std():>10.3f} "
                      f"{vals.min():>10.3f} {vals.max():>10.3f}")

        print(f"\nМетрики:          {metrics_path}  ({len(metrics_df)} строк)")
        print(f"Таблица районов:  {districts_path}  ({len(districts_df)} строк)")

    return metrics_df, districts_df


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Seed-прогоны для кандидата {CANDIDATE_LABEL}"
    )
    parser.add_argument("--agents", type=int, default=5000, help="Агентов на прогон")
    parser.add_argument("--ticks", type=int, default=36, help="Тиков на прогон")
    parser.add_argument("--runs", type=int, default=30, help="Число прогонов с разными сидами")
    parser.add_argument("--start-seed", type=int, default=0, help="Начальный seed")
    parser.add_argument("--output", default="seed_results", help="Префикс выходных файлов")
    parser.add_argument("--parallel", action="store_true", help="Параллельный режим (10 воркеров)")
    args = parser.parse_args()

    seed_run(
        n_agents=args.agents,
        n_ticks=args.ticks,
        n_runs=args.runs,
        start_seed=args.start_seed,
        output_prefix=args.output,
        parallel=args.parallel,
    )
