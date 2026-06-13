"""
run.py v2 — точка входа для запуска симуляции (вся Словакия, 70k агентов).

Использование:
  python run.py                            # дефолт: 70000 агентов, 60 тиков
  python run.py --agents 10000 --ticks 24  # быстрый тест
  python run.py --output report.txt

В Google Colab:
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

from graph   import build_graph, print_graph_summary
from agents  import create_agents, JOBS_CAPACITY
from engine  import run_simulation
from report  import demographic_portrait, compare_snapshots, summary_report, agent_parameters_table


def run(
    env_path:       str  = "environment.json",
    commuting_path: str  = "commuting_filtered_with_travel.csv",
    agent_dist_path: str = "agent_init_distributions.json",
    n_agents:       int  = 70000,
    n_ticks:        int  = 60,
    seed:           int  = 42,
    output_file:    str  = None,
    verbose:        bool = True,
    detail:         bool = False,
) -> tuple:
    t0 = time.time()

    if verbose:
        print("\n[1/4] Строим граф Словакии из commuting-матрицы...")
    G = build_graph(env_path, commuting_path)
    if verbose:
        print_graph_summary(G)

    if verbose:
        print(f"\n[2/4] Создаём агентов (n={n_agents:,}, seed={seed})...")
    df = create_agents(agent_dist_path, n_agents=n_agents, seed=seed, commuting_path=commuting_path)

    # Синхронизируем jobs_capacity в узлах графа с отмасштабированным JOBS_CAPACITY.
    # G.nodes использует свою оценку (pop × 0.45), а JOBS_CAPACITY — из реальной
    # commuting-матрицы. Приводим к единому знаменателю.
    for d in G.nodes:
        if d in JOBS_CAPACITY:
            G.nodes[d]["jobs_capacity"] = JOBS_CAPACITY[d]

    # Загружаем init_dists для graduation (отрасль выпускников)
    dist_path = Path(agent_dist_path)
    if not dist_path.exists():
        dist_path = SIM_DIR / agent_dist_path
    with open(dist_path, encoding="utf-8") as f:
        init_dists = json.load(f).get("districts", {})

    snapshot_ticks = [0, 6, n_ticks // 4, n_ticks // 2, n_ticks]

    if verbose:
        print(f"\n[3/4] Запуск симуляции ({n_ticks} тиков = {n_ticks//12} лет {n_ticks%12} мес)...")
    df_final, snapshots, tick_stats, all_action_log = run_simulation(
        df, G,
        n_ticks=n_ticks,
        snapshot_ticks=snapshot_ticks,
        seed=seed,
        verbose=verbose,
        jobs_capacity=JOBS_CAPACITY,
        init_dists=init_dists,
    )

    if verbose:
        print(f"\n[4/4] Генерируем отчёт...")

    # Снимки по тикам (демографические портреты)
    report_parts = []

    # ═══ МАТРИЦА ПАРАМЕТРОВ АГЕНТОВ: Тик 0 → Тик 6 ═══
    agent_table = agent_parameters_table(snapshots, n_show=20, tick_a=0, tick_b=6, seed=seed)
    report_parts.append(agent_table)

    for t in sorted(snapshots.keys()):
        label = {0: "НАЧАЛО"}.get(t, f"Тик {t}")
        portrait = demographic_portrait(snapshots[t], label=label, tick_num=t, detail=detail)
        report_parts.append(portrait)

    # Итоговая сводка
    final_summary = summary_report(df_final, tick_stats, all_action_log, snapshots, detail=detail)
    report_parts.append(final_summary)
    # Также межрегиональный баланс (compare_snapshots)
    comparison = compare_snapshots(snapshots, tick_stats, all_action_log, detail=detail)
    report_parts.append(comparison)

    full_report = "\n\n".join(report_parts)
    elapsed = time.time() - t0
    full_report += f"\n\n⏱  {elapsed:.1f} сек | {n_ticks} тиков | {n_agents:,} агентов"
    if detail:
        full_report += " | режим: полный (detail=True)"
    else:
        full_report += " | режим: summary"

    print(full_report)

    if output_file:
        Path(output_file).write_text(full_report, encoding="utf-8")
        print(f"\n  Отчёт сохранён: {output_file}")

    return df_final, snapshots, tick_stats, all_action_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABM Миграция — Словакия")
    parser.add_argument("--env",       default="environment.json")
    parser.add_argument("--commuting", default="commuting_filtered_with_travel.csv")
    parser.add_argument("--agent_dist", default="agent_init_distributions.json")
    parser.add_argument("--agents",    type=int, default=70000)
    parser.add_argument("--ticks",     type=int, default=60)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--quiet",     action="store_true")
    parser.add_argument("--detail",    action="store_true", help="Полный отчёт с аудитом и доп. метриками")
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
