"""
run.py — точка входа для запуска симуляции через CLI или Google Colab.

Использование:
  python run.py                          # дефолт: 5000 агентов, 60 тиков
  python run.py --agents 3000 --ticks 24
  python run.py --env /path/to/environment.json --output report.txt

В Google Colab:
  !python run.py --ticks 60
  или импортируй напрямую:
    from run import run
    df_final, snapshots, stats = run(n_agents=5000, n_ticks=60)
"""

import argparse
import sys
import time
from pathlib import Path

# Позволяет запускать из Colab с другим путём
import os
SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(SIM_DIR))

from graph import build_graph, print_graph_summary
from agents import create_agents
from engine import run_simulation
from report import demographic_portrait, compare_snapshots


def run(
    env_path: str = "environment.json",
    n_agents: int = 5000,
    n_ticks: int = 60,
    seed: int = 42,
    output_file: str = None,
    verbose: bool = True,
) -> tuple:
    """
    Полный цикл симуляции.

    Returns:
        df_final, snapshots, tick_stats
    """
    t0 = time.time()

    # ── 1. Строим граф среды ──────────────────────────────────────────────────
    if verbose:
        print("\n[1/4] Загружаем среду и строим граф...")
    G = build_graph(env_path)
    if verbose:
        print_graph_summary(G)

    # ── 2. Создаём агентов ────────────────────────────────────────────────────
    if verbose:
        print(f"\n[2/4] Создаём агентов (n={n_agents:,}, seed={seed})...")
    df = create_agents(env_path, n_agents=n_agents, seed=seed)

    # Снимок начального состояния
    snapshot_ticks = [0, n_ticks // 2, n_ticks]

    # ── 3. Запускаем симуляцию ────────────────────────────────────────────────
    if verbose:
        print(f"\n[3/4] Запуск симуляции ({n_ticks} тиков = {n_ticks//12} лет {n_ticks%12} мес)...")
    df_final, snapshots, tick_stats = run_simulation(
        df, G,
        n_ticks=n_ticks,
        snapshot_ticks=snapshot_ticks,
        seed=seed,
        verbose=verbose,
    )

    # ── 4. Генерируем отчёт ───────────────────────────────────────────────────
    if verbose:
        print(f"\n[4/4] Генерируем отчёт...")

    report_parts = []

    # Портреты для каждого снимка
    snapshot_labels = {
        0: "НАЧАЛО",
        n_ticks // 2: f"СЕРЕДИНА (тик {n_ticks//2})",
        n_ticks: f"КОНЕЦ (тик {n_ticks})",
    }
    for t in sorted(snapshots.keys()):
        label = snapshot_labels.get(t, f"Тик {t}")
        portrait = demographic_portrait(snapshots[t], label=label, tick_num=t)
        report_parts.append(portrait)

    # Сравнительная таблица + миграция
    comparison = compare_snapshots(snapshots, tick_stats)
    report_parts.append(comparison)

    full_report = "\n\n".join(report_parts)

    elapsed = time.time() - t0
    footer = f"\n⏱  Время выполнения: {elapsed:.1f} сек | {n_ticks} тиков | {n_agents:,} агентов"
    full_report += footer

    # Вывод в консоль
    print(full_report)

    # Сохранение в файл
    if output_file:
        Path(output_file).write_text(full_report, encoding="utf-8")
        print(f"\n  Отчёт сохранён: {output_file}")

    return df_final, snapshots, tick_stats


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Симуляция миграции — Трнавский край")
    parser.add_argument("--env",     default="environment.json", help="Путь к environment.json")
    parser.add_argument("--agents",  type=int, default=5000,     help="Количество агентов")
    parser.add_argument("--ticks",   type=int, default=60,       help="Количество тиков (месяцев)")
    parser.add_argument("--seed",    type=int, default=42,       help="Random seed")
    parser.add_argument("--output",  default=None,               help="Файл для сохранения отчёта")
    parser.add_argument("--quiet",   action="store_true",        help="Без подробного вывода")
    args = parser.parse_args()

    run(
        env_path=args.env,
        n_agents=args.agents,
        n_ticks=args.ticks,
        seed=args.seed,
        output_file=args.output,
        verbose=not args.quiet,
    )
