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
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Dict, List, Optional
from dataclasses import dataclass

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
    18 ядерных комбинаций + 9 специализированных.
    """
    fixed = spec["fixed"]
    core = spec["grid"]["core"]
    specialized = spec["grid"]["specialized"]

    plans = []

    # ── Ядерная сетка: 3×3×2 = 18 комбинаций ───────────────────────────
    core_keys = list(core.keys())
    core_values = [core[k] for k in core_keys]

    for combo in product(*core_values):
        plan = dict(fixed)
        for k, v in zip(core_keys, combo):
            plan[k] = v
        # Специализированные — на дефолтах (из fixed уже есть)
        plan["_group"] = "core"
        plans.append(plan)

    # ── Специализированные: варьируем по одному, ядерные на дефолтах ───
    core_defaults = {
        "base_appetite_min": 0.10,
        "social_boost_move": 0.06,
        "max_work_candidates": 12,
    }
    for spec_name, spec_values in specialized.items():
        for val in spec_values:
            plan = dict(fixed)
            plan.update(core_defaults)
            plan[spec_name] = val
            plan["_group"] = f"specialized:{spec_name}"
            plans.append(plan)

    return plans


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Подмена параметров (переиспользуем ParamPatcher из lhs_runner)
# ═══════════════════════════════════════════════════════════════════════════════

from lhs_runner import ParamPatcher, create_patched_dispatcher


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Таблица по 79 районам
# ═══════════════════════════════════════════════════════════════════════════════

# Карта: район → регион (8 краёв) — копия из agents.py для автономности
DISTRICT_TO_REGION = {
    "District of Bratislava I": "BA", "District of Bratislava II": "BA",
    "District of Bratislava III": "BA", "District of Bratislava IV": "BA",
    "District of Bratislava V": "BA", "District of Malacky": "BA",
    "District of Pezinok": "BA", "District of Senec": "BA",
    "District of Trnava": "TT", "District of Dunajská Streda": "TT",
    "District of Galanta": "TT", "District of Hlohovec": "TT",
    "District of Piešťany": "TT", "District of Senica": "TT",
    "District of Skalica": "TT",
    "District of Trenčín": "TN", "District of Bánovce nad Bebravou": "TN",
    "District of Ilava": "TN", "District of Myjava": "TN",
    "District of Nové Mesto nad Váhom": "TN", "District of Partizánske": "TN",
    "District of Považská Bystrica": "TN", "District of Púchov": "TN",
    "District of Prievidza": "TN",
    "District of Nitra": "NR", "District of Komárno": "NR",
    "District of Levice": "NR", "District of Nové Zámky": "NR",
    "District of Šaľa": "NR", "District of Topoľčany": "NR",
    "District of Zlaté Moravce": "NR",
    "District of Žilina": "ZA", "District of Bytča": "ZA",
    "District of Čadca": "ZA", "District of Dolný Kubín": "ZA",
    "District of Kysucké Nové Mesto": "ZA", "District of Liptovský Mikuláš": "ZA",
    "District of Martin": "ZA", "District of Námestovo": "ZA",
    "District of Ružomberok": "ZA", "District of Turčianske Teplice": "ZA",
    "District of Tvrdošín": "ZA",
    "District of Banská Bystrica": "BB", "District of Banská Štiavnica": "BB",
    "District of Brezno": "BB", "District of Detva": "BB",
    "District of Krupina": "BB", "District of Lučenec": "BB",
    "District of Poltár": "BB", "District of Revúca": "BB",
    "District of Rimavská Sobota": "BB", "District of Veľký Krtíš": "BB",
    "District of Zvolen": "BB", "District of Žiar nad Hronom": "BB",
    "District of Žarnovica": "BB",
    "District of Prešov": "PO", "District of Bardejov": "PO",
    "District of Humenné": "PO", "District of Kežmarok": "PO",
    "District of Levoča": "PO", "District of Medzilaborce": "PO",
    "District of Poprad": "PO", "District of Sabinov": "PO",
    "District of Snina": "PO", "District of Stará Ľubovňa": "PO",
    "District of Stropkov": "PO", "District of Svidník": "PO",
    "District of Vranov nad Topľou": "PO",
    "District of Košice I": "KE", "District of Košice II": "KE",
    "District of Košice III": "KE", "District of Košice IV": "KE",
    "District of Košice-okolie": "KE", "District of Gelnica": "KE",
    "District of Rožňava": "KE", "District of Sobrance": "KE",
    "District of Spišská Nová Ves": "KE", "District of Trebišov": "KE",
    "District of Michalovce": "KE",
    # Альтернативные написания
    "District of Košice - okolie": "KE",
    "District of Śaľa": "NR",
}


def build_district_table(
    snapshots: dict,
    run_id: int,
    run_label: str,
    fixed_params: Dict[str, float],
    variant_params: Dict[str, float],
    all_action_log: List[dict],
    district_col: str = "district",
) -> pd.DataFrame:
    """
    Строит таблицу: один район = одна строка, столбцы:
      district, region, pop_tick0, pop_tickN, delta_abs, delta_pct,
      n_moved_in, n_moved_out, net_flow,
      run_id, run_label, ...параметры...
    """
    ticks = sorted(snapshots.keys())
    if len(ticks) < 2:
        return pd.DataFrame()

    t0, tN = ticks[0], ticks[-1]
    df0 = snapshots[t0]
    dfN = snapshots[tN]

    # Население по районам
    pop0 = df0[district_col].value_counts()
    popN = dfN[district_col].value_counts()

    # Потоки из лога действий
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

    all_districts = sorted(set(pop0.index) | set(popN.index))
    rows = []
    for d in all_districts:
        p0 = pop0.get(d, 0)
        pN = popN.get(d, 0)
        delta_abs = pN - p0
        delta_pct = round(delta_abs / max(p0, 1) * 100, 2)
        region = DISTRICT_TO_REGION.get(d, "??")

        rows.append({
            "district": d,
            "region": region,
            "pop_tick0": p0,
            f"pop_tick{tN}": pN,
            "delta_abs": delta_abs,
            "delta_pct": delta_pct,
            "n_moved_in": moved_in.get(d, 0),
            "n_moved_out": moved_out.get(d, 0),
            "net_flow": moved_in.get(d, 0) - moved_out.get(d, 0),
            "run_id": run_id,
            "run_label": run_label,
            **{f"p_{k}": v for k, v in variant_params.items()},
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Сбор метрик (аналог collect_metrics из lhs_runner)
# ═══════════════════════════════════════════════════════════════════════════════

from lhs_runner import collect_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Главный цикл
# ═══════════════════════════════════════════════════════════════════════════════

def grid_run(
    n_agents: int = 5000,
    n_ticks: int = 36,
    seed: int = 42,
    spec_path: str = "grid_parameters.json",
    output_prefix: str = "grid_results",
    verbose: bool = True,
):
    from graph import build_graph
    from agents import create_agents, JOBS_CAPACITY, INDUSTRY_JOBS_CAPACITY
    from engine import run_simulation
    from signals import EventBus
    from graph import sync_industry_jobs_to_graph, initialize_industry_pressure_from_agents

    t_start = time.time()

    # Загружаем план
    spec = load_grid_spec(spec_path)
    plans = build_grid_plan(spec)
    n_runs = len(plans)
    fixed = spec["fixed"]

    if verbose:
        n_core = sum(1 for p in plans if p["_group"] == "core")
        n_spec = n_runs - n_core
        print(f"План прогонов: {n_core} ядерных + {n_spec} специализированных = {n_runs} всего")
        print(f"Агентов: {n_agents:,}  |  Тиков: {n_ticks}  |  Seed: {seed}\n")

    # Однократно строим граф
    if verbose:
        print("Строим граф Словакии...")
    G = build_graph(
        str(SIM_DIR / "environment.json"),
        str(SIM_DIR / "commuting_filtered_with_travel.csv"),
    )

    # Загружаем init_dists
    dist_path = SIM_DIR / "agent_init_distributions.json"
    with open(dist_path, encoding="utf-8") as f:
        init_dists = json.load(f).get("districts", {})

    all_metrics = []
    all_district_rows = []

    for run_idx, plan in enumerate(plans):
        group = plan.pop("_group")
        t_run = time.time()

        # Разделяем фиксированные и варьируемые
        variant = {k: v for k, v in plan.items() if k not in fixed or plan[k] != fixed.get(k)}
        # Определяем, какие параметры реально варьируются
        variant_names = sorted(variant.keys())

        # Патчим
        patcher = ParamPatcher(plan)
        signal_params = patcher.apply()

        # Создаём агентов
        df = create_agents(
            str(SIM_DIR / "agent_init_distributions.json"),
            str(SIM_DIR / "agent_params_from_survey.json"),
            str(SIM_DIR / "commuting_filtered_with_travel.csv"),
            n_agents=n_agents,
            seed=seed,
        )

        sync_industry_jobs_to_graph(G, INDUSTRY_JOBS_CAPACITY, JOBS_CAPACITY)
        initialize_industry_pressure_from_agents(G, df)

        dispatcher = create_patched_dispatcher(signal_params)
        bus = EventBus(dispatcher=dispatcher)

        # Снимки: тик 0, каждый 6-й тик, последний тик
        snapshot_ticks = sorted(set([0] + list(range(6, n_ticks, 6)) + [n_ticks]))
        df_final, snapshots, tick_stats, all_action_log = run_simulation(
            df, G,
            n_ticks=n_ticks,
            snapshot_ticks=snapshot_ticks,
            seed=seed,
            verbose=False,
            jobs_capacity=JOBS_CAPACITY,
            init_dists=init_dists,
            bus=bus,
            scenario=None,
        )

        # Метрики
        metrics = collect_metrics(df_final, snapshots, tick_stats, all_action_log)

        # Формируем метку прогона
        if group == "core":
            label = f"core_bamin{plan['base_appetite_min']}_sbm{plan['social_boost_move']}_mwc{plan['max_work_candidates']}"
        else:
            label = group.replace(":", "_")

        row = {
            "run_id": run_idx,
            "run_label": label,
            "group": group,
            **{f"v_{k}": v for k, v in variant.items()},
            **metrics,
        }
        all_metrics.append(row)

        # Таблица по районам
        dist_table = build_district_table(
            snapshots, run_idx, label, fixed, variant, all_action_log
        )
        all_district_rows.append(dist_table)

        # Восстанавливаем
        patcher.restore()

        elapsed = time.time() - t_run
        if verbose:
            moves = metrics.get("n_moved_economic", 0) + metrics.get("n_moved_place", 0)
            commutes = metrics.get("n_commute_started", 0)
            print(f"  [{run_idx+1:3d}/{n_runs}] {label:<55} "
                  f"moves={moves:5d}  commutes={commutes:4d}  "
                  f"time={elapsed:.1f}s")

    # ── Сохраняем ──────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)

    # Переставляем колонки: run_*, group, v_*, затем метрики
    meta_cols = [c for c in metrics_df.columns if c.startswith(("run_", "group", "v_"))]
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
    parser.add_argument("--agents", type=int, default=5000, help="Агентов на прогон")
    parser.add_argument("--ticks", type=int, default=36, help="Тиков на прогон")
    parser.add_argument("--seed", type=int, default=42, help="Базовый seed")
    parser.add_argument("--output", default="grid_results", help="Префикс выходных файлов")
    parser.add_argument("--spec", default="grid_parameters.json", help="JSON-спецификация сетки")
    args = parser.parse_args()

    grid_run(
        n_agents=args.n_agents,
        n_ticks=args.ticks,
        seed=args.seed,
        spec_path=args.spec,
        output_prefix=args.output,
    )
