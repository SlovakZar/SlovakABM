"""
lhs_runner.py — Фреймворк для LHS-тестирования параметров SlovakABM.

Использование:
  python lhs_runner.py --samples 100 --agents 5000 --ticks 36 --output lhs_results.csv

Или из кода:
  from lhs_runner import lhs_test
  results = lhs_test(n_samples=50, n_agents=5000, n_ticks=36)
"""

import json
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ── Путь к проекту ───────────────────────────────────────────────────────────
SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(SIM_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Загрузка спецификации параметров
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParamSpec:
    """Спецификация одного параметра для LHS."""
    name: str
    default: float
    low: float
    high: float
    category: str
    description: str = ""
    is_integer: bool = False


def load_param_specs(spec_path: str = "lhs_parameters.json") -> List[ParamSpec]:
    """Загружает спецификацию параметров из JSON, извлекая только float/int параметры."""
    p = Path(spec_path)
    if not p.exists():
        p = SIM_DIR / spec_path
    with open(p, encoding="utf-8") as f:
        spec = json.load(f)

    params = []
    # Рекурсивно обходим JSON, собираем только объекты с range/default
    def _walk(obj, category="", prefix="", leaf_key=None):
        if isinstance(obj, dict):
            if "range" in obj and "default" in obj and "type" in obj:
                ptype = obj["type"]
                if ptype in ("float", "int"):
                    rng = obj["range"]
                    name = leaf_key  # короткое имя без пути секций
                    params.append(ParamSpec(
                        name=name,
                        default=obj["default"],
                        low=rng[0],
                        high=rng[1],
                        category=category,
                        description=obj.get("description", ""),
                        is_integer=(ptype == "int"),
                    ))
            else:
                for k, v in obj.items():
                    if k.startswith("_"):
                        continue
                    next_cat = k if category == "" else f"{category}.{k}"
                    _walk(v, next_cat, f"{prefix}{k}.", leaf_key=k)

    _walk(spec)
    print(f"  Загружено {len(params)} параметров для LHS из {spec_path}")
    return params


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LHS-сэмплирование
# ═══════════════════════════════════════════════════════════════════════════════

def lhs_sample(params: List[ParamSpec], n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Генерирует LHS-матрицу: n_samples строк × len(params) столбцов.
    Каждый столбец — равномерное распределение в [low, high] параметра.
    """
    rng = np.random.default_rng(seed)
    n_params = len(params)
    
    # LHS: для каждого параметра — перестановка страт
    lhs = np.zeros((n_samples, n_params))
    for j in range(n_params):
        # Равномерные страты
        strata = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
        low, high = params[j].low, params[j].high
        lhs[:, j] = low + strata * (high - low)
        if params[j].is_integer:
            lhs[:, j] = np.round(lhs[:, j]).astype(int)

    return lhs


def sample_to_dict(params: List[ParamSpec], sample_row: np.ndarray) -> Dict[str, float]:
    """Преобразует строку LHS-матрицы в словарь {имя_параметра: значение}."""
    result = {}
    for j, p in enumerate(params):
        val = float(sample_row[j])
        if p.is_integer:
            val = int(round(val))
        result[p.name] = val
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Патчинг параметров в код симуляции
# ═══════════════════════════════════════════════════════════════════════════════

class ParamPatcher:
    """
    Подменяет константы в модулях engine, graph, signals, agents
    на переданный словарь значений.
    """

    # Маппинг: имя в LHS → (модуль, имя_константы_в_модуле)
    MAPPING = {
        # ── BARRIER_1 ─────────────────────────────────────────────────────
        "aspirations_alpha":             ("engine", "ASPIRATIONS_ALPHA"),
        "signal_decay":                  ("engine", "SIGNAL_DECAY"),
        "gap_adapt_lambda":              ("engine", "GAP_ADAPT_LAMBDA"),
        "sat_smoothing":                 ("engine", "SAT_SMOOTHING"),

        "hub_weak_ties_bonus":           ("engine", "HUB_WEAK_TIES_BONUS"),
        "move_weak_ties_penalty":        ("engine", "MOVE_WEAK_TIES_PENALTY"),

        # ── BARRIER_2 ─────────────────────────────────────────────────────
        "migration_pressure_p_min":      ("engine", "MIGRATION_PRESSURE_P_MIN"),
        "migration_pressure_p_max":      ("engine", "MIGRATION_PRESSURE_P_MAX"),
        "migration_pressure_divisor":    ("engine", "MIGRATION_PRESSURE_DIVISOR"),
        "pc_d_perceived_modifier":       ("engine", "PC_D_PERCEIVED_MODIFIER"),
        "migration_cooldown_ticks":      ("engine", "MIGRATION_COOLDOWN_TICKS"),
        "sb_move_total_ticks":           ("engine", "SB_MOVE_TOTAL_TICKS"),
        "social_boost_decay":            ("engine", "SOCIAL_BOOST_DECAY"),

        # ── DECAY (затухание динамических переменных) ──────────────────
        "decay_social_boost_move":        ("engine", "SB_MOVE_DECAY_PER_TICK"),
        "decay_inertia_mobility":         ("engine", "INERTIA_MOB_DECAY_PER_TICK"),
        "decay_econ_penalty":             ("engine", "ECON_PENALTY_DECAY_PER_TICK"),
        "jobloss_ramp_step":              ("engine", "JOBLOSS_RAMP_STEP"),

        # ── HEURISTIC_SEARCH ──────────────────────────────────────────────
        "max_jobs_pressure":             ("engine", "MAX_JOBS_PRESSURE"),
        "max_work_candidates":           ("engine", "MAX_WORK_CANDIDATES"),
        "housing_budget_ratio":          ("engine", "HOUSING_BUDGET_RATIO"),
        "move_stress_factor":            ("engine", "MOVE_STRESS_FACTOR"),
        "adapt_flex_threshold":          ("engine", "ADAPT_FLEX_THRESHOLD"),
        "adapt_sat_boost":               ("engine", "ADAPT_SAT_BOOST"),
        "commuter_gate_ref":             ("engine", "COMMUTER_GATE_REF"),
        "job_flex_gate_ref":             ("engine", "JOB_FLEX_GATE_REF"),
        "base_appetite_min":             ("engine", "BASE_APPETITE_MIN"),
        "base_appetite_max":             ("engine", "BASE_APPETITE_MAX"),
        "min_desired_raise":             ("engine", "MIN_DESIRED_RAISE"),
        "unemployed_wage_floor":         ("engine", "UNEMPLOYED_WAGE_FLOOR"),
        "unemployed_wage_ceil":          ("engine", "UNEMPLOYED_WAGE_CEIL"),

        # ── ENVIRONMENT (graph) ───────────────────────────────────────────
        "housing_alpha":                 ("graph", "HOUSING_ALPHA"),
        "wage_alpha":                    ("graph", "WAGE_ALPHA"),
        "national_avg_wage":             ("engine", "NATIONAL_AVG_WAGE"),
        "spillover_weight":              ("graph", "SPILLOVER_WEIGHT"),
        "agent_housing_footprint":       ("graph", "AGENT_HOUSING_FOOTPRINT"),
        "housing_remaining_floor":       ("graph", "HOUSING_REMAINING_FLOOR"),

        # ── AGENT_CREATION (Rogers-Castro — константы модуля agents) ──────
        "rogers_castro_a1":              ("agents", "RC_A1"),
        "rogers_castro_mu1":             ("agents", "RC_MU1"),
        "rogers_castro_alpha1":          ("agents", "RC_ALPHA1"),
        "rogers_castro_a2":              ("agents", "RC_A2"),
        "rogers_castro_mu2":             ("agents", "RC_MU2"),
        "rogers_castro_alpha2":          ("agents", "RC_ALPHA2"),
        "rogers_castro_c":               ("agents", "RC_C"),

        # ── SIGNAL_SYSTEM (патчим через пересоздание dispatcher) ─────────
        # Эти параметры не константы модуля, а параметры create_default_dispatcher().
        # Они обрабатываются отдельно — через пересоздание шины с новыми параметрами.
    }

    def __init__(self, param_dict: Dict[str, float]):
        self.param_dict = param_dict
        self._originals = {}  # для восстановления

    def apply(self) -> Dict:
        """Применяет параметры к модулям. Возвращает словарь сигнальных параметров."""
        import engine
        import graph
        import agents

        signal_params = {}

        for lhs_name, (mod_name, const_name) in self.MAPPING.items():
            if lhs_name not in self.param_dict:
                continue
            val = self.param_dict[lhs_name]

            if mod_name == "engine":
                if not hasattr(engine, const_name):
                    continue
                orig = getattr(engine, const_name)
                self._originals[(mod_name, const_name)] = orig
                setattr(engine, const_name, val)

            elif mod_name == "graph":
                if not hasattr(graph, const_name):
                    continue
                orig = getattr(graph, const_name)
                self._originals[(mod_name, const_name)] = orig
                setattr(graph, const_name, val)

            elif mod_name == "agents":
                if not hasattr(agents, const_name):
                    continue
                orig = getattr(agents, const_name)
                self._originals[(mod_name, const_name)] = orig
                setattr(agents, const_name, val)

        # Собираем сигнальные параметры отдельно
        signal_keys = [
            "social_boost_move", "social_boost_commute",
            "unemployed_signal", "neighbor_signal_coef",
            "inertia_mobility_penalty_move", "inertia_loss_jobloss",
            "econ_gap_jobloss", "place_deficit_penalty_move",
            "infra_bonus_delta", "social_boost_new_employer",
            "aspirations_closed_employer",
            "decay_social_boost_move", "decay_inertia_mobility",
            "decay_econ_penalty",
        ]
        for k in signal_keys:
            if k in self.param_dict:
                signal_params[k] = self.param_dict[k]

        return signal_params

    def restore(self):
        """Восстанавливает оригинальные значения констант."""
        import engine
        import graph
        import agents

        for (mod_name, const_name), orig_val in self._originals.items():
            if mod_name == "engine":
                setattr(engine, const_name, orig_val)
            elif mod_name == "graph":
                setattr(graph, const_name, orig_val)
            elif mod_name == "agents":
                setattr(agents, const_name, orig_val)
        self._originals.clear()


def create_patched_dispatcher(signal_params: Dict[str, float]):
    """
    Создаёт Dispatcher с переопределёнными параметрами сигнальной системы.
    Копирует create_default_dispatcher() из signals.py, подставляя значения.
    """
    from signals import (
        Dispatcher, EventType, Rule,
        SCOPE_SELF, SCOPE_RESIDENCE_NEIGHBORS, SCOPE_TARGET_NEIGHBORS,
        SCOPE_WORKPLACE_COLLEAGUES, SCOPE_SAME_INDUSTRY_DISTRICT,
        SCOPE_SAME_SETTLEMENT_TYPE, SCOPE_WHOLE_REGION,
    )

    sp = signal_params

    social_boost_move          = sp.get("social_boost_move", 0.06)
    social_boost_commute       = sp.get("social_boost_commute", 0.02)
    unemployed_signal          = sp.get("unemployed_signal", 0.35)
    neighbor_signal_coef       = sp.get("neighbor_signal_coef", 0.04)
    inertia_mob_pen_move       = sp.get("inertia_mobility_penalty_move", 0.06)
    inertia_loss_jobloss       = sp.get("inertia_loss_jobloss", -0.25)
    econ_gap_jobloss           = sp.get("econ_gap_jobloss", 0.25)
    place_deficit_pen_move     = sp.get("place_deficit_penalty_move", 0.03)
    infra_bonus_delta          = sp.get("infra_bonus_delta", 0.05)
    social_boost_new_employer  = sp.get("social_boost_new_employer", 0.05)
    aspirations_closed_employer = sp.get("aspirations_closed_employer", 0.08)

    d = Dispatcher()

    # AGENT_MOVED
    d.add_rule(Rule(EventType.AGENT_MOVED, SCOPE_RESIDENCE_NEIGHBORS, "social_boost", base_delta=social_boost_move))
    # Отрицательный знак: переезд соседа понижает инерцию, делая миграцию более вероятной
    d.add_rule(Rule(EventType.AGENT_MOVED, SCOPE_RESIDENCE_NEIGHBORS, "inertia_mobility_penalty",
                    base_delta=-inertia_mob_pen_move, clip_min=-1.0, clip_max=1.0))
    d.add_rule(Rule(EventType.AGENT_MOVED, SCOPE_TARGET_NEIGHBORS, "social_boost", base_delta=social_boost_move))
    d.add_rule(Rule(EventType.AGENT_MOVED, SCOPE_SAME_SETTLEMENT_TYPE, "place_deficit_penalty",
                    base_delta=place_deficit_pen_move, motivation="place", delay_ticks=1, clip_min=0.0, clip_max=5.0))
    d.add_rule(Rule(EventType.AGENT_MOVED, SCOPE_RESIDENCE_NEIGHBORS, "signal_reduction",
                    base_delta=neighbor_signal_coef, scale_by_field="net_signal_susc"))
    # v3: soc_calibration_signal соседям при AGENT_MOVED
    d.add_rule(Rule(EventType.AGENT_MOVED, SCOPE_RESIDENCE_NEIGHBORS, "soc_calibration_signal",
                    base_delta=0.04, scale_by_field="net_signal_susc", clip_min=0.0, clip_max=1.0))
    # v3: AGENT_MOVED (economic) → econ_penalty низкообразованным соседям той же отрасли
    d.add_rule(Rule(EventType.AGENT_MOVED, SCOPE_RESIDENCE_NEIGHBORS, "econ_penalty",
                    base_delta=0.05, motivation="economic", filter_education="low",
                    filter_same_industry=True, clip_min=0.0, clip_max=0.5))

    # AGENT_COMMUTE_STARTED
    d.add_rule(Rule(EventType.AGENT_COMMUTE_STARTED, SCOPE_RESIDENCE_NEIGHBORS, "social_boost", base_delta=social_boost_commute))
    # v3: soc_calibration_signal соседям при AGENT_COMMUTE_STARTED
    d.add_rule(Rule(EventType.AGENT_COMMUTE_STARTED, SCOPE_RESIDENCE_NEIGHBORS, "soc_calibration_signal",
                    base_delta=0.02, scale_by_field="net_signal_susc", clip_min=0.0, clip_max=1.0))

    # JOB_CHANGED
    d.add_rule(Rule(EventType.JOB_CHANGED, SCOPE_WORKPLACE_COLLEAGUES, "social_boost", base_delta=social_boost_move * 0.8))
    # v3: soc_calibration_signal коллегам при JOB_CHANGED
    d.add_rule(Rule(EventType.JOB_CHANGED, SCOPE_WORKPLACE_COLLEAGUES, "soc_calibration_signal",
                    base_delta=0.03, scale_by_field="net_signal_susc", clip_min=0.0, clip_max=1.0))
    # v3: JOB_CHANGED → econ_penalty низкообразованным коллегам той же отрасли
    d.add_rule(Rule(EventType.JOB_CHANGED, SCOPE_WORKPLACE_COLLEAGUES, "econ_penalty",
                    base_delta=0.03, filter_education="low",
                    filter_same_industry=True, clip_min=0.0, clip_max=0.5))

    # LOST_JOB
    d.add_rule(Rule(EventType.LOST_JOB, SCOPE_SELF, "inertia", base_delta=inertia_loss_jobloss, clip_min=0.05, clip_max=0.95))
    d.add_rule(Rule(EventType.LOST_JOB, SCOPE_SELF, "econ_gap", base_delta=econ_gap_jobloss, clip_min=0.0, clip_max=1.0))
    d.add_rule(Rule(EventType.LOST_JOB, SCOPE_SELF, "signal_reduction", base_delta=unemployed_signal))
    d.add_rule(Rule(EventType.LOST_JOB, SCOPE_SELF, "intention_state", mode="set", value="seeking_work"))
    d.add_rule(Rule(EventType.LOST_JOB, SCOPE_WORKPLACE_COLLEAGUES, "inertia_mobility_penalty", base_delta=0.02, clip_min=0.0, clip_max=1.0))

    # NEW_EMPLOYER
    d.add_rule(Rule(EventType.NEW_EMPLOYER, SCOPE_WHOLE_REGION, "social_boost", base_delta=social_boost_new_employer))
    # v3: soc_calibration_signal всему региону при NEW_EMPLOYER
    d.add_rule(Rule(EventType.NEW_EMPLOYER, SCOPE_WHOLE_REGION, "soc_calibration_signal",
                    base_delta=0.03, scale_by_field="net_signal_susc", clip_min=0.0, clip_max=1.0))
    # v3: econ_penalty той же отрасли в том же районе при NEW_EMPLOYER (wage_pressure>1)
    d.add_rule(Rule(EventType.NEW_EMPLOYER, SCOPE_SAME_INDUSTRY_DISTRICT, "econ_penalty",
                    base_delta=0.02, filter_wage_pressure=True, clip_min=0.0, clip_max=1.0))

    # CLOSED_EMPLOYER
    d.add_rule(Rule(EventType.CLOSED_EMPLOYER, SCOPE_WHOLE_REGION, "aspirations", base_delta=aspirations_closed_employer, filter_status="employed"))
    # v3: soc_calibration_signal снижается в регионе при CLOSED_EMPLOYER
    d.add_rule(Rule(EventType.CLOSED_EMPLOYER, SCOPE_WHOLE_REGION, "soc_calibration_signal",
                    base_delta=-0.03, scale_by_field="net_signal_susc", clip_min=0.0, clip_max=1.0))
    # v3: econ_penalty сброс той же отрасли в том же районе при CLOSED_EMPLOYER
    d.add_rule(Rule(EventType.CLOSED_EMPLOYER, SCOPE_SAME_INDUSTRY_DISTRICT, "econ_penalty",
                    mode="set", value=0.0, filter_wage_pressure=True))

    # NEW_INFRA / CLOSED_INFRA
    d.add_rule(Rule(EventType.NEW_INFRA, SCOPE_RESIDENCE_NEIGHBORS, "infra_bonus", base_delta=infra_bonus_delta, clip_min=-1.0, clip_max=1.0))
    d.add_rule(Rule(EventType.CLOSED_INFRA, SCOPE_RESIDENCE_NEIGHBORS, "infra_bonus", base_delta=-infra_bonus_delta, clip_min=-1.0, clip_max=1.0))

    # GRADUATED
    d.add_rule(Rule(EventType.GRADUATED, SCOPE_SELF, "intention_state", mode="set", value="seeking_work"))

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Сбор выходных метрик
# ═══════════════════════════════════════════════════════════════════════════════

def collect_metrics(df_final: pd.DataFrame, snapshots: Dict,
                    tick_stats: list, all_action_log: list) -> Dict[str, float]:
    """Собирает ключевые метрики из финального состояния симуляции."""

    df = df_final.copy()
    employed = df[df["wage"] > 0]
    unemployed = df[df["status"] == "unemployed"]
    total = len(df)

    metrics = {}

    # Занятость
    metrics["unemployment_rate"] = len(unemployed) / total if total > 0 else 0.0
    metrics["n_commuters"] = int((df["status"] == "commute").sum())
    metrics["commuter_share"] = metrics["n_commuters"] / total if total > 0 else 0.0
    metrics["n_stay"] = int((df["status"] == "stay").sum())

    # Зарплаты
    if len(employed) > 0:
        metrics["avg_wage_employed"] = float(employed["wage"].mean())
        metrics["median_wage_employed"] = float(employed["wage"].median())
        metrics["wage_q25"] = float(employed["wage"].quantile(0.25))
        metrics["wage_q75"] = float(employed["wage"].quantile(0.75))
        metrics["wage_std"] = float(employed["wage"].std())
    else:
        metrics["avg_wage_employed"] = 0.0
        metrics["median_wage_employed"] = 0.0
        metrics["wage_q25"] = 0.0
        metrics["wage_q75"] = 0.0
        metrics["wage_std"] = 0.0

    # Satisfaction
    for col in ["sat_economic", "sat_social", "sat_family", "sat_place"]:
        if col in df.columns:
            metrics[f"avg_{col}"] = float(df[col].mean())

    # Dissatisfaction
    if "aspirations" in df.columns:
        metrics["avg_aspirations"] = float(df["aspirations"].mean())
    if "signal_reduction" in df.columns:
        metrics["avg_signal_reduction"] = float(df["signal_reduction"].mean())

    # TPB
    if "tpb_active" in df.columns:
        metrics["tpb_active_share"] = float(df["tpb_active"].mean())

    # ── ИНЕРЦИЯ (inertia) ────────────────────────────────────────────────
    if "inertia" in df.columns:
        metrics["avg_inertia"] = float(df["inertia"].mean())
    if "inertia_social" in df.columns:
        metrics["avg_inertia_social"] = float(df["inertia_social"].mean())
    if "inertia_mobility_penalty" in df.columns:
        metrics["avg_inertia_mob_penalty"] = float(df["inertia_mobility_penalty"].mean())

    # ── НЕДОВОЛЬСТВО (dissatisfaction / gaps) ────────────────────────────
    if "migration_pressure" in df.columns:
        metrics["avg_migration_pressure"] = float(df["migration_pressure"].mean())
    if "econ_gap" in df.columns:
        metrics["avg_econ_gap"] = float(df["econ_gap"].mean())
    if "place_deficit_penalty" in df.columns:
        metrics["avg_place_deficit_penalty"] = float(df["place_deficit_penalty"].mean())
    if "signal_reduction" in df.columns:
        metrics["avg_signal_reduction"] = float(df["signal_reduction"].mean())

    # ── ВОЗМОЖНОСТИ (capabilities / perceived_control / weak_ties) ───────
    if "perceived_control" in df.columns:
        metrics["avg_perceived_control"] = float(df["perceived_control"].mean())
    if "econ_perceived_control" in df.columns:
        metrics["avg_econ_perceived_control"] = float(df["econ_perceived_control"].mean())
    if "weak_ties_utility" in df.columns:
        metrics["avg_weak_ties_utility"] = float(df["weak_ties_utility"].mean())
    if "info_quality" in df.columns:
        metrics["avg_info_quality"] = float(df["info_quality"].mean())
    if "job_flexibility" in df.columns:
        metrics["avg_job_flexibility"] = float(df["job_flexibility"].mean())
    if "internal_mig_thr" in df.columns:
        metrics["avg_internal_mig_thr"] = float(df["internal_mig_thr"].mean())

    # ── ЖЕЛАНИЯ (aspirations / domain_future_place / social_boost) ───────
    if "domain_future_place" in df.columns:
        metrics["avg_domain_future_place"] = float(df["domain_future_place"].mean())
    if "social_boost" in df.columns:
        metrics["avg_social_boost"] = float(df["social_boost"].mean())
    if "soc_calibration_signal" in df.columns:
        metrics["avg_soc_calibration_signal"] = float(df["soc_calibration_signal"].mean())

    # Сводный dissatisfaction (среднее по 4 доменам)
    sat_cols = ["sat_economic", "sat_social", "sat_family", "sat_place"]
    thr_cols = ["thr_economic", "thr_social", "thr_family", "thr_place"]
    w_cols   = ["w_economic", "w_social", "w_family", "w_future"]
    dissat_vals = []
    for sv, tv, wv in zip(sat_cols, thr_cols, w_cols):
        if sv in df.columns and tv in df.columns and wv in df.columns:
            gap = np.maximum(0, df[tv].values - df[sv].values) / np.maximum(df[tv].values, 0.01)
            dissat_vals.append((df[wv].values * gap) ** 2)
    if dissat_vals:
        combined = np.clip(np.sqrt(sum(dissat_vals)), 0.0, 1.0)
        metrics["avg_dissatisfaction_weighted"] = float(combined.mean())

    # Динамические переменные v2
    for col in ["econ_penalty", "infra_bonus", "inertia_mobility_penalty", "jobloss_econ_gap_bonus"]:
        if col in df.columns:
            metrics[f"avg_{col}"] = float(df[col].mean())

    # Региональный баланс
    if "region" in df.columns:
        region_counts = df["region"].value_counts(normalize=True)
        metrics["regional_population_gini"] = float(
            1.0 - (region_counts ** 2).sum()  # упрощённый Simpson = 1 - Σ(pᵢ²)
        )
        if "wage" in df.columns:
            region_wages = df[df["wage"] > 0].groupby("region")["wage"].mean()
            metrics["regional_wage_std"] = float(region_wages.std()) if len(region_wages) > 1 else 0.0

    # Статистика действий из лога
    n_moved_econ = sum(1 for a in all_action_log if a.get("decision") in ("move", "satellite_move")
                       and a.get("activation_domain") == "economic")
    n_moved_place = sum(1 for a in all_action_log if a.get("decision") in ("move", "satellite_move")
                        and a.get("activation_domain") == "place")
    n_commuted = sum(1 for a in all_action_log if a.get("decision") == "commute")
    n_adapted = sum(1 for a in all_action_log if a.get("decision") == "adapt")
    n_stay_dec = sum(1 for a in all_action_log if a.get("decision") == "stay")

    metrics["n_moved_economic"] = n_moved_econ
    metrics["n_moved_place"] = n_moved_place
    metrics["n_commute_started"] = n_commuted
    metrics["n_adapted"] = n_adapted
    metrics["n_stay_decision"] = n_stay_dec
    metrics["n_actions_total"] = len(all_action_log)

    # Тренды по тикам
    if tick_stats:
        last_tick = tick_stats[-1]
        metrics["jobs_pressure_max_final"] = float(last_tick.get("jobs_pressure_max", 0.0))
        metrics["avg_dissat_final"] = float(last_tick.get("avg_dissat", 0.0))
        metrics["n_unemployed_final"] = int(last_tick.get("n_unemployed", 0))
        metrics["avg_inertia_final"] = float(last_tick.get("avg_inertia", 0.0))
        metrics["avg_aspirations_final"] = float(last_tick.get("avg_aspirations", 0.0))
        metrics["avg_signal_red_final"] = float(last_tick.get("avg_signal_red", 0.0))
        metrics["avg_age_final"] = float(last_tick.get("avg_age", 0.0))
        metrics["avg_wage_final"] = float(last_tick.get("avg_wage", 0.0))
        # Средние за всю симуляцию
        metrics["avg_inertia_sim"] = float(np.mean([s.get("avg_inertia", 0.0) for s in tick_stats]))
        metrics["avg_dissat_sim"] = float(np.mean([s.get("avg_dissat", 0.0) for s in tick_stats]))
        metrics["avg_aspirations_sim"] = float(np.mean([s.get("avg_aspirations", 0.0) for s in tick_stats]))

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Главный цикл LHS
# ═══════════════════════════════════════════════════════════════════════════════

def lhs_test(
    n_samples: int = 50,
    n_agents: int = 5000,
    n_ticks: int = 36,
    seed: int = 42,
    spec_path: str = "lhs_parameters.json",
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Запускает LHS-тестирование параметров.

    Args:
        n_samples: число LHS-выборок
        n_agents: число агентов на прогон (рекомендуется 3000-10000 для скорости)
        n_ticks: число тиков на прогон (рекомендуется 24-48)
        seed: базовый seed
        spec_path: путь к JSON-спецификации параметров
        output_csv: путь для сохранения результатов (опционально)
        verbose: выводить прогресс

    Returns:
        DataFrame со всеми результатами: строки = прогоны, столбцы = параметры + метрики
    """
    from graph import build_graph
    from agents import create_agents, JOBS_CAPACITY, INDUSTRY_JOBS_CAPACITY
    from engine import run_simulation
    from signals import EventBus
    from graph import sync_industry_jobs_to_graph

    t_start = time.time()

    # Загружаем спецификацию
    params = load_param_specs(spec_path)
    if verbose:
        print(f"\nПараметров для варьирования: {len(params)}")
        for p in params:
            print(f"  {p.name}: [{p.low}, {p.high}] (default={p.default})")

    # Генерируем LHS-матрицу
    lhs_matrix = lhs_sample(params, n_samples, seed=seed)
    if verbose:
        print(f"\nLHS-матрица: {lhs_matrix.shape}")

    # Предварительно строим граф (он не зависит от параметров)
    if verbose:
        print("\nСтроим граф Словакии (однократно)...")
    G = build_graph(
        str(SIM_DIR / "environment.json"),
        str(SIM_DIR / "commuting_filtered_with_travel.csv"),
    )
    if verbose:
        print(f"  Узлов: {G.number_of_nodes()}, Рёбер: {G.number_of_edges()}")

    # Загружаем init_dists для graduation
    dist_path = SIM_DIR / "agent_init_distributions.json"
    with open(dist_path, encoding="utf-8") as f:
        init_dists = json.load(f).get("districts", {})

    all_results = []

    for run_idx in range(n_samples):
        run_seed = seed
        t_run = time.time()

        # Параметры этого прогона
        param_dict = sample_to_dict(params, lhs_matrix[run_idx])

        # Патчим константы
        patcher = ParamPatcher(param_dict)
        signal_params = patcher.apply()

        # Создаём агентов
        df = create_agents(
            str(SIM_DIR / "agent_init_distributions.json"),
            str(SIM_DIR / "agent_params_from_survey.json"),
            str(SIM_DIR / "commuting_filtered_with_travel.csv"),
            n_agents=n_agents,
            seed=run_seed,
        )

        # v3: Синхронизируем industry_jobs (occupied+vacant) и jobs_capacity
        sync_industry_jobs_to_graph(G, INDUSTRY_JOBS_CAPACITY, JOBS_CAPACITY)

        # Создаём шину с кастомным dispatcher
        dispatcher = create_patched_dispatcher(signal_params)
        bus = EventBus(dispatcher=dispatcher)

        # Запускаем симуляцию
        snapshot_ticks = [0, n_ticks // 2, n_ticks]
        df_final, snapshots, tick_stats, all_action_log = run_simulation(
            df, G,
            n_ticks=n_ticks,
            snapshot_ticks=snapshot_ticks,
            seed=run_seed,
            verbose=False,
            jobs_capacity=JOBS_CAPACITY,
            init_dists=init_dists,
            bus=bus,
            scenario=None,
        )

        # Собираем метрики
        metrics = collect_metrics(df_final, snapshots, tick_stats, all_action_log)

        # Объединяем параметры и метрики
        row = {**param_dict, **metrics, "run": run_idx, "seed": run_seed}
        all_results.append(row)

        # Восстанавливаем константы
        patcher.restore()

        elapsed_run = time.time() - t_run
        if verbose:
            unemp = metrics.get("unemployment_rate", 0)
            avg_w = metrics.get("avg_wage_employed", 0)
            moves = metrics.get("n_moved_economic", 0) + metrics.get("n_moved_place", 0)
            print(f"  [{run_idx+1:4d}/{n_samples}] "
                  f"unemp={unemp:.3f}  wage={avg_w:.0f}  moves={moves:4d}  "
                  f"time={elapsed_run:.1f}s")

    # Формируем итоговый DataFrame
    results_df = pd.DataFrame(all_results)

    total_elapsed = time.time() - t_start
    if verbose:
        print(f"\n{'='*60}")
        print(f"LHS завершён: {n_samples} прогонов за {total_elapsed:.0f} сек "
              f"({total_elapsed/n_samples:.1f} с/прогон)")
        print(f"Метрик собрано: {len([c for c in results_df.columns if c not in ('run', 'seed')])}")

    # Сохраняем
    if output_csv:
        results_df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Результаты сохранены: {output_csv}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Анализ чувствительности
# ═══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(results_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Быстрый корреляционный анализ чувствительности:
    корреляция Пирсона каждого параметра с каждой выходной метрикой.
    """
    param_cols = [c for c in results_df.columns
                  if c not in ("run", "seed") and not c.startswith(("n_", "avg_", "unemployment", "commuter", "wage_", "median", "regional", "jobs_pressure", "tpb_"))]
    metric_cols = [c for c in results_df.columns if c not in param_cols and c not in ("run", "seed")]

    # Строим матрицу корреляций: параметры × метрики
    corr_rows = []
    for pcol in param_cols:
        for mcol in metric_cols:
            corr = results_df[pcol].corr(results_df[mcol])
            if not np.isnan(corr):
                corr_rows.append({
                    "parameter": pcol,
                    "metric": mcol,
                    "pearson_r": round(corr, 4),
                })

    corr_df = pd.DataFrame(corr_rows)
    corr_df["abs_r"] = corr_df["pearson_r"].abs()
    corr_df = corr_df.sort_values("abs_r", ascending=False).head(top_n)
    return corr_df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LHS-тестирование параметров SlovakABM")
    parser.add_argument("--samples", type=int, default=50, help="Число LHS-выборок (прогонов)")
    parser.add_argument("--agents", type=int, default=5000, help="Агентов на прогон")
    parser.add_argument("--ticks", type=int, default=36, help="Тиков на прогон")
    parser.add_argument("--seed", type=int, default=42, help="Базовый seed")
    parser.add_argument("--output", default="lhs_results.csv", help="CSV для результатов")
    parser.add_argument("--spec", default="lhs_parameters.json", help="JSON-спецификация параметров")
    parser.add_argument("--sensitivity", action="store_true", help="Показать топ корреляций после прогона")
    args = parser.parse_args()

    results = lhs_test(
        n_samples=args.samples,
        n_agents=args.agents,
        n_ticks=args.ticks,
        seed=args.seed,
        spec_path=args.spec,
        output_csv=args.output,
    )

    if args.sensitivity and len(results) > 0:
        print("\nАнализ чувствительности (топ-20 корреляций):")
        sens = sensitivity_analysis(results)
        print(sens.to_string(index=False))
