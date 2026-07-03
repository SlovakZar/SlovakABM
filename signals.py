"""
signals.py — Event signal system for ABM Migration Slovakia.

Architecture:
  Event       — atomic event (agent or environmental)
  EventBus    — bus with two queues (pending + scheduled) and Dispatcher
  Rule        — propagation rule: whom, what to change, with what strength
  Dispatcher  — rule table event_type → list[Rule]

Imports only pandas and numpy. Does not depend on engine, agents, graph.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1. Event types
# ══════════════════════════════════════════════════════════════════════════════

class EventType(str, enum.Enum):
    """Event types in the system (v2 — signal system)."""
    # Agent
    AGENT_MOVED          = "AGENT_MOVED"
    AGENT_COMMUTE_STARTED = "AGENT_COMMUTE_STARTED"
    JOB_CHANGED          = "JOB_CHANGED"
    JOB_LOST             = "JOB_LOST"           # deprecated → LOST_JOB
    GRADUATED            = "GRADUATED"
    ADAPTED              = "ADAPTED"

    # Agent v2
    LOST_JOB             = "LOST_JOB"           # job loss (agent, not scenario)

    # Environmental (scenario)
    FACTORY_CLOSED       = "FACTORY_CLOSED"     # deprecated → CLOSED_EMPLOYER
    EMPLOYER_OPENED      = "EMPLOYER_OPENED"    # deprecated → NEW_EMPLOYER
    HOUSING_SHOCK        = "HOUSING_SHOCK"
    INFRASTRUCTURE_CHANGE = "INFRASTRUCTURE_CHANGE"

    # Environmental v2
    NEW_EMPLOYER         = "NEW_EMPLOYER"       # new employer (size: small/medium/big)
    CLOSED_EMPLOYER      = "CLOSED_EMPLOYER"    # employer closure (size: small/medium/big)
    NEW_INFRA            = "NEW_INFRA"          # new infrastructure
    CLOSED_INFRA         = "CLOSED_INFRA"       # infrastructure closure


# ══════════════════════════════════════════════════════════════════════════════
# 2. Event class
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Event:
    """Atomic event in the system.

    Поля:
      event_type       — тип события (EventType)
      tick_emitted     — at which tick it occurred
      source_agent_id  — source agent ID (None для средовых)
      source_district  — source district of event
      target_district  — target district (если применимо)
      industry         — industry (если применимо)
      settlement_type  — settlement type: metro/city/town/rural
      motivation       — economic / place
      magnitude        — event magnitude [0, 1]
      size             — employer size: small/medium/big (для NEW_EMPLOYER/CLOSED_EMPLOYER)
      n_agents_affected — число затронутых agents/рабочих moт (для CLOSED_EMPLOYER/FACTORY_CLOSED)
      deliver_at_tick  — delivery tick (>= tick_emitted, для буферизации)
    """
    event_type: EventType
    tick_emitted: int
    source_agent_id: Optional[int] = None
    source_district: Optional[str] = None
    target_district: Optional[str] = None
    industry: Optional[str] = None
    settlement_type: Optional[str] = None
    motivation: Optional[str] = None            # "economic" | "place"
    magnitude: float = 0.5
    size: Optional[str] = None                  # "small" | "medium" | "big"
    n_agents_affected: int = 0                  # v3: число рабочих moт/agents
    deliver_at_tick: Optional[int] = None       # None = instantly (tick_emitted)

    def __post_init__(self):
        if self.deliver_at_tick is None:
            self.deliver_at_tick = self.tick_emitted


# ══════════════════════════════════════════════════════════════════════════════
# 3. Signal — result of event processing
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """Signal to apply to DataFrame — result of event dispatching.

    Поля:
      target_mask      — boolean array or callable(df)→mask: whom to apply to
      field            — column name in df for modification
      delta            — magnitude of change (для add/multiply)
      mode             — "add" (прибавить) | "set" (установить) | "multiply"
      value            — value for mode="set" (может быть строкой)
      clip_min         — minimum value after application
      clip_max         — maximum value after application
      scale_by_field   — column name for scaling delta (опционально)
      event_type       — source event type (для sb_pending tracking, v2)

    v3 — graph operations (signals to environment, not just agents):
      graph_op         — "add_vacant" | "sub_occupied" | None
      graph_district   — district for graph operation
      graph_industry   — industry для граф-операции
      graph_delta      — magnitude of change in graph (int)
    """
    target_mask: Union[np.ndarray, Callable[[pd.DataFrame], np.ndarray]]
    field: str
    delta: float = 0.0
    mode: str = "add"           # "add" | "set" | "multiply"
    value: object = None        # value for mode="set" (строка, число)
    clip_min: float = 0.0
    clip_max: float = 1.0
    scale_by_field: Optional[str] = None
    event_type: Optional["EventType"] = None  # v2: для отслеживания decay

    # v3: graph operations
    graph_op: Optional[str] = None           # "add_vacant" | "sub_occupied"
    graph_district: Optional[str] = None
    graph_industry: Optional[str] = None
    graph_delta: float = 0.0

    def apply(self, df: pd.DataFrame,
              G: "nx.DiGraph | None" = None) -> pd.DataFrame:
        """Applies signal to DataFrame (and optionally to graph G).

        v3: if graph_op is set — modifies industry_jobs in graph nodes.
        v4: WITHOUT df.copy() — in-place modification. flush() передаёт df
            по цепочке, копирование на каждом сигнале избыточно.
        """
        # ── Agent part (df) ──────────────────────────────────────────
        if self.mode:
            if callable(self.target_mask):
                mask = self.target_mask(df)
            else:
                mask = self.target_mask

            if mask.any():
                col = df[self.field].values
                # v5: pandas .values may be read-only (numpy >= 1.24)
                # or StringArray (pandas extension). Check via isinstance.
                need_writeback = False
                if isinstance(col, np.ndarray) and not col.flags.writeable:
                    col = col.copy()
                    need_writeback = True

                if self.mode == "set":
                    col[mask] = self.value if self.value is not None else self.delta
                else:
                    if self.scale_by_field and self.scale_by_field in df.columns:
                        per_agent_delta = self.delta * df[self.scale_by_field].values
                    else:
                        per_agent_delta = np.full(len(df), self.delta)

                    if self.mode == "add":
                        col[mask] = col[mask] + per_agent_delta[mask]
                    elif self.mode == "multiply":
                        col[mask] = col[mask] * per_agent_delta[mask]

                    col[mask] = np.clip(col[mask], self.clip_min, self.clip_max)

                if need_writeback:
                    df[self.field] = col

        # ── v3: Graph operation ────────────────────────────────────────────
        if self.graph_op and G is not None and self.graph_district:
            self._apply_graph(G)

        return df

    def _apply_graph(self, G: "nx.DiGraph") -> None:
        """v3: Applies graph operation to node industry_jobs."""
        district = self.graph_district
        industry = self.graph_industry

        if district not in G.nodes:
            return

        ind_jobs = G.nodes[district].get("industry_jobs", {})
        if not ind_jobs:
            return

        if self.graph_op == "add_vacant":
            if industry and industry in ind_jobs:
                ind_jobs[industry]["vacant"] = max(
                    0, ind_jobs[industry].get("vacant", 0) + int(self.graph_delta)
                )
        elif self.graph_op == "sub_occupied":
            if industry and industry in ind_jobs:
                ind_jobs[industry]["occupied"] = max(
                    0, ind_jobs[industry].get("occupied", 0) - int(self.graph_delta)
                )

        # Recalculate total jobs_capacity of node
        total = sum(v["occupied"] + v["vacant"] for v in ind_jobs.values())
        G.nodes[district]["jobs_capacity"] = max(1, total)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Rule — event propagation rule
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rule:
    """Dispatching rule: how an event becomes signals.

    Поля:
      event_type       — which event type it applies to
      target_scope     — whom to propagate to
      field            — what to change in df
      base_delta       — base change strength
      mode             — "add" | "set" | "multiply"
      value            — value for mode="set" (строка, число)
      distance_decay   — distance decay (0 = без затухания)
      filter_status    — status filter: None / "unemployed" / "employed"
      filter_industry  — industry filter: None или конкретная industry
      delay_ticks      — delay in ticks (0 = instantly)
      motivation       — motivation filter: None / "economic" / "place"
      scale_by_field   — column name for per-agent delta scaling
      clip_min         — min. value after application (по умолчанию 0.0)
      clip_max         — max. value after application (по умолчанию 1.0)

    v3 — graph rules:
      graph_op         — "add_vacant" | "sub_occupied" | None
      graph_delta      — magnitude of change in graph (int)
    """
    event_type: EventType
    target_scope: str                           # см. выше
    field: str                                  # имя колонки в df
    base_delta: float = 0.05
    mode: str = "add"                           # "add" | "set" | "multiply"
    value: object = None                        # value for mode="set"
    distance_decay: float = 0.0
    filter_status: Optional[str] = None
    filter_industry: Optional[str] = None
    delay_ticks: int = 0
    motivation: Optional[str] = None
    scale_by_field: Optional[str] = None
    clip_min: float = 0.0
    clip_max: float = 1.0

    # v3: graph rules
    graph_op: Optional[str] = None              # "add_vacant" | "sub_occupied"
    graph_delta: float = 0.0                    # величина in graph

    # v3: additional filters
    filter_wage_pressure: bool = False          # фильтровать agents с wage_pressure > 1
    filter_education: Optional[str] = None      # "low" | "medium" | "high"
    filter_same_industry: bool = False          # фильтровать agents той же отрасли, что и event.industry


# ══════════════════════════════════════════════════════════════════════════════
# 5. Dispatcher — rule table
# ══════════════════════════════════════════════════════════════════════════════

# ── Cache for district → settlement_type mapping ────────────────────────────────
# Filled externally (engine.py passes SETTLEMENT_MAP from agents.py)
_SETTLEMENT_MAP: dict[str, str] = {}

# ── Cache for district → region mapping ─────────────────────────────────────────
# Filled from G.nodes on first access
_REGION_CACHE: dict[str, str] = {}


def set_settlement_map(sm: dict[str, str]) -> None:
    """Sets district → settlement_type mapping."""
    global _SETTLEMENT_MAP
    _SETTLEMENT_MAP = dict(sm)


def _get_settlement(district: str) -> str:
    """Returns settlement_type for district (town by default)."""
    return _SETTLEMENT_MAP.get(district, "town")


def _get_region(district: str, G: "nx.DiGraph | None" = None) -> str:
    """Returns region code for district."""
    global _REGION_CACHE
    if district in _REGION_CACHE:
        return _REGION_CACHE[district]
    if G is not None and district in G.nodes:
        region = G.nodes[district].get("region", "XX")
        _REGION_CACHE[district] = region
        return region
    return "XX"


# ── Public scope constants ──────────────────────────────────────────────

SCOPE_SELF                   = "self"
SCOPE_RESIDENCE_NEIGHBORS    = "residence_neighbors"     # живут в source_district
SCOPE_TARGET_NEIGHBORS       = "target_neighbors"        # живут в target_district
SCOPE_WORKPLACE_COLLEAGUES   = "same_workplace_district" # работают в source_district
SCOPE_SAME_INDUSTRY_DISTRICT = "same_industry_district"  # та же industry + source_district
SCOPE_SAME_SETTLEMENT_TYPE   = "same_settlement_type"    # тот же settlement + source_district
SCOPE_WHOLE_REGION           = "whole_region"            # весь регион source_district


def _industry_wage_in_district_signal(G, district: str, industry: str) -> float:
    """Industry wage in graph node; fallback → avg_wage → 1614."""
    attr = G.nodes.get(district, {})
    sal = attr.get("salary_by_industry", {})
    if sal:
        return float(sal.get(industry, attr.get("avg_wage", 1614.0)))
    return float(attr.get("avg_wage", 1614.0))


def _wage_pressure_mask(df: pd.DataFrame, G, event: "Event",
                        base_mask: np.ndarray) -> np.ndarray:
    """v3: Filters mask — keeps only agents with wage_pressure > 1.

    wage_pressure = отраслевая_зарплата_в_районе / зарплата_агента.
    If agent wage is 0 — wage_pressure = 1.0 (not filtered).
    """
    import numpy as np
    indices = np.where(base_mask)[0]
    if len(indices) == 0:
        return base_mask

    district = event.source_district
    industry = event.industry
    ind_wage = _industry_wage_in_district_signal(G, district, industry)

    # Vectorized: instead of loop over indices
    result = base_mask.copy()
    wages = df["wage"].values[indices].astype(float)
    if ind_wage > 0:
        wp = np.divide(ind_wage, wages, out=np.ones_like(wages, dtype=float), where=wages > 0)
        # Keep only agents with wage_pressure > 1
        keep = wp > 1.0
        result[indices[~keep]] = False
    return result


class Dispatcher:
    """Stores rule table event_type → list[Rule] and dispatches events."""

    def __init__(self):
        self._rules: dict[EventType, list[Rule]] = {}

    def add_rule(self, rule: Rule) -> None:
        """Adds rule to table."""
        self._rules.setdefault(rule.event_type, []).append(rule)

    def add_rules(self, rules: list[Rule]) -> None:
        """Adds multiple rules."""
        for r in rules:
            self.add_rule(r)

    def dispatch(self, event: Event, df: pd.DataFrame,
                 G: "nx.DiGraph | None" = None) -> tuple[list[Signal], list[tuple[Signal, int]]]:
        """Converts event into signals: immediate and delayed.

        v3: For rules with graph_op — creates graph signal (district/industry из Event).
        For rules with field — creates agent signal (как раньше).

        Returns:
          (signals_now, delayed) — где delayed = list[(Signal, deliver_at_tick)]
        """
        signals_now: list[Signal] = []
        delayed: list[tuple[Signal, int]] = []

        rules = self._rules.get(event.event_type, [])
        for rule in rules:
            # Фильтр по motivation
            if rule.motivation is not None and rule.motivation != event.motivation:
                continue

            # v3: Pure graph rule (no field/delta for df)
            if rule.graph_op:
                # Граф-сигнал: не нужна маска agents, district/industry из Event
                if event.source_district is None:
                    continue

                # graph_delta: из Event.n_agents_affected, иначе из Event.size → _size_to_jobs
                g_delta = rule.graph_delta
                if event.n_agents_affected > 0:
                    g_delta = float(event.n_agents_affected)
                elif event.size:
                    g_delta = float(_size_to_jobs(event.size))

                sig = Signal(
                    target_mask=np.zeros(len(df), dtype=bool),  # empty mask
                    field="",                                     # no df field
                    mode="",                                      # no df mode
                    graph_op=rule.graph_op,
                    graph_district=event.source_district,
                    graph_industry=event.industry,
                    graph_delta=g_delta,
                    event_type=event.event_type,
                )

                if rule.delay_ticks > 0:
                    deliver_at = event.tick_emitted + rule.delay_ticks
                    delayed.append((sig, deliver_at))
                else:
                    signals_now.append(sig)
                continue

            # ── Agent rule (standard path) ───────────────────────
            mask = self._build_mask(rule, event, df, G)
            if not mask.any():
                continue

            sig = Signal(
                target_mask=mask,
                field=rule.field,
                delta=rule.base_delta,
                mode=rule.mode,
                value=rule.value,
                clip_min=rule.clip_min,
                clip_max=rule.clip_max,
                scale_by_field=rule.scale_by_field,
                event_type=event.event_type,      # v2: для sb_pending tracking
            )

            if rule.delay_ticks > 0:
                deliver_at = event.tick_emitted + rule.delay_ticks
                delayed.append((sig, deliver_at))
            else:
                signals_now.append(sig)

        return signals_now, delayed

    def _build_mask(self, rule: Rule, event: Event,
                    df: pd.DataFrame, G: "nx.DiGraph | None" = None) -> np.ndarray:
        """Builds boolean mask for rule based on target_scope and filters."""
        n = len(df)
        scope = rule.target_scope

        # ── Base geo-scope ────────────────────────────────────────────
        if scope == SCOPE_SELF:
            base = np.zeros(n, dtype=bool)
            if event.source_agent_id is not None:
                base[df["id"].values == event.source_agent_id] = True
        elif scope == SCOPE_RESIDENCE_NEIGHBORS:
            if event.source_district is None:
                return np.zeros(n, dtype=bool)
            base = (df["district"].values == event.source_district)
        elif scope == SCOPE_TARGET_NEIGHBORS:
            if event.target_district is None:
                return np.zeros(n, dtype=bool)
            base = (df["district"].values == event.target_district)
        elif scope == SCOPE_WORKPLACE_COLLEAGUES:
            if event.source_district is None:
                return np.zeros(n, dtype=bool)
            base = (df["workplace_district"].values == event.source_district)
        elif scope == SCOPE_SAME_INDUSTRY_DISTRICT:
            if event.source_district is None or event.industry is None:
                return np.zeros(n, dtype=bool)
            base = (
                (df["workplace_district"].values == event.source_district) &
                (df["industry"].values == event.industry)
            )
        elif scope == SCOPE_SAME_SETTLEMENT_TYPE:
            if event.settlement_type is None:
                return np.zeros(n, dtype=bool)
            # Векторизовано: pandas .map вmoто цикла по агентам
            base = (df["district"].map(_SETTLEMENT_MAP).fillna("town").values == event.settlement_type)
        elif scope == SCOPE_WHOLE_REGION:
            if event.source_district is None:
                return np.zeros(n, dtype=bool)
            region = _get_region(event.source_district, G)
            # Векторизовано: строим region-map из G.nodes один раз
            if G is not None:
                region_map = {d: str(G.nodes[d].get("region", "XX")) for d in G.nodes}
            else:
                region_map = {}
            agent_regions = df["district"].map(region_map).fillna("XX").values
            base = agent_regions == region
        else:
            return np.zeros(n, dtype=bool)

        # ── Status filter ────────────────────────────────────────────
        if rule.filter_status == "unemployed":
            base = base & (df["status"].values == "unemployed")
        elif rule.filter_status == "employed":
            base = base & (df["status"].values != "unemployed")

        # ── Industry filter ────────────────────────────────────────────
        if rule.filter_industry is not None:
            base = base & (df["industry"].values == rule.filter_industry)

        # ── v3: wage_pressure > 1 filter ─────────────────────────────────
        if rule.filter_wage_pressure and G is not None and event.industry:
            # Нужен импорт _industry_wage_in_district из engine — ленивый
            base = base & _wage_pressure_mask(df, G, event, base)

        # ── Education filter ───────────────────────────────────────
        if rule.filter_education is not None:
            base = base & (df["education"].values == rule.filter_education)

        # ── Same-industry as event filter ─────────────────────
        if rule.filter_same_industry and event.industry:
            base = base & (df["industry"].values == event.industry)

        # ── Exclude source agent (кроме scope=self) ─────────
        if scope != SCOPE_SELF and event.source_agent_id is not None:
            base = base & (df["id"].values != event.source_agent_id)

        return base


# ══════════════════════════════════════════════════════════════════════════════
# 6. EventBus — шина событий
# ══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """Central event bus.

    Two event queues:
      pending   — events received this tick, not yet processed
      scheduled — delayed events, waiting for their tick

    Delayed signal queue:
      _delayed_signals — {tick: [Signal]} — signals awaiting delivery

    Методы:
      emit(event)                — add event to pending
      process(current_tick, df, G) — обработать pending + scheduled,
                                     вернуть список Signal
      flush(df, signals)         — apply signals to DataFrame
    """

    def __init__(self, dispatcher: Dispatcher | None = None):
        self.pending: list[Event] = []
        self.scheduled: list[Event] = []
        self._delayed_signals: dict[int, list[Signal]] = {}  # tick → signals
        self.dispatcher = dispatcher or Dispatcher()
        self._stats = {
            "emitted": 0,
            "processed": 0,
            "signals_generated": 0,
            "scheduled_count": 0,
        }

    def emit(self, event: Event) -> None:
        """Adds event to pending queue."""
        self.pending.append(event)
        self._stats["emitted"] += 1

    def process(self, current_tick: int, df: pd.DataFrame,
                G: "nx.DiGraph | None" = None) -> list[Signal]:
        """Processes all pending events + scheduled ready for this tick.

        Returns list of Signal for immediate application.
        Delayed signals (delay_ticks > 0) are stored in _delayed_signals.
        """
        all_signals: list[Signal] = []

        # 0. Extract delayed signals for this tick
        if current_tick in self._delayed_signals:
            all_signals.extend(self._delayed_signals.pop(current_tick))

        # 1. Move scheduled → pending for events whose time has come
        due = [e for e in self.scheduled if e.deliver_at_tick <= current_tick]
        still_waiting = [e for e in self.scheduled if e.deliver_at_tick > current_tick]
        self.scheduled = still_waiting
        self.pending.extend(due)

        # 2. Pending dispatching
        for event in self.pending:
            signals_now, delayed_signals = self.dispatcher.dispatch(event, df, G)
            all_signals.extend(signals_now)
            self._stats["processed"] += 1
            self._stats["signals_generated"] += len(signals_now)

            # Store delayed signals for future ticks
            for sig, deliver_at in delayed_signals:
                self._delayed_signals.setdefault(deliver_at, []).append(sig)
                self._stats["scheduled_count"] += 1

        self.pending.clear()
        return all_signals

    def flush(self, df: pd.DataFrame, signals: list[Signal],
              G: "nx.DiGraph | None" = None) -> pd.DataFrame:
        """Applies signals to DataFrame (and optionally to graph G). v3: +G."""
        for sig in signals:
            df = sig.apply(df, G)

            # v2: decay tracking for social_boost
            if sig.event_type is not None and sig.field == "social_boost" and sig.mode == "add":
                self._update_sb_pending(df, sig)

        return df

    def _update_sb_pending(self, df: pd.DataFrame, sig: Signal) -> None:
        """Updates sb_pending for linear-decay social_boost tracking. (ВЕКТОРИЗОВАНО)"""
        import numpy as np

        if callable(sig.target_mask):
            mask = sig.target_mask(df)
        else:
            mask = sig.target_mask

        if not mask.any():
            return

        et = sig.event_type
        if et == EventType.AGENT_MOVED:
            suffix = "M6"
        elif et == EventType.AGENT_COMMUTE_STARTED:
            suffix = "C3"
        else:
            return

        # Vectorized: work only with affected agents (mask), не со всеми n
        current = df["sb_pending"].values.copy()
        affected = current[mask]
        # Mask for empty/NaN values among affected
        empty_mask = np.array([
            v is None or str(v) in ("", "nan", "None")
            for v in affected
        ], dtype=bool)
        new_vals = affected.copy()
        new_vals[empty_mask] = suffix
        if (~empty_mask).any():
            new_vals[~empty_mask] = np.array(
                [str(v) + "," + suffix for v in affected[~empty_mask]]
            )
        current[mask] = new_vals
        df["sb_pending"] = current

    def schedule(self, event: Event, deliver_at_tick: int) -> None:
        """Puts event into scheduled queue with delay."""
        event.deliver_at_tick = deliver_at_tick
        self.scheduled.append(event)
        self._stats["scheduled_count"] += 1

    @property
    def stats(self) -> dict:
        """Bus statistics."""
        return dict(self._stats)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Default rule factory
# ══════════════════════════════════════════════════════════════════════════════

# Constants — соответствуют текущим значениям из engine.py
EVENT_SOCIAL_BOOST    = 0.08   # base social_boost addition from event
UNEMPLOYED_SIGNAL     = 0.35   # addition to signal_reduction on job loss
NEIGHBOR_SIGNAL_COEF  = 0.04   # signal coefficient from moved neighbor

# v5: company sizes → jobs (synchronized with graph.py)
_SIZE_TO_JOBS = {"small": 25, "medium": 130, "big": 400}


def _size_to_jobs(size: str) -> int:
    """Converts company size to approximate number of jobs."""
    return _SIZE_TO_JOBS.get(size, 50)


def create_default_dispatcher() -> Dispatcher:
    """Creates Dispatcher v2 with updated signal system rules.

    New in v2:
      - social_boost: separate decays for MOVE (+0.06, −0.01×6) и COMMUTE (+0.02, reset через 3)
      - inertia_mobility_penalty: accumulation from neighbor MOVEs (−0.06, −0.01×6)
      - LOST_JOB: immediate inertia decrease (−0.25) + econ_gap increase (+0.25) + ramp
      - NEW_EMPLOYER/CLOSED_EMPLOYER: econ_penalty with wage_pressure>1 condition
      - NEW_INFRA/CLOSED_INFRA: infra_bonus ±0.05
    """
    d = Dispatcher()

    # ═══════════════════════════════════════════════════════════════════════
    # AGENT_MOVED — v2: social_boost +0.06 (было 0.08), новый decay
    # ═══════════════════════════════════════════════════════════════════════
    # Old district neighbors: social_boost +0.02 (MOVE decay: −0.005×6)
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="social_boost",
        base_delta=0.02,
    ))
    # Old district neighbors: inertia_mobility_penalty −0.01 (decay: −0.005×6)
    # Negative sign: neighbor move lowers inertia, making migration more likely
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="inertia_mobility_penalty",
        base_delta=-0.01,
        clip_min=-1.0,
        clip_max=1.0,
    ))
    # ── AGENT_MOVED (place): place_deficit_penalty to neighbors of same settlement ─
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_SAME_SETTLEMENT_TYPE,
        field="place_deficit_penalty",
        base_delta=0.03,
        motivation="place",
        delay_ticks=1,
        clip_min=0.0,
        clip_max=5.0,
    ))

    # ── AGENT_MOVED: signal_reduction to old-district neighbors ──────────────
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="signal_reduction",
        base_delta=NEIGHBOR_SIGNAL_COEF,         # 0.04
        scale_by_field="net_signal_susc",
    ))
    # ── v3: AGENT_MOVED → soc_calibration_signal соседям ──────────────────
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="soc_calibration_signal",
        base_delta=0.04,
        scale_by_field="net_signal_susc",
        clip_min=0.0,
        clip_max=1.0,
    ))
    # ── AGENT_MOVED (economic) → econ_penalty to low-education neighbors of same industry ─
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="econ_penalty",
        base_delta=0.05,
        motivation="economic",
        filter_education="low",
        filter_same_industry=True,
        clip_min=0.0,
        clip_max=0.5,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # AGENT_COMMUTE_STARTED — v2: social_boost +0.02, reset через 3 тика
    # ═══════════════════════════════════════════════════════════════════════
    d.add_rule(Rule(
        event_type=EventType.AGENT_COMMUTE_STARTED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="social_boost",
        base_delta=0.02,
    ))
    # ── v3: AGENT_COMMUTE_STARTED → soc_calibration_signal ─────────────────
    d.add_rule(Rule(
        event_type=EventType.AGENT_COMMUTE_STARTED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="soc_calibration_signal",
        base_delta=0.02,
        scale_by_field="net_signal_susc",
        clip_min=0.0,
        clip_max=1.0,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # JOB_CHANGED
    # ═══════════════════════════════════════════════════════════════════════
    # ── v3: JOB_CHANGED → soc_calibration_signal to colleagues ─────────────────
    d.add_rule(Rule(
        event_type=EventType.JOB_CHANGED,
        target_scope=SCOPE_WORKPLACE_COLLEAGUES,
        field="soc_calibration_signal",
        base_delta=0.03,
        scale_by_field="net_signal_susc",
        clip_min=0.0,
        clip_max=1.0,
    ))
    # ── JOB_CHANGED → econ_penalty низкообразованным to colleagues той же отрасли ─
    d.add_rule(Rule(
        event_type=EventType.JOB_CHANGED,
        target_scope=SCOPE_WORKPLACE_COLLEAGUES,
        field="econ_penalty",
        base_delta=0.03,
        filter_education="low",
        filter_same_industry=True,
        clip_min=0.0,
        clip_max=0.5,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # LOST_JOB — v2: new agent job-loss signal
    # ═══════════════════════════════════════════════════════════════════════
    # Agent self: inertia −0.25 (immediately)
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_SELF,
        field="inertia",
        base_delta=-0.25,
        clip_min=0.05,
        clip_max=0.95,
    ))
    # Agent self: econ_gap +0.25 (immediately, + ramp in tick)
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_SELF,
        field="econ_gap",
        base_delta=0.25,
        clip_min=0.0,
        clip_max=1.0,
    ))
    # Agent self: signal_reduction +0.35 (job loss shock)
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_SELF,
        field="signal_reduction",
        base_delta=UNEMPLOYED_SIGNAL,            # 0.35
    ))
    # Agent self: intention_state → seeking_work
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_SELF,
        field="intention_state",
        mode="set",
        value="seeking_work",
    ))
    # Colleagues by workplace_district: inertia +0.02 (decay: −0.01×2)
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_WORKPLACE_COLLEAGUES,
        field="inertia_mobility_penalty",
        base_delta=0.02,
        clip_min=0.0,
        clip_max=1.0,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # JOB_LOST (deprecated, kept for backward compatibility)
    # ═══════════════════════════════════════════════════════════════════════
    d.add_rule(Rule(
        event_type=EventType.JOB_LOST,
        target_scope=SCOPE_SELF,
        field="signal_reduction",
        base_delta=UNEMPLOYED_SIGNAL,
    ))
    d.add_rule(Rule(
        event_type=EventType.JOB_LOST,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="aspirations",
        base_delta=0.06,
        filter_status="employed",
    ))
    d.add_rule(Rule(
        event_type=EventType.JOB_LOST,
        target_scope=SCOPE_WHOLE_REGION,
        field="aspirations",
        base_delta=0.02,
        filter_status="employed",
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # NEW_EMPLOYER — v5: граф меняется через _execute_new_employer(),
    # здесь только агент-сигналы (social_boost, econ_penalty)
    # ═══════════════════════════════════════════════════════════════════════
    # Агенты total региона: social_boost
    d.add_rule(Rule(
        event_type=EventType.NEW_EMPLOYER,
        target_scope=SCOPE_WHOLE_REGION,
        field="social_boost",
        base_delta=0.05,
    ))
    # v3: NEW_EMPLOYER → soc_calibration_signal to whole region
    d.add_rule(Rule(
        event_type=EventType.NEW_EMPLOYER,
        target_scope=SCOPE_WHOLE_REGION,
        field="soc_calibration_signal",
        base_delta=0.03,
        scale_by_field="net_signal_susc",
        clip_min=0.0,
        clip_max=1.0,
    ))
    # Same-industry same-district agents with wage_pressure>1: econ_penalty
    d.add_rule(Rule(
        event_type=EventType.NEW_EMPLOYER,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="econ_penalty",
        base_delta=0.02,
        filter_wage_pressure=True,
        clip_min=0.0,
        clip_max=1.0,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # CLOSED_EMPLOYER — v5: граф меняется через _execute_closed_employer(),
    # здесь только агент-сигналы (aspirations, soc_calibration)
    # ═══════════════════════════════════════════════════════════════════════
    # Агенты total региона (занятые): aspirations ↑
    d.add_rule(Rule(
        event_type=EventType.CLOSED_EMPLOYER,
        target_scope=SCOPE_WHOLE_REGION,
        field="aspirations",
        base_delta=0.08,
        filter_status="employed",
    ))
    # v3: CLOSED_EMPLOYER → soc_calibration_signal decreases in region ─────
    d.add_rule(Rule(
        event_type=EventType.CLOSED_EMPLOYER,
        target_scope=SCOPE_WHOLE_REGION,
        field="soc_calibration_signal",
        base_delta=-0.03,
        scale_by_field="net_signal_susc",
        clip_min=0.0,
        clip_max=1.0,
    ))
    # Агенты той же отрасли в том же районе: econ_penalty reset
    d.add_rule(Rule(
        event_type=EventType.CLOSED_EMPLOYER,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="econ_penalty",
        mode="set",
        value=0.0,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # NEW_INFRA — v2: infra_bonus +0.05
    # ═══════════════════════════════════════════════════════════════════════
    d.add_rule(Rule(
        event_type=EventType.NEW_INFRA,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="infra_bonus",
        base_delta=0.05,
        clip_min=-1.0,
        clip_max=1.0,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # CLOSED_INFRA — v2: infra_bonus −0.05
    # ═══════════════════════════════════════════════════════════════════════
    d.add_rule(Rule(
        event_type=EventType.CLOSED_INFRA,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="infra_bonus",
        base_delta=-0.05,
        clip_min=-1.0,
        clip_max=1.0,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # GRADUATED
    # ═══════════════════════════════════════════════════════════════════════
    d.add_rule(Rule(
        event_type=EventType.GRADUATED,
        target_scope=SCOPE_SELF,
        field="intention_state",
        mode="set",
        value="seeking_work",
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # Old rules for backward compatibility (HOUSING_SHOCK,
    # EMPLOYER_OPENED)
    # ═══════════════════════════════════════════════════════════════════════

    # ── HOUSING_SHOCK ─────────────────────────────────────────────────────
    d.add_rule(Rule(
        event_type=EventType.HOUSING_SHOCK,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="place_deficit_penalty",
        base_delta=0.15,
        clip_min=0.0,
        clip_max=5.0,
    ))
    d.add_rule(Rule(
        event_type=EventType.HOUSING_SHOCK,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="inertia",
        base_delta=-0.04,
    ))

    # ── EMPLOYER_OPENED ───────────────────────────────────────────────────
    d.add_rule(Rule(
        event_type=EventType.EMPLOYER_OPENED,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="aspirations",
        base_delta=-0.06,
        filter_status="unemployed",
    ))
    d.add_rule(Rule(
        event_type=EventType.EMPLOYER_OPENED,
        target_scope=SCOPE_WHOLE_REGION,
        field="social_boost",
        base_delta=0.05,
    ))

    return d
