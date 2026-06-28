"""
signals.py — Сигнальная система событий для ABM Миграция Словакии.

Архитектура:
  Event       — атомарное событие (агентное или средовое)
  EventBus    — шина с двумя очередями (pending + scheduled) и Dispatcher
  Rule        — правило распространения: кому, что менять, с какой силой
  Dispatcher  — таблица правил event_type → list[Rule]

Импортирует только pandas и numpy. Не зависит от engine, agents, graph.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1. Типы событий
# ══════════════════════════════════════════════════════════════════════════════

class EventType(str, enum.Enum):
    """Типы событий в системе (v2 — сигнальная система)."""
    # Агентные
    AGENT_MOVED          = "AGENT_MOVED"
    AGENT_COMMUTE_STARTED = "AGENT_COMMUTE_STARTED"
    JOB_CHANGED          = "JOB_CHANGED"
    JOB_LOST             = "JOB_LOST"           # deprecated → LOST_JOB
    GRADUATED            = "GRADUATED"
    ADAPTED              = "ADAPTED"

    # Агентные v2
    LOST_JOB             = "LOST_JOB"           # потеря работы (агентное, не сценарное)

    # Средовые (сценарные)
    FACTORY_CLOSED       = "FACTORY_CLOSED"     # deprecated → CLOSED_EMPLOYER
    EMPLOYER_OPENED      = "EMPLOYER_OPENED"    # deprecated → NEW_EMPLOYER
    HOUSING_SHOCK        = "HOUSING_SHOCK"
    INFRASTRUCTURE_CHANGE = "INFRASTRUCTURE_CHANGE"

    # Средовые v2
    NEW_EMPLOYER         = "NEW_EMPLOYER"       # новый работодатель (size: small/medium/big)
    CLOSED_EMPLOYER      = "CLOSED_EMPLOYER"    # закрытие работодателя (size: small/medium/big)
    NEW_INFRA            = "NEW_INFRA"          # новая инфраструктура
    CLOSED_INFRA         = "CLOSED_INFRA"       # закрытие инфраструктуры


# ══════════════════════════════════════════════════════════════════════════════
# 2. Класс Event
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Event:
    """Атомарное событие в системе.

    Поля:
      event_type       — тип события (EventType)
      tick_emitted     — на каком тике произошло
      source_agent_id  — ID агента-источника (None для средовых)
      source_district  — район-источник события
      target_district  — целевой район (если применимо)
      industry         — отрасль (если применимо)
      settlement_type  — тип поселения: metro/city/town/rural
      motivation       — economic / place
      magnitude        — сила события [0, 1]
      size             — размер работодателя: small/medium/big (для NEW_EMPLOYER/CLOSED_EMPLOYER)
      n_agents_affected — число затронутых агентов/рабочих мест (для CLOSED_EMPLOYER/FACTORY_CLOSED)
      deliver_at_tick  — тик доставки (>= tick_emitted, для буферизации)
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
    n_agents_affected: int = 0                  # v3: число рабочих мест/агентов
    deliver_at_tick: Optional[int] = None       # None = мгновенно (tick_emitted)

    def __post_init__(self):
        if self.deliver_at_tick is None:
            self.deliver_at_tick = self.tick_emitted


# ══════════════════════════════════════════════════════════════════════════════
# 3. Signal — результат обработки события
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """Сигнал к применению на DataFrame — результат диспетчеризации события.

    Поля:
      target_mask      — булев массив или callable(df)→mask: кому применять
      field            — имя колонки в df для изменения
      delta            — величина изменения (для add/multiply)
      mode             — "add" (прибавить) | "set" (установить) | "multiply"
      value            — значение для mode="set" (может быть строкой)
      clip_min         — минимальное значение после применения
      clip_max         — максимальное значение после применения
      scale_by_field   — имя колонки для масштабирования delta (опционально)
      event_type       — тип события-источника (для sb_pending tracking, v2)

    v3 — граф-операции (сигналы среде, не только агентам):
      graph_op         — "add_vacant" | "sub_occupied" | None
      graph_district   — район для граф-операции
      graph_industry   — отрасль для граф-операции
      graph_delta      — величина изменения в графе (int)
    """
    target_mask: Union[np.ndarray, Callable[[pd.DataFrame], np.ndarray]]
    field: str
    delta: float = 0.0
    mode: str = "add"           # "add" | "set" | "multiply"
    value: object = None        # значение для mode="set" (строка, число)
    clip_min: float = 0.0
    clip_max: float = 1.0
    scale_by_field: Optional[str] = None
    event_type: Optional["EventType"] = None  # v2: для отслеживания decay

    # v3: граф-операции
    graph_op: Optional[str] = None           # "add_vacant" | "sub_occupied"
    graph_district: Optional[str] = None
    graph_industry: Optional[str] = None
    graph_delta: float = 0.0

    def apply(self, df: pd.DataFrame,
              G: "nx.DiGraph | None" = None) -> pd.DataFrame:
        """Применяет сигнал к DataFrame (и опционально к графу G).

        v3: если graph_op задан — модифицирует industry_jobs в узлах графа.
        v4: БЕЗ df.copy() — модификация in-place. flush() передаёт df
            по цепочке, копирование на каждом сигнале избыточно.
        """
        # ── Агентная часть (df) ──────────────────────────────────────────
        if self.mode:
            if callable(self.target_mask):
                mask = self.target_mask(df)
            else:
                mask = self.target_mask

            if mask.any():
                col = df[self.field].values
                # v5: pandas .values может быть read-only (numpy >= 1.24)
                # или StringArray (pandas extension). Проверяем через isinstance.
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

        # ── v3: Граф-операция ────────────────────────────────────────────
        if self.graph_op and G is not None and self.graph_district:
            self._apply_graph(G)

        return df

    def _apply_graph(self, G: "nx.DiGraph") -> None:
        """v3: Применяет граф-операцию к industry_jobs узла."""
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

        # Пересчитываем общую jobs_capacity узла
        total = sum(v["occupied"] + v["vacant"] for v in ind_jobs.values())
        G.nodes[district]["jobs_capacity"] = max(1, total)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Rule — правило распространения события
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rule:
    """Правило диспетчеризации: как событие превращается в сигналы.

    Поля:
      event_type       — к какому типу события применяется
      target_scope     — кому распространять
      field            — что менять в df
      base_delta       — базовая сила изменения
      mode             — "add" | "set" | "multiply"
      value            — значение для mode="set" (строка, число)
      distance_decay   — затухание по расстоянию (0 = без затухания)
      filter_status    — фильтр по статусу: None / "unemployed" / "employed"
      filter_industry  — фильтр по отрасли: None или конкретная отрасль
      delay_ticks      — задержка в тиках (0 = мгновенно)
      motivation       — фильтр по мотивации: None / "economic" / "place"
      scale_by_field   — имя колонки для per-agent масштабирования delta
      clip_min         — мин. значение после применения (по умолчанию 0.0)
      clip_max         — макс. значение после применения (по умолчанию 1.0)

    v3 — граф-правила:
      graph_op         — "add_vacant" | "sub_occupied" | None
      graph_delta      — величина изменения в графе (int)
    """
    event_type: EventType
    target_scope: str                           # см. выше
    field: str                                  # имя колонки в df
    base_delta: float = 0.05
    mode: str = "add"                           # "add" | "set" | "multiply"
    value: object = None                        # значение для mode="set"
    distance_decay: float = 0.0
    filter_status: Optional[str] = None
    filter_industry: Optional[str] = None
    delay_ticks: int = 0
    motivation: Optional[str] = None
    scale_by_field: Optional[str] = None
    clip_min: float = 0.0
    clip_max: float = 1.0

    # v3: граф-правила
    graph_op: Optional[str] = None              # "add_vacant" | "sub_occupied"
    graph_delta: float = 0.0                    # величина в графе

    # v3: дополнительные фильтры
    filter_wage_pressure: bool = False          # фильтровать агентов с wage_pressure > 1
    filter_education: Optional[str] = None      # "low" | "medium" | "high"
    filter_same_industry: bool = False          # фильтровать агентов той же отрасли, что и event.industry


# ══════════════════════════════════════════════════════════════════════════════
# 5. Dispatcher — таблица правил
# ══════════════════════════════════════════════════════════════════════════════

# ── Кэш маппинга district → settlement_type ────────────────────────────────
# Заполняется извне (engine.py передаёт SETTLEMENT_MAP из agents.py)
_SETTLEMENT_MAP: dict[str, str] = {}

# ── Кэш маппинга district → region ─────────────────────────────────────────
# Заполняется из G.nodes при первом обращении
_REGION_CACHE: dict[str, str] = {}


def set_settlement_map(sm: dict[str, str]) -> None:
    """Устанавливает маппинг district → settlement_type."""
    global _SETTLEMENT_MAP
    _SETTLEMENT_MAP = dict(sm)


def _get_settlement(district: str) -> str:
    """Возвращает settlement_type для района (town по умолчанию)."""
    return _SETTLEMENT_MAP.get(district, "town")


def _get_region(district: str, G: "nx.DiGraph | None" = None) -> str:
    """Возвращает код региона для района."""
    global _REGION_CACHE
    if district in _REGION_CACHE:
        return _REGION_CACHE[district]
    if G is not None and district in G.nodes:
        region = G.nodes[district].get("region", "XX")
        _REGION_CACHE[district] = region
        return region
    return "XX"


# ── Публичные константы scope ──────────────────────────────────────────────

SCOPE_SELF                   = "self"
SCOPE_RESIDENCE_NEIGHBORS    = "residence_neighbors"     # живут в source_district
SCOPE_TARGET_NEIGHBORS       = "target_neighbors"        # живут в target_district
SCOPE_WORKPLACE_COLLEAGUES   = "same_workplace_district" # работают в source_district
SCOPE_SAME_INDUSTRY_DISTRICT = "same_industry_district"  # та же отрасль + source_district
SCOPE_SAME_SETTLEMENT_TYPE   = "same_settlement_type"    # тот же settlement + source_district
SCOPE_WHOLE_REGION           = "whole_region"            # весь регион source_district


def _industry_wage_in_district_signal(G, district: str, industry: str) -> float:
    """Отраслевая зарплата в узле графа; fallback → avg_wage → 1614."""
    attr = G.nodes.get(district, {})
    sal = attr.get("salary_by_industry", {})
    if sal:
        return float(sal.get(industry, attr.get("avg_wage", 1614.0)))
    return float(attr.get("avg_wage", 1614.0))


def _wage_pressure_mask(df: pd.DataFrame, G, event: "Event",
                        base_mask: np.ndarray) -> np.ndarray:
    """v3: Фильтрует маску — оставляет только агентов с wage_pressure > 1.

    wage_pressure = отраслевая_зарплата_в_районе / зарплата_агента.
    Если у агента зарплата 0 — wage_pressure = 1.0 (не фильтруется).
    """
    import numpy as np
    indices = np.where(base_mask)[0]
    if len(indices) == 0:
        return base_mask

    district = event.source_district
    industry = event.industry
    ind_wage = _industry_wage_in_district_signal(G, district, industry)

    # Векторизовано: вместо цикла по индексам
    result = base_mask.copy()
    wages = df["wage"].values[indices].astype(float)
    if ind_wage > 0:
        wp = np.where(wages > 0, ind_wage / wages, 1.0)
        # Оставляем только агентов с wage_pressure > 1
        keep = wp > 1.0
        result[indices[~keep]] = False
    return result


class Dispatcher:
    """Хранит таблицу правил event_type → list[Rule] и диспетчеризует события."""

    def __init__(self):
        self._rules: dict[EventType, list[Rule]] = {}

    def add_rule(self, rule: Rule) -> None:
        """Добавляет правило в таблицу."""
        self._rules.setdefault(rule.event_type, []).append(rule)

    def add_rules(self, rules: list[Rule]) -> None:
        """Добавляет несколько правил."""
        for r in rules:
            self.add_rule(r)

    def dispatch(self, event: Event, df: pd.DataFrame,
                 G: "nx.DiGraph | None" = None) -> tuple[list[Signal], list[tuple[Signal, int]]]:
        """Превращает событие в сигналы: немедленные и отложенные.

        v3: Для правил с graph_op — создаёт граф-сигнал (district/industry из Event).
        Для правил с field — создаёт агентный сигнал (как раньше).

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

            # v3: Чисто граф-правило (нет field/delta для df)
            if rule.graph_op:
                # Граф-сигнал: не нужна маска агентов, district/industry из Event
                if event.source_district is None:
                    continue

                # graph_delta: из Event.n_agents_affected, иначе из Event.size → _size_to_jobs
                g_delta = rule.graph_delta
                if event.n_agents_affected > 0:
                    g_delta = float(event.n_agents_affected)
                elif event.size:
                    g_delta = float(_size_to_jobs(event.size))

                sig = Signal(
                    target_mask=np.zeros(len(df), dtype=bool),  # пустая маска
                    field="",                                     # нет поля df
                    mode="",                                      # нет режима df
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

            # ── Агентное правило (стандартный путь) ───────────────────────
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
        """Строит булеву маску для правила на основе target_scope и фильтров."""
        n = len(df)
        scope = rule.target_scope

        # ── Базовый гео-скоуп ────────────────────────────────────────────
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
            # Векторизовано: pandas .map вместо цикла по агентам
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

        # ── Фильтр по статусу ────────────────────────────────────────────
        if rule.filter_status == "unemployed":
            base = base & (df["status"].values == "unemployed")
        elif rule.filter_status == "employed":
            base = base & (df["status"].values != "unemployed")

        # ── Фильтр по отрасли ────────────────────────────────────────────
        if rule.filter_industry is not None:
            base = base & (df["industry"].values == rule.filter_industry)

        # ── v3: Фильтр wage_pressure > 1 ─────────────────────────────────
        if rule.filter_wage_pressure and G is not None and event.industry:
            # Нужен импорт _industry_wage_in_district из engine — ленивый
            base = base & _wage_pressure_mask(df, G, event, base)

        # ── Фильтр по образованию ───────────────────────────────────────
        if rule.filter_education is not None:
            base = base & (df["education"].values == rule.filter_education)

        # ── Фильтр по той же отрасли, что и событие ─────────────────────
        if rule.filter_same_industry and event.industry:
            base = base & (df["industry"].values == event.industry)

        # ── Исключаем самого агента-источника (кроме scope=self) ─────────
        if scope != SCOPE_SELF and event.source_agent_id is not None:
            base = base & (df["id"].values != event.source_agent_id)

        return base


# ══════════════════════════════════════════════════════════════════════════════
# 6. EventBus — шина событий
# ══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """Центральная шина событий.

    Две очереди событий:
      pending   — события, поступившие в этом тике, ещё не обработаны
      scheduled — события с задержкой, ждут своего тика

    Очередь отложенных сигналов:
      _delayed_signals — {tick: [Signal]} — сигналы, ждущие доставки

    Методы:
      emit(event)                — добавить событие в pending
      process(current_tick, df, G) — обработать pending + scheduled,
                                     вернуть список Signal
      flush(df, signals)         — применить сигналы к DataFrame
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
        """Добавляет событие в очередь pending."""
        self.pending.append(event)
        self._stats["emitted"] += 1

    def process(self, current_tick: int, df: pd.DataFrame,
                G: "nx.DiGraph | None" = None) -> list[Signal]:
        """Обрабатывает все pending-события + scheduled, готовые к этому тику.

        Возвращает список Signal для немедленного применения.
        Отложенные сигналы (delay_ticks > 0) сохраняются в _delayed_signals.
        """
        all_signals: list[Signal] = []

        # 0. Извлекаем отложенные сигналы для этого тика
        if current_tick in self._delayed_signals:
            all_signals.extend(self._delayed_signals.pop(current_tick))

        # 1. Перенос scheduled → pending для событий, чьё время пришло
        due = [e for e in self.scheduled if e.deliver_at_tick <= current_tick]
        still_waiting = [e for e in self.scheduled if e.deliver_at_tick > current_tick]
        self.scheduled = still_waiting
        self.pending.extend(due)

        # 2. Диспетчеризация pending
        for event in self.pending:
            signals_now, delayed_signals = self.dispatcher.dispatch(event, df, G)
            all_signals.extend(signals_now)
            self._stats["processed"] += 1
            self._stats["signals_generated"] += len(signals_now)

            # Сохраняем отложенные сигналы для будущих тиков
            for sig, deliver_at in delayed_signals:
                self._delayed_signals.setdefault(deliver_at, []).append(sig)
                self._stats["scheduled_count"] += 1

        self.pending.clear()
        return all_signals

    def flush(self, df: pd.DataFrame, signals: list[Signal],
              G: "nx.DiGraph | None" = None) -> pd.DataFrame:
        """Применяет сигналы к DataFrame (и опционально к графу G). v3: +G."""
        for sig in signals:
            df = sig.apply(df, G)

            # v2: отслеживание decay для social_boost
            if sig.event_type is not None and sig.field == "social_boost" and sig.mode == "add":
                self._update_sb_pending(df, sig)

        return df

    def _update_sb_pending(self, df: pd.DataFrame, sig: Signal) -> None:
        """Обновляет sb_pending для отслеживания linear-decay social_boost. (ВЕКТОРИЗОВАНО)"""
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

        # Векторизовано: работаем только с затронутыми агентами (маска), не со всеми n
        current = df["sb_pending"].values.copy()
        affected = current[mask]
        # Маска для пустых/NaN значений среди затронутых
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
        """Помещает событие в scheduled-очередь с задержкой."""
        event.deliver_at_tick = deliver_at_tick
        self.scheduled.append(event)
        self._stats["scheduled_count"] += 1

    @property
    def stats(self) -> dict:
        """Статистика шины."""
        return dict(self._stats)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Фабрика правил по умолчанию
# ══════════════════════════════════════════════════════════════════════════════

# Константы — соответствуют текущим значениям из engine.py
EVENT_SOCIAL_BOOST    = 0.08   # базовая добавка social_boost от события
UNEMPLOYED_SIGNAL     = 0.35   # добавка к signal_reduction при потере работы
NEIGHBOR_SIGNAL_COEF  = 0.04   # коэфф. сигнала от переехавшего соседа

# v3: размеры компаний → рабочие места
_SIZE_TO_JOBS = {"small": 50, "medium": 250, "big": 1000}


def _size_to_jobs(size: str) -> int:
    """Переводит размер компании в примерное число рабочих мест."""
    return _SIZE_TO_JOBS.get(size, 50)


def create_default_dispatcher() -> Dispatcher:
    """Создаёт Dispatcher v2 с обновлёнными правилами сигнальной системы.

    Новое в v2:
      - social_boost: раздельные decay для MOVE (+0.06, −0.01×6) и COMMUTE (+0.02, сброс через 3)
      - inertia_mobility_penalty: накопление от MOVE соседей (−0.06, −0.01×6)
      - LOST_JOB: немедленное снижение inertia (−0.25) + рост econ_gap (+0.25) + ramp
      - NEW_EMPLOYER/CLOSED_EMPLOYER: econ_penalty с условием wage_pressure>1
      - NEW_INFRA/CLOSED_INFRA: infra_bonus ±0.05
    """
    d = Dispatcher()

    # ═══════════════════════════════════════════════════════════════════════
    # AGENT_MOVED — v2: social_boost +0.06 (было 0.08), новый decay
    # ═══════════════════════════════════════════════════════════════════════
    # Соседи по старому району: social_boost +0.01 (MOVE decay: −0.005×6)
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="social_boost",
        base_delta=0.01,
    ))
    # Соседи по старому району: inertia_mobility_penalty −0.01 (decay: −0.005×6)
    # Отрицательный знак: переезд соседа понижает инерцию, делая миграцию более вероятной
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="inertia_mobility_penalty",
        base_delta=-0.01,
        clip_min=-1.0,
        clip_max=1.0,
    ))
    # Соседи по новому району: позитивный сигнал → social_boost +0.01
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_TARGET_NEIGHBORS,
        field="social_boost",
        base_delta=0.01,
    ))

    # ── AGENT_MOVED (place): place_deficit_penalty соседям того же settlement ─
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

    # ── AGENT_MOVED: signal_reduction соседям старого района ──────────────
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
    # ── AGENT_MOVED (economic) → econ_penalty низкообразованным соседям той же отрасли ─
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
    # AGENT_COMMUTE_STARTED — v2: social_boost +0.02, сброс через 3 тика
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
    d.add_rule(Rule(
        event_type=EventType.JOB_CHANGED,
        target_scope=SCOPE_WORKPLACE_COLLEAGUES,
        field="social_boost",
        base_delta=EVENT_SOCIAL_BOOST * 0.8,       # 0.064
    ))
    # ── v3: JOB_CHANGED → soc_calibration_signal коллегам ─────────────────
    d.add_rule(Rule(
        event_type=EventType.JOB_CHANGED,
        target_scope=SCOPE_WORKPLACE_COLLEAGUES,
        field="soc_calibration_signal",
        base_delta=0.03,
        scale_by_field="net_signal_susc",
        clip_min=0.0,
        clip_max=1.0,
    ))
    # ── JOB_CHANGED → econ_penalty низкообразованным коллегам той же отрасли ─
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
    # LOST_JOB — v2: новый агентный сигнал потери работы
    # ═══════════════════════════════════════════════════════════════════════
    # Сам агент: inertia −0.25 (немедленно)
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_SELF,
        field="inertia",
        base_delta=-0.25,
        clip_min=0.05,
        clip_max=0.95,
    ))
    # Сам агент: econ_gap +0.25 (немедленно, + ramp в tick)
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_SELF,
        field="econ_gap",
        base_delta=0.25,
        clip_min=0.0,
        clip_max=1.0,
    ))
    # Сам агент: signal_reduction +0.35 (шок от потери работы)
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_SELF,
        field="signal_reduction",
        base_delta=UNEMPLOYED_SIGNAL,            # 0.35
    ))
    # Сам агент: intention_state → seeking_work
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_SELF,
        field="intention_state",
        mode="set",
        value="seeking_work",
    ))
    # Коллеги по workplace_district: inertia +0.02 (decay: −0.01×2)
    d.add_rule(Rule(
        event_type=EventType.LOST_JOB,
        target_scope=SCOPE_WORKPLACE_COLLEAGUES,
        field="inertia_mobility_penalty",
        base_delta=0.02,
        clip_min=0.0,
        clip_max=1.0,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # JOB_LOST (deprecated, оставлен для обратной совместимости)
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
    # NEW_EMPLOYER — v3: граф-сигнал (vacant_jobs) + агент-сигналы (social_boost, econ_penalty)
    # ═══════════════════════════════════════════════════════════════════════
    # Граф: добавляем вакансии в отрасли района
    d.add_rule(Rule(
        event_type=EventType.NEW_EMPLOYER,
        target_scope=SCOPE_SELF,           # игнорируется для graph_op
        field="",                           # нет поля df
        graph_op="add_vacant",
        graph_delta=50,                     # переопределяется из Event.size
    ))
    # Агенты всего региона: social_boost
    d.add_rule(Rule(
        event_type=EventType.NEW_EMPLOYER,
        target_scope=SCOPE_WHOLE_REGION,
        field="social_boost",
        base_delta=0.05,
    ))
    # v3: NEW_EMPLOYER → soc_calibration_signal всему региону
    d.add_rule(Rule(
        event_type=EventType.NEW_EMPLOYER,
        target_scope=SCOPE_WHOLE_REGION,
        field="soc_calibration_signal",
        base_delta=0.03,
        scale_by_field="net_signal_susc",
        clip_min=0.0,
        clip_max=1.0,
    ))
    # Агенты той же отрасли в том же районе с wage_pressure>1: econ_penalty
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
    # CLOSED_EMPLOYER — v3: граф-сигнал (occupied_jobs) + агент-сигналы
    # ═══════════════════════════════════════════════════════════════════════
    # Граф: уменьшаем занятые места в отрасли района
    d.add_rule(Rule(
        event_type=EventType.CLOSED_EMPLOYER,
        target_scope=SCOPE_SELF,
        field="",
        graph_op="sub_occupied",
        graph_delta=50,                     # переопределяется из Event.size / n_agents_affected
    ))
    # Агенты всего региона (занятые): aspirations ↑
    d.add_rule(Rule(
        event_type=EventType.CLOSED_EMPLOYER,
        target_scope=SCOPE_WHOLE_REGION,
        field="aspirations",
        base_delta=0.08,
        filter_status="employed",
    ))
    # v3: CLOSED_EMPLOYER → soc_calibration_signal снижается в регионе ─────
    d.add_rule(Rule(
        event_type=EventType.CLOSED_EMPLOYER,
        target_scope=SCOPE_WHOLE_REGION,
        field="soc_calibration_signal",
        base_delta=-0.03,
        scale_by_field="net_signal_susc",
        clip_min=0.0,
        clip_max=1.0,
    ))
    # Агенты той же отрасли в том же районе с wage_pressure>1: econ_penalty сброс
    d.add_rule(Rule(
        event_type=EventType.CLOSED_EMPLOYER,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="econ_penalty",
        mode="set",
        value=0.0,
        filter_wage_pressure=True,
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
    # Старые правила для обратной совместимости (FACTORY_CLOSED, HOUSING_SHOCK,
    # EMPLOYER_OPENED)
    # ═══════════════════════════════════════════════════════════════════════

    # ── FACTORY_CLOSED ─────────────────────────────────────────────────────
    d.add_rule(Rule(
        event_type=EventType.FACTORY_CLOSED,
        target_scope=SCOPE_WHOLE_REGION,
        field="aspirations",
        base_delta=0.12,
        filter_status="employed",
    ))
    d.add_rule(Rule(
        event_type=EventType.FACTORY_CLOSED,
        target_scope=SCOPE_WHOLE_REGION,
        field="inertia",
        base_delta=-0.08,
        filter_status="employed",
    ))
    d.add_rule(Rule(
        event_type=EventType.FACTORY_CLOSED,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="aspirations",
        base_delta=0.20,
        filter_status="employed",
    ))
    d.add_rule(Rule(
        event_type=EventType.FACTORY_CLOSED,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="signal_reduction",
        base_delta=0.25,
    ))

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
