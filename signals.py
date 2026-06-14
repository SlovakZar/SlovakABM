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
    """Типы событий в системе."""
    # Агентные
    AGENT_MOVED          = "AGENT_MOVED"
    AGENT_COMMUTE_STARTED = "AGENT_COMMUTE_STARTED"
    JOB_CHANGED          = "JOB_CHANGED"
    JOB_LOST             = "JOB_LOST"
    GRADUATED            = "GRADUATED"
    ADAPTED              = "ADAPTED"

    # Средовые (сценарные)
    FACTORY_CLOSED       = "FACTORY_CLOSED"
    EMPLOYER_OPENED      = "EMPLOYER_OPENED"
    HOUSING_SHOCK        = "HOUSING_SHOCK"
    INFRASTRUCTURE_CHANGE = "INFRASTRUCTURE_CHANGE"


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
    """
    target_mask: Union[np.ndarray, Callable[[pd.DataFrame], np.ndarray]]
    field: str
    delta: float = 0.0
    mode: str = "add"           # "add" | "set" | "multiply"
    value: object = None        # значение для mode="set" (строка, число)
    clip_min: float = 0.0
    clip_max: float = 1.0
    scale_by_field: Optional[str] = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет сигнал к DataFrame, возвращает изменённый df."""
        df = df.copy()
        if callable(self.target_mask):
            mask = self.target_mask(df)
        else:
            mask = self.target_mask

        if not mask.any():
            return df

        col = df[self.field].values.copy()

        if self.mode == "set":
            # Установка явного значения (может быть строкой)
            col[mask] = self.value if self.value is not None else self.delta
        else:
            # Вычисляем per-agent delta для add/multiply
            if self.scale_by_field and self.scale_by_field in df.columns:
                per_agent_delta = self.delta * df[self.scale_by_field].values
            else:
                per_agent_delta = np.full(len(df), self.delta)

            if self.mode == "add":
                col[mask] = col[mask] + per_agent_delta[mask]
            elif self.mode == "multiply":
                col[mask] = col[mask] * per_agent_delta[mask]

            col[mask] = np.clip(col[mask], self.clip_min, self.clip_max)

        df[self.field] = col
        return df


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
            # Вычисляем settlement для каждого агента по его району
            agent_settlements = np.array([
                _get_settlement(str(df.at[i, "district"]))
                for i in range(n)
            ])
            base = agent_settlements == event.settlement_type
        elif scope == SCOPE_WHOLE_REGION:
            if event.source_district is None:
                return np.zeros(n, dtype=bool)
            region = _get_region(event.source_district, G)
            agent_regions = np.array([
                _get_region(str(df.at[i, "district"]), G)
                for i in range(n)
            ])
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

    def flush(self, df: pd.DataFrame, signals: list[Signal]) -> pd.DataFrame:
        """Применяет сигналы к DataFrame. На шаге 1 — возвращает df as-is."""
        for sig in signals:
            df = sig.apply(df)
        return df

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


def create_default_dispatcher() -> Dispatcher:
    """Создаёт Dispatcher с правилами, воспроизводящими логику Блоков B и C.

    Правила:
      AGENT_MOVED (economic)  → social_boost соседям старого/нового района
      AGENT_MOVED (place)     → place_deficit_penalty соседям старого района
                                 (та же settlement, задержка 1)
      AGENT_COMMUTE_STARTED   → social_boost соседям по проживанию
      JOB_CHANGED             → social_boost коллегам по новому workplace
    """
    d = Dispatcher()

    # ── AGENT_MOVED (economic) ─────────────────────────────────────────────
    # Соседи по старому району: лёгкий шок → social_boost +0.048
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="social_boost",
        base_delta=EVENT_SOCIAL_BOOST * 0.6,       # 0.048
        motivation="economic",
    ))
    # Соседи по новому району: позитивный сигнал → social_boost +0.08
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_TARGET_NEIGHBORS,
        field="social_boost",
        base_delta=EVENT_SOCIAL_BOOST,             # 0.08
        motivation="economic",
    ))

    # ── AGENT_MOVED (place) ────────────────────────────────────────────────
    # Соседи по старому району (та же settlement): place_deficit_penalty +delta
    # Задержка 1 тик (будет учтена в Шаге 6; пока — мгновенно)
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

    # ── AGENT_COMMUTE_STARTED ──────────────────────────────────────────────
    # Соседи: умеренный позитивный сигнал → social_boost +0.04
    d.add_rule(Rule(
        event_type=EventType.AGENT_COMMUTE_STARTED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="social_boost",
        base_delta=EVENT_SOCIAL_BOOST * 0.5,       # 0.04
    ))

    # ── JOB_CHANGED (значимый рост зарплаты >20%) ─────────────────────────
    # Коллеги по новому workplace → social_boost +0.064
    d.add_rule(Rule(
        event_type=EventType.JOB_CHANGED,
        target_scope=SCOPE_WORKPLACE_COLLEAGUES,
        field="social_boost",
        base_delta=EVENT_SOCIAL_BOOST * 0.8,       # 0.064
    ))

    # ── JOB_LOST (Шаг 3: перенос Блока C) ─────────────────────────────────
    # Сам агент: signal_reduction +0.35
    d.add_rule(Rule(
        event_type=EventType.JOB_LOST,
        target_scope=SCOPE_SELF,
        field="signal_reduction",
        base_delta=UNEMPLOYED_SIGNAL,            # 0.35
    ))
    # Коллеги по отрасли в том же районе: aspirations +delta
    d.add_rule(Rule(
        event_type=EventType.JOB_LOST,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="aspirations",
        base_delta=0.06,
        filter_status="employed",
    ))
    # Весь регион: aspirations +слабее
    d.add_rule(Rule(
        event_type=EventType.JOB_LOST,
        target_scope=SCOPE_WHOLE_REGION,
        field="aspirations",
        base_delta=0.02,
        filter_status="employed",
    ))

    # ── AGENT_MOVED: signal_reduction соседям старого района ──────────────
    # Масштабируется на net_signal_susc каждого соседа
    d.add_rule(Rule(
        event_type=EventType.AGENT_MOVED,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="signal_reduction",
        base_delta=NEIGHBOR_SIGNAL_COEF,         # 0.04
        scale_by_field="net_signal_susc",        # × net_signal_susc
    ))

    # ── GRADUATED (Шаг 4) ─────────────────────────────────────────────────
    # Только сам агент: intention_state → seeking_work
    d.add_rule(Rule(
        event_type=EventType.GRADUATED,
        target_scope=SCOPE_SELF,
        field="intention_state",
        mode="set",
        value="seeking_work",
    ))

    # ── FACTORY_CLOSED (Шаг 5) ─────────────────────────────────────────────
    # Весь регион: aspirations +strong, inertia −delta
    d.add_rule(Rule(
        event_type=EventType.FACTORY_CLOSED,
        target_scope=SCOPE_WHOLE_REGION,
        field="aspirations",
        base_delta=0.12,                          # strong delta
        filter_status="employed",
    ))
    d.add_rule(Rule(
        event_type=EventType.FACTORY_CLOSED,
        target_scope=SCOPE_WHOLE_REGION,
        field="inertia",
        base_delta=-0.08,                         # снижение инерции
        filter_status="employed",
    ))
    # Та же отрасль + район: aspirations +очень сильно
    d.add_rule(Rule(
        event_type=EventType.FACTORY_CLOSED,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="aspirations",
        base_delta=0.20,
        filter_status="employed",
    ))
    # Та же отрасль + район: signal_reduction (шок)
    d.add_rule(Rule(
        event_type=EventType.FACTORY_CLOSED,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="signal_reduction",
        base_delta=0.25,
    ))

    # ── HOUSING_SHOCK (Шаг 5) ─────────────────────────────────────────────
    # Все агенты в районе: place_deficit_penalty +delta
    d.add_rule(Rule(
        event_type=EventType.HOUSING_SHOCK,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="place_deficit_penalty",
        base_delta=0.15,
        clip_min=0.0,
        clip_max=5.0,
    ))
    # Все агенты в районе: inertia −слабо
    d.add_rule(Rule(
        event_type=EventType.HOUSING_SHOCK,
        target_scope=SCOPE_RESIDENCE_NEIGHBORS,
        field="inertia",
        base_delta=-0.04,
    ))

    # ── EMPLOYER_OPENED (Шаг 5) ───────────────────────────────────────────
    # Та же отрасль + район: aspirations −delta (давление спадает)
    d.add_rule(Rule(
        event_type=EventType.EMPLOYER_OPENED,
        target_scope=SCOPE_SAME_INDUSTRY_DISTRICT,
        field="aspirations",
        base_delta=-0.06,
        filter_status="unemployed",
    ))
    # Весь регион: social_boost +delta
    d.add_rule(Rule(
        event_type=EventType.EMPLOYER_OPENED,
        target_scope=SCOPE_WHOLE_REGION,
        field="social_boost",
        base_delta=0.05,
    ))

    return d
