"""
scenario.py — Сценарные события по расписанию.

Классы:
  ScenarioEvent — одно событие с привязкой к тику
  Scenario      — загрузка и выдача событий по тикам

Формат JSON:
  [
    {"tick": 6, "type": "FACTORY_CLOSED",
     "district": "District of Žilina",
     "industry": "Manufacturing", "magnitude": 0.8, "n_agents": 400},
    {"tick": 18, "type": "EMPLOYER_OPENED",
     "district": "District of Bratislava I",
     "industry": "ICT", "magnitude": 0.6, "n_agents": 200}
  ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Forward reference — импортируется при необходимости
# from signals import EventType, Event


@dataclass
class ScenarioEvent:
    """Событие сценария, привязанное к конкретному тику.

    Поля:
      tick              — тик, на котором происходит событие
      event_type        — строка типа события (ключ EventType)
      district          — район, где происходит
      industry          — отрасль (опционально)
      magnitude         — сила [0, 1]
      n_agents_affected — сколько агентов затронуть (для FACTORY_CLOSED и т.п.)
      size              — размер работодателя: small/medium/big (для NEW_EMPLOYER/CLOSED_EMPLOYER)
    """
    tick: int
    event_type: str
    district: str
    industry: Optional[str] = None
    magnitude: float = 0.5
    n_agents_affected: int = 0
    size: Optional[str] = None  # v2: small/medium/big

    def to_event(self, tick_num: int):
        """Конвертирует в signals.Event (ленивый импорт)."""
        from signals import Event, EventType

        et = EventType(self.event_type)
        return Event(
            event_type=et,
            tick_emitted=tick_num,
            source_district=self.district,
            industry=self.industry,
            magnitude=self.magnitude,
            size=self.size,
        )


class Scenario:
    """Хранит список сценарных событий и выдаёт их по тикам."""

    def __init__(self, events: list[ScenarioEvent] | None = None):
        self._events: list[ScenarioEvent] = list(events) if events else []
        # Индекс: tick → list[ScenarioEvent]
        self._by_tick: dict[int, list[ScenarioEvent]] = {}
        for e in self._events:
            self._by_tick.setdefault(e.tick, []).append(e)

    def add(self, event: ScenarioEvent) -> None:
        """Добавляет одно событие."""
        self._events.append(event)
        self._by_tick.setdefault(event.tick, []).append(event)

    def get_events(self, tick: int) -> list[ScenarioEvent]:
        """Возвращает все события для данного тика."""
        return self._by_tick.get(tick, [])

    @staticmethod
    def from_json(path: str) -> "Scenario":
        """Загружает сценарий из JSON-файла."""
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        if not p.exists():
            # Файл не найден — возвращаем пустой сценарий
            return Scenario()

        with open(p, encoding="utf-8") as f:
            raw = json.load(f)

        events = []
        for item in raw:
            events.append(ScenarioEvent(
                tick=item["tick"],
                event_type=item["type"],
                district=item["district"],
                industry=item.get("industry"),
                magnitude=item.get("magnitude", 0.5),
                n_agents_affected=item.get("n_agents", 0),
                size=item.get("size"),
            ))
        return Scenario(events)

    @staticmethod
    def from_list(events: list[dict]) -> "Scenario":
        """Создаёт сценарий из списка словарей (для встраивания в код)."""
        scenario = Scenario()
        for item in events:
            scenario.add(ScenarioEvent(
                tick=item["tick"],
                event_type=item["type"],
                district=item["district"],
                industry=item.get("industry"),
                magnitude=item.get("magnitude", 0.5),
                n_agents_affected=item.get("n_agents", 0),
                size=item.get("size"),
            ))
        return scenario

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"Scenario({len(self._events)} events)"
