"""
scenario.py — Scheduled scenario events.

Classes:
  ScenarioEvent — one event tied to a tick
  Scenario      — loading and dispatching events by ticks

JSON format:
  [
    {"tick": 6, "type": "CLOSED_EMPLOYER",
     "district": "District of Žilina",
     "industry": "Manufacturing", "size": "medium"},
    {"tick": 18, "type": "NEW_EMPLOYER",
     "district": "District of Bratislava I",
     "industry": "ICT", "size": "big"}
  ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Forward reference — imported when needed
# from signals import EventType, Event


@dataclass
class ScenarioEvent:
    """Scenario event tied to a specific tick.

    Поля:
      tick              — tick at which the event occurs
      event_type        — event type string (ключ EventType)
      district          — district where it occurs
      industry          — industry (опционально)
      magnitude         — magnitude [0, 1]
      n_agents_affected — сколько agents затронуть (для FACTORY_CLOSED и т.п.)
      size              — employer size: small/medium/big (для NEW_EMPLOYER/CLOSED_EMPLOYER)
    """
    tick: int
    event_type: str
    district: str
    industry: Optional[str] = None
    magnitude: float = 0.5
    n_agents_affected: int = 0
    size: Optional[str] = None  # v2: small/medium/big

    def to_event(self, tick_num: int):
        """Converts to signals.Event (lazy import)."""
        from signals import Event, EventType

        et = EventType(self.event_type)
        return Event(
            event_type=et,
            tick_emitted=tick_num,
            source_district=self.district,
            industry=self.industry,
            magnitude=self.magnitude,
            size=self.size,
            n_agents_affected=self.n_agents_affected,
        )


class Scenario:
    """Stores scenario events and dispatches them by ticks."""

    def __init__(self, events: list[ScenarioEvent] | None = None):
        self._events: list[ScenarioEvent] = list(events) if events else []
        # Index: tick → list[ScenarioEvent]
        self._by_tick: dict[int, list[ScenarioEvent]] = {}
        for e in self._events:
            self._by_tick.setdefault(e.tick, []).append(e)

    def add(self, event: ScenarioEvent) -> None:
        """Adds one event."""
        self._events.append(event)
        self._by_tick.setdefault(event.tick, []).append(event)

    def get_events(self, tick: int) -> list[ScenarioEvent]:
        """Returns all events for the given tick."""
        return self._by_tick.get(tick, [])

    @staticmethod
    def from_json(path: str) -> "Scenario":
        """Loads scenario from JSON file."""
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        if not p.exists():
            # File not found — return empty scenario
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
        """Creates scenario from list of dictionaries (for embedding in code)."""
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
