"""Runtime types for the event catalog and individual event instances."""

from __future__ import annotations

from dataclasses import dataclass

from src.scripts.persona.config.schema import (
    Category,
    EventConfig,
    EventDefinition,
    EventOverride,
)


@dataclass(frozen=True, slots=True)
class EventInstance:
    """A concrete scheduled event (start + duration in minutes).

    `label` carries the per-episode display name when the catalog event sets
    `calendar_variations.task_labels` (e.g. an office_work episode shown as
    "standup"); it stays `None` for events with no task labels.
    """

    event_name: str
    start: int
    duration: int
    label: str | None = None


@dataclass(frozen=True)
class Catalog:
    """Indexed view of an EventConfig: O(1) lookup by event name and category."""

    categories: dict[str, Category]
    events_by_name: dict[str, EventDefinition]

    @classmethod
    def from_event_config(cls, event_config: EventConfig) -> Catalog:
        events_by_name: dict[str, EventDefinition] = {}
        for cat in event_config.categories.values():
            for event_name, event in cat.events.items():
                events_by_name[event_name] = event
        return cls(
            categories=dict(event_config.categories),
            events_by_name=events_by_name,
        )

    def get(self, event_name: str) -> EventDefinition:
        if event_name not in self.events_by_name:
            raise KeyError(f"event not found in catalog: {event_name!r}")
        return self.events_by_name[event_name]

    def by_category(self, category_name: str) -> list[EventDefinition]:
        if category_name not in self.categories:
            raise KeyError(f"category not found in catalog: {category_name!r}")
        return list(self.categories[category_name].events.values())

    def __contains__(self, event_name: object) -> bool:
        return event_name in self.events_by_name

    def __len__(self) -> int:
        return len(self.events_by_name)


def _merge_event_definition(
    base: EventDefinition, override: EventOverride
) -> EventDefinition:
    """Apply every set field on `override` over the corresponding field of `base`."""
    update: dict[str, object] = {}
    for field_name in EventOverride.model_fields:
        value = getattr(override, field_name)
        if value is not None:
            update[field_name] = value
    return base.model_copy(update=update)


def apply_event_overrides(
    catalog: Catalog, overrides: dict[str, EventOverride]
) -> Catalog:
    """Return a per-person catalog with persona-stated overrides applied.

    Overrides for event names absent from the global catalog are ignored:
    the persona schema does not let you introduce a brand new event from a
    partial override (no required fields like `per_event_duration` would
    be set), only retarget the constraints of an event the catalog
    already defines.
    """
    if not overrides:
        return catalog
    new_events: dict[str, EventDefinition] = {}
    for name, event_def in catalog.events_by_name.items():
        if name in overrides:
            new_events[name] = _merge_event_definition(event_def, overrides[name])
        else:
            new_events[name] = event_def
    new_categories: dict[str, Category] = {}
    for cat_name, cat in catalog.categories.items():
        new_cat_events: dict[str, EventDefinition] = {}
        for ev_name in cat.events:
            new_cat_events[ev_name] = new_events[ev_name]
        new_categories[cat_name] = Category(name=cat.name, events=new_cat_events)
    return Catalog(categories=new_categories, events_by_name=new_events)
