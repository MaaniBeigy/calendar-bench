"""Per-person task entrypoint, picklable for use with `ProcessPoolExecutor`."""

from __future__ import annotations

from src.scripts.persona.config.schema import EnvironmentConfig, TemporalRelationRules
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.planner.horizon import plan_horizon


def run_person_task(
    person: Person,
    catalog: Catalog,
    environment: EnvironmentConfig,
    rules: TemporalRelationRules,
) -> PersonSchedule:
    """Solve one person's horizon; positional signature so it pickles for ProcessPoolExecutor."""
    window_map = WindowMap.from_config(environment.time_windows)
    return plan_horizon(
        person,
        catalog,
        environment,
        window_map=window_map,
        ltl_rules=list(rules.rules) if rules and rules.rules else None,
    )
