"""Scheduling solution: the full decision-variable assignment for one person."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.scripts.scenarios.domain.calendar import AugmentedCalendar
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask


@dataclass
class SchedulingSolution:
    """Full decision-variable assignment for one person's calendar augmentation.

    Captures the binary variables h_k, u_k, m_kj from the scheduling
    formulation without materialising them as explicit arrays; instead,
    membership in `scheduled` encodes h_k = 1, membership in
    `unscheduled` encodes h_k = 0, and
    `ScheduledTask.is_standalone / merged_into` encode u_k and m_kj.

    Fields:
        person_id: identifier matching CalendarTrace.person_id.
        augmented_calendar: the resulting X^G_aug (always provided by the
            augmenter; placed second so required fields group together).
        tasks: the full recommended-task set Y (scheduled + unscheduled).
        scheduled: tasks with h_k = 1 (placed as standalone or merged).
        unscheduled: tasks with h_k = 0 (no feasible slot was found).
    """

    person_id: str
    augmented_calendar: AugmentedCalendar
    tasks: list[RecommendedTask] = field(default_factory=list)
    scheduled: list[ScheduledTask] = field(default_factory=list)
    unscheduled: list[RecommendedTask] = field(default_factory=list)
