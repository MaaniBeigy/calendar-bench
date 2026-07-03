"""Recommended-task and scheduled-task domain types."""

from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass
class RecommendedTask:
    """Desired behavior task y_k = (label, duration band, intensity, flags, ontology)."""

    label: str
    duration_min: int
    duration_max: int
    intensity: int = 1
    is_dividable: bool = False
    is_concurrent: bool = False
    ontology_uri: str | None = None
    display_name: str = ""
    description: str = ""
    difficulty_level: int = 0

    @property
    def effective_display_name(self) -> str:
        """Human-readable title; falls back to title-cased label."""
        return self.display_name or self.label.replace("_", " ").title()

    @property
    def effective_description(self) -> str:
        """Description text with ontology citation, ready for display.

        * `description` set to use it; append `[uri]` if not already present.
        * `description` empty, `ontology_uri` set to `effective_display_name [uri]`.
        * Both empty to empty string (callers provide their own fallback).
        """
        if self.description:
            if self.ontology_uri and f"[{self.ontology_uri}]" not in self.description:
                return f"{self.description} [{self.ontology_uri}]"
            return self.description
        if self.ontology_uri:
            return f"{self.effective_display_name} [{self.ontology_uri}]"
        return ""


@dataclass
class ScheduledTask:
    """Realised event ê_k = (ℓ_k, τ_s,k, τ_e,k) plus decision-variable flags.

    Fields:
        task: the originating RecommendedTask.
        start_minutes: realised start in minutes from midnight (τ_s,k).
        end_minutes: realised end in minutes from midnight (τ_e,k).
        is_standalone: True when placed as an independent event (u_k = 1).
        concurrent_with: label of the existing calendar event this task is
            co-scheduled with (concurrent_with); None if standalone.
        date: calendar date on which the task is realised.
        parent_task_label: when set, this scheduled instance is one piece
            of a larger split; the label refers to the original
            `RecommendedTask.label`.  Used by `L_divide` to credit
            augmenters that honour `hb:isDividable` by emitting two or
            more `ScheduledTask` rows for one `RecommendedTask`.  Default
            `None` (the scheduled instance refers to its own label).
    """

    task: RecommendedTask
    start_minutes: int
    end_minutes: int
    is_standalone: bool
    concurrent_with: str | None
    date: datetime.date
    parent_task_label: str | None = None
