"""Shared fixtures for persona-pipeline unit tests."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable

import pytest

from src.scripts.persona.config.schema import JitterConfig, PersonaEventStage, Weekday
from src.scripts.persona.domain.persona import Person

EXAMPLES_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "scripts"
    / "persona"
    / "config"
    / "examples"
)


@pytest.fixture
def environment_yaml() -> Path:
    return EXAMPLES_DIR / "environment.yaml"


@pytest.fixture
def persona_yaml() -> Path:
    return EXAMPLES_DIR / "persona_config.yaml"


@pytest.fixture
def event_yaml() -> Path:
    return EXAMPLES_DIR / "event_config.yaml"


@pytest.fixture
def rules_yaml() -> Path:
    return EXAMPLES_DIR / "temporal_relation_rules.yaml"


# -------------------------------------------------------------------------------------
# Shared person/stage factories
# -------------------------------------------------------------------------------------

ALL_DAYS: list[Weekday] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def make_stage(
    name: str,
    *,
    time: str | None = None,
    duration_minutes: int | None = None,
    days: Iterable[Weekday] = ALL_DAYS,
    date: _dt.date | None = None,
) -> PersonaEventStage:
    """Compact helper for building a PersonaEventStage in tests."""
    return PersonaEventStage(
        name=name,
        time=time,
        duration_minutes=duration_minutes,
        days=list(days),
        date=date,
    )


def routine_stages() -> list[PersonaEventStage]:
    """Default daily routine: sleep + 3 meals on every day of the week.

    Mirrors the legacy "workdays_routine" defaults so tests that just
    need a person with a plausible daily routine can drop this in.
    """
    return [
        make_stage("sleep", time="23:00"),
        make_stage("first_eat", time="07:30", duration_minutes=20),
        make_stage("lunch", time="12:30", duration_minutes=45),
        make_stage("dinner", time="19:00", duration_minutes=60),
    ]


def make_person(
    *,
    person_id: str = "alice_0000",
    persona_id: str = "alice",
    person_seed: int = 42,
    instance_index: int = 0,
    occupation_status: str = "student",
    stages: Iterable[PersonaEventStage] | None = None,
    event_overrides: dict | None = None,
) -> Person:
    """Build a `Person` with a sensible default set of stages.

    `stages` defaults to `routine_stages()`. Pass an explicit list to
    override (e.g. add a `gym` stage on Mon/Wed/Fri).
    """
    return Person(
        person_id=person_id,
        persona_id=persona_id,
        person_seed=person_seed,
        instance_index=instance_index,
        occupation_status=occupation_status,
        stages=list(stages) if stages is not None else routine_stages(),
        jitter_applied=JitterConfig(),
        event_overrides=event_overrides or {},
    )
