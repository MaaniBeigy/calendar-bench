"""Unit tests for src.scripts.persona.concurrency.tasks."""

from __future__ import annotations

import datetime as _dt
import pickle

from src.scripts.persona.concurrency.tasks import run_person_task
from src.scripts.persona.config.schema import (
    EnvironmentConfig,
    HorizonConfig,
    OutputConfig,
    ParallelismConfig,
    SolverConfig,
    TemporalRelationRules,
    WindowRange,
)
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule
from src.scripts.persona.event_config.loader import load_catalog
from tests.unit.persona.conftest import make_person, make_stage


def _env() -> EnvironmentConfig:
    return EnvironmentConfig(
        seed=1,
        horizon=HorizonConfig(start_date=_dt.date(2026, 5, 4), weeks=1),
        output=OutputConfig(dir="./out"),
        solver=SolverConfig(),
        time_windows={
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        },
        parallelism=ParallelismConfig(workers=1, executor="process"),
    )


def _person() -> Person:
    return make_person(
        person_id="alice_0000",
        persona_id="alice",
        person_seed=42,
        occupation_status="student",
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("first_eat", time="07:30", duration_minutes=20),
            make_stage("lunch", time="12:30", duration_minutes=45),
            make_stage("dinner", time="19:00", duration_minutes=60),
        ],
    )


def test_run_person_task_returns_person_schedule(event_yaml):
    catalog = load_catalog(event_yaml)
    out = run_person_task(_person(), catalog, _env(), TemporalRelationRules())
    assert isinstance(out, PersonSchedule)
    assert out.person_id == "alice_0000"
    assert len(out.days) == 7


def test_run_person_task_is_deterministic(event_yaml):
    catalog = load_catalog(event_yaml)
    a = run_person_task(_person(), catalog, _env(), TemporalRelationRules())
    b = run_person_task(_person(), catalog, _env(), TemporalRelationRules())
    assert a == b


def test_run_person_task_inputs_are_picklable(event_yaml):
    """All inputs must round-trip through pickle for `ProcessPoolExecutor`."""
    catalog = load_catalog(event_yaml)
    args = (_person(), catalog, _env(), TemporalRelationRules())
    restored = pickle.loads(pickle.dumps(args))
    out = run_person_task(*restored)
    assert isinstance(out, PersonSchedule)


def test_run_person_task_output_is_picklable(event_yaml):
    """The schedule must round-trip through pickle to cross the process boundary."""
    catalog = load_catalog(event_yaml)
    out = run_person_task(_person(), catalog, _env(), TemporalRelationRules())
    restored = pickle.loads(pickle.dumps(out))
    assert restored == out


def test_run_person_task_ignores_rules_in_phase_three(event_yaml):
    """LTL rules are accepted but not yet enforced; passing extras is a no-op."""
    catalog = load_catalog(event_yaml)
    rules_a = TemporalRelationRules(rules=[])
    rules_b = TemporalRelationRules.model_validate(
        {"rules": [{"id": "r", "formula": "G ¬(sleep ∧ lunch)"}]}
    )
    a = run_person_task(_person(), catalog, _env(), rules_a)
    b = run_person_task(_person(), catalog, _env(), rules_b)
    assert a == b
