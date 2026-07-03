"""Unit tests for `applies_to` rule scoping (personas: + characteristic axes)."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from src.scripts.persona.config.loader import ConfigError, load_config
from src.scripts.persona.config.schema import (
    AllenPairRule,
    JitterConfig,
    SelectorPredicate,
    TemporalRule,
)
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.validation.check_ltl import check_ltl_rules


def _person(person_id: str, persona_id: str, **chars) -> Person:
    return Person(
        person_id=person_id,
        persona_id=persona_id,
        person_seed=1,
        instance_index=0,
        characteristics=chars,
        jitter_applied=JitterConfig(),
    )


def _schedule(person_id: str) -> PersonSchedule:
    day = DaySchedule(
        day_index=0,
        date=_dt.date(2026, 5, 4),
        weekday="Mon",
        events={
            "sleep": [EventInstance(event_name="sleep", start=0, duration=420)],
            "study": [EventInstance(event_name="study", start=0, duration=120)],
        },
        spillovers=[],
    )
    return PersonSchedule(
        person_id=person_id, persona_id="x", person_seed=1, days=[day]
    )


def test_allen_pair_rule_accepts_empty_applies_to_by_default():
    r = AllenPairRule(
        id="r",
        event_a=SelectorPredicate(name="a"),
        event_b=SelectorPredicate(name="b"),
        admissible_relations=["p"],
    )
    assert r.applies_to == {}


def test_allen_pair_rule_accepts_applies_to_personas_block():
    r = AllenPairRule(
        id="r",
        event_a=SelectorPredicate(name="a"),
        event_b=SelectorPredicate(name="b"),
        admissible_relations=["p"],
        applies_to={"personas": ["alice", "bob"]},
    )
    assert r.applies_to == {"personas": ["alice", "bob"]}


def test_temporal_rule_applies_to_accepts_int_and_bool_values():
    r = TemporalRule(
        id="r",
        formula="G ¬(sleep ∧ study)",
        applies_to={"age": [18, 25, 30], "has_kids": [True]},
    )
    assert r.applies_to["age"] == [18, 25, 30]
    assert r.applies_to["has_kids"] == [True]


def test_check_ltl_rules_filters_by_personas_field():
    rule = TemporalRule(
        id="no_sleep_study",
        formula="G ¬(sleep ∧ study)",
        applies_to={"personas": ["gym_rat"]},
    )
    sched = _schedule("couch_potato_0000")
    persons = {"couch_potato_0000": _person("couch_potato_0000", "couch_potato")}
    out = check_ltl_rules([sched], [rule], persons)
    assert out == []  # scoping skips couch_potato


def test_check_ltl_rules_fires_for_matching_personas_field():
    rule = TemporalRule(
        id="no_sleep_study",
        formula="G ¬(sleep ∧ study)",
        applies_to={"personas": ["gym_rat"]},
    )
    sched = _schedule("gym_rat_0000")
    persons = {"gym_rat_0000": _person("gym_rat_0000", "gym_rat")}
    out = check_ltl_rules([sched], [rule], persons)
    assert len(out) == 1


def test_check_ltl_rules_filters_by_characteristic_axis():
    rule = TemporalRule(
        id="no_sleep_study",
        formula="G ¬(sleep ∧ study)",
        applies_to={"socioeconomic_status": ["low"]},
    )
    sched = _schedule("rich_0000")
    persons = {
        "rich_0000": _person("rich_0000", "rich", socioeconomic_status="high"),
    }
    assert check_ltl_rules([sched], [rule], persons) == []


def test_check_ltl_rules_personas_and_characteristic_compose_with_and():
    rule = TemporalRule(
        id="r",
        formula="G ¬(sleep ∧ study)",
        applies_to={"personas": ["c"], "socioeconomic_status": ["low"]},
    )
    sched = _schedule("c_0000")
    persons_no_match = {"c_0000": _person("c_0000", "c", socioeconomic_status="high")}
    assert check_ltl_rules([sched], [rule], persons_no_match) == []
    persons_match = {"c_0000": _person("c_0000", "c", socioeconomic_status="low")}
    assert len(check_ltl_rules([sched], [rule], persons_match)) == 1


def test_check_ltl_rules_legacy_string_dict_still_works():
    """Old callers passing `{id: occupation_status}` strings keep working."""
    rule = TemporalRule(
        id="r",
        formula="G ¬(sleep ∧ study)",
        applies_to={"occupation_status": ["student"]},
    )
    sched = _schedule("p_0000")
    out = check_ltl_rules([sched], [rule], {"p_0000": "student"})
    assert len(out) == 1


def test_check_ltl_rules_returns_no_violation_when_person_missing_from_map():
    rule = TemporalRule(
        id="r",
        formula="G ¬(sleep ∧ study)",
        applies_to={"personas": ["x"]},
    )
    sched = _schedule("ghost_0000")
    out = check_ltl_rules([sched], [rule], {})
    assert out == []


def test_loader_rejects_applies_to_personas_unknown_id(tmp_path: Path):
    """The cross-check between rules and persona_config catches typos."""
    (tmp_path / "environment.yaml").write_text(
        """seed: 1
horizon:
  start_date: 2026-05-04
  weeks: 1
output:
  dir: ./out
time_windows:
  early_morning: [0, 400]
  morning: [400, 600]
  afternoon: [600, 960]
  evening: [960, 1260]
  night: [1260, 1440]
""",
        encoding="utf-8",
    )
    (tmp_path / "persona_config.yaml").write_text(
        """personas:
  - id: gym_rat
    instances: 1
    characteristics:
      occupation_status: { type: categorical, values: { fulltime: 1.0 } }
    stages:
      - { name: sleep, time: "23:00", days: [Mon, Tue, Wed, Thu, Fri, Sat, Sun] }
""",
        encoding="utf-8",
    )
    (tmp_path / "event_config.yaml").write_text(
        """categories:
  sleep:
    events:
      sleep:
        per_event_duration: { min: 6, max: 9, unit: hours }
        total_event_duration: { scale: day, min: 6, max: 9, unit: hours }
        total_event_episodes: { scale: day, min: 1, max: 1 }
""",
        encoding="utf-8",
    )
    (tmp_path / "rules.yaml").write_text(
        """rules:
  - id: typo
    formula: "G ¬(sleep ∧ sleep)"
    applies_to:
      personas: [gymm_rat]
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown persona"):
        load_config(
            environment=tmp_path / "environment.yaml",
            persona=tmp_path / "persona_config.yaml",
            event=tmp_path / "event_config.yaml",
            rules=tmp_path / "rules.yaml",
        )


def test_loader_accepts_applies_to_personas_on_allen_pair_rule(tmp_path: Path):
    (tmp_path / "environment.yaml").write_text(
        """seed: 1
horizon:
  start_date: 2026-05-04
  weeks: 1
output:
  dir: ./out
time_windows:
  early_morning: [0, 400]
  morning: [400, 600]
  afternoon: [600, 960]
  evening: [960, 1260]
  night: [1260, 1440]
""",
        encoding="utf-8",
    )
    (tmp_path / "persona_config.yaml").write_text(
        """personas:
  - id: gym_rat
    instances: 1
    characteristics:
      occupation_status: { type: categorical, values: { fulltime: 1.0 } }
    stages:
      - { name: sleep, time: "23:00", days: [Mon, Tue, Wed, Thu, Fri, Sat, Sun] }
""",
        encoding="utf-8",
    )
    (tmp_path / "event_config.yaml").write_text(
        """categories:
  sleep:
    events:
      sleep:
        per_event_duration: { min: 6, max: 9, unit: hours }
        total_event_duration: { scale: day, min: 6, max: 9, unit: hours }
        total_event_episodes: { scale: day, min: 1, max: 1 }
""",
        encoding="utf-8",
    )
    (tmp_path / "rules.yaml").write_text(
        """allen_pair_rules:
  - id: p
    event_a: sleep
    event_b: sleep
    admissible_relations: [p]
    applies_to:
      personas: [gym_rat]
""",
        encoding="utf-8",
    )
    cfg = load_config(
        environment=tmp_path / "environment.yaml",
        persona=tmp_path / "persona_config.yaml",
        event=tmp_path / "event_config.yaml",
        rules=tmp_path / "rules.yaml",
    )
    assert cfg.rules.allen_pair_rules[0].applies_to == {"personas": ["gym_rat"]}
