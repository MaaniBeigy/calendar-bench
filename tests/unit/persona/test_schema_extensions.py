"""Tests for the persona-pipeline schema extensions.

Covers:
  - New EventDefinition fields: is_concurrent, is_dividable, concurrent_with
  - New EventOverride fields: is_concurrent, is_dividable, concurrent_with
  - New AllenPairRule model and AllenRelationToken validation
  - Extended TemporalRelationRules with allen_pair_rules
  - _merge_event_definition handling of new fields (via apply_event_overrides)
  - Loader cross-validation: _validate_concurrent_with, _validate_allen_pair_rules
  - Backward compatibility: experiments A/B/C and examples load without error
  - LTL checker regression: check_ltl_rules is unaffected by allen_pair_rules
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.loader import (
    ConfigError,
    _validate_allen_pair_rules,
    _validate_concurrent_with,
    load_config,
    load_event,
    load_rules,
)
from src.scripts.persona.config.schema import (
    AllenPairRule,
    EventConfig,
    EventOverride,
    PersonaConfig,
    TemporalRelationRules,
    TemporalRule,
)
from src.scripts.persona.domain.event import Catalog, apply_event_overrides
from src.scripts.persona.domain.schedule import (
    DaySchedule,
    EventInstance,
    PersonSchedule,
)
from src.scripts.persona.validation.check_ltl import check_ltl_rules

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS = _ROOT / "tests" / "fixtures" / "persona"
_EXAMPLES = _ROOT / "src" / "scripts" / "persona" / "config" / "examples"


def _exp(name: str) -> Path:
    return _EXPERIMENTS / name


# ---------------------------------------------------------------------------
# Minimal YAML helpers
# ---------------------------------------------------------------------------


def _minimal_event_raw(extra: dict | None = None) -> dict:
    """Minimal EventConfig dict with one event (sleep)."""
    event = {
        "per_event_duration": {"min": 6, "max": 9, "unit": "hours"},
        "total_event_duration": {"scale": "day", "min": 6, "max": 9, "unit": "hours"},
        "total_event_episodes": {"scale": "day", "min": 1, "max": 1},
    }
    if extra:
        event.update(extra)
    return {"categories": {"sleep": {"events": {"sleep": event}}}}


def _two_event_raw(second_extra: dict | None = None) -> dict:
    """EventConfig with sleep and lunch events."""
    sleep = {
        "per_event_duration": {"min": 6, "max": 9, "unit": "hours"},
        "total_event_duration": {"scale": "day", "min": 6, "max": 9, "unit": "hours"},
        "total_event_episodes": {"scale": "day", "min": 1, "max": 1},
    }
    lunch = {
        "per_event_duration": {"min": 30, "max": 90, "unit": "minutes"},
        "total_event_duration": {
            "scale": "day",
            "min": 30,
            "max": 90,
            "unit": "minutes",
        },
        "total_event_episodes": {"scale": "day", "min": 1, "max": 1},
    }
    if second_extra:
        lunch.update(second_extra)
    return {
        "categories": {
            "sleep": {"events": {"sleep": sleep}},
            "eat": {"events": {"lunch": lunch}},
        }
    }


def _minimal_persona_raw(event_overrides: dict | None = None) -> dict:
    persona = {
        "id": "p1",
        "instances": 1,
        "occupation_status": "student",
        "stages": [],
    }
    if event_overrides:
        persona["event_overrides"] = event_overrides
    return {"personas": [persona]}


def _make_rules(
    ltl_rules: list | None = None, allen_pair_rules: list | None = None
) -> TemporalRelationRules:
    return TemporalRelationRules.model_validate(
        {
            "rules": ltl_rules or [],
            "allen_pair_rules": allen_pair_rules or [],
        }
    )


# ---------------------------------------------------------------------------
# EventDefinition new fields
# ---------------------------------------------------------------------------


class TestEventDefinitionNewFields:
    def test_defaults_are_false_and_empty(self):
        cfg = EventConfig.model_validate(_minimal_event_raw())
        defn = cfg.categories["sleep"].events["sleep"]
        assert defn.is_concurrent is False
        assert defn.is_dividable is False
        assert defn.concurrent_with == []

    def test_is_concurrent_true(self):
        cfg = EventConfig.model_validate(_minimal_event_raw({"is_concurrent": True}))
        assert cfg.categories["sleep"].events["sleep"].is_concurrent is True

    def test_is_dividable_true(self):
        cfg = EventConfig.model_validate(_minimal_event_raw({"is_dividable": True}))
        assert cfg.categories["sleep"].events["sleep"].is_dividable is True

    def test_concurrent_with_list(self):
        cfg = EventConfig.model_validate(_two_event_raw({"concurrent_with": ["sleep"]}))
        assert cfg.categories["eat"].events["lunch"].concurrent_with == ["sleep"]

    def test_concurrent_with_empty_list_accepted(self):
        cfg = EventConfig.model_validate(_minimal_event_raw({"concurrent_with": []}))
        assert cfg.categories["sleep"].events["sleep"].concurrent_with == []

    def test_concurrent_with_defaults_are_independent_instances(self):
        """Two EventDefinitions must not share a mutable list object."""
        cfg = EventConfig.model_validate(_two_event_raw())
        sleep = cfg.categories["sleep"].events["sleep"]
        lunch = cfg.categories["eat"].events["lunch"]
        assert sleep.concurrent_with is not lunch.concurrent_with

    def test_existing_yaml_without_new_fields_gets_defaults(self):
        """Backward compat: absence of new fields to defaults applied cleanly."""
        raw = _minimal_event_raw()  # no is_concurrent / is_dividable / concurrent_with
        cfg = EventConfig.model_validate(raw)
        defn = cfg.categories["sleep"].events["sleep"]
        assert defn.is_concurrent is False
        assert defn.is_dividable is False
        assert defn.concurrent_with == []

    def test_all_three_new_fields_together(self):
        cfg = EventConfig.model_validate(
            _two_event_raw(
                {
                    "is_concurrent": True,
                    "is_dividable": True,
                    "concurrent_with": ["sleep"],
                }
            )
        )
        lunch = cfg.categories["eat"].events["lunch"]
        assert lunch.is_concurrent is True
        assert lunch.is_dividable is True
        assert lunch.concurrent_with == ["sleep"]


# ---------------------------------------------------------------------------
# EventOverride new fields
# ---------------------------------------------------------------------------


class TestEventOverrideNewFields:
    def test_new_fields_default_to_none(self):
        ov = EventOverride()
        assert ov.is_concurrent is None
        assert ov.is_dividable is None
        assert ov.concurrent_with is None

    def test_is_concurrent_override_true(self):
        ov = EventOverride(is_concurrent=True)
        assert ov.is_concurrent is True

    def test_is_concurrent_override_false(self):
        ov = EventOverride(is_concurrent=False)
        assert ov.is_concurrent is False

    def test_is_dividable_override_true(self):
        ov = EventOverride(is_dividable=True)
        assert ov.is_dividable is True

    def test_concurrent_with_override_list(self):
        ov = EventOverride(concurrent_with=["lunch"])
        assert ov.concurrent_with == ["lunch"]

    def test_concurrent_with_override_empty_list(self):
        ov = EventOverride(concurrent_with=[])
        assert ov.concurrent_with == []

    def test_concurrent_with_override_none(self):
        ov = EventOverride(concurrent_with=None)
        assert ov.concurrent_with is None


# ---------------------------------------------------------------------------
# AllenPairRule
# ---------------------------------------------------------------------------


class TestAllenPairRule:
    def test_valid_rule(self):
        rule = AllenPairRule(
            id="r1",
            event_a="sleep",
            event_b="running",
            admissible_relations=["p", "m", "M", "P"],
        )
        assert rule.id == "r1"
        # v4: event_a / event_b are SelectorPredicate; literal strings
        # are auto-promoted to `{"name": <str>}`.
        assert rule.event_a.name == "sleep"
        assert rule.event_b.name == "running"
        assert rule.admissible_relations == ["p", "m", "M", "P"]

    def test_all_thirteen_relation_tokens_accepted(self):
        tokens = ["p", "m", "o", "s", "d", "f", "e", "P", "M", "O", "S", "D", "F"]
        rule = AllenPairRule(
            id="r_all",
            event_a="a",
            event_b="b",
            admissible_relations=tokens,
        )
        assert set(rule.admissible_relations) == set(tokens)

    def test_empty_admissible_relations_rejected(self):
        with pytest.raises(ValidationError):
            AllenPairRule(
                id="r1", event_a="sleep", event_b="running", admissible_relations=[]
            )

    def test_invalid_relation_token_rejected(self):
        with pytest.raises(ValidationError):
            AllenPairRule(
                id="r1",
                event_a="sleep",
                event_b="running",
                admissible_relations=["x"],
            )

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            AllenPairRule(id="", event_a="a", event_b="b", admissible_relations=["p"])

    def test_single_relation_token(self):
        rule = AllenPairRule(
            id="r1", event_a="a", event_b="b", admissible_relations=["e"]
        )
        assert rule.admissible_relations == ["e"]


# ---------------------------------------------------------------------------
# TemporalRelationRules with allen_pair_rules
# ---------------------------------------------------------------------------


class TestTemporalRelationRulesExtended:
    def test_allen_pair_rules_defaults_to_empty(self):
        trr = TemporalRelationRules()
        assert trr.allen_pair_rules == []

    def test_allen_pair_rules_populated(self):
        trr = _make_rules(
            allen_pair_rules=[
                {
                    "id": "r1",
                    "event_a": "sleep",
                    "event_b": "running",
                    "admissible_relations": ["p", "m", "M", "P"],
                }
            ]
        )
        assert len(trr.allen_pair_rules) == 1
        assert trr.allen_pair_rules[0].id == "r1"

    def test_duplicate_id_within_allen_pair_rules_rejected(self):
        rule = {
            "id": "x",
            "event_a": "a",
            "event_b": "b",
            "admissible_relations": ["p"],
        }
        with pytest.raises(ValidationError, match="duplicate rule id"):
            _make_rules(allen_pair_rules=[rule, rule])

    def test_duplicate_id_across_rules_and_allen_pair_rules_rejected(self):
        with pytest.raises(ValidationError, match="duplicate rule id"):
            _make_rules(
                ltl_rules=[{"id": "shared", "formula": "G ¬(a ∧ b)"}],
                allen_pair_rules=[
                    {
                        "id": "shared",
                        "event_a": "a",
                        "event_b": "b",
                        "admissible_relations": ["p"],
                    }
                ],
            )

    def test_duplicate_id_in_rules_still_rejected(self):
        """Regression: existing behaviour for duplicates in rules is unchanged."""
        rule = {"id": "x", "formula": "G ¬(a ∧ b)"}
        with pytest.raises(ValidationError, match="duplicate temporal-rule id"):
            _make_rules(ltl_rules=[rule, rule])

    def test_yaml_without_allen_pair_rules_key_parses(self):
        """Backward compat: absence of allen_pair_rules key to empty list."""
        trr = TemporalRelationRules.model_validate(
            {"rules": [{"id": "r1", "formula": "G ¬(a ∧ b)"}]}
        )
        assert trr.allen_pair_rules == []
        assert len(trr.rules) == 1

    def test_mixed_rules_and_allen_pair_rules_unique_ids_pass(self):
        trr = _make_rules(
            ltl_rules=[{"id": "ltl1", "formula": "G ¬(a ∧ b)"}],
            allen_pair_rules=[
                {
                    "id": "apr1",
                    "event_a": "a",
                    "event_b": "b",
                    "admissible_relations": ["p"],
                }
            ],
        )
        assert len(trr.rules) == 1
        assert len(trr.allen_pair_rules) == 1


# ---------------------------------------------------------------------------
# _merge_event_definition via apply_event_overrides
# ---------------------------------------------------------------------------


class TestMergeEventDefinitionWithNewFields:
    """Verify that _merge_event_definition handles the new fields correctly.

    Tested indirectly through apply_event_overrides, which is the public API
    used by the calendar loader.
    """

    def _make_catalog(self, extra: dict | None = None) -> Catalog:
        return Catalog.from_event_config(
            EventConfig.model_validate(_minimal_event_raw(extra))
        )

    def test_is_concurrent_override_applied(self):
        catalog = self._make_catalog({"is_concurrent": False})
        overrides = {"sleep": EventOverride(is_concurrent=True)}
        result = apply_event_overrides(catalog, overrides)
        assert result.events_by_name["sleep"].is_concurrent is True

    def test_is_dividable_override_applied(self):
        catalog = self._make_catalog({"is_dividable": False})
        overrides = {"sleep": EventOverride(is_dividable=True)}
        result = apply_event_overrides(catalog, overrides)
        assert result.events_by_name["sleep"].is_dividable is True

    def test_concurrent_with_override_applied(self):
        catalog = self._make_catalog({"concurrent_with": []})
        overrides = {"sleep": EventOverride(concurrent_with=["lunch"])}
        result = apply_event_overrides(catalog, overrides)
        assert result.events_by_name["sleep"].concurrent_with == ["lunch"]

    def test_false_override_overrides_true_base(self):
        """is_concurrent=False on override should override base True."""
        catalog = self._make_catalog({"is_concurrent": True})
        overrides = {"sleep": EventOverride(is_concurrent=False)}
        result = apply_event_overrides(catalog, overrides)
        assert result.events_by_name["sleep"].is_concurrent is False

    def test_none_is_concurrent_preserves_base(self):
        """is_concurrent=None (default) must not override the base value."""
        catalog = self._make_catalog({"is_concurrent": True})
        overrides = {"sleep": EventOverride()}  # is_concurrent=None
        result = apply_event_overrides(catalog, overrides)
        assert result.events_by_name["sleep"].is_concurrent is True

    def test_none_is_dividable_preserves_base(self):
        catalog = self._make_catalog({"is_dividable": True})
        overrides = {"sleep": EventOverride()}
        result = apply_event_overrides(catalog, overrides)
        assert result.events_by_name["sleep"].is_dividable is True

    def test_none_concurrent_with_preserves_base(self):
        catalog = self._make_catalog({"concurrent_with": ["lunch"]})
        overrides = {"sleep": EventOverride()}  # concurrent_with=None
        result = apply_event_overrides(catalog, overrides)
        assert result.events_by_name["sleep"].concurrent_with == ["lunch"]

    def test_empty_list_concurrent_with_clears_base(self):
        """Explicit [] on override should clear the base list."""
        catalog = self._make_catalog({"concurrent_with": ["lunch"]})
        overrides = {"sleep": EventOverride(concurrent_with=[])}
        result = apply_event_overrides(catalog, overrides)
        assert result.events_by_name["sleep"].concurrent_with == []

    def test_no_overrides_returns_same_values(self):
        catalog = self._make_catalog(
            {"is_concurrent": True, "is_dividable": True, "concurrent_with": ["lunch"]}
        )
        result = apply_event_overrides(catalog, {})
        assert result.events_by_name["sleep"].is_concurrent is True
        assert result.events_by_name["sleep"].is_dividable is True
        assert result.events_by_name["sleep"].concurrent_with == ["lunch"]


# ---------------------------------------------------------------------------
# Loader validation: _validate_concurrent_with
# ---------------------------------------------------------------------------


class TestValidateConcurrentWith:
    def _make_event_config(self, extra: dict | None = None) -> EventConfig:
        return EventConfig.model_validate(_two_event_raw(extra))

    def _make_persona_config(
        self, event_overrides: dict | None = None
    ) -> PersonaConfig:
        return PersonaConfig.model_validate(_minimal_persona_raw(event_overrides))

    def test_no_concurrent_with_entries_passes(self):
        ec = self._make_event_config()
        pc = self._make_persona_config()
        _validate_concurrent_with(ec, pc)  # must not raise

    def test_valid_concurrent_with_in_event_definition_passes(self):
        ec = self._make_event_config({"concurrent_with": ["sleep"]})
        pc = self._make_persona_config()
        _validate_concurrent_with(ec, pc)  # must not raise

    def test_invalid_concurrent_with_in_event_definition_raises(self):
        ec = self._make_event_config({"concurrent_with": ["no_such_event"]})
        pc = self._make_persona_config()
        with pytest.raises(ConfigError, match="unknown event"):
            _validate_concurrent_with(ec, pc)

    def test_valid_concurrent_with_in_override_passes(self):
        ec = self._make_event_config()
        pc = self._make_persona_config(
            event_overrides={"lunch": {"concurrent_with": ["sleep"]}}
        )
        _validate_concurrent_with(ec, pc)  # must not raise

    def test_invalid_concurrent_with_in_override_raises(self):
        ec = self._make_event_config()
        pc = self._make_persona_config(
            event_overrides={"lunch": {"concurrent_with": ["ghost_event"]}}
        )
        with pytest.raises(ConfigError, match="unknown event"):
            _validate_concurrent_with(ec, pc)

    def test_none_concurrent_with_in_override_is_skipped(self):
        """override.concurrent_with = None must not trigger validation."""
        ec = self._make_event_config()
        pc = self._make_persona_config(
            event_overrides={"lunch": {"is_concurrent": True}}  # no concurrent_with key
        )
        _validate_concurrent_with(ec, pc)  # must not raise


# ---------------------------------------------------------------------------
# Loader validation: _validate_allen_pair_rules
# ---------------------------------------------------------------------------


class TestValidateAllenPairRules:
    def _make_event_config(self) -> EventConfig:
        return EventConfig.model_validate(_two_event_raw())

    def test_empty_allen_pair_rules_passes(self):
        ec = self._make_event_config()
        rules = _make_rules()
        _validate_allen_pair_rules(ec, rules)  # must not raise

    def test_valid_event_names_pass(self):
        ec = self._make_event_config()
        rules = _make_rules(
            allen_pair_rules=[
                {
                    "id": "r1",
                    "event_a": "sleep",
                    "event_b": "lunch",
                    "admissible_relations": ["p"],
                }
            ]
        )
        _validate_allen_pair_rules(ec, rules)  # must not raise

    def test_invalid_event_a_raises(self):
        ec = self._make_event_config()
        rules = _make_rules(
            allen_pair_rules=[
                {
                    "id": "r1",
                    "event_a": "ghost",
                    "event_b": "lunch",
                    "admissible_relations": ["p"],
                }
            ]
        )
        with pytest.raises(ConfigError, match="event_a"):
            _validate_allen_pair_rules(ec, rules)

    def test_invalid_event_b_raises(self):
        ec = self._make_event_config()
        rules = _make_rules(
            allen_pair_rules=[
                {
                    "id": "r1",
                    "event_a": "sleep",
                    "event_b": "phantom",
                    "admissible_relations": ["p"],
                }
            ]
        )
        with pytest.raises(ConfigError, match="event_b"):
            _validate_allen_pair_rules(ec, rules)


# ---------------------------------------------------------------------------
# Backward compatibility: existing experiments and examples load
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure every existing YAML set loads without error."""

    @pytest.mark.parametrize("exp", ["experiment_c"])
    def test_experiment_without_new_fields_loads(self, exp):
        """Experiment C has no new fields; defaults must kick in.

         Experiment B used to be here, but on 2026-05-14 it got explicit
         `is_concurrent` declarations (lunch / dinner / reading set to
         `true`) so the L_merge metric can score against this cohort
        ; without them every (task, event) pair was excluded by the
         catalog flag and `G_semantic_coscheduling_merge` reported
         `n/a` for every persona.  Experiment B is now covered by
         :meth:`test_experiment_b_with_new_fields_loads` below.
        """
        event_cfg = load_event(_exp(exp) / "event_config.yaml")
        # All events must have the new fields with their defaults.
        for cat in event_cfg.categories.values():
            for defn in cat.events.values():
                assert defn.is_concurrent is False
                assert defn.is_dividable is False
                assert defn.concurrent_with == []

    @pytest.mark.parametrize("exp", ["experiment_b", "experiment_c"])
    def test_experiment_rules_without_allen_pair_rules_loads(self, exp):
        """Experiments B and C have no allen_pair_rules; defaults to []."""
        rules = load_rules(_exp(exp) / "temporal_relation_rules.yaml")
        assert rules.allen_pair_rules == []

    def test_experiment_a_with_new_fields_loads(self):
        """Experiment A now has explicit new fields; must parse correctly."""
        event_cfg = load_event(_exp("experiment_a") / "event_config.yaml")
        sleep = event_cfg.categories["sleep"].events["sleep"]
        assert sleep.is_concurrent is False
        assert sleep.is_dividable is False
        lunch = event_cfg.categories["eat"].events["lunch"]
        assert lunch.is_concurrent is True
        assert "reading" in lunch.concurrent_with

    def test_experiment_b_with_new_fields_loads(self):
        """Experiment B declares `is_concurrent` flags so L_merge can
        score against the senior-leisure cohort.  Without these flags
        every (task, event) pair gets dropped by `is_excluded_pair`
        and `G_semantic_coscheduling_merge` reports `n/a` for every
        persona; the bug fixed on 2026-05-14.  This regression test
        guards the declarations.

        Researcher's design intent:
        Concurrency-friendly hosts; sustained activities that can
        host a layered habit OR a refinement of the activity itself:
          * `lunch` / `dinner`; mindful eating, posture check,
            hydration sip
          * `office_work`; 5 squats, hydration sip, looking outside
          * `reading`; breathing exercise, meditation, posture check
          * `walking` / `cycling` / `yoga`; REFINEMENT tasks
            specify how to perform the host (`Walk 5,000 Steps`
            during a Walking event hits a step-count goal within the
            slot; `Light Cycling Session` specifies intensity within
            the slot).
        Exclusive hosts:
          * `sleep` / `first_eat`; unconscious or too short.
          * `gardening`; hands and posture occupied.
          * `visit_family` / `meet_friends`; social commitments
            belong before/after.
        """
        event_cfg = load_event(_exp("experiment_b") / "event_config.yaml")
        # Concurrency-friendly hosts.
        lunch = event_cfg.categories["eat"].events["lunch"]
        assert lunch.is_concurrent is True
        dinner = event_cfg.categories["eat"].events["dinner"]
        assert dinner.is_concurrent is True
        office = event_cfg.categories["work"].events["office_work"]
        assert office.is_concurrent is True
        reading = event_cfg.categories["leisure"].events["reading"]
        assert reading.is_concurrent is True
        walking = event_cfg.categories["sports"].events["walking"]
        assert walking.is_concurrent is True
        cycling = event_cfg.categories["sports"].events["cycling"]
        assert cycling.is_concurrent is True
        yoga = event_cfg.categories["sports"].events["yoga"]
        assert yoga.is_concurrent is True
        # Exclusive hosts.
        sleep = event_cfg.categories["sleep"].events["sleep"]
        assert sleep.is_concurrent is False
        first_eat = event_cfg.categories["eat"].events["first_eat"]
        assert first_eat.is_concurrent is False
        gardening = event_cfg.categories["leisure"].events["gardening"]
        assert gardening.is_concurrent is False
        visit_family = event_cfg.categories["social"].events["visit_family"]
        assert visit_family.is_concurrent is False
        meet_friends = event_cfg.categories["social"].events["meet_friends"]
        assert meet_friends.is_concurrent is False

    def test_experiment_a_allen_pair_rules_loads(self):
        rules = load_rules(_exp("experiment_a") / "temporal_relation_rules.yaml")
        # v4: 6 rules; 2 literal back-compat + 4 selector-driven.
        assert len(rules.allen_pair_rules) == 6
        ids = {r.id for r in rules.allen_pair_rules}
        # Literal carryovers (event_a / event_b auto-promoted to SelectorPredicate).
        assert "sleep_separates_from_running" in ids
        assert "reading_may_overlap_lunch" in ids
        # Selector-driven additions.
        assert "intensive_far_from_intensive" in ids
        assert "lunch_far_from_intensive" in ids
        assert "dinner_far_from_intensive" in ids
        assert "reading_adjacent_to_lunch" in ids
        # Verify the selector grammar parsed correctly: `intensive_far_*`
        # carries an intensity predicate on both endpoints, not a literal
        # name match.
        by_id = {r.id: r for r in rules.allen_pair_rules}
        intensive = by_id["intensive_far_from_intensive"]
        assert intensive.event_a.name is None
        assert intensive.event_a.intensity == [3, 4]
        assert intensive.buffer == 60

    def test_examples_load_without_error(
        self,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        """Example config bundle must load and pass the new cross-validation."""
        cfg = load_config(environment_yaml, persona_yaml, event_yaml, rules_yaml)
        # New fields default to False/[]
        for cat in cfg.event.categories.values():
            for defn in cat.events.values():
                assert defn.is_concurrent is False
        assert cfg.rules.allen_pair_rules == []


# ---------------------------------------------------------------------------
# LTL checker regression
# ---------------------------------------------------------------------------


class TestLTLCheckerRegression:
    """Verify that allen_pair_rules being added to TemporalRelationRules
    has zero effect on check_ltl_rules, which only reads rules.rules."""

    def _make_schedule(self, person_id: str = "p001") -> PersonSchedule:
        import datetime

        day = DaySchedule(
            day_index=0,
            date=datetime.date(2026, 5, 4),
            weekday="Mon",
            events={
                "sleep": [EventInstance(event_name="sleep", start=1380, duration=480)],
                "office_work": [
                    EventInstance(event_name="office_work", start=540, duration=480)
                ],
            },
        )
        return PersonSchedule(
            person_id=person_id, persona_id="p1", person_seed=0, days=[day]
        )

    def test_no_violations_without_allen_pair_rules(self):
        """Baseline: the schedule has no overlap between sleep and office_work."""
        schedule = self._make_schedule()
        rule = TemporalRule(id="no_overlap", formula="G ¬(sleep ∧ office_work)")
        violations = check_ltl_rules([schedule], [rule], {"p001": "fulltime"})
        assert violations == []

    def test_ltl_checker_sees_only_rules_not_allen_pair_rules(self):
        """check_ltl_rules takes list[TemporalRule]; allen_pair_rules is a
        separate field and must not be passed to it (not even accidentally)."""
        trr = _make_rules(
            ltl_rules=[{"id": "r1", "formula": "G ¬(sleep ∧ office_work)"}],
            allen_pair_rules=[
                {
                    "id": "apr1",
                    "event_a": "sleep",
                    "event_b": "office_work",
                    "admissible_relations": ["p"],
                }
            ],
        )
        schedule = self._make_schedule()
        # Pass only trr.rules (list[TemporalRule]); this is the correct call pattern.
        violations = check_ltl_rules([schedule], trr.rules, {"p001": "fulltime"})
        assert violations == []

    def test_ltl_violation_still_detected_after_schema_change(self):
        """Adding allen_pair_rules to the schema must not suppress LTL violations."""
        import datetime

        # Create a schedule where sleep overlaps office_work (violation).
        day = DaySchedule(
            day_index=0,
            date=datetime.date(2026, 5, 4),
            weekday="Mon",
            events={
                "sleep": [EventInstance(event_name="sleep", start=0, duration=600)],
                "office_work": [
                    EventInstance(event_name="office_work", start=300, duration=300)
                ],
            },
        )
        schedule = PersonSchedule(
            person_id="p001", persona_id="p1", person_seed=0, days=[day]
        )
        rule = TemporalRule(id="no_overlap", formula="G ¬(sleep ∧ office_work)")
        violations = check_ltl_rules([schedule], [rule], {"p001": "fulltime"})
        assert len(violations) == 1
        assert violations[0].rule_id == "no_overlap"
