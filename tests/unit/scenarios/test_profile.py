"""Unit tests for src.scripts.scenarios.task_generation.profile.

Coverage targets:
  _format_value:
    - bool to true/false, float to two decimals, int / str passthrough
  profile_block:
    - every axis in declaration order by default
    - include list filters and orders the axes
    - axes the person does not carry are skipped silently
    - empty selection renders the (none) placeholder
  profile_summary:
    - one-line axis=value form, same selection rules
"""

from __future__ import annotations

from src.scripts.persona.config.schema import JitterConfig
from src.scripts.persona.domain.persona import Person
from src.scripts.scenarios.task_generation.profile import (
    profile_block,
    profile_summary,
)


def _make_person(characteristics: dict | None = None) -> Person:
    return Person(
        person_id="p_0000",
        persona_id="p",
        person_seed=42,
        instance_index=0,
        characteristics=characteristics or {},
        stages=[],
        jitter_applied=JitterConfig(),
        event_overrides={},
    )


_RICH = {
    "occupation_status": "fulltime",
    "gender": "female",
    "has_kids": True,
    "age": 50,
    "neuroticism": 0.38884503449169394,
    "annual_income_eur": 36127,
}


class TestProfileBlock:
    def test_renders_every_axis_in_declaration_order(self):
        block = profile_block(_make_person(_RICH))
        assert block == (
            "  - occupation_status: fulltime\n"
            "  - gender: female\n"
            "  - has_kids: true\n"
            "  - age: 50\n"
            "  - neuroticism: 0.39\n"
            "  - annual_income_eur: 36127"
        )

    def test_include_filters_and_orders_axes(self):
        block = profile_block(_make_person(_RICH), include=["age", "gender"])
        assert block == "  - age: 50\n  - gender: female"

    def test_include_skips_axes_person_lacks(self):
        block = profile_block(_make_person(_RICH), include=["age", "field_of_study"])
        assert block == "  - age: 50"

    def test_no_characteristics_renders_placeholder(self):
        assert profile_block(_make_person()) == "  - (none)"

    def test_empty_include_renders_placeholder(self):
        assert profile_block(_make_person(_RICH), include=[]) == "  - (none)"

    def test_false_boolean_rendered_as_false(self):
        block = profile_block(_make_person({"has_kids": False}))
        assert block == "  - has_kids: false"

    def test_occupation_shim_appears_via_characteristics_mirror(self):
        """A legacy occupation-only Person still returns a profile line."""
        person = Person(
            person_id="p_0000",
            persona_id="p",
            person_seed=42,
            instance_index=0,
            occupation_status="student",
            stages=[],
            jitter_applied=JitterConfig(),
            event_overrides={},
        )
        assert profile_block(person) == "  - occupation_status: student"


class TestProfileSummary:
    def test_one_line_axis_value_pairs(self):
        summary = profile_summary(_make_person(_RICH))
        assert summary == (
            "occupation_status=fulltime; gender=female; has_kids=true; "
            "age=50; neuroticism=0.39; annual_income_eur=36127"
        )

    def test_include_filters_and_orders_axes(self):
        summary = profile_summary(_make_person(_RICH), include=["gender", "age"])
        assert summary == "gender=female; age=50"

    def test_no_characteristics_renders_placeholder(self):
        assert profile_summary(_make_person()) == "(none)"
