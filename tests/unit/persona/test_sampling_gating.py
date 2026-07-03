"""Unit tests for the generic `matches_requires` evaluator."""

from __future__ import annotations

from src.scripts.persona.config.schema import JitterConfig, RolePredicate
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.sampling.gating import matches_requires


def _person(**characteristics) -> Person:
    return Person(
        person_id="p_0000",
        persona_id="p",
        person_seed=1,
        instance_index=0,
        characteristics=characteristics,
        stages=[],
        jitter_applied=JitterConfig(),
    )


def _pred(**fields) -> RolePredicate:
    return RolePredicate.model_validate(fields)


def test_empty_requires_always_matches():
    assert matches_requires(_person(), {}) is True


def test_in_predicate_matches_when_value_in_list():
    person = _person(occupation_status="student")
    requires = {"occupation_status": _pred(**{"in": ["student", "parttime"]})}
    assert matches_requires(person, requires) is True


def test_in_predicate_rejects_when_value_not_in_list():
    person = _person(occupation_status="fulltime")
    requires = {"occupation_status": _pred(**{"in": ["student"]})}
    assert matches_requires(person, requires) is False


def test_missing_characteristic_fails_the_predicate():
    person = _person(age=30)
    requires = {"socioeconomic_status": _pred(eq="low")}
    assert matches_requires(person, requires) is False


def test_eq_predicate_on_boolean_value():
    person = _person(has_kids=True)
    assert matches_requires(person, {"has_kids": _pred(eq=True)}) is True
    assert matches_requires(person, {"has_kids": _pred(eq=False)}) is False


def test_ge_le_predicates_on_int_characteristic():
    person = _person(age=42)
    assert matches_requires(person, {"age": _pred(ge=18, le=65)}) is True
    assert matches_requires(person, {"age": _pred(ge=50)}) is False
    assert matches_requires(person, {"age": _pred(le=30)}) is False


def test_gt_lt_predicates_on_float_characteristic():
    person = _person(bmi=24.7)
    assert matches_requires(person, {"bmi": _pred(gt=18.5, lt=30.0)}) is True
    assert matches_requires(person, {"bmi": _pred(gt=25.0)}) is False
    assert matches_requires(person, {"bmi": _pred(lt=20.0)}) is False


def test_numeric_predicate_against_string_value_fails():
    person = _person(occupation_status="student")
    assert matches_requires(person, {"occupation_status": _pred(ge=1)}) is False


def test_numeric_predicate_against_boolean_value_fails():
    person = _person(flag=True)
    assert matches_requires(person, {"flag": _pred(ge=0)}) is False


def test_string_in_predicate_falls_back_to_string_equality_for_bools():
    """YAML `in: ['true']` should match a Python `True` characteristic."""
    person = _person(has_kids=True)
    assert matches_requires(person, {"has_kids": _pred(**{"in": ["true"]})}) is True


def test_eq_predicate_string_yaml_value_matches_python_bool():
    person = _person(has_kids=False)
    assert matches_requires(person, {"has_kids": _pred(eq="false")}) is True


def test_multiple_predicates_compose_with_and():
    person = _person(occupation_status="fulltime", age=40, has_kids=True)
    requires = {
        "occupation_status": _pred(**{"in": ["fulltime"]}),
        "age": _pred(ge=30, le=50),
        "has_kids": _pred(eq=True),
    }
    assert matches_requires(person, requires) is True
    person2 = _person(occupation_status="fulltime", age=40, has_kids=False)
    assert matches_requires(person2, requires) is False
