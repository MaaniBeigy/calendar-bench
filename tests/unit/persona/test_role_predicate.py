"""Unit tests for the extended `RolePredicate` predicate grammar."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.schema import RolePredicate


def test_in_predicate_accepts_string_list():
    p = RolePredicate.model_validate({"in": ["student", "parttime"]})
    assert p.in_ == ["student", "parttime"]


def test_in_predicate_accepts_mixed_value_types():
    p = RolePredicate.model_validate({"in": [True, 1, "yes"]})
    assert p.in_ == [True, 1, "yes"]


def test_eq_predicate_accepts_bool():
    p = RolePredicate.model_validate({"eq": True})
    assert p.eq is True


def test_eq_predicate_accepts_int_and_float():
    assert RolePredicate.model_validate({"eq": 18}).eq == 18
    assert RolePredicate.model_validate({"eq": 3.14}).eq == 3.14


def test_ge_predicate_is_recognised():
    p = RolePredicate.model_validate({"ge": 18})
    assert p.ge == 18
    assert p.le is None


def test_le_predicate_is_recognised():
    p = RolePredicate.model_validate({"le": 65})
    assert p.le == 65


def test_gt_lt_combination_is_allowed():
    p = RolePredicate.model_validate({"gt": 10, "lt": 50})
    assert p.gt == 10
    assert p.lt == 50


def test_ge_and_le_combination_is_allowed():
    p = RolePredicate.model_validate({"ge": 18, "le": 65})
    assert (p.ge, p.le) == (18, 65)


def test_empty_predicate_is_rejected():
    with pytest.raises(ValidationError, match="must set one of"):
        RolePredicate.model_validate({})


def test_unknown_predicate_key_is_rejected():
    with pytest.raises(ValidationError):
        RolePredicate.model_validate({"between": [1, 5]})
