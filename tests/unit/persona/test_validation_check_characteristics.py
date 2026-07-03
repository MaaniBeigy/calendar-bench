"""Unit tests for `check_characteristic_distributions` and its report row."""

from __future__ import annotations

from src.scripts.persona.config.schema import (
    BooleanDist,
    CategoricalDist,
    ClipRange,
    JitterConfig,
    Persona,
    PersonaConfig,
    ScipyDist,
)
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.validation.check_characteristics import (
    check_characteristic_distributions,
)


def _person(person_id: str, persona_id: str, **chars) -> Person:
    return Person(
        person_id=person_id,
        persona_id=persona_id,
        person_seed=1,
        instance_index=int(person_id[-4:]),
        characteristics=chars,
        jitter_applied=JitterConfig(),
    )


def _config(persona: Persona) -> PersonaConfig:
    return PersonaConfig(personas=[persona])


def test_categorical_passes_when_realized_counts_match_within_tolerance():
    persona = Persona(
        id="p",
        instances=10,
        characteristics={
            "socioeconomic": CategoricalDist(
                type="categorical", values={"low": 0.7, "high": 0.3}
            ),
        },
        stages=[],
    )
    persons = [
        _person(f"p_{i:04d}", "p", socioeconomic="low" if i < 7 else "high")
        for i in range(10)
    ]
    out = check_characteristic_distributions(_config(persona), persons)
    assert out == []


def test_categorical_flags_deviation_above_tolerance():
    persona = Persona(
        id="p",
        instances=10,
        characteristics={
            "x": CategoricalDist(type="categorical", values={"a": 0.5, "b": 0.5})
        },
        stages=[],
    )
    persons = [_person(f"p_{i:04d}", "p", x="a" if i < 8 else "b") for i in range(10)]
    out = check_characteristic_distributions(_config(persona), persons)
    kinds = {(v.axis, v.kind) for v in out}
    assert ("x", "count_mismatch") in kinds


def test_boolean_distribution_flags_count_mismatch():
    persona = Persona(
        id="p",
        instances=10,
        characteristics={"has_kids": BooleanDist(type="boolean", p_true=0.3)},
        stages=[],
    )
    persons = [
        _person(f"p_{i:04d}", "p", has_kids=True if i < 8 else False) for i in range(10)
    ]
    out = check_characteristic_distributions(_config(persona), persons)
    assert any(v.axis == "has_kids" and v.kind == "count_mismatch" for v in out)


def test_missing_value_for_some_instances_is_flagged():
    persona = Persona(
        id="p",
        instances=4,
        characteristics={"x": CategoricalDist(type="categorical", values={"a": 1.0})},
        stages=[],
    )
    persons = [_person(f"p_{i:04d}", "p", x="a") for i in range(3)]
    persons.append(_person("p_0003", "p"))  # missing x
    out = check_characteristic_distributions(_config(persona), persons)
    assert any(v.kind == "missing" for v in out)


def test_scipy_clip_violation_flagged_when_value_exceeds_bounds():
    persona = Persona(
        id="p",
        instances=2,
        characteristics={
            "age": ScipyDist(
                type="norm",
                params={"loc": 40, "scale": 8},
                clip=ClipRange(min=18, max=65),
                dtype="int",
            )
        },
        stages=[],
    )
    persons = [
        _person("p_0000", "p", age=40),
        _person("p_0001", "p", age=200),  # out of range
    ]
    out = check_characteristic_distributions(_config(persona), persons)
    assert any(v.axis == "age" and v.kind == "out_of_range" for v in out)


def test_scipy_without_clip_skips_range_check():
    persona = Persona(
        id="p",
        instances=2,
        characteristics={"steps": ScipyDist(type="poisson", params={"mu": 100})},
        stages=[],
    )
    persons = [_person(f"p_{i:04d}", "p", steps=100) for i in range(2)]
    out = check_characteristic_distributions(_config(persona), persons)
    assert out == []


def test_scipy_missing_value_under_clip_is_flagged():
    persona = Persona(
        id="p",
        instances=2,
        characteristics={
            "age": ScipyDist(
                type="norm",
                params={"loc": 40, "scale": 8},
                clip=ClipRange(min=18, max=65),
            )
        },
        stages=[],
    )
    persons = [
        _person("p_0000", "p", age=40),
        _person("p_0001", "p"),  # missing
    ]
    out = check_characteristic_distributions(_config(persona), persons)
    assert any(v.kind == "missing" for v in out)


def test_persona_with_no_instances_returns_zero_violations():
    persona = Persona(
        id="p",
        instances=4,
        characteristics={"x": CategoricalDist(type="categorical", values={"a": 1.0})},
        stages=[],
    )
    out = check_characteristic_distributions(_config(persona), [])
    assert out == []


def test_persona_with_multiple_axes_iterates_to_completion():
    """Scipy axis followed by a categorical axis exercises the loop tail."""
    persona = Persona(
        id="p",
        instances=2,
        characteristics={
            "age": ScipyDist(type="poisson", params={"mu": 40}),
            "tier": CategoricalDist(type="categorical", values={"a": 1.0}),
        },
        stages=[],
    )
    persons = [_person(f"p_{i:04d}", "p", age=40, tier="a") for i in range(2)]
    assert check_characteristic_distributions(_config(persona), persons) == []


def test_tolerance_zero_flags_every_off_by_one():
    persona = Persona(
        id="p",
        instances=10,
        characteristics={
            "x": CategoricalDist(type="categorical", values={"a": 0.5, "b": 0.5})
        },
        stages=[],
    )
    persons = [_person(f"p_{i:04d}", "p", x="a" if i < 6 else "b") for i in range(10)]
    out = check_characteristic_distributions(_config(persona), persons, tolerance=0)
    assert len(out) >= 1


def test_unexpected_label_flagged_on_pinned_axis():
    """A value outside the declared label set must not hide in the tolerance."""
    persona = Persona(
        id="p",
        instances=8,
        characteristics={
            "occupation_status": CategoricalDist(
                type="categorical", values={"fulltime": 1.0}
            )
        },
        stages=[],
    )
    persons = [
        _person(f"p_{i:04d}", "p", occupation_status="fulltime") for i in range(7)
    ] + [_person("p_0007", "p", occupation_status="student")]
    out = check_characteristic_distributions(_config(persona), persons)
    unexpected = [v for v in out if v.kind == "unexpected_label"]
    assert len(unexpected) == 1
    assert unexpected[0].axis == "occupation_status"
    assert "'student'" in unexpected[0].detail


def test_unexpected_label_flagged_for_boolean_axis():
    persona = Persona(
        id="p",
        instances=2,
        characteristics={"has_kids": BooleanDist(type="boolean", p_true=0.5)},
        stages=[],
    )
    persons = [
        _person("p_0000", "p", has_kids=True),
        _person("p_0001", "p", has_kids="yes"),
    ]
    out = check_characteristic_distributions(_config(persona), persons)
    assert any(v.kind == "unexpected_label" and v.axis == "has_kids" for v in out)


def test_declared_labels_never_flagged_as_unexpected():
    persona = Persona(
        id="p",
        instances=4,
        characteristics={
            "tier": CategoricalDist(type="categorical", values={"a": 0.5, "b": 0.5})
        },
        stages=[],
    )
    persons = [_person(f"p_{i:04d}", "p", tier="a" if i < 2 else "b") for i in range(4)]
    out = check_characteristic_distributions(_config(persona), persons)
    assert not any(v.kind == "unexpected_label" for v in out)
