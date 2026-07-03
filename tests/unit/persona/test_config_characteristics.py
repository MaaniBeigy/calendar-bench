"""Unit tests for the per-persona characteristic distribution schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.schema import (
    BooleanDist,
    CategoricalDist,
    ClipRange,
    JitterConfig,
    Persona,
    ScipyDist,
)


def _persona(**characteristics) -> Persona:
    return Persona(
        id="p",
        instances=4,
        characteristics=characteristics,
        stages=[],
    )


def test_categorical_dist_accepts_weights_that_sum_to_one():
    d = CategoricalDist(type="categorical", values={"a": 0.7, "b": 0.3})
    assert d.values == {"a": 0.7, "b": 0.3}


def test_categorical_dist_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        CategoricalDist(type="categorical", values={"a": 0.6, "b": 0.3})


def test_categorical_dist_rejects_negative_weight():
    with pytest.raises(ValidationError, match="negative weight"):
        CategoricalDist(type="categorical", values={"a": 1.2, "b": -0.2})


def test_categorical_dist_rejects_empty_values():
    with pytest.raises(ValidationError):
        CategoricalDist(type="categorical", values={})


def test_boolean_dist_accepts_p_true_in_unit_interval():
    assert BooleanDist(type="boolean", p_true=0.0).p_true == 0.0
    assert BooleanDist(type="boolean", p_true=1.0).p_true == 1.0


def test_boolean_dist_rejects_p_true_outside_unit_interval():
    with pytest.raises(ValidationError):
        BooleanDist(type="boolean", p_true=1.5)


def test_clip_range_requires_max_above_min():
    with pytest.raises(ValidationError, match="must exceed"):
        ClipRange(min=5, max=5)


def test_scipy_dist_norm_round_trips():
    d = ScipyDist(type="norm", params={"loc": 40, "scale": 8})
    assert d.type == "norm"
    assert d.params["loc"] == 40


def test_scipy_dist_rejects_unknown_distribution_name():
    with pytest.raises(ValidationError, match="unknown scipy.stats"):
        ScipyDist(type="gaussian", params={})


def test_scipy_dist_rejects_bad_params_kwarg():
    with pytest.raises(ValidationError, match="rejected"):
        ScipyDist(type="norm", params={"not_a_kwarg": 1.0})


def test_scipy_dist_rejects_reserved_type_names():
    for reserved in ("categorical", "boolean"):
        with pytest.raises(ValidationError, match="reserved"):
            ScipyDist(type=reserved, params={})


def test_scipy_dist_accepts_clip_and_dtype():
    d = ScipyDist(
        type="norm",
        params={"loc": 40, "scale": 8},
        clip={"min": 18, "max": 80},
        dtype="int",
    )
    assert d.clip is not None
    assert d.dtype == "int"


def test_scipy_dist_supports_poisson():
    ScipyDist(type="poisson", params={"mu": 7500})


def test_scipy_dist_supports_gamma_with_shape_and_scale():
    ScipyDist(type="gamma", params={"a": 2.5, "loc": 1000, "scale": 5000})


def test_persona_coerces_characteristics_dict_to_models():
    p = _persona(
        occupation=dict(type="categorical", values={"a": 1.0}),
        age=dict(type="norm", params={"loc": 40, "scale": 8}),
        has_kids=dict(type="boolean", p_true=0.3),
    )
    assert isinstance(p.characteristics["occupation"], CategoricalDist)
    assert isinstance(p.characteristics["age"], ScipyDist)
    assert isinstance(p.characteristics["has_kids"], BooleanDist)


def test_persona_legacy_occupation_status_desugars_into_characteristics():
    p = Persona(id="a_student", instances=4, occupation_status="student", stages=[])
    assert "occupation_status" in p.characteristics
    cat = p.characteristics["occupation_status"]
    assert isinstance(cat, CategoricalDist)
    assert cat.values == {"student": 1.0}


def test_persona_rejects_legacy_and_new_occupation_status_when_disagreeing():
    with pytest.raises(ValidationError, match="disagrees"):
        Persona(
            id="dual",
            instances=1,
            occupation_status="fulltime",
            characteristics={
                "occupation_status": {
                    "type": "categorical",
                    "values": {"parttime": 1.0},
                }
            },
            stages=[],
        )


def test_persona_accepts_legacy_and_matching_characteristics_round_trip():
    """Dumping then reloading a desugared persona must not raise."""
    p = Persona(id="r", instances=1, occupation_status="student", stages=[])
    Persona.model_validate(p.model_dump())


def test_persona_with_only_new_characteristics_block_loads():
    p = _persona(
        socioeconomic_status=dict(type="categorical", values={"low": 0.7, "high": 0.3}),
    )
    assert p.occupation_status is None
    assert "socioeconomic_status" in p.characteristics


def test_coerce_skips_non_dict_input():
    """The desugar leaves non-dict input alone."""
    p = Persona.model_validate(
        Persona(id="x", instances=1, occupation_status="student", stages=[])
    )
    assert isinstance(p, Persona)


def test_persona_rejects_legacy_with_non_categorical_existing_entry():
    with pytest.raises(ValidationError, match="disagrees"):
        Persona.model_validate(
            {
                "id": "n",
                "instances": 1,
                "occupation_status": "student",
                "characteristics": {
                    "occupation_status": {"type": "boolean", "p_true": 1.0}
                },
                "stages": [],
            }
        )


def test_persona_rejects_legacy_when_characteristic_has_wider_distribution():
    with pytest.raises(ValidationError, match="disagrees"):
        Persona.model_validate(
            {
                "id": "n",
                "instances": 1,
                "occupation_status": "student",
                "characteristics": {
                    "occupation_status": {
                        "type": "categorical",
                        "values": {"student": 0.5, "parttime": 0.5},
                    }
                },
                "stages": [],
            }
        )


def test_persona_characteristic_value_already_model_instance_round_trips():
    pre = CategoricalDist(type="categorical", values={"a": 1.0})
    p = Persona(id="m", instances=1, characteristics={"axis": pre}, stages=[])
    assert isinstance(p.characteristics["axis"], CategoricalDist)


def test_persona_characteristics_with_non_dict_value_is_left_alone():
    """A bad characteristics dict (non-dict entry) surfaces as a validation error."""
    with pytest.raises(ValidationError):
        Persona.model_validate(
            {
                "id": "x",
                "instances": 1,
                "characteristics": {"axis": "not_a_dict"},
                "stages": [],
            }
        )


def test_persona_legacy_value_matches_prebuilt_categorical_dist():
    """Legacy field and a hand-built `CategoricalDist` instance match cleanly."""
    pre = CategoricalDist(type="categorical", values={"fulltime": 1.0})
    p = Persona(
        id="m",
        instances=1,
        occupation_status="fulltime",
        characteristics={"occupation_status": pre},
        stages=[],
    )
    assert isinstance(p.characteristics["occupation_status"], CategoricalDist)


def test_persona_legacy_value_disagrees_with_prebuilt_boolean_dist():
    """Legacy `occupation_status` plus a hand-built non-categorical dist is rejected."""
    pre = BooleanDist(type="boolean", p_true=1.0)
    with pytest.raises(ValidationError, match="disagrees"):
        Persona(
            id="m",
            instances=1,
            occupation_status="fulltime",
            characteristics={"occupation_status": pre},
            stages=[],
        )


def test_persona_legacy_value_disagrees_with_prebuilt_categorical_dist():
    pre = CategoricalDist(type="categorical", values={"parttime": 1.0})
    with pytest.raises(ValidationError, match="disagrees"):
        Persona(
            id="m",
            instances=1,
            occupation_status="fulltime",
            characteristics={"occupation_status": pre},
            stages=[],
        )


def test_person_keeps_non_string_existing_characteristic_alongside_typed_field():
    """A non-string value in `characteristics['occupation_status']` leaves the typed field untouched."""
    from src.scripts.persona.domain.persona import Person

    p = Person(
        person_id="x_0000",
        persona_id="x",
        person_seed=1,
        instance_index=0,
        characteristics={"occupation_status": 1},  # int, not a str
        jitter_applied=JitterConfig(),
    )
    assert p.characteristics["occupation_status"] == 1
    assert p.occupation_status is None


def test_persona_independence_two_disjoint_axis_sets_load():
    couch = _persona(
        occupation=dict(type="categorical", values={"unemployed": 1.0}),
        screen_time=dict(type="expon", params={"scale": 240}),
    )
    gym = _persona(
        occupation=dict(type="categorical", values={"fulltime": 1.0}),
        weekly_workouts=dict(type="poisson", params={"mu": 4}),
    )
    assert set(couch.characteristics) == {"occupation", "screen_time"}
    assert set(gym.characteristics) == {"occupation", "weekly_workouts"}
