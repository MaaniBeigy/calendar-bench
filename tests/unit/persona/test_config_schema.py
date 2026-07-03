"""Unit tests for src.scripts.persona.config.schema invariants."""

from __future__ import annotations

import datetime as _dt

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.schema import (
    DurationRange,
    EnvironmentConfig,
    EpisodeRange,
    EventConfig,
    HorizonConfig,
    JitterConfig,
    OutputConfig,
    Persona,
    PersonaCommon,
    PersonaConfig,
    PersonaEventStage,
    RolePredicate,
    SolverConfig,
    TemporalRelationRules,
    TemporalRule,
    WindowRange,
)

# -------------------------------------------------------------------------------------
# ------------------------------ stage time-token validation --------------------------
# -------------------------------------------------------------------------------------


def test_stage_accepts_hhmm():
    stage = PersonaEventStage(
        name="lunch", time="07:30", duration_minutes=20, days=["Mon"]
    )
    assert stage.time == "07:30"


def test_stage_accepts_named_window():
    stage = PersonaEventStage(
        name="walk", time="morning", duration_minutes=30, days=["Mon"]
    )
    assert stage.time == "morning"


def test_stage_rejects_relative_anchor_token():
    """`after_dinner` / `after_lunch` are no longer valid time tokens -
    the schema is event-name agnostic."""
    with pytest.raises(ValidationError):
        PersonaEventStage(
            name="reading", time="after_dinner", duration_minutes=60, days=["Mon"]
        )


def test_stage_rejects_unknown_token():
    with pytest.raises(ValidationError):
        PersonaEventStage(name="x", time="whenever", duration_minutes=30, days=["Mon"])


def test_stage_rejects_bad_hour():
    with pytest.raises(ValidationError):
        PersonaEventStage(name="x", time="25:00", duration_minutes=30, days=["Mon"])


def test_stage_rejects_zero_duration():
    with pytest.raises(ValidationError):
        PersonaEventStage(name="x", time="07:00", duration_minutes=0, days=["Mon"])


def test_stage_requires_calendar_trigger():
    """A stage must have either `days` or `date` - the trigger that
    makes it fire on at least one day."""
    with pytest.raises(ValidationError):
        PersonaEventStage(name="x", time="07:00", duration_minutes=30, days=[])


def test_stage_accepts_date_only():
    stage = PersonaEventStage(
        name="dentist",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 12),
    )
    assert stage.date == _dt.date(2026, 5, 12)


def test_stage_optional_time_and_duration():
    """Both `time` and `duration_minutes` are optional - a stage can
    fire on its `days` trigger and let the catalog/solver pick start
    and duration."""
    stage = PersonaEventStage(name="x", days=["Mon"])
    assert stage.time is None
    assert stage.duration_minutes is None


# -------------------------------------------------------------------------------------
# ------------------------------------ window range -----------------------------------
# -------------------------------------------------------------------------------------


def test_window_range_rejects_inverted():
    with pytest.raises(ValidationError):
        WindowRange(start=600, end=400)


def test_window_range_rejects_zero_span():
    with pytest.raises(ValidationError):
        WindowRange(start=400, end=400)


# -------------------------------------------------------------------------------------
# --------------------------------- duration / episode --------------------------------
# -------------------------------------------------------------------------------------


def test_duration_rejects_max_lt_min():
    with pytest.raises(ValidationError):
        DurationRange(min=10, max=5, unit="minutes")


def test_episode_rejects_max_lt_min():
    with pytest.raises(ValidationError):
        EpisodeRange(min=2, max=1)


# -------------------------------------------------------------------------------------
# ------------------------------------ environment ------------------------------------
# -------------------------------------------------------------------------------------


def _environment_kwargs(**overrides):
    base = {
        "seed": 1,
        "horizon": HorizonConfig(start_date=_dt.date(2026, 5, 4), weeks=1),
        "output": OutputConfig(dir="./out"),
        "solver": SolverConfig(),
        "time_windows": {
            "early_morning": [0, 400],
            "morning": [400, 600],
            "afternoon": [600, 960],
            "evening": [960, 1260],
            "night": [1260, 1440],
        },
    }
    base.update(overrides)
    return base


def test_environment_requires_named_windows():
    kwargs = _environment_kwargs(time_windows={"morning": [400, 600]})
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate(kwargs)


def test_environment_seed_must_be_non_negative():
    kwargs = _environment_kwargs(seed=-1)
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate(kwargs)


def test_environment_window_list_coerced():
    kwargs = _environment_kwargs()
    cfg = EnvironmentConfig.model_validate(kwargs)
    assert cfg.time_windows["morning"].start == 400
    assert cfg.time_windows["morning"].end == 600


def test_environment_time_windows_must_be_dict():
    """Non-dict time_windows passes through the coercer untouched, then fails later."""
    kwargs = _environment_kwargs(time_windows=["early_morning", 0, 400])
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate(kwargs)


def test_environment_time_windows_already_dict_shape():
    """A window already in {start, end} dict form passes the coercer's else arm."""
    kwargs = _environment_kwargs(
        time_windows={
            "early_morning": {"start": 0, "end": 400},
            "morning": [400, 600],
            "afternoon": [600, 960],
            "evening": [960, 1260],
            "night": [1260, 1440],
        }
    )
    cfg = EnvironmentConfig.model_validate(kwargs)
    assert cfg.time_windows["early_morning"].start == 0
    assert cfg.time_windows["early_morning"].end == 400


# -------------------------------------------------------------------------------------
# ------------------------------------ daily window -----------------------------------
# -------------------------------------------------------------------------------------


def test_daily_window_defaults_06_00_to_22_00():
    from src.scripts.persona.config.schema import DailyWindow

    dw = DailyWindow()
    assert dw.wake_minutes == 360  # 06:00
    assert dw.sleep_minutes == 1320  # 22:00


def test_daily_window_custom_values():
    from src.scripts.persona.config.schema import DailyWindow

    dw = DailyWindow(wake_minutes=420, sleep_minutes=1380)
    assert dw.wake_minutes == 420
    assert dw.sleep_minutes == 1380


def test_daily_window_rejects_inverted():
    from src.scripts.persona.config.schema import DailyWindow

    with pytest.raises(ValidationError):
        DailyWindow(wake_minutes=1320, sleep_minutes=360)


def test_daily_window_rejects_zero_span():
    from src.scripts.persona.config.schema import DailyWindow

    with pytest.raises(ValidationError):
        DailyWindow(wake_minutes=600, sleep_minutes=600)


def test_daily_window_rejects_out_of_range():
    from src.scripts.persona.config.schema import DailyWindow

    with pytest.raises(ValidationError):
        DailyWindow(wake_minutes=-1)
    with pytest.raises(ValidationError):
        DailyWindow(sleep_minutes=1500)


def test_environment_includes_daily_window_default():
    kwargs = _environment_kwargs()
    cfg = EnvironmentConfig.model_validate(kwargs)
    assert cfg.daily_window.wake_minutes == 360
    assert cfg.daily_window.sleep_minutes == 1320


def test_environment_accepts_custom_daily_window():
    kwargs = _environment_kwargs(
        daily_window={"wake_minutes": 480, "sleep_minutes": 1380}
    )
    cfg = EnvironmentConfig.model_validate(kwargs)
    assert cfg.daily_window.wake_minutes == 480
    assert cfg.daily_window.sleep_minutes == 1380


# -------------------------------------------------------------------------------------
# -------------------------------------- compute --------------------------------------
# -------------------------------------------------------------------------------------


def test_compute_config_defaults_to_auto():
    from src.scripts.persona.config.schema import ComputeConfig

    c = ComputeConfig()
    assert c.device == "auto"
    assert c.cuda_device_index == 0


def test_compute_config_accepts_cpu_and_cuda():
    from src.scripts.persona.config.schema import ComputeConfig

    assert ComputeConfig(device="cpu").device == "cpu"
    assert ComputeConfig(device="cuda").device == "cuda"


def test_compute_config_rejects_unknown_device():
    from src.scripts.persona.config.schema import ComputeConfig

    with pytest.raises(ValidationError):
        ComputeConfig(device="tpu")


def test_compute_config_rejects_negative_cuda_index():
    from src.scripts.persona.config.schema import ComputeConfig

    with pytest.raises(ValidationError):
        ComputeConfig(cuda_device_index=-1)


def test_environment_includes_compute_default():
    kwargs = _environment_kwargs()
    cfg = EnvironmentConfig.model_validate(kwargs)
    assert cfg.compute.device == "auto"
    assert cfg.compute.cuda_device_index == 0


def test_environment_accepts_custom_compute_block():
    kwargs = _environment_kwargs(compute={"device": "cuda", "cuda_device_index": 1})
    cfg = EnvironmentConfig.model_validate(kwargs)
    assert cfg.compute.device == "cuda"
    assert cfg.compute.cuda_device_index == 1


# -------------------------------------------------------------------------------------
# -------------------------------------- persona --------------------------------------
# -------------------------------------------------------------------------------------


def _persona(id_="alice", **overrides):
    base = {
        "id": id_,
        "instances": 1,
        "occupation_status": "student",
        "stages": [],
    }
    base.update(overrides)
    return Persona(**base)


def test_persona_minimal_has_empty_stages():
    p = _persona()
    assert p.stages == []
    assert p.event_overrides == {}


def test_persona_with_stage_list():
    p = _persona(
        stages=[PersonaEventStage(name="sleep", time="23:00", days=["Mon", "Tue"])]
    )
    assert len(p.stages) == 1
    assert p.stages[0].name == "sleep"


def test_persona_id_pattern_rejects_spaces():
    with pytest.raises(ValidationError):
        _persona(id_="alice student")


def test_persona_config_rejects_duplicate_ids():
    p1 = _persona(id_="alice")
    p2 = _persona(id_="alice")
    with pytest.raises(ValidationError):
        PersonaConfig(common=PersonaCommon(), personas=[p1, p2])


def test_persona_config_requires_at_least_one():
    with pytest.raises(ValidationError):
        PersonaConfig(common=PersonaCommon(), personas=[])


def test_jitter_config_defaults():
    j = JitterConfig()
    assert j.time_minutes == 15
    assert j.duration_minutes == 10


# -------------------------------------------------------------------------------------
# ----------------------------------- event catalog -----------------------------------
# -------------------------------------------------------------------------------------


def _minimal_event_yaml() -> dict:
    return {
        "categories": {
            "sleep": {
                "events": {
                    "sleep": {
                        "per_event_duration": {"min": 6, "max": 9, "unit": "hours"},
                        "total_event_duration": {
                            "scale": "day",
                            "min": 6,
                            "max": 9,
                            "unit": "hours",
                        },
                        "total_event_episodes": {"scale": "day", "min": 1, "max": 1},
                    }
                }
            }
        }
    }


def test_event_config_injects_dict_keys():
    cfg = EventConfig.model_validate(_minimal_event_yaml())
    sleep = cfg.categories["sleep"].events["sleep"]
    assert sleep.name == "sleep"
    assert sleep.category == "sleep"


def test_event_config_rejects_duplicate_event_names_across_categories():
    raw = _minimal_event_yaml()
    raw["categories"]["other"] = {
        "events": {"sleep": raw["categories"]["sleep"]["events"]["sleep"]}
    }
    with pytest.raises(ValidationError):
        EventConfig.model_validate(raw)


def test_role_predicate_requires_in_or_eq():
    """A predicate that sets neither 'in' nor 'eq' fails the at-least-one rule."""
    with pytest.raises(ValidationError):
        RolePredicate.model_validate({})


def test_event_config_inject_skips_when_not_a_dict():
    """A non-dict top-level input is left untouched by the injector."""
    with pytest.raises(ValidationError):
        EventConfig.model_validate(["categories"])


def test_event_config_inject_skips_when_no_categories_key():
    """An input dict without 'categories' is left untouched by the injector."""
    with pytest.raises(ValidationError):
        EventConfig.model_validate({})


def test_event_config_inject_skips_when_categories_not_dict():
    """Non-dict 'categories' field is left untouched by the injector."""
    with pytest.raises(ValidationError):
        EventConfig.model_validate({"categories": ["not", "a", "dict"]})


def test_event_config_inject_skips_non_dict_category():
    """A category value that isn't a dict is skipped by the injector."""
    with pytest.raises(ValidationError):
        EventConfig.model_validate({"categories": {"sleep": "not-a-dict"}})


def test_event_config_inject_skips_non_dict_events_block():
    """An events block that isn't a dict is left alone (False arm of inner if)."""
    with pytest.raises(ValidationError):
        EventConfig.model_validate({"categories": {"sleep": {"events": "not-a-dict"}}})


def test_event_config_inject_skips_non_dict_event_value():
    """A single event value that isn't a dict is skipped (continues to next event)."""
    with pytest.raises(ValidationError):
        EventConfig.model_validate(
            {"categories": {"sleep": {"events": {"sleep": "not-a-dict"}}}}
        )


# -------------------------------------------------------------------------------------
# ------------------------------ temporal-relation rules ------------------------------
# -------------------------------------------------------------------------------------


def test_temporal_rules_reject_duplicate_ids():
    r1 = TemporalRule(id="x", formula="G ¬(a ∧ b)")
    r2 = TemporalRule(id="x", formula="G ¬(c ∧ d)")
    with pytest.raises(ValidationError):
        TemporalRelationRules(rules=[r1, r2])


def test_temporal_rules_min_fraction_out_of_range():
    with pytest.raises(ValidationError):
        TemporalRule(id="x", formula="G x", min_fraction=2.0)


def test_temporal_rules_reject_allen_id_colliding_with_temporal_rule_id():
    """`allen_pair_rules` ids share the same namespace as `rules` ids:
    duplicating across the two lists must still be rejected by the
    cross-list dedup pass."""
    from src.scripts.persona.config.schema import AllenPairRule

    rule = TemporalRule(id="x", formula="G x")
    allen = AllenPairRule(
        id="x", event_a="sleep", event_b="lunch", admissible_relations=["p"]
    )
    with pytest.raises(ValidationError):
        TemporalRelationRules(rules=[rule], allen_pair_rules=[allen])


def test_temporal_rules_accept_distinct_allen_and_rule_ids():
    """Non-colliding ids across `rules` and `allen_pair_rules` pass:
    exercises the `seen.add` happy path on the allen branch of
    `_check_unique_ids`."""
    from src.scripts.persona.config.schema import AllenPairRule

    rule = TemporalRule(id="r1", formula="G x")
    allen = AllenPairRule(
        id="a1", event_a="sleep", event_b="lunch", admissible_relations=["p"]
    )
    bundle = TemporalRelationRules(rules=[rule], allen_pair_rules=[allen])
    assert [r.id for r in bundle.rules] == ["r1"]
    assert [r.id for r in bundle.allen_pair_rules] == ["a1"]


# -------------------------------------------------------------------------------------
# ----------- ontology IRI fields on EventDefinition / Override / Stage ---------------
# -------------------------------------------------------------------------------------


def test_event_definition_accepts_optional_iri_fields():
    from src.scripts.persona.config.schema import EventDefinition, TotalDuration

    ev = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=15, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=15, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        human_activity_iri=(
            "https://w3id.org/calendar-bench/human-activities/"
            "activity/walking-3-5-mph-mod-pace"
        ),
        health_task_iri=(
            "https://w3id.org/calendar-bench/health/task/schedule-a-30-minute-walk"
        ),
    )
    assert ev.human_activity_iri.endswith("walking-3-5-mph-mod-pace")
    assert ev.health_task_iri.endswith("schedule-a-30-minute-walk")


def test_event_definition_iri_fields_default_none():
    from src.scripts.persona.config.schema import EventDefinition, TotalDuration

    ev = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=15, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=15, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
    )
    assert ev.human_activity_iri is None
    assert ev.health_task_iri is None


def test_event_override_accepts_iri_fields():
    from src.scripts.persona.config.schema import EventOverride

    override = EventOverride(
        human_activity_iri="https://w3id.org/calendar-bench/human-activities/activity/yoga",
        health_task_iri="https://w3id.org/calendar-bench/health/task/practice-yoga",
    )
    assert override.human_activity_iri.endswith("/yoga")
    assert override.health_task_iri.endswith("/practice-yoga")


def test_persona_event_stage_accepts_iri_fields():
    stage = PersonaEventStage(
        name="walking",
        time="morning",
        days=["Mon", "Tue"],
        human_activity_iri="https://w3id.org/calendar-bench/human-activities/activity/walking",
        health_task_iri="https://w3id.org/calendar-bench/health/task/schedule-a-walk",
    )
    assert stage.human_activity_iri.endswith("/walking")
    assert stage.health_task_iri.endswith("/schedule-a-walk")


def test_persona_event_stage_iri_fields_default_none():
    stage = PersonaEventStage(name="walking", time="morning", days=["Mon"])
    assert stage.human_activity_iri is None
    assert stage.health_task_iri is None


def test_scale_literal_now_includes_month():
    """The Scale enum gained `"month"`; the literal
    is consumed by `DurationRange.scale` / `EpisodeRange.scale`."""
    from src.scripts.persona.config.schema import TotalDuration

    td = TotalDuration(min=60, max=120, scale="month", unit="minutes")
    assert td.scale == "month"


def test_scale_literal_rejects_unknown_value():
    """Adding `"month"` to the literal must not silently accept other values."""
    from src.scripts.persona.config.schema import TotalDuration

    with pytest.raises(ValidationError):
        TotalDuration(min=60, max=120, scale="lustrum", unit="minutes")
