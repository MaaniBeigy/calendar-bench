"""Unit tests for src.scripts.persona.config.loader."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.scripts.persona.config.loader import (
    Config,
    ConfigError,
    _iri_exists_in_neo4j,
    _iter_declared_iris,
    load_config,
    load_environment,
    load_event,
    load_persona,
    load_rules,
    validate_ontology_iris,
)
from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    EventOverride,
    Persona,
    PersonaCommon,
    PersonaConfig,
    PersonaEventStage,
    TotalDuration,
)


def test_load_config_round_trips_examples(
    environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    cfg = load_config(environment_yaml, persona_yaml, event_yaml, rules_yaml)
    assert isinstance(cfg, Config)
    assert cfg.environment.horizon.weeks == 4
    assert cfg.persona.personas[0].id == "test_student"
    assert "sleep" in cfg.event.categories
    assert any(r.id == "no_overlap_sleep_work" for r in cfg.rules.rules)


def test_load_config_uses_supplied_driver_when_persona_declares_contexts(
    environment_yaml, event_yaml, rules_yaml, tmp_path, monkeypatch
):
    """When a persona has contexts and a driver is provided, no auto-create happens."""
    import src.scripts.persona.context.catalog as catalog_mod

    persona_path = tmp_path / "p.yaml"
    persona_path.write_text(_persona_with_context_yaml(), encoding="utf-8")
    monkeypatch.setattr(
        catalog_mod, "load_catalog", lambda driver: _FakeCatalog({"http://x/k"})
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("auto-create must not be called when driver is supplied")

    monkeypatch.setattr("src.graphrag.neo4j_client.make_driver", boom)
    cfg = load_config(
        environment_yaml,
        persona_path,
        event_yaml,
        rules_yaml,
        neo4j_driver=MagicMock(name="passed-in-driver"),
    )
    assert isinstance(cfg, Config)


def test_load_config_auto_creates_driver_when_persona_has_contexts(
    environment_yaml, event_yaml, rules_yaml, tmp_path, monkeypatch
):
    """When no driver is passed and a persona has contexts, one is built from env."""
    import src.scripts.persona.config.loader as loader_mod
    import src.scripts.persona.context.catalog as catalog_mod

    persona_path = tmp_path / "p.yaml"
    persona_path.write_text(_persona_with_context_yaml(), encoding="utf-8")
    monkeypatch.setattr(
        catalog_mod, "load_catalog", lambda driver: _FakeCatalog({"http://x/k"})
    )
    fake_driver = MagicMock(name="auto-created-driver")
    monkeypatch.setattr(
        "src.graphrag.config.Neo4jSettings.from_env", lambda: MagicMock()
    )
    monkeypatch.setattr("src.graphrag.neo4j_client.make_driver", lambda _s: fake_driver)
    cfg = load_config(environment_yaml, persona_path, event_yaml, rules_yaml)
    assert isinstance(cfg, Config)
    fake_driver.close.assert_called_once()


def test_load_config_skips_context_validation_when_no_persona_has_contexts(
    environment_yaml, persona_yaml, event_yaml, rules_yaml, monkeypatch
):
    """The example persona has no contexts, so no driver is touched."""
    called: list[bool] = []
    monkeypatch.setattr(
        "src.graphrag.neo4j_client.make_driver",
        lambda _s: called.append(True) or MagicMock(),
    )
    load_config(environment_yaml, persona_yaml, event_yaml, rules_yaml)
    assert called == []


class _FakeCatalog:
    def __init__(self, iris: set[str]) -> None:
        self._iris = iris

    def has(self, iri: str) -> bool:
        return iri in self._iris


def _persona_with_context_yaml() -> str:
    return (
        "personas:\n"
        "  - id: p\n"
        "    instances: 1\n"
        "    contexts:\n"
        "      mood_emotion:\n"
        "        mutually_exclusive: true\n"
        "        members:\n"
        "          happy:\n"
        "            per_event_duration: {min: 15, max: 60}\n"
        "            total_event_duration: {min: 30, max: 180, scale: day}\n"
        "            total_event_episodes: {min: 1, max: 3, scale: day}\n"
        "            ontology_uri: http://x/k\n"
    )


def test_load_environment_only(environment_yaml):
    e = load_environment(environment_yaml)
    assert e.parallelism.executor == "process"
    assert e.solver.optimize_objective == "maximize_sleep"
    # named windows present
    assert {"early_morning", "morning", "afternoon", "evening", "night"} <= set(
        e.time_windows
    )


def test_load_persona_only(persona_yaml):
    p = load_persona(persona_yaml)
    ids = [persona.id for persona in p.personas]
    assert ids == ["test_student", "bob_fulltime"]
    alice = p.personas[0]
    # Persona is a flat list of stages; verify structure rather than
    # specific slot names.
    assert isinstance(alice.stages, list)
    assert any(s.name == "padel" for s in alice.stages)
    # The example student has a sleep stage at 23:00.
    sleep = next(s for s in alice.stages if s.name == "sleep")
    assert sleep.time == "23:00"


def test_load_event_only(event_yaml):
    e = load_event(event_yaml)
    assert "sleep" in e.categories
    assert "office_work" in e.categories["work"].events
    work = e.categories["work"].events["office_work"]
    assert work.name == "office_work"
    assert work.category == "work"
    assert work.weekdays == ["Mon", "Tue", "Wed", "Thu", "Fri"]


def test_load_rules_only(rules_yaml):
    r = load_rules(rules_yaml)
    ids = [rule.id for rule in r.rules]
    assert "no_overlap_sleep_work" in ids
    weekly = next(rule for rule in r.rules if rule.id == "weekly_running_target")
    assert weekly.applies_to == {
        "occupation_status": ["student", "fulltime", "parttime"]
    }


def test_load_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError, match="not found"):
        load_environment(missing)


def test_load_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\n  : invalid:\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML parse error"):
        load_environment(bad)


def test_load_empty_yaml_raises(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_environment(empty)


def test_load_validation_error_wrapped(tmp_path):
    """Schema validation failures must surface as ConfigError, not ValidationError."""
    invalid = tmp_path / "environment.yaml"
    invalid.write_text(
        "seed: 1\n"
        "horizon: { start_date: 2026-05-04, weeks: 1 }\n"
        "output: { dir: ./out }\n"
        "time_windows:\n"
        "  morning: [400, 600]\n",  # missing the other named windows
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="validation error"):
        load_environment(invalid)


def test_load_oserror_wrapped(tmp_path):
    """A non-file path (here, a directory) surfaces as ConfigError, not OSError."""
    target = tmp_path / "as_directory"
    target.mkdir()
    with pytest.raises(ConfigError, match="could not read"):
        load_environment(target)


# -------------------------------------------------------------------------------------
# ----------- ontology-IRI validator (loader-side, Neo4j-mocked) ----------------------
# -------------------------------------------------------------------------------------


def _minimal_event_config(
    *, human_activity_iri: str | None = None, health_task_iri: str | None = None
) -> EventConfig:
    ev = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=15, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=15, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        human_activity_iri=human_activity_iri,
        health_task_iri=health_task_iri,
    )
    return EventConfig(
        categories={"sports": Category(name="sports", events={"walking": ev})}
    )


def _minimal_persona_config(
    *,
    stage_health_iri: str | None = None,
    stage_human_iri: str | None = None,
    override_human_iri: str | None = None,
    override_health_iri: str | None = None,
) -> PersonaConfig:
    overrides: dict[str, EventOverride] = {}
    if override_human_iri is not None or override_health_iri is not None:
        overrides["walking"] = EventOverride(
            human_activity_iri=override_human_iri,
            health_task_iri=override_health_iri,
        )
    stage = PersonaEventStage(
        name="walking",
        time="morning",
        days=["Mon"],
        health_task_iri=stage_health_iri,
        human_activity_iri=stage_human_iri,
    )
    persona = Persona(
        id="p1",
        occupation_status="parttime",
        stages=[stage],
        event_overrides=overrides,
    )
    return PersonaConfig(common=PersonaCommon(), personas=[persona])


def test_validate_ontology_iris_no_driver_is_noop():
    ev_cfg = _minimal_event_config(
        human_activity_iri="https://example.org/iri/a",
        health_task_iri="https://example.org/iri/b",
    )
    p_cfg = _minimal_persona_config(stage_health_iri="https://example.org/iri/c")
    # No driver to silent no-op, returns [].
    out = validate_ontology_iris(ev_cfg, p_cfg, neo4j_driver=None)
    assert out == []


def _make_driver(resolves: dict[str, bool]):
    """Mock a Neo4j driver whose `run` returns `{exists: bool}` per IRI."""
    driver = MagicMock()

    @contextmanager
    def session_cm():
        sess = MagicMock()

        def run(_query, iri: str):
            single = MagicMock()
            single.single.return_value = {"exists": resolves.get(iri, False)}
            return single

        sess.run.side_effect = lambda q, iri: run(q, iri)
        yield sess

    driver.session = session_cm
    return driver


def test_validate_ontology_iris_emits_warning_for_missing_iri():
    iri_a = "https://w3id.org/calendar-bench/human-activities/activity/walking"
    iri_b = "https://w3id.org/calendar-bench/health/task/missing-one"
    ev_cfg = _minimal_event_config(human_activity_iri=iri_a, health_task_iri=iri_b)
    p_cfg = _minimal_persona_config()
    driver = _make_driver({iri_a: True, iri_b: False})
    with warnings.catch_warnings(record=True) as recs:
        warnings.simplefilter("always")
        missing = validate_ontology_iris(ev_cfg, p_cfg, neo4j_driver=driver)
    assert len(missing) == 1
    origin, missed = missing[0]
    assert missed == iri_b
    assert "health_task_iri" in origin
    assert any("did not resolve" in str(r.message) for r in recs)


def test_validate_ontology_iris_resolves_all_returns_empty():
    iri = "https://w3id.org/calendar-bench/human-activities/activity/yoga"
    ev_cfg = _minimal_event_config(human_activity_iri=iri)
    p_cfg = _minimal_persona_config()
    driver = _make_driver({iri: True})
    assert validate_ontology_iris(ev_cfg, p_cfg, neo4j_driver=driver) == []


def test_validate_ontology_iris_covers_stage_and_override_origins():
    stage_iri = "https://w3id.org/calendar-bench/health/task/stage-iri"
    override_iri = (
        "https://w3id.org/calendar-bench/human-activities/activity/override-iri"
    )
    ev_cfg = _minimal_event_config()
    p_cfg = _minimal_persona_config(
        stage_health_iri=stage_iri, override_human_iri=override_iri
    )
    driver = _make_driver({})  # both missing
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        missing = validate_ontology_iris(ev_cfg, p_cfg, neo4j_driver=driver)
    origins = {origin for origin, _ in missing}
    assert any("stage:" in o for o in origins)
    assert any("override:" in o for o in origins)


def test_iter_declared_iris_handles_override_with_only_health_iri():
    """Override with only `health_task_iri` set must still be collected.

    Exercises the 204->211 branch in `_iter_declared_iris` where the
    `human_activity_iri` check is skipped and only `health_task_iri`
    contributes to the output.
    """
    ev_cfg = _minimal_event_config()
    p_cfg = _minimal_persona_config(
        override_health_iri="https://example.org/iri/only-health"
    )
    pairs = _iter_declared_iris(ev_cfg, p_cfg)
    iris = {iri for _, iri in pairs}
    assert iris == {"https://example.org/iri/only-health"}


def test_iter_declared_iris_collects_all_four_field_kinds():
    """Cover every (override / stage) × (human_activity / health_task) branch."""
    ev_cfg = _minimal_event_config(
        human_activity_iri="https://example.org/iri/event-h",
        health_task_iri="https://example.org/iri/event-t",
    )
    p_cfg = _minimal_persona_config(
        stage_health_iri="https://example.org/iri/stage-t",
        stage_human_iri="https://example.org/iri/stage-h",
        override_human_iri="https://example.org/iri/override-h",
        override_health_iri="https://example.org/iri/override-t",
    )
    pairs = _iter_declared_iris(ev_cfg, p_cfg)
    iris = {iri for _, iri in pairs}
    assert "https://example.org/iri/event-h" in iris
    assert "https://example.org/iri/event-t" in iris
    assert "https://example.org/iri/stage-t" in iris
    assert "https://example.org/iri/stage-h" in iris
    assert "https://example.org/iri/override-h" in iris
    assert "https://example.org/iri/override-t" in iris
    # Each origin breadcrumb is distinct.
    assert len({origin for origin, _ in pairs}) == len(pairs)


def test_iter_declared_iris_skips_unset_fields():
    ev_cfg = _minimal_event_config()  # no IRIs declared
    p_cfg = _minimal_persona_config()
    assert _iter_declared_iris(ev_cfg, p_cfg) == []


def test_iri_exists_in_neo4j_handles_driver_exception_as_unknown():
    """A driver that raises during `run` must be treated as "not resolved",
    not crash the validator."""
    driver = MagicMock()

    @contextmanager
    def bad_session():
        sess = MagicMock()
        sess.run.side_effect = RuntimeError("network down")
        yield sess

    driver.session = bad_session
    assert _iri_exists_in_neo4j(driver, "https://example.org/iri") is False
