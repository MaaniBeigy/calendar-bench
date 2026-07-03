"""Unit tests for src.scripts.persona.sampling.persona_sampler."""

from __future__ import annotations

from src.scripts.persona.config.loader import load_persona
from src.scripts.persona.config.schema import Persona, PersonaCommon, PersonaConfig
from src.scripts.persona.sampling.persona_sampler import sample_population
from src.scripts.persona.sampling.seeds import mix_seed


def test_sample_population_count_matches_instances(persona_yaml):
    cfg = load_persona(persona_yaml)
    expected = sum(p.instances for p in cfg.personas)
    population = sample_population(cfg, root_seed=42)
    assert len(population) == expected


def test_sample_population_is_deterministic(persona_yaml):
    cfg = load_persona(persona_yaml)
    a = sample_population(cfg, root_seed=42)
    b = sample_population(cfg, root_seed=42)
    assert [(p.person_id, p.person_seed) for p in a] == [
        (p.person_id, p.person_seed) for p in b
    ]
    # And every jittered stage matches too.
    for pa, pb in zip(a, b):
        assert pa.stages == pb.stages


def test_sample_population_different_root_seeds_diverge(persona_yaml):
    cfg = load_persona(persona_yaml)
    a = sample_population(cfg, root_seed=1)
    b = sample_population(cfg, root_seed=2)
    # Identity is the same…
    assert [p.person_id for p in a] == [p.person_id for p in b]
    # …but at least one jittered stage differs across the population.
    assert any(pa.stages != pb.stages for pa, pb in zip(a, b))


def test_sample_population_seed_is_mix_of_root_persona_index(persona_yaml):
    cfg = load_persona(persona_yaml)
    population = sample_population(cfg, root_seed=42)
    for person in population:
        expected = mix_seed(42, person.persona_id, person.instance_index)
        assert person.person_seed == expected


def test_sample_population_person_ids_are_unique(persona_yaml):
    cfg = load_persona(persona_yaml)
    population = sample_population(cfg, root_seed=42)
    ids = [p.person_id for p in population]
    assert len(ids) == len(set(ids))


def test_sample_population_preserves_non_jittered_fields(persona_yaml):
    cfg = load_persona(persona_yaml)
    population = sample_population(cfg, root_seed=42)
    by_persona: dict[str, list] = {}
    for p in population:
        by_persona.setdefault(p.persona_id, []).append(p)
    # All instances of the same persona share occupation_status and the
    # ordered list of stage names (only times/durations get jittered).
    for instances in by_persona.values():
        first = instances[0]
        for other in instances[1:]:
            assert other.occupation_status == first.occupation_status
            assert [s.name for s in other.stages] == [s.name for s in first.stages]
            assert [s.days for s in other.stages] == [s.days for s in first.stages]
            assert [s.date for s in other.stages] == [s.date for s in first.stages]


def test_sample_population_jitter_on_hhmm_within_bounds(persona_yaml):
    """Every jittered HH:MM must stay within ±jitter.time_minutes of the template."""
    cfg = load_persona(persona_yaml)
    population = sample_population(cfg, root_seed=42)
    template_by_id = {p.id: p for p in cfg.personas}
    for person in population:
        tpl = template_by_id[person.persona_id]
        jitter = tpl.jitter or cfg.common.default_jitter
        for tpl_stage, person_stage in zip(tpl.stages, person.stages):
            if tpl_stage.time is None or ":" not in tpl_stage.time:
                continue
            assert person_stage.time is not None
            tpl_min = _hhmm_to_min(tpl_stage.time)
            person_min = _hhmm_to_min(person_stage.time)
            assert abs(person_min - tpl_min) <= jitter.time_minutes


def _hhmm_to_min(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


def test_sample_population_named_windows_pass_through(persona_yaml):
    """Named-window tokens (e.g. `morning`) carry no scalar to perturb,
    so the jittered stage keeps the same token string."""
    cfg = load_persona(persona_yaml)
    population = sample_population(cfg, root_seed=42)
    template_by_id = {p.id: p for p in cfg.personas}
    for person in population:
        tpl = template_by_id[person.persona_id]
        for tpl_stage, person_stage in zip(tpl.stages, person.stages):
            if tpl_stage.time is None or ":" in tpl_stage.time:
                continue
            assert person_stage.time == tpl_stage.time


def test_sample_population_handles_persona_with_no_stages():
    """A persona with no stages returns persons whose `stages` list is empty."""
    persona = Persona(
        id="solo",
        instances=2,
        occupation_status="parttime",
        stages=[],
    )
    cfg = PersonaConfig(common=PersonaCommon(), personas=[persona])
    population = sample_population(cfg, root_seed=42)
    assert len(population) == 2
    for p in population:
        assert p.stages == []


def test_sample_population_preserves_stage_with_no_time():
    """A stage that omits `time` (free-floating start) round-trips
    through the sampler unchanged - jitter has no scalar to perturb,
    so the per-person stage keeps `time=None`."""
    from src.scripts.persona.config.schema import PersonaEventStage

    persona = Persona(
        id="freefloat",
        instances=1,
        occupation_status="student",
        stages=[
            PersonaEventStage(
                name="study",
                time=None,
                duration_minutes=60,
                days=["Mon", "Tue", "Wed", "Thu", "Fri"],
            )
        ],
    )
    cfg = PersonaConfig(common=PersonaCommon(), personas=[persona])
    [person] = sample_population(cfg, root_seed=42)
    [stage] = person.stages
    assert stage.name == "study"
    assert stage.time is None
    assert stage.days == ["Mon", "Tue", "Wed", "Thu", "Fri"]
