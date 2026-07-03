"""Expand persona templates into concrete Person instances with bounded jitter.

The sampler is pure with respect to its inputs and the root seed: same
inputs give the same population, regardless of execution order or
platform. Jitter is applied only to scalar fields (HH:MM times and
minute durations); named windows pass through unchanged because there
is no scalar to perturb.
"""

from __future__ import annotations

import random

from src.scripts.persona.config.schema import (
    JitterConfig,
    Persona,
    PersonaConfig,
    PersonaEventStage,
)
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.sampling.characteristics import draw_axis_values
from src.scripts.persona.sampling.distributions import (
    jittered_duration_minutes,
    jittered_time_minutes,
)
from src.scripts.persona.sampling.seeds import mix_seed


def _jitter_time_token(
    token: str | None, jitter: JitterConfig, rng: random.Random
) -> str | None:
    """Apply time jitter to HH:MM; pass window/None tokens through unchanged."""
    if token is None:
        return None
    if ":" not in token:
        return token
    hh, mm = token.split(":", 1)
    new_minutes = jittered_time_minutes(
        rng, base_minutes=int(hh) * 60 + int(mm), bound=jitter.time_minutes
    )
    return f"{new_minutes // 60:02d}:{new_minutes % 60:02d}"


def _jitter_stage(
    stage: PersonaEventStage, jitter: JitterConfig, rng: random.Random
) -> PersonaEventStage:
    """Return a stage with HH:MM time and duration jittered."""
    return PersonaEventStage(
        name=stage.name,
        time=_jitter_time_token(stage.time, jitter, rng),
        duration_minutes=(
            jittered_duration_minutes(
                rng,
                base_minutes=stage.duration_minutes,
                bound=jitter.duration_minutes,
            )
            if stage.duration_minutes is not None
            else None
        ),
        days=list(stage.days),
        date=stage.date,
    )


def _resolve_characteristics(
    persona: Persona, root_seed: int
) -> list[dict[str, str | bool | int | float]]:
    """Draw a per-instance characteristic dict for every instance of `persona`."""
    drawn: dict[str, list] = {
        axis: draw_axis_values(axis, dist, persona.id, persona.instances, root_seed)
        for axis, dist in persona.characteristics.items()
    }
    return [{axis: drawn[axis][i] for axis in drawn} for i in range(persona.instances)]


def _build_person(
    persona: Persona,
    instance_index: int,
    person_seed: int,
    jitter: JitterConfig,
    characteristics: dict[str, str | bool | int | float],
) -> Person:
    rng = random.Random(person_seed)
    occupation = characteristics.get("occupation_status")
    return Person(
        person_id=f"{persona.id}_{instance_index:04d}",
        persona_id=persona.id,
        person_seed=person_seed,
        instance_index=instance_index,
        occupation_status=occupation if isinstance(occupation, str) else None,
        characteristics=characteristics,
        stages=[_jitter_stage(s, jitter, rng) for s in persona.stages],
        jitter_applied=jitter,
        event_overrides=dict(persona.event_overrides),
        contexts=dict(persona.contexts),
    )


def sample_population(persona_config: PersonaConfig, root_seed: int) -> list[Person]:
    """Expand each persona template into `persona.instances` Person records.

    Output order is deterministic: personas in declaration order,
    instances by ascending instance_index.
    """
    population: list[Person] = []
    default_jitter = persona_config.common.default_jitter
    for persona in persona_config.personas:
        jitter = persona.jitter or default_jitter
        characteristics = _resolve_characteristics(persona, root_seed)
        for idx in range(persona.instances):
            person_seed = mix_seed(root_seed, persona.id, idx)
            population.append(
                _build_person(persona, idx, person_seed, jitter, characteristics[idx])
            )
    return population
