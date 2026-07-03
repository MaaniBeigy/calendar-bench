"""Runtime Person type: a single jittered instance of a Persona template."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.scripts.persona.config.schema import (
    ContextCategory,
    EventOverride,
    JitterConfig,
    OccupationStatus,
    PersonaEventStage,
)


class Person(BaseModel):
    """One sampled instance of a persona template.

    `characteristics` carries the per-instance value for every axis the
    persona declared. `occupation_status` is a backward-compat shim
    populated from `characteristics["occupation_status"]` when present.
    Setting `occupation_status` at construction also mirrors it into
    `characteristics["occupation_status"]` for callers that gate on
    `requires:`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    person_id: str
    persona_id: str
    person_seed: int
    instance_index: int
    occupation_status: OccupationStatus | None = None
    characteristics: dict[str, str | bool | int | float] = Field(default_factory=dict)
    stages: list[PersonaEventStage] = Field(default_factory=list)
    jitter_applied: JitterConfig
    event_overrides: dict[str, EventOverride] = Field(default_factory=dict)
    contexts: dict[str, ContextCategory] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _mirror_legacy_occupation(cls, data: Any) -> Any:
        """Keep `occupation_status` and `characteristics['occupation_status']` aligned."""
        if not isinstance(data, dict):  # pragma: no cover - defensive
            return data
        legacy = data.get("occupation_status")
        chars = dict(data.get("characteristics") or {})
        existing = chars.get("occupation_status")
        if legacy is not None and existing is None:
            chars["occupation_status"] = legacy
        elif existing is not None and legacy is None:
            if isinstance(existing, str):
                data = dict(data)
                data["occupation_status"] = existing
        if chars:
            data = dict(data)
            data["characteristics"] = chars
        return data
