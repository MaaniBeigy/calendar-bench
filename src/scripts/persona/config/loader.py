"""Load and validate the four YAML configuration files into typed models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from src.scripts.persona.config.schema import (
    EnvironmentConfig,
    EventConfig,
    PersonaConfig,
    TemporalRelationRules,
)


class ConfigError(ValueError):
    """Raised when a YAML config file cannot be read or fails validation."""


@dataclass(frozen=True)
class Config:
    """All four validated configs for a single run."""

    environment: EnvironmentConfig
    persona: PersonaConfig
    event: EventConfig
    rules: TemporalRelationRules


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"empty config file: {path}")
    return data


def _validate(model: type[BaseModel], data: Any, path: Path) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"validation error in {path}:\n{exc}") from exc


def load_environment(path: Path) -> EnvironmentConfig:
    return _validate(EnvironmentConfig, _read_yaml(path), path)


def load_persona(path: Path) -> PersonaConfig:
    return _validate(PersonaConfig, _read_yaml(path), path)


def load_event(path: Path) -> EventConfig:
    return _validate(EventConfig, _read_yaml(path), path)


def load_rules(path: Path) -> TemporalRelationRules:
    return _validate(TemporalRelationRules, _read_yaml(path), path)


def _validate_concurrent_with(
    event_config: EventConfig, persona_config: PersonaConfig
) -> None:
    """Raise ConfigError if any concurrent_with entry names a non-existent event."""
    catalog_names: set[str] = {
        event_name
        for cat in event_config.categories.values()
        for event_name in cat.events
    }
    for cat in event_config.categories.values():
        for event_name, event_def in cat.events.items():
            for ref in event_def.concurrent_with:
                if ref not in catalog_names:
                    raise ConfigError(
                        f"event {event_name!r}: concurrent_with references "
                        f"unknown event {ref!r}"
                    )
    for persona in persona_config.personas:
        for override_name, override in persona.event_overrides.items():
            if override.concurrent_with is None:
                continue
            for ref in override.concurrent_with:
                if ref not in catalog_names:
                    raise ConfigError(
                        f"persona {persona.id!r}, override {override_name!r}: "
                        f"concurrent_with references unknown event {ref!r}"
                    )


def _validate_allen_pair_rules(
    event_config: EventConfig, rules: TemporalRelationRules
) -> None:
    """Validate that any literal-name endpoints in `AllenPairRule` exist in the catalog.

    Selector predicates may name events that come from the *task* side
    (HealthTasks ontology) which is not in the calendar catalog; those
    are intentionally not validated here.  Only literal `name` fields
    on a selector are cross-checked.  Predicate-only selectors (using
    `intensity` / `domain` / `met_*`) are allowed through.
    """
    catalog_names: set[str] = {
        event_name
        for cat in event_config.categories.values()
        for event_name in cat.events
    }
    for rule in rules.allen_pair_rules:
        for attr, selector in (("event_a", rule.event_a), ("event_b", rule.event_b)):
            if selector.kind == "context":
                continue
            name_field = selector.name
            if name_field is None:
                continue  # predicate-only; nothing to cross-check.
            literal_names = (
                [name_field] if isinstance(name_field, str) else list(name_field)
            )
            for n in literal_names:
                # Allow names that refer to task ontology entries; only
                # reject when the name appears to be an event-catalog
                # literal that the user typo'd.  Heuristic: names with
                # an underscore or the same shape as catalog events are
                # validated; we keep it simple by validating every
                # literal but only when at least one candidate exists.
                if catalog_names and n not in catalog_names:
                    # If the rule names something that isn't in the catalog,
                    # that's likely a task-side label; let it pass.  We
                    # only flag when the name looks like an event but has
                    # no matching catalog entry AND the rule otherwise has
                    # a literal-vs-literal shape (both endpoints are
                    # name-only).  This keeps backwards compatibility
                    # while not over-restricting predicate-driven rules.
                    if (
                        rule.event_a.name is not None
                        and rule.event_b.name is not None
                        and rule.event_a.intensity is None
                        and rule.event_b.intensity is None
                        and rule.event_a.domain is None
                        and rule.event_b.domain is None
                        and rule.event_a.met_min is None
                        and rule.event_b.met_min is None
                        and rule.event_a.met_max is None
                        and rule.event_b.met_max is None
                    ):
                        # Both endpoints purely literal: validate strictly.
                        raise ConfigError(
                            f"allen_pair_rule {rule.id!r}: {attr}={n!r} "
                            f"not found in event catalog"
                        )


def _validate_applies_to_personas(
    persona_config: PersonaConfig, rules: TemporalRelationRules
) -> None:
    """Reject any `applies_to.personas: [...]` that names an undeclared persona."""
    known = {p.id for p in persona_config.personas}
    for rule in list(rules.rules) + list(rules.allen_pair_rules):
        ids = rule.applies_to.get("personas")
        if not ids:
            continue
        for pid in ids:
            if not isinstance(pid, str) or pid not in known:
                raise ConfigError(
                    f"rule {rule.id!r}: applies_to.personas references "
                    f"unknown persona {pid!r}; declared: {sorted(known)}"
                )


def _validate_context_iris(persona_config: PersonaConfig, driver: Any) -> None:
    """Reject any context `ontology_uri` not present in the Neo4j Context dictionary."""
    has_context = any(p.contexts for p in persona_config.personas)
    if not has_context:
        return
    from src.scripts.persona.context.catalog import load_catalog

    catalog = load_catalog(driver)
    for persona in persona_config.personas:
        for cat_name, category in persona.contexts.items():
            for member_name, member in category.members.items():
                iri = member.ontology_uri
                if iri is None:
                    continue
                if not catalog.has(iri):
                    raise ConfigError(
                        f"persona {persona.id!r}: context "
                        f"{cat_name}.{member_name} ontology_uri {iri!r} "
                        f"not in Context dictionary"
                    )


def load_config(
    environment: Path,
    persona: Path,
    event: Path,
    rules: Path,
    *,
    neo4j_driver: Any | None = None,
) -> Config:
    """Load and validate all four YAML files, returning a single Config.

    Persona context IRIs are validated against the Context dictionary in
    Neo4j; pass an existing driver to reuse, otherwise one is created from
    `Neo4jSettings.from_env()` for the duration of the call.
    """
    config = Config(
        environment=load_environment(Path(environment)),
        persona=load_persona(Path(persona)),
        event=load_event(Path(event)),
        rules=load_rules(Path(rules)),
    )
    _validate_concurrent_with(config.event, config.persona)
    _validate_allen_pair_rules(config.event, config.rules)
    _validate_applies_to_personas(config.persona, config.rules)
    needs_context = any(p.contexts for p in config.persona.personas)
    if not needs_context:
        _validate_context_iris(config.persona, neo4j_driver)
        return config
    if neo4j_driver is not None:
        _validate_context_iris(config.persona, neo4j_driver)
        return config
    from src.graphrag.config import Neo4jSettings
    from src.graphrag.neo4j_client import make_driver

    driver = make_driver(Neo4jSettings.from_env())
    try:
        _validate_context_iris(config.persona, driver)
    finally:
        driver.close()
    return config


def _iter_declared_iris(
    event_config: EventConfig, persona_config: PersonaConfig
) -> list[tuple[str, str]]:
    """Return every `(origin, iri)` declared on the catalog or personas.

    `origin` is a human-readable breadcrumb (`"event:walking"`,
    `"stage:b_parttime_morning:walking"`, …) used to anchor the warning
    issued when an IRI does not resolve.
    """
    pairs: list[tuple[str, str]] = []
    for cat in event_config.categories.values():
        for event_name, event_def in cat.events.items():
            if event_def.human_activity_iri:
                pairs.append(
                    (
                        f"event:{event_name}:human_activity_iri",
                        event_def.human_activity_iri,
                    )
                )
            if event_def.health_task_iri:
                pairs.append(
                    (f"event:{event_name}:health_task_iri", event_def.health_task_iri)
                )
    for persona in persona_config.personas:
        for override_name, override in persona.event_overrides.items():
            if override.human_activity_iri:
                pairs.append(
                    (
                        f"override:{persona.id}:{override_name}:human_activity_iri",
                        override.human_activity_iri,
                    )
                )
            if override.health_task_iri:
                pairs.append(
                    (
                        f"override:{persona.id}:{override_name}:health_task_iri",
                        override.health_task_iri,
                    )
                )
        for stage in persona.stages:
            if stage.human_activity_iri:
                pairs.append(
                    (
                        f"stage:{persona.id}:{stage.name}:human_activity_iri",
                        stage.human_activity_iri,
                    )
                )
            if stage.health_task_iri:
                pairs.append(
                    (
                        f"stage:{persona.id}:{stage.name}:health_task_iri",
                        stage.health_task_iri,
                    )
                )
    return pairs


def _iri_exists_in_neo4j(driver: Any, iri: str) -> bool:
    """Cheap "does this IRI exist as a node" probe against Neo4j.

    Matches any node whose `uri` or `iri` property equals `iri`;
    returns `True` when the count is positive.  Errors and timeouts are
    treated as "unknown" to `False` (we want to warn rather than crash
    the loader).
    """
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE coalesce(n.uri, n.iri) = $iri
                RETURN count(n) > 0 AS exists
                """,
                iri=iri,
            ).single()
            return bool(result and result["exists"])
    except Exception:  # pragma: no cover - defensive
        return False


def validate_ontology_iris(
    event_config: EventConfig,
    persona_config: PersonaConfig,
    *,
    neo4j_driver: Any | None = None,
) -> list[tuple[str, str]]:
    """Validate every declared `human_activity_iri` / `health_task_iri`.

    When `neo4j_driver` is `None` this is a no-op (CI / offline mode).
    When provided, every declared IRI is probed in Neo4j; mismatches are
    surfaced via :func:`warnings.warn` with one warning per missing IRI.
    Returns the list of `(origin, iri)` pairs that failed the probe so
    callers can also assert on them programmatically.
    """
    if neo4j_driver is None:
        return []
    declared = _iter_declared_iris(event_config, persona_config)
    missing: list[tuple[str, str]] = []
    for origin, iri in declared:
        if not _iri_exists_in_neo4j(neo4j_driver, iri):
            missing.append((origin, iri))
            warnings.warn(
                f"ontology IRI declared on {origin} did not resolve in Neo4j: {iri!r}",
                stacklevel=2,
            )
    return missing
