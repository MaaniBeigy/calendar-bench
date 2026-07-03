"""End-to-end smoke for the persona pipeline with the context pass enabled.

Loads a rich persona fixture, runs persona generation
under one worker for determinism, and asserts the per-person artefacts
carry context episodes plus a non-empty validation report block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.persona.cli import _attach_contexts, _run_validation
from src.scripts.persona.concurrency.pool import run_pool
from src.scripts.persona.config.loader import load_config
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.export.json_writer import write_population_json
from src.scripts.persona.sampling.persona_sampler import sample_population
from src.scripts.persona.validation.report import report_to_dict, write_report

_EXPERIMENT_H = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "persona"
    / "experiment_h"
)


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    cfg = load_config(
        _EXPERIMENT_H / "environment.yaml",
        _EXPERIMENT_H / "persona_config.yaml",
        _EXPERIMENT_H / "event_config.yaml",
        _EXPERIMENT_H / "temporal_relation_rules.yaml",
    )
    catalog = Catalog.from_event_config(cfg.event)
    persons = sample_population(cfg.persona, cfg.environment.seed)
    schedules = run_pool(persons, catalog, cfg.environment, cfg.rules, workers=1)
    schedules = _attach_contexts(persons, schedules, cfg)
    out_dir = tmp_path_factory.mktemp("experiment_h_run")
    write_population_json(persons, schedules, out_dir / "persons")
    report = _run_validation(cfg, persons, schedules, catalog, workers=1)
    write_report(report, out_dir)
    return {
        "out_dir": out_dir,
        "persons": persons,
        "schedules": schedules,
        "report": report,
    }


def test_each_person_has_context_episodes(cohort):
    schedules = cohort["schedules"]
    assert schedules
    for s in schedules:
        assert s.contexts, f"{s.person_id} has no context episodes"


def test_every_enabled_category_appears_at_least_once(cohort):
    schedules = cohort["schedules"]
    seen_categories = set()
    for s in schedules:
        for ep in s.contexts:
            seen_categories.add(ep.category)
    # Stress, behaviour_state, capability_opportunity, goal_intention are
    # `mutually_exclusive=False` by default so they always have room; trait
    # state polarities are gated by `extraversion`, may be absent for some
    # cohorts. Require >= 8 of 11 to keep the smoke test stable.
    assert len(seen_categories) >= 8


def test_per_person_json_carries_contexts(cohort):
    persons_dir = cohort["out_dir"] / "persons"
    first = next(iter(persons_dir.glob("*.json")))
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert "contexts" in payload
    assert payload["contexts"], "per-person JSON contexts list is empty"


def test_validation_report_has_contexts_block(cohort):
    rendered = report_to_dict(cohort["report"])
    assert "contexts" in rendered
    assert "contexts" in rendered["totals"]


def test_mutually_exclusive_violations_under_tolerance(cohort):
    report = cohort["report"]
    overlaps = [v for v in report.contexts if v.kind == "mutually_exclusive_overlap"]
    assert overlaps == [], f"unexpected mutually-exclusive overlaps: {overlaps[:3]!r}"
