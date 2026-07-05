"""Smoke test: filter pool returns >= 30% dividable tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios.config.loader import load_experiment_scenarios

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_F = _ROOT / "tests" / "fixtures" / "persona" / "experiment_f"
_SCENARIOS = _EXPERIMENT_F / "scenarios.yaml"
_ONTOLOGY = _ROOT / "src" / "assets" / "ontologies" / "HealthTasks_2026.05.19.json"

# Map scenario YAML domain filter values to ontology top-level keys.
_DOMAIN_TO_BRANCH = {
    "PhysicalActivityTask": "PhysicalActivity",
    "NutritionTask": "Nutrition",
    "MentalWellbeingTask": "MentalWellbeing",
}

DIVIDABLE_SHARE_THRESHOLD = 0.30
LONG_DURATION_MINUTES = 30


def _pool(domain_filter: list[str], level_filter: list[str]) -> list[dict]:
    """Walk HealthTasks and return tasks matching the domain+level filter."""
    data = json.loads(_ONTOLOGY.read_text(encoding="utf-8"))
    branches = {_DOMAIN_TO_BRANCH[d] for d in domain_filter}
    levels = set(level_filter)
    out: list[dict] = []

    def walk(node, path: list[str]) -> None:
        if isinstance(node, dict):
            if "isDividable" in node and len(path) >= 3:
                if path[0] in branches and path[2] in levels:
                    out.append(node)
            for k, v in node.items():
                walk(v, path + [k])

    walk(data, [])
    return out


@pytest.fixture(scope="module")
def task_gen_filter() -> tuple[list[str], list[str], int]:
    """Read the shared filter and num_tasks from the fixture scenarios.yaml."""
    cfg = load_experiment_scenarios(_SCENARIOS)
    baseline = next(s for s in cfg.scenarios if s.id == "oneshot_gpt_4o_mini")
    return (
        baseline.task_generation.filters.domains,
        baseline.task_generation.filters.difficulty,
        baseline.task_generation.num_tasks,
    )


def test_candidate_pool_meets_dividable_share(task_gen_filter):
    domains, levels, _ = task_gen_filter
    pool = _pool(domains, levels)
    assert pool, "candidate pool is empty"
    dividable = [t for t in pool if t.get("isDividable") is True]
    share = len(dividable) / len(pool)
    assert share >= DIVIDABLE_SHARE_THRESHOLD, (
        f"dividable share {share:.1%} below the {DIVIDABLE_SHARE_THRESHOLD:.0%} "
        f"gate; pool={len(pool)} dividable={len(dividable)} "
        f"(domains={domains}, difficulty={levels})"
    )


def test_pool_contains_long_dividable_tasks(task_gen_filter):
    """L_divide is most informative on dividable tasks of meaningful length."""
    domains, levels, _ = task_gen_filter
    pool = _pool(domains, levels)
    long_div = [
        t
        for t in pool
        if t.get("isDividable") is True
        and (t.get("estimatedDuration") or 0) >= LONG_DURATION_MINUTES
    ]
    assert len(long_div) >= 3, (
        f"need >=3 dividable tasks with duration >= {LONG_DURATION_MINUTES}min; "
        f"got {len(long_div)} in pool of {len(pool)}"
    )


def test_num_tasks_per_week_is_at_least_20(task_gen_filter):
    _, _, num_tasks = task_gen_filter
    assert num_tasks >= 20, (
        f"num_tasks={num_tasks}; bump to >=20 so L_divide has enough "
        f"dividable instances per week"
    )
