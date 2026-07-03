"""Integration: L_context_fit member-level matching over the real ontology assets.

These parse the committed HealthTasks TTL plus context_iris.json through the
production loaders (rather than fixtures), so they run outside the unit gate.
They lock in two properties of the fixed metric on real data: links join on
ontology IRIs (members), not category slugs, and the denominator counts only
the context categories the persona actually generates.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.context_fit import (
    load_context_categories_by_iri,
    load_context_links_by_uri,
    per_task_fit,
)

HEALTHTASKS_TTL = Path("src/assets/ontologies/HealthTasks_2026.05.19.ttl")
CONTEXT_IRIS = Path("src/assets/ontologies/context_iris.json")

# Social member IRIs the persona pipeline generates (also Context dictionary entries).
ALONE = "https://github.com/EBehaviourChange-COPPER/ontology/blob/main/COPPER_2004"
WITH_FAMILY = "http://humanbehaviourchange.org/ontology/BCIO_006002"
D = datetime.date(2026, 5, 4)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (HEALTHTASKS_TTL.is_file() and CONTEXT_IRIS.is_file()),
        reason="ontology assets not on disk",
    ),
]


def _scheduled(uri: str, *, start: int, end: int) -> ScheduledTask:
    task = RecommendedTask(
        label="t", duration_min=10, duration_max=20, intensity=2, ontology_uri=uri
    )
    return ScheduledTask(
        task=task,
        start_minutes=start,
        end_minutes=end,
        is_standalone=True,
        concurrent_with=None,
        date=D,
    )


def _episode(iri: str, *, start: int, end: int) -> ContextEpisode:
    return ContextEpisode(
        name="social_context",
        category="social_context",
        date=D,
        start_minutes=start,
        end_minutes=end,
        ontology_uri=iri,
    )


def _real_links() -> dict[str, dict[str, frozenset[str]]]:
    cats = load_context_categories_by_iri(CONTEXT_IRIS)
    return load_context_links_by_uri(HEALTHTASKS_TTL, cats)


def test_real_ontology_links_are_member_level_iris() -> None:
    cats = load_context_categories_by_iri(CONTEXT_IRIS)
    assert cats[ALONE] == "social_context"
    assert cats[WITH_FAMILY] == "social_context"
    links = _real_links()
    assert links, "expected real tasks to carry resolvable context links"
    # Link values are {category: frozenset(IRI)} with IRI-shaped members, not slugs.
    sample = next(iter(links.values()))
    for category, iris in sample.items():
        assert isinstance(category, str)
        assert iris and all(i.startswith(("http://", "https://")) for i in iris)
    # The persona's `alone` member is actually linked by real tasks.
    assert any(ALONE in d.get("social_context", ()) for d in links.values())


def test_generated_categories_are_the_only_denominator() -> None:
    """A task links many categories, but a social-only trace scores social only."""
    links = _real_links()
    uri = next(u for u, d in links.items() if ALONE in d.get("social_context", ()))
    task = _scheduled(uri, start=540, end=560)
    verdict = per_task_fit(task, [_episode(ALONE, start=540, end=600)], links)
    assert verdict is not None
    assert verdict.recommended_categories == ("social_context",)
    assert verdict.fit == pytest.approx(1.0)


def test_wrong_member_earns_no_credit_on_real_links() -> None:
    """Linking social_context:alone earns nothing when placed during with_family."""
    links = _real_links()
    uri = next(
        (
            u
            for u, d in links.items()
            if ALONE in d.get("social_context", ())
            and WITH_FAMILY not in d.get("social_context", ())
        ),
        None,
    )
    if uri is None:
        pytest.skip("no real task links alone without also linking with_family")
    task = _scheduled(uri, start=540, end=560)
    # alone is generated (realizable) but the placement overlaps with_family.
    trace = [
        _episode(WITH_FAMILY, start=540, end=600),
        _episode(ALONE, start=0, end=100),
    ]
    verdict = per_task_fit(task, trace, links)
    assert verdict is not None
    assert verdict.recommended_categories == ("social_context",)
    assert verdict.fit == pytest.approx(0.0)
