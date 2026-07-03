"""Validation report carries the contexts section."""

from __future__ import annotations

from src.scripts.persona.validation.check_contexts import ContextViolation
from src.scripts.persona.validation.report import (
    ValidationReport,
    report_to_dict,
    report_to_text,
)


def _v(**overrides) -> ContextViolation:
    payload: dict = {
        "person_id": "p_0001",
        "category": "mood_emotion",
        "member": "happy",
        "kind": "episode_count_mismatch",
        "detail": "realized 2 episodes, expected >= 5 (scale=day)",
    }
    payload.update(overrides)
    return ContextViolation(**payload)


def test_report_total_includes_contexts():
    r = ValidationReport(contexts=[_v(), _v()])
    assert r.total == 2
    assert r.has_violations is True


def test_text_section_present_when_no_contexts():
    r = ValidationReport()
    text = report_to_text(r)
    assert "Context Distribution Check" in text
    assert "No violations found." in text


def test_text_section_renders_violations():
    r = ValidationReport(contexts=[_v()])
    text = report_to_text(r)
    assert "Context Distribution Check" in text
    assert "episode_count_mismatch" in text
    assert "category=mood_emotion" in text
    assert "member=happy" in text


def test_text_section_drops_member_field_for_overlap_kind():
    r = ValidationReport(
        contexts=[
            _v(
                member=None,
                kind="mutually_exclusive_overlap",
                detail="happy[540-620] overlaps sad[600-700] on 2026-05-04",
            )
        ]
    )
    text = report_to_text(r)
    assert "member=" not in text


def test_dict_contains_contexts_block_and_totals():
    r = ValidationReport(contexts=[_v()])
    d = report_to_dict(r)
    assert "contexts" in d
    assert d["contexts"][0]["category"] == "mood_emotion"
    assert d["totals"]["contexts"] == 1
    assert d["totals"]["all"] == 1
