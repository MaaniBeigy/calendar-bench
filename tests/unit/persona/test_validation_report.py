"""Unit tests for src.scripts.persona.validation.report."""

from __future__ import annotations

import json

from src.scripts.persona.validation.check_characteristics import CharacteristicViolation
from src.scripts.persona.validation.check_continuity import ContinuityViolation
from src.scripts.persona.validation.check_event import EventViolation
from src.scripts.persona.validation.check_ltl import LTLViolation
from src.scripts.persona.validation.check_persona import PersonaViolation
from src.scripts.persona.validation.report import (
    ValidationReport,
    report_to_dict,
    report_to_text,
    write_report,
)


def _full_report() -> ValidationReport:
    return ValidationReport(
        persona=[
            PersonaViolation(
                persona_id="a",
                person_id="a_0000",
                kind="cadence",
                detail="missing padel",
            )
        ],
        event=[
            EventViolation(
                person_id="a_0000",
                day_index=2,
                event_name="lunch",
                kind="per_event_duration",
                detail="duration 10 < min 30",
                event_idx=0,
            )
        ],
        ltl=[
            LTLViolation(
                person_id="a_0000",
                rule_id="r1",
                day_index=0,
                detail="overlap",
            )
        ],
        continuity=[
            ContinuityViolation(
                person_id="a_0000",
                day_index=1,
                kind="overlap",
                detail="event hits spillover",
            )
        ],
    )


def test_total_and_has_violations_track_all_streams():
    report = _full_report()
    assert report.total == 4
    assert report.has_violations is True


def test_clean_report_has_no_violations():
    report = ValidationReport()
    assert report.total == 0
    assert report.has_violations is False


def test_report_to_text_contains_each_section():
    text = report_to_text(_full_report())
    for header in (
        "=== Persona Feature Check ===",
        "=== Event Constraint Violations ===",
        "=== LTL (Event Model) Constraint Violations ===",
        "=== Continuity Check ===",
    ):
        assert header in text
    assert "missing padel" in text
    assert "duration 10 < min 30" in text


def test_report_to_text_clean_run_says_no_violations_per_section():
    text = report_to_text(ValidationReport())
    assert text.count("No violations found.") == 6


def test_report_to_dict_carries_totals():
    rendered = report_to_dict(_full_report())
    assert rendered["totals"] == {
        "persona": 1,
        "event": 1,
        "ltl": 1,
        "continuity": 1,
        "characteristics": 0,
        "contexts": 0,
        "all": 4,
    }
    # Round-trips through JSON.
    json.dumps(rendered)


def test_report_to_dict_event_includes_event_idx():
    rendered = report_to_dict(_full_report())
    assert rendered["event"][0]["event_idx"] == 0


def test_event_section_omits_event_idx_when_none():
    report = ValidationReport(
        event=[
            EventViolation(
                person_id="a_0000",
                day_index=2,
                event_name="lunch",
                kind="per_day",
                detail="total too small",
            )
        ]
    )
    text = report_to_text(report)
    assert "event_idx" not in text


def test_ltl_section_omits_day_when_none():
    report = ValidationReport(
        ltl=[
            LTLViolation(
                person_id="a_0000",
                rule_id="r1",
                day_index=None,
                detail="malformed",
            )
        ]
    )
    text = report_to_text(report)
    line = next(line for line in text.splitlines() if "rule=r1" in line)
    assert "day=" not in line


def test_report_includes_characteristic_violations_in_text_and_dict():
    report = ValidationReport(
        characteristics=[
            CharacteristicViolation(
                persona_id="gym_rat",
                axis="socioeconomic_status",
                kind="count_mismatch",
                detail="label 'low': configured 70.00% -> expected 7, realized 4",
            )
        ]
    )
    text = report_to_text(report)
    assert "=== Characteristic Distribution Check ===" in text
    assert "axis=socioeconomic_status" in text
    payload = report_to_dict(report)
    assert payload["totals"]["characteristics"] == 1
    assert payload["characteristics"][0]["axis"] == "socioeconomic_status"


def test_write_report_creates_two_files(tmp_path):
    txt, jsonp = write_report(_full_report(), tmp_path)
    assert txt.name == "summary_report.txt"
    assert jsonp.name == "summary_report.json"
    assert "missing padel" in txt.read_text(encoding="utf-8")
    payload = json.loads(jsonp.read_text(encoding="utf-8"))
    assert payload["totals"]["all"] == 4
