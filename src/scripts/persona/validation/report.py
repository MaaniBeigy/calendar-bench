"""Bundle every check's violations into a deterministic text + JSON report.

The text section order mirrors the legacy `Summarize.py` so existing reviewers
can scan a generated run with the same eyes. The JSON form is the same data
in machine-readable shape so CI and dashboards can ingest it directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.scripts.persona.validation.check_characteristics import CharacteristicViolation
from src.scripts.persona.validation.check_contexts import ContextViolation
from src.scripts.persona.validation.check_continuity import ContinuityViolation
from src.scripts.persona.validation.check_event import EventViolation
from src.scripts.persona.validation.check_ltl import LTLViolation
from src.scripts.persona.validation.check_persona import PersonaViolation


@dataclass(frozen=True)
class ValidationReport:
    """All six streams of violations from a single run."""

    persona: list[PersonaViolation] = field(default_factory=list)
    event: list[EventViolation] = field(default_factory=list)
    ltl: list[LTLViolation] = field(default_factory=list)
    continuity: list[ContinuityViolation] = field(default_factory=list)
    characteristics: list[CharacteristicViolation] = field(default_factory=list)
    contexts: list[ContextViolation] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            len(self.persona)
            + len(self.event)
            + len(self.ltl)
            + len(self.continuity)
            + len(self.characteristics)
            + len(self.contexts)
        )

    @property
    def has_violations(self) -> bool:
        return self.total > 0


def _persona_lines(violations: list[PersonaViolation]) -> list[str]:
    out = ["=== Persona Feature Check ==="]
    if not violations:
        out.append("No violations found.")
        return out
    for v in violations:
        who = v.person_id or "(persona)"
        out.append(f"{v.kind} | persona={v.persona_id} person={who}: {v.detail}")
    return out


def _event_lines(violations: list[EventViolation]) -> list[str]:
    out = ["=== Event Constraint Violations ==="]
    if not violations:
        out.append("No violations found.")
        return out
    for v in violations:
        suffix = f" event_idx={v.event_idx}" if v.event_idx is not None else ""
        out.append(
            f"{v.kind} | person={v.person_id} day={v.day_index} "
            f"event={v.event_name}{suffix}: {v.detail}"
        )
    return out


def _ltl_lines(violations: list[LTLViolation]) -> list[str]:
    out = ["=== LTL (Event Model) Constraint Violations ==="]
    if not violations:
        out.append("No violations found.")
        return out
    for v in violations:
        day_part = f" day={v.day_index}" if v.day_index is not None else ""
        out.append(f"rule={v.rule_id} | person={v.person_id}{day_part}: {v.detail}")
    return out


def _continuity_lines(violations: list[ContinuityViolation]) -> list[str]:
    out = ["=== Continuity Check ==="]
    if not violations:
        out.append("No violations found.")
        return out
    for v in violations:
        out.append(f"{v.kind} | person={v.person_id} day={v.day_index}: {v.detail}")
    return out


def _characteristic_lines(
    violations: list[CharacteristicViolation],
) -> list[str]:
    out = ["=== Characteristic Distribution Check ==="]
    if not violations:
        out.append("No violations found.")
        return out
    for v in violations:
        out.append(f"{v.kind} | persona={v.persona_id} axis={v.axis}: {v.detail}")
    return out


def _context_lines(violations: list[ContextViolation]) -> list[str]:
    out = ["=== Context Distribution Check ==="]
    if not violations:
        out.append("No violations found.")
        return out
    for v in violations:
        member = f" member={v.member}" if v.member else ""
        out.append(
            f"{v.kind} | person={v.person_id} category={v.category}{member}: "
            f"{v.detail}"
        )
    return out


def report_to_text(report: ValidationReport) -> str:
    """Render the report as a human-readable text block."""
    sections: list[list[str]] = [
        _persona_lines(report.persona),
        _event_lines(report.event),
        _ltl_lines(report.ltl),
        _continuity_lines(report.continuity),
        _characteristic_lines(report.characteristics),
        _context_lines(report.contexts),
    ]
    chunks: list[str] = []
    for section in sections:
        chunks.append("\n".join(section))
    return "\n\n".join(chunks) + "\n"


def report_to_dict(report: ValidationReport) -> dict[str, object]:
    """Render the report as a JSON-serializable dict."""
    return {
        "persona": [
            {
                "persona_id": v.persona_id,
                "person_id": v.person_id,
                "kind": v.kind,
                "detail": v.detail,
            }
            for v in report.persona
        ],
        "event": [
            {
                "person_id": v.person_id,
                "day_index": v.day_index,
                "event_name": v.event_name,
                "kind": v.kind,
                "detail": v.detail,
                "event_idx": v.event_idx,
            }
            for v in report.event
        ],
        "ltl": [
            {
                "person_id": v.person_id,
                "rule_id": v.rule_id,
                "day_index": v.day_index,
                "detail": v.detail,
            }
            for v in report.ltl
        ],
        "continuity": [
            {
                "person_id": v.person_id,
                "day_index": v.day_index,
                "kind": v.kind,
                "detail": v.detail,
            }
            for v in report.continuity
        ],
        "characteristics": [
            {
                "persona_id": v.persona_id,
                "axis": v.axis,
                "kind": v.kind,
                "detail": v.detail,
            }
            for v in report.characteristics
        ],
        "contexts": [
            {
                "person_id": v.person_id,
                "category": v.category,
                "member": v.member,
                "kind": v.kind,
                "detail": v.detail,
            }
            for v in report.contexts
        ],
        "totals": {
            "persona": len(report.persona),
            "event": len(report.event),
            "ltl": len(report.ltl),
            "continuity": len(report.continuity),
            "characteristics": len(report.characteristics),
            "contexts": len(report.contexts),
            "all": report.total,
        },
    }


def write_report(report: ValidationReport, out_dir: Path | str) -> tuple[Path, Path]:
    """Write `summary_report.txt` + `summary_report.json`. Returns the pair."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir_path / "summary_report.txt"
    json_path = out_dir_path / "summary_report.json"
    txt_path.write_text(report_to_text(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return txt_path, json_path


__all__ = [
    "ValidationReport",
    "report_to_dict",
    "report_to_text",
    "write_report",
]
