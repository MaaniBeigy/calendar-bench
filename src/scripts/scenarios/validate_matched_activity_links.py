"""Validate `hb:matchedActivity` links against a loaded HumanActivities TTL.

Two checks per HealthTask instance:

1. Resolution: every `hb:matchedActivity` target IRI resolves to a real
   HumanActivities instance.
2. Coverage: every HealthTask instance carries at least one
   `hb:matchedActivity` triple.

Emits a `matched_activity_violations.{txt,json}` sidecar.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Namespace
from rdflib.namespace import RDF

HB = Namespace("https://w3id.org/calendar-bench/health/")
HB_TK = Namespace("https://w3id.org/calendar-bench/health/task/")
HA = Namespace("https://w3id.org/calendar-bench/human-activities/")
HA_ACT = Namespace("https://w3id.org/calendar-bench/human-activities/activity/")

REASON_LABEL_NOT_FOUND = "iri_not_in_human_activities"
REASON_NO_MATCHED_ACTIVITY = "task_has_no_matched_activity"


@dataclass(frozen=True, slots=True)
class Violation:
    """One drift entry. Either a missing target IRI or a coverage gap."""

    task_iri: str
    task_label: str
    missing_target_iri: str
    missing_target_label: str
    reason: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate validator result with version pins for reproducibility."""

    health_tasks_version: str
    human_activities_version: str
    checked: int
    violations: tuple[Violation, ...]


def _ontology_version(graph: Graph, ontology_iri: str) -> str:
    """Return the `owl:versionInfo` value or `unknown`."""
    owl_version = Namespace("http://www.w3.org/2002/07/owl#").versionInfo
    for _, _, value in graph.triples((None, owl_version, None)):
        return str(value)
    return "unknown"


def _label_of(graph: Graph, subject) -> str:
    rdfs_label = Namespace("http://www.w3.org/2000/01/rdf-schema#").label
    for _, _, value in graph.triples((subject, rdfs_label, None)):
        return str(value)
    return ""


def _is_health_task_instance(graph: Graph, subject) -> bool:
    """True iff the subject's class chain reaches `hb:HealthTask`.

    Walks `rdf:type` then `rdfs:subClassOf*` to handle the LevelN to
    Sub-task to Task class hierarchy.
    """
    if not str(subject).startswith(str(HB_TK)):
        return False
    rdfs_subclass = Namespace("http://www.w3.org/2000/01/rdf-schema#").subClassOf
    seen: set = set()
    stack = [obj for _, _, obj in graph.triples((subject, RDF.type, None))]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if cls == HB.HealthTask:
            return True
        for _, _, parent in graph.triples((cls, rdfs_subclass, None)):
            stack.append(parent)
    return False


def validate(
    health_tasks_ttl: Path,
    human_activities_ttl: Path,
) -> ValidationReport:
    """Run both checks and return a populated :class:`ValidationReport`."""
    hb_graph = Graph()
    hb_graph.parse(health_tasks_ttl.as_posix(), format="turtle")
    ha_graph = Graph()
    ha_graph.parse(human_activities_ttl.as_posix(), format="turtle")

    ha_known: set[str] = {
        str(s)
        for s, _, _ in ha_graph.triples((None, RDF.type, None))
        if str(s).startswith(str(HA_ACT))
    }
    ha_label_by_iri: dict[str, str] = {}
    for iri in ha_known:
        from rdflib import URIRef

        ha_label_by_iri[iri] = _label_of(ha_graph, URIRef(iri))

    violations: list[Violation] = []
    checked = 0
    for subject, _, _ in hb_graph.triples((None, RDF.type, None)):
        if not _is_health_task_instance(hb_graph, subject):
            continue
        checked += 1
        task_iri = str(subject)
        task_label = _label_of(hb_graph, subject)
        targets = [
            str(target)
            for _, _, target in hb_graph.triples((subject, HB.matchedActivity, None))
        ]
        if not targets:
            violations.append(
                Violation(
                    task_iri=task_iri,
                    task_label=task_label,
                    missing_target_iri="",
                    missing_target_label="",
                    reason=REASON_NO_MATCHED_ACTIVITY,
                )
            )
            continue
        for target in targets:
            if target not in ha_known:
                violations.append(
                    Violation(
                        task_iri=task_iri,
                        task_label=task_label,
                        missing_target_iri=target,
                        missing_target_label=ha_label_by_iri.get(target, ""),
                        reason=REASON_LABEL_NOT_FOUND,
                    )
                )

    return ValidationReport(
        health_tasks_version=_ontology_version(hb_graph, str(HB)),
        human_activities_version=_ontology_version(ha_graph, str(HA)),
        checked=checked,
        violations=tuple(violations),
    )


def write_sidecar(
    report: ValidationReport,
    out_dir: Path,
    *,
    name: str = "matched_activity_violations",
) -> tuple[Path, Path]:
    """Persist the report as `{name}.json` plus `{name}.txt`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    txt_path = out_dir / f"{name}.txt"
    payload = {
        "ontology_versions": {
            "health_tasks": report.health_tasks_version,
            "human_activities": report.human_activities_version,
        },
        "checked": report.checked,
        "violations": [asdict(v) for v in report.violations],
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(_format_text(report), encoding="utf-8")
    return (json_path, txt_path)


def _format_text(report: ValidationReport) -> str:
    """Render the report as a human-readable summary."""
    lines = [
        "matched_activity validation",
        f"  health_tasks version: {report.health_tasks_version}",
        f"  human_activities version: {report.human_activities_version}",
        f"  tasks checked: {report.checked}",
        f"  violations:   {len(report.violations)}",
    ]
    counts: dict[str, int] = {}
    for v in report.violations:
        counts[v.reason] = counts.get(v.reason, 0) + 1
    for reason, n in sorted(counts.items()):
        lines.append(f"    {reason}: {n}")
    if report.violations:
        lines.append("")
        lines.append("details:")
        for v in report.violations:
            if v.reason == REASON_NO_MATCHED_ACTIVITY:
                lines.append(f"  - {v.task_iri} ({v.task_label}): no matched_activity")
            else:
                lines.append(
                    f"  - {v.task_iri} ({v.task_label}): "
                    f"missing target {v.missing_target_iri}"
                )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-ttl", required=True, type=Path)
    parser.add_argument("--ha-ttl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = validate(args.health_ttl, args.ha_ttl)
    write_sidecar(report, args.out_dir)
    sys.stdout.write(
        f"validated {report.checked} tasks; {len(report.violations)} violations\n"
    )
    return 0 if not report.violations else 1


if __name__ == "__main__":
    sys.exit(main())
