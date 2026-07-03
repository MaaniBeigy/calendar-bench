"""Bake `hb:matchedActivity` triples into HealthTasks TTL + JSON artefacts.

Reads `matched_activity.jsonl` (output of `precompute_matched_activity`)
plus the current HealthTasks TTL and JSON, writes a new versioned pair
that carries `hb:matchedActivity` per task instance.

Validates with rdflib before writing; aborts on parse errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rdflib import Graph

HB_TASK_PREFIX = "https://w3id.org/calendar-bench/health/task/"
HA_INSTANCE_PREFIX = "https://w3id.org/calendar-bench/human-activities/activity/"

_PROPERTY_DECLARATION_BLOCK = """hb:matchedActivity a owl:ObjectProperty ;
    rdfs:domain hb:HealthTask ;
    rdfs:range  ha:HumanActivity ;
    rdfs:label  "matched activity"@en ;
    rdfs:comment "Bridge from a HealthTask to the most representative Compendium of Physical Activities instance."@en .
"""

_HA_PREFIX_DIRECTIVE = "@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .\n@prefix ha:      <https://w3id.org/calendar-bench/human-activities/> .\n"

_VERSION_LINE = re.compile(r'owl:versionInfo\s+"[^"]*"')


@dataclass(frozen=True, slots=True)
class MatchedActivityRecord:
    """One resolved row from `matched_activity.jsonl`."""

    task_slug: str
    task_iri: str
    matched_activity_iris: tuple[str, ...]


def load_records(jsonl_path: Path) -> list[MatchedActivityRecord]:
    """Read `matched_activity.jsonl` and return one record per line."""
    out: list[MatchedActivityRecord] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        iris = tuple(data.get("matched_activity_iris") or ())
        if not iris:
            continue
        out.append(
            MatchedActivityRecord(
                task_slug=str(data["task_slug"]),
                task_iri=str(data["task_iri"]),
                matched_activity_iris=iris,
            )
        )
    return out


def _iri_to_curie(iri: str) -> str:
    """Render a HumanActivities activity IRI as a `ha-act:slug` CURIE."""
    if not iri.startswith(HA_INSTANCE_PREFIX):
        raise ValueError(f"expected HumanActivities activity IRI, got {iri!r}")
    return f"ha-act:{iri[len(HA_INSTANCE_PREFIX):]}"


def _instance_block_pattern(slug: str) -> re.Pattern[str]:
    """Match the full instance block ending at the first standalone `.`."""
    escaped = re.escape(slug)
    return re.compile(
        rf"^(hb-tk:{escaped}\s+a\s+[^;]+;.*?)\.\s*$",
        re.DOTALL | re.MULTILINE,
    )


def _matched_activity_clause(iris: Iterable[str]) -> str:
    """Render the `hb:matchedActivity` clause for one or more targets."""
    curies = ", ".join(_iri_to_curie(iri) for iri in iris)
    return f"hb:matchedActivity {curies}"


def inject_into_ttl(
    ttl_text: str,
    records: Iterable[MatchedActivityRecord],
    *,
    new_version: str,
) -> str:
    """Return the TTL with the property declaration and per-task triples added."""
    if "ha-act:" not in ttl_text:
        ttl_text = _HA_PREFIX_DIRECTIVE + ttl_text
    if "hb:matchedActivity" not in ttl_text:
        # Insert the property declaration just before the first class definition.
        marker = "## Class hierarchy"
        if marker in ttl_text:
            ttl_text = ttl_text.replace(
                marker,
                _PROPERTY_DECLARATION_BLOCK + "\n" + marker,
                1,
            )
        else:
            ttl_text = ttl_text + "\n" + _PROPERTY_DECLARATION_BLOCK
    if _VERSION_LINE.search(ttl_text):
        ttl_text = _VERSION_LINE.sub(
            f'owl:versionInfo     "{new_version}"', ttl_text, count=1
        )
    for record in records:
        pattern = _instance_block_pattern(record.task_slug)
        clause = _matched_activity_clause(record.matched_activity_iris)
        replacement = rf"\1;\n    {clause} ."
        new_text, n = pattern.subn(replacement, ttl_text, count=1)
        if n == 0:
            raise ValueError(
                f"instance block for hb-tk:{record.task_slug} not found in TTL"
            )
        ttl_text = new_text
    return ttl_text


def inject_into_json(
    payload: dict[str, Any],
    records: Iterable[MatchedActivityRecord],
) -> dict[str, Any]:
    """Walk the nested JSON and attach `matchedActivity` to each task entry."""
    slug_to_iris = {r.task_slug: list(r.matched_activity_iris) for r in records}
    seen: set[str] = set()
    for intervention in payload.values():
        if not isinstance(intervention, dict):
            continue
        for challenge in intervention.values():
            if not isinstance(challenge, dict):
                continue
            for level in challenge.values():
                if not isinstance(level, dict):
                    continue
                for task_slug, task in level.items():
                    if not isinstance(task, dict):
                        continue
                    iris = slug_to_iris.get(task_slug)
                    if iris is None:
                        continue
                    task["matchedActivity"] = list(iris)
                    seen.add(task_slug)
    missing = set(slug_to_iris) - seen
    if missing:
        raise ValueError(
            f"matched_activity records reference unknown JSON task slugs: "
            f"{sorted(missing)}"
        )
    return payload


def _validate_ttl(ttl_text: str) -> None:
    Graph().parse(data=ttl_text, format="turtle")


def run(
    jsonl_path: Path,
    ttl_in: Path,
    json_in: Path,
    ttl_out: Path,
    json_out: Path,
    *,
    new_version: str,
) -> int:
    """End-to-end: rewrite the TTL + JSON, return record count."""
    records = load_records(jsonl_path)
    ttl_text = ttl_in.read_text(encoding="utf-8")
    new_ttl = inject_into_ttl(ttl_text, records, new_version=new_version)
    _validate_ttl(new_ttl)
    payload = json.loads(json_in.read_text(encoding="utf-8"))
    new_payload = inject_into_json(payload, records)
    ttl_out.parent.mkdir(parents=True, exist_ok=True)
    ttl_out.write_text(new_ttl, encoding="utf-8")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(new_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(records)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--ttl-in", required=True, type=Path)
    parser.add_argument("--json-in", required=True, type=Path)
    parser.add_argument("--ttl-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--version", required=True, dest="new_version")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        count = run(
            args.jsonl,
            args.ttl_in,
            args.json_in,
            args.ttl_out,
            args.json_out,
            new_version=args.new_version,
        )
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"enrichment failed: {exc}\n")
        return 2
    sys.stdout.write(
        f"enriched {count} tasks; wrote {args.ttl_out} and {args.json_out}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
