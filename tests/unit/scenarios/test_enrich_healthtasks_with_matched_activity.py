"""Tests for `scenarios.enrich_healthtasks_with_matched_activity`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Graph, Namespace, URIRef

from src.scripts.scenarios import enrich_healthtasks_with_matched_activity as mod

HB = Namespace("https://w3id.org/calendar-bench/health/")
HB_TK = Namespace(mod.HB_TASK_PREFIX)
HA_ACT = Namespace(mod.HA_INSTANCE_PREFIX)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_ttl() -> str:
    return """## ----------------------------------------------------------------------
## Calendar-Bench HealthTasks knowledge graph
## ----------------------------------------------------------------------

@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .


<https://w3id.org/calendar-bench/health/> a owl:Ontology ;
    dcterms:title       "HealthTasks" ;
    owl:versionInfo     "2026.05.06" .


## Datatype properties

hb:isConcurrent a owl:DatatypeProperty .

## Class hierarchy
hb:HealthTask a owl:Class .
hb:PhysicalActivityTask a owl:Class ;
    rdfs:subClassOf hb:HealthTask .
hb:PhysicalActivityCardioAndStaminaTask a owl:Class ;
    rdfs:subClassOf hb:PhysicalActivityTask .
hb:PhysicalActivityCardioAndStaminaLevel1 a owl:Class ;
    rdfs:subClassOf hb:PhysicalActivityCardioAndStaminaTask .


hb-tk:do-10-minutes-of-cardio a hb:PhysicalActivityCardioAndStaminaLevel1 ;
    dcterms:title       "Do 10 Minutes of Cardio" ;
    dcterms:description "Choose one cardio activity." ;
    hb:estimatedDurationMinutes 10 ;
    hb:isConcurrent     false ;
    hb:isDividable      false .

hb-tk:do-15-minutes-of-cardio a hb:PhysicalActivityCardioAndStaminaLevel1 ;
    dcterms:title       "Do 15 Minutes of Cardio" ;
    dcterms:description "Choose one cardio activity." ;
    hb:estimatedDurationMinutes 15 ;
    hb:isConcurrent     false ;
    hb:isDividable      false .
"""


@pytest.fixture()
def base_json() -> dict:
    return {
        "PhysicalActivity": {
            "CardioAndStamina": {
                "Level1": {
                    "do-10-minutes-of-cardio": {
                        "title": "Do 10 Minutes of Cardio",
                        "description": "Choose one cardio activity.",
                        "isConcurrent": False,
                        "isDividable": False,
                        "estimatedDuration": 10,
                    },
                    "do-15-minutes-of-cardio": {
                        "title": "Do 15 Minutes of Cardio",
                        "description": "Choose one cardio activity.",
                        "isConcurrent": False,
                        "isDividable": False,
                        "estimatedDuration": 15,
                    },
                }
            }
        }
    }


@pytest.fixture()
def records() -> list[mod.MatchedActivityRecord]:
    return [
        mod.MatchedActivityRecord(
            task_slug="do-10-minutes-of-cardio",
            task_iri=f"{mod.HB_TASK_PREFIX}do-10-minutes-of-cardio",
            matched_activity_iris=(f"{mod.HA_INSTANCE_PREFIX}aerobic-general",),
        ),
        mod.MatchedActivityRecord(
            task_slug="do-15-minutes-of-cardio",
            task_iri=f"{mod.HB_TASK_PREFIX}do-15-minutes-of-cardio",
            matched_activity_iris=(
                f"{mod.HA_INSTANCE_PREFIX}aerobic-general",
                f"{mod.HA_INSTANCE_PREFIX}bicycling-general",
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------


def test_load_records_round_trip(tmp_path: Path):
    path = tmp_path / "in.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_slug": "t1",
                "task_iri": f"{mod.HB_TASK_PREFIX}t1",
                "matched_activity_iris": [f"{mod.HA_INSTANCE_PREFIX}aerobic-general"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = mod.load_records(path)
    assert out[0].task_slug == "t1"
    assert out[0].matched_activity_iris == (f"{mod.HA_INSTANCE_PREFIX}aerobic-general",)


def test_load_records_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "blank.jsonl"
    path.write_text(
        "\n"
        + json.dumps(
            {
                "task_slug": "t1",
                "task_iri": "x",
                "matched_activity_iris": [
                    "https://w3id.org/calendar-bench/human-activities/activity/a"
                ],
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    assert len(mod.load_records(path)) == 1


def test_load_records_drops_rows_with_no_iris(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text(
        json.dumps({"task_slug": "t1", "task_iri": "x", "matched_activity_iris": []})
        + "\n",
        encoding="utf-8",
    )
    assert mod.load_records(path) == []


# ---------------------------------------------------------------------------
# CURIE helpers
# ---------------------------------------------------------------------------


def test_iri_to_curie_happy_path():
    assert (
        mod._iri_to_curie(f"{mod.HA_INSTANCE_PREFIX}aerobic-general")
        == "ha-act:aerobic-general"
    )


def test_iri_to_curie_rejects_off_prefix():
    with pytest.raises(ValueError):
        mod._iri_to_curie("https://example.com/other")


def test_matched_activity_clause_single():
    assert (
        mod._matched_activity_clause([f"{mod.HA_INSTANCE_PREFIX}aerobic-general"])
        == "hb:matchedActivity ha-act:aerobic-general"
    )


def test_matched_activity_clause_multi():
    out = mod._matched_activity_clause(
        [
            f"{mod.HA_INSTANCE_PREFIX}aerobic-general",
            f"{mod.HA_INSTANCE_PREFIX}bicycling-general",
        ]
    )
    assert out == "hb:matchedActivity ha-act:aerobic-general, ha-act:bicycling-general"


# ---------------------------------------------------------------------------
# inject_into_ttl
# ---------------------------------------------------------------------------


def test_inject_into_ttl_adds_property_declaration(
    base_ttl: str, records: list[mod.MatchedActivityRecord]
):
    out = mod.inject_into_ttl(base_ttl, records, new_version="2026.05.18")
    assert "hb:matchedActivity a owl:ObjectProperty" in out
    # Declaration is anchored before the class hierarchy comment.
    assert out.index("hb:matchedActivity a") < out.index("hb:HealthTask a owl:Class")


def test_inject_into_ttl_bumps_version(
    base_ttl: str, records: list[mod.MatchedActivityRecord]
):
    out = mod.inject_into_ttl(base_ttl, records, new_version="2026.05.18")
    assert '"2026.05.18"' in out
    assert '"2026.05.06"' not in out


def test_inject_into_ttl_attaches_single_target(
    base_ttl: str, records: list[mod.MatchedActivityRecord]
):
    out = mod.inject_into_ttl(base_ttl, [records[0]], new_version="2026.05.18")
    assert "hb-tk:do-10-minutes-of-cardio" in out
    assert "hb:matchedActivity ha-act:aerobic-general ." in out
    # Validates as RDF.
    graph = Graph()
    graph.parse(data=out, format="turtle")
    matched = list(
        graph.triples(
            (
                HB_TK["do-10-minutes-of-cardio"],
                HB["matchedActivity"],
                None,
            )
        )
    )
    assert len(matched) == 1
    assert matched[0][2] == URIRef(f"{mod.HA_INSTANCE_PREFIX}aerobic-general")


def test_inject_into_ttl_attaches_multi_target(
    base_ttl: str, records: list[mod.MatchedActivityRecord]
):
    out = mod.inject_into_ttl(base_ttl, [records[1]], new_version="2026.05.18")
    graph = Graph()
    graph.parse(data=out, format="turtle")
    targets = {
        str(o)
        for _, _, o in graph.triples(
            (HB_TK["do-15-minutes-of-cardio"], HB["matchedActivity"], None)
        )
    }
    assert targets == {
        f"{mod.HA_INSTANCE_PREFIX}aerobic-general",
        f"{mod.HA_INSTANCE_PREFIX}bicycling-general",
    }


def test_inject_into_ttl_unknown_slug_raises(
    base_ttl: str,
):
    bad = mod.MatchedActivityRecord(
        task_slug="missing-task",
        task_iri=f"{mod.HB_TASK_PREFIX}missing-task",
        matched_activity_iris=(f"{mod.HA_INSTANCE_PREFIX}aerobic-general",),
    )
    with pytest.raises(ValueError) as exc:
        mod.inject_into_ttl(base_ttl, [bad], new_version="2026.05.18")
    assert "missing-task" in str(exc.value)


def test_inject_into_ttl_inserts_prefix_when_absent(
    records: list[mod.MatchedActivityRecord],
):
    ttl_without_ha = """@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .

<https://w3id.org/calendar-bench/health/> a owl:Ontology ;
    owl:versionInfo "2026.05.06" .

## Class hierarchy
hb:HealthTask a owl:Class .
hb:PhysicalActivityCardioAndStaminaLevel1 a owl:Class .

hb-tk:do-10-minutes-of-cardio a hb:PhysicalActivityCardioAndStaminaLevel1 ;
    hb:isConcurrent false .
"""
    out = mod.inject_into_ttl(ttl_without_ha, [records[0]], new_version="2026.05.18")
    assert "@prefix ha-act:" in out


def test_inject_into_ttl_appends_declaration_when_no_class_marker(
    records: list[mod.MatchedActivityRecord],
):
    ttl_no_marker = """@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix ha:      <https://w3id.org/calendar-bench/human-activities/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .

<https://w3id.org/calendar-bench/health/> a owl:Ontology ;
    owl:versionInfo "2026.05.06" .

hb:HealthTask a owl:Class .
hb:PhysicalActivityCardioAndStaminaLevel1 a owl:Class .

hb-tk:do-10-minutes-of-cardio a hb:PhysicalActivityCardioAndStaminaLevel1 ;
    hb:isConcurrent false .
"""
    out = mod.inject_into_ttl(ttl_no_marker, [records[0]], new_version="2026.05.18")
    assert "hb:matchedActivity a owl:ObjectProperty" in out


def test_inject_into_ttl_skips_property_block_when_already_present(
    base_ttl: str, records: list[mod.MatchedActivityRecord]
):
    # First pass adds the property block.
    once = mod.inject_into_ttl(base_ttl, [records[0]], new_version="2026.05.18")
    # Second pass on a fresh TTL that ALREADY carries the declaration
    # to verify the block is not duplicated.  Use a TTL where the slug
    # block has not been mutated yet but the declaration is present.
    seeded = base_ttl.replace(
        "## Class hierarchy",
        mod._PROPERTY_DECLARATION_BLOCK + "\n## Class hierarchy",
        1,
    )
    twice = mod.inject_into_ttl(seeded, [records[0]], new_version="2026.05.18")
    assert twice.count("hb:matchedActivity a owl:ObjectProperty") == 1
    assert once.count("hb:matchedActivity a owl:ObjectProperty") == 1


def test_inject_into_ttl_skips_version_bump_when_absent(
    records: list[mod.MatchedActivityRecord],
):
    ttl = """@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix ha:      <https://w3id.org/calendar-bench/human-activities/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .

## Class hierarchy
hb:HealthTask a owl:Class .
hb:PhysicalActivityCardioAndStaminaLevel1 a owl:Class .

hb-tk:do-10-minutes-of-cardio a hb:PhysicalActivityCardioAndStaminaLevel1 ;
    hb:isConcurrent false .
"""
    out = mod.inject_into_ttl(ttl, [records[0]], new_version="2026.05.18")
    # No version line existed; injection should not crash and not add one.
    assert "owl:versionInfo" not in out


# ---------------------------------------------------------------------------
# inject_into_json
# ---------------------------------------------------------------------------


def test_inject_into_json_attaches_lists(base_json, records):
    out = mod.inject_into_json(base_json, records)
    cardio = out["PhysicalActivity"]["CardioAndStamina"]["Level1"]
    assert cardio["do-10-minutes-of-cardio"]["matchedActivity"] == [
        f"{mod.HA_INSTANCE_PREFIX}aerobic-general"
    ]
    assert cardio["do-15-minutes-of-cardio"]["matchedActivity"] == [
        f"{mod.HA_INSTANCE_PREFIX}aerobic-general",
        f"{mod.HA_INSTANCE_PREFIX}bicycling-general",
    ]


def test_inject_into_json_raises_on_missing_slug(base_json):
    record = mod.MatchedActivityRecord(
        task_slug="not-in-json",
        task_iri="x",
        matched_activity_iris=("y",),
    )
    with pytest.raises(ValueError) as exc:
        mod.inject_into_json(base_json, [record])
    assert "not-in-json" in str(exc.value)


def test_inject_into_json_tolerates_non_dict_nesting(records):
    payload = {
        "stringy": "skip me",
        "PhysicalActivity": {
            "stringy": "skip me",
            "CardioAndStamina": {
                "stringy": "skip me",
                "Level1": {
                    "stringy": "skip me",
                    "do-10-minutes-of-cardio": {
                        "title": "x",
                    },
                    "do-15-minutes-of-cardio": {"title": "y"},
                },
            },
        },
    }
    mod.inject_into_json(payload, records)
    assert payload["PhysicalActivity"]["CardioAndStamina"]["Level1"][
        "do-10-minutes-of-cardio"
    ]["matchedActivity"] == [f"{mod.HA_INSTANCE_PREFIX}aerobic-general"]


# ---------------------------------------------------------------------------
# End-to-end run + CLI
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_run_writes_ttl_and_json(tmp_path: Path, base_ttl: str, base_json, records):
    jsonl = tmp_path / "matched.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_slug": r.task_slug,
                    "task_iri": r.task_iri,
                    "matched_activity_iris": list(r.matched_activity_iris),
                }
            )
            for r in records
        )
        + "\n",
        encoding="utf-8",
    )
    ttl_in = _write(tmp_path / "in.ttl", base_ttl)
    json_in = _write(tmp_path / "in.json", json.dumps(base_json))
    ttl_out = tmp_path / "out.ttl"
    json_out = tmp_path / "out.json"
    count = mod.run(jsonl, ttl_in, json_in, ttl_out, json_out, new_version="2026.05.18")
    assert count == 2
    assert ttl_out.exists() and json_out.exists()
    assert '"2026.05.18"' in ttl_out.read_text(encoding="utf-8")
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert (
        "matchedActivity"
        in payload["PhysicalActivity"]["CardioAndStamina"]["Level1"][
            "do-10-minutes-of-cardio"
        ]
    )


def test_run_aborts_on_invalid_ttl(tmp_path: Path, base_json, records, monkeypatch):
    jsonl = tmp_path / "matched.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "task_slug": records[0].task_slug,
                "task_iri": records[0].task_iri,
                "matched_activity_iris": list(records[0].matched_activity_iris),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ttl_in = _write(tmp_path / "in.ttl", "not valid turtle <<")
    json_in = _write(tmp_path / "in.json", json.dumps(base_json))
    with pytest.raises(ValueError):
        mod.run(
            jsonl,
            ttl_in,
            json_in,
            tmp_path / "out.ttl",
            tmp_path / "out.json",
            new_version="2026.05.18",
        )


def test_main_success(tmp_path: Path, base_ttl: str, base_json, records, capsys):
    jsonl = tmp_path / "matched.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "task_slug": records[0].task_slug,
                "task_iri": records[0].task_iri,
                "matched_activity_iris": list(records[0].matched_activity_iris),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ttl_in = _write(tmp_path / "in.ttl", base_ttl)
    json_in = _write(tmp_path / "in.json", json.dumps(base_json))
    rc = mod.main(
        [
            "--jsonl",
            jsonl.as_posix(),
            "--ttl-in",
            ttl_in.as_posix(),
            "--json-in",
            json_in.as_posix(),
            "--ttl-out",
            (tmp_path / "out.ttl").as_posix(),
            "--json-out",
            (tmp_path / "out.json").as_posix(),
            "--version",
            "2026.05.18",
        ]
    )
    assert rc == 0
    assert "enriched 1 tasks" in capsys.readouterr().out


def test_main_failure_propagates(tmp_path: Path, base_json, records, capsys):
    jsonl = tmp_path / "matched.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "task_slug": records[0].task_slug,
                "task_iri": records[0].task_iri,
                "matched_activity_iris": list(records[0].matched_activity_iris),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rc = mod.main(
        [
            "--jsonl",
            jsonl.as_posix(),
            "--ttl-in",
            (tmp_path / "missing.ttl").as_posix(),
            "--json-in",
            (tmp_path / "missing.json").as_posix(),
            "--ttl-out",
            (tmp_path / "out.ttl").as_posix(),
            "--json-out",
            (tmp_path / "out.json").as_posix(),
            "--version",
            "2026.05.18",
        ]
    )
    assert rc == 2
    assert "enrichment failed" in capsys.readouterr().err
