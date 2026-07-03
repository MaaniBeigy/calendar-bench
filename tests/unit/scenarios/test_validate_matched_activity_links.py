"""Tests for `scenarios.validate_matched_activity_links`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios import validate_matched_activity_links as mod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


HEALTH_TTL_BASE = """
@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix ha:      <https://w3id.org/calendar-bench/human-activities/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

<https://w3id.org/calendar-bench/health/> a owl:Ontology ;
    owl:versionInfo "2026.05.18" .

hb:matchedActivity a owl:ObjectProperty .

hb:HealthTask a owl:Class .
hb:PhysicalActivityTask a owl:Class ;
    rdfs:subClassOf hb:HealthTask .
hb:PhysicalActivityCardioAndStaminaTask a owl:Class ;
    rdfs:subClassOf hb:PhysicalActivityTask .
hb:PhysicalActivityCardioAndStaminaLevel1 a owl:Class ;
    rdfs:subClassOf hb:PhysicalActivityCardioAndStaminaTask .

hb-tk:do-10-minutes-of-cardio a hb:PhysicalActivityCardioAndStaminaLevel1 ;
    rdfs:label "Do 10 Minutes of Cardio" ;
    hb:matchedActivity ha-act:aerobic-general .

hb-tk:no-bridge a hb:PhysicalActivityCardioAndStaminaLevel1 ;
    rdfs:label "No Bridge Task" .

hb-tk:bad-bridge a hb:PhysicalActivityCardioAndStaminaLevel1 ;
    rdfs:label "Bad Bridge Task" ;
    hb:matchedActivity ha-act:missing-activity .
"""

HA_TTL_BASE = """
@prefix ha:      <https://w3id.org/calendar-bench/human-activities/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .

<https://w3id.org/calendar-bench/human-activities/> a owl:Ontology ;
    owl:versionInfo "2026.05.03" .

ha:SportsExerciseWorkoutConditioningExercise a owl:Class .

ha-act:aerobic-general a ha:SportsExerciseWorkoutConditioningExercise ;
    rdfs:label "Aerobic, general" .
"""


@pytest.fixture()
def health_ttl(tmp_path: Path) -> Path:
    path = tmp_path / "hb.ttl"
    path.write_text(HEALTH_TTL_BASE, encoding="utf-8")
    return path


@pytest.fixture()
def ha_ttl(tmp_path: Path) -> Path:
    path = tmp_path / "ha.ttl"
    path.write_text(HA_TTL_BASE, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_returns_versions(self, health_ttl: Path, ha_ttl: Path):
        report = mod.validate(health_ttl, ha_ttl)
        assert report.health_tasks_version == "2026.05.18"
        assert report.human_activities_version == "2026.05.03"

    def test_counts_health_task_instances(self, health_ttl: Path, ha_ttl: Path):
        report = mod.validate(health_ttl, ha_ttl)
        assert report.checked == 3

    def test_reports_missing_target(self, health_ttl: Path, ha_ttl: Path):
        report = mod.validate(health_ttl, ha_ttl)
        bad = [v for v in report.violations if v.reason == mod.REASON_LABEL_NOT_FOUND]
        assert len(bad) == 1
        assert bad[0].task_iri.endswith("bad-bridge")
        assert "missing-activity" in bad[0].missing_target_iri

    def test_reports_coverage_gap(self, health_ttl: Path, ha_ttl: Path):
        report = mod.validate(health_ttl, ha_ttl)
        gaps = [
            v for v in report.violations if v.reason == mod.REASON_NO_MATCHED_ACTIVITY
        ]
        assert len(gaps) == 1
        assert gaps[0].task_iri.endswith("no-bridge")

    def test_no_violations_when_all_clean(self, tmp_path: Path, ha_ttl: Path):
        clean_ttl = tmp_path / "clean.ttl"
        clean_ttl.write_text(
            """
@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .

<https://w3id.org/calendar-bench/health/> a owl:Ontology ;
    owl:versionInfo "2026.05.18" .

hb:matchedActivity a owl:ObjectProperty .

hb:HealthTask a owl:Class .
hb:LevelClass a owl:Class ;
    <http://www.w3.org/2000/01/rdf-schema#subClassOf> hb:HealthTask .

hb-tk:ok a hb:LevelClass ;
    hb:matchedActivity ha-act:aerobic-general .
""",
            encoding="utf-8",
        )
        report = mod.validate(clean_ttl, ha_ttl)
        assert report.violations == ()

    def test_skips_non_task_subjects(self, tmp_path: Path, ha_ttl: Path):
        # `hb:HealthTask` is the root class; only `hb-tk:*` instances
        # should be counted.
        ttl = tmp_path / "x.ttl"
        ttl.write_text(
            """
@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .

hb:matchedActivity a owl:ObjectProperty .

hb:HealthTask a owl:Class .
hb:Sub a owl:Class ;
    <http://www.w3.org/2000/01/rdf-schema#subClassOf> hb:HealthTask .

hb-tk:in a hb:Sub ;
    hb:matchedActivity ha-act:aerobic-general .

hb:NotATask a hb:Sub .
""",
            encoding="utf-8",
        )
        report = mod.validate(ttl, ha_ttl)
        assert report.checked == 1

    def test_ontology_version_defaults_to_unknown(self, tmp_path: Path, ha_ttl: Path):
        ttl = tmp_path / "noversion.ttl"
        ttl.write_text(
            """
@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .

hb:matchedActivity a owl:ObjectProperty .
hb:HealthTask a owl:Class .
hb:Sub a owl:Class ;
    <http://www.w3.org/2000/01/rdf-schema#subClassOf> hb:HealthTask .

hb-tk:ok a hb:Sub ;
    hb:matchedActivity ha-act:aerobic-general .
""",
            encoding="utf-8",
        )
        report = mod.validate(ttl, ha_ttl)
        assert report.health_tasks_version == "unknown"

    def test_label_of_returns_empty_when_missing(self, tmp_path: Path, ha_ttl: Path):
        ttl = tmp_path / "nolabel.ttl"
        ttl.write_text(
            """
@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .

hb:HealthTask a owl:Class .
hb:Sub a owl:Class ;
    <http://www.w3.org/2000/01/rdf-schema#subClassOf> hb:HealthTask .

hb-tk:plain a hb:Sub .
""",
            encoding="utf-8",
        )
        report = mod.validate(ttl, ha_ttl)
        # Coverage check fires; label_of returns ""
        gap = report.violations[0]
        assert gap.task_label == ""

    def test_is_health_task_instance_handles_cycles(self, tmp_path: Path, ha_ttl: Path):
        # Pathological self-referencing subClassOf must NOT loop forever.
        ttl = tmp_path / "cycle.ttl"
        ttl.write_text(
            """
@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .

hb:HealthTask a owl:Class .
hb:Cycle a owl:Class ;
    <http://www.w3.org/2000/01/rdf-schema#subClassOf> hb:Cycle .

hb-tk:cyc a hb:Cycle ;
    hb:matchedActivity ha-act:aerobic-general .
""",
            encoding="utf-8",
        )
        # Should terminate; `cyc` is NOT a HealthTask, so checked stays 0.
        report = mod.validate(ttl, ha_ttl)
        assert report.checked == 0


# ---------------------------------------------------------------------------
# write_sidecar + text rendering
# ---------------------------------------------------------------------------


def test_write_sidecar_emits_json_and_text(
    tmp_path: Path, health_ttl: Path, ha_ttl: Path
):
    report = mod.validate(health_ttl, ha_ttl)
    json_path, txt_path = mod.write_sidecar(report, tmp_path / "out")
    assert json_path.exists() and txt_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ontology_versions"]["health_tasks"] == "2026.05.18"
    assert payload["checked"] == 3
    assert len(payload["violations"]) == 2
    assert {v["reason"] for v in payload["violations"]} == {
        mod.REASON_LABEL_NOT_FOUND,
        mod.REASON_NO_MATCHED_ACTIVITY,
    }


def test_format_text_lists_per_reason_counts(
    tmp_path: Path, health_ttl: Path, ha_ttl: Path
):
    report = mod.validate(health_ttl, ha_ttl)
    text = mod._format_text(report)
    assert "tasks checked: 3" in text
    assert "violations:   2" in text
    assert mod.REASON_LABEL_NOT_FOUND in text
    assert mod.REASON_NO_MATCHED_ACTIVITY in text
    assert "no matched_activity" in text


def test_format_text_handles_zero_violations(tmp_path: Path, ha_ttl: Path):
    clean_ttl = tmp_path / "clean.ttl"
    clean_ttl.write_text(
        """
@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .

hb:matchedActivity a owl:ObjectProperty .
hb:HealthTask a owl:Class .
hb:Sub a owl:Class ;
    <http://www.w3.org/2000/01/rdf-schema#subClassOf> hb:HealthTask .

hb-tk:ok a hb:Sub ;
    hb:matchedActivity ha-act:aerobic-general .
""",
        encoding="utf-8",
    )
    report = mod.validate(clean_ttl, ha_ttl)
    text = mod._format_text(report)
    assert "details:" not in text
    assert "violations:   0" in text


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_clean(tmp_path: Path, ha_ttl: Path, capsys):
    clean_ttl = tmp_path / "clean.ttl"
    clean_ttl.write_text(
        """
@prefix hb:      <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk:   <https://w3id.org/calendar-bench/health/task/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .

hb:matchedActivity a owl:ObjectProperty .
hb:HealthTask a owl:Class .
hb:Sub a owl:Class ;
    <http://www.w3.org/2000/01/rdf-schema#subClassOf> hb:HealthTask .

hb-tk:ok a hb:Sub ;
    hb:matchedActivity ha-act:aerobic-general .
""",
        encoding="utf-8",
    )
    rc = mod.main(
        [
            "--health-ttl",
            clean_ttl.as_posix(),
            "--ha-ttl",
            ha_ttl.as_posix(),
            "--out-dir",
            (tmp_path / "out").as_posix(),
        ]
    )
    assert rc == 0
    assert "0 violations" in capsys.readouterr().out


def test_main_returns_one_when_violations_exist(
    tmp_path: Path, health_ttl: Path, ha_ttl: Path, capsys
):
    rc = mod.main(
        [
            "--health-ttl",
            health_ttl.as_posix(),
            "--ha-ttl",
            ha_ttl.as_posix(),
            "--out-dir",
            (tmp_path / "out").as_posix(),
        ]
    )
    assert rc == 1
    assert "2 violations" in capsys.readouterr().out
