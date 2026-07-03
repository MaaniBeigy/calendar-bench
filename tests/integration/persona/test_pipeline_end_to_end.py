"""End-to-end integration tests for the persona generation pipeline.

These run the CLI on the bundled example configs against a single worker, so
the test footprint is the same shape a real run would have, just smaller.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.scripts.persona import cli


def _run_generate(
    out_dir: Path,
    environment: Path,
    personas: Path,
    events: Path,
    rules: Path,
    *,
    workers: int = 1,
    executor: str = "thread",
    seed: int = 20260503,
) -> int:
    return cli.main(
        [
            "generate",
            "--environment",
            str(environment),
            "--personas",
            str(personas),
            "--events",
            str(events),
            "--rules",
            str(rules),
            "--out-dir",
            str(out_dir),
            "--workers",
            str(workers),
            "--executor",
            executor,
            "--seed",
            str(seed),
            "--allow-violations",
        ]
    )


def _read_index(out_dir: Path) -> dict:
    return json.loads((out_dir / "index.json").read_text(encoding="utf-8"))


def _read_persons_canonical(out_dir: Path) -> list[str]:
    """Return the canonical-string form of every per-person json, sorted by id."""
    persons_dir = out_dir / "persons"
    out = []
    for path in sorted(persons_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out.append(json.dumps(payload, sort_keys=True))
    return out


def test_pipeline_writes_full_run_artifacts(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    code = _run_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    assert code == 0
    for required in (
        "index.json",
        "summary_report.txt",
        "summary_report.json",
        "used_configs.json",
    ):
        assert (out_dir / required).exists(), required
    persons = list((out_dir / "persons").glob("*.json"))
    ics = list((out_dir / "ics_per_person").glob("*.ics"))
    assert len(persons) == len(ics)
    assert len(persons) >= 5  # 3 students + 2 fulltime in the example


def test_pipeline_index_lists_every_person(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    _run_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    index = _read_index(out_dir)
    on_disk = {p.stem for p in (out_dir / "persons").glob("*.json")}
    in_index = {entry["person_id"] for entry in index["persons"]}
    assert on_disk == in_index


def test_pipeline_each_person_json_has_full_horizon(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    _run_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    # The example horizon is 4 weeks => 28 days.
    for path in (out_dir / "persons").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["days"]) == 28
        assert payload["days"][0]["weekday"] == "Mon"


def test_pipeline_is_deterministic_across_runs(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run_generate(out_a, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    _run_generate(out_b, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    assert _read_persons_canonical(out_a) == _read_persons_canonical(out_b)


def test_pipeline_is_deterministic_across_worker_counts(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run_generate(
        out_a, environment_yaml, persona_yaml, event_yaml, rules_yaml, workers=1
    )
    _run_generate(
        out_a, environment_yaml, persona_yaml, event_yaml, rules_yaml, workers=1
    )
    _run_generate(
        out_b, environment_yaml, persona_yaml, event_yaml, rules_yaml, workers=2
    )
    assert _read_persons_canonical(out_a) == _read_persons_canonical(out_b)


def test_validate_subcommand_re_emits_report(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    _run_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    code = cli.main(
        [
            "validate",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--allow-violations",
        ]
    )
    assert code == 0
