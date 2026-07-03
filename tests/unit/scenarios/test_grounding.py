"""Unit tests for src.scripts.scenarios.metrics.grounding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios.metrics.grounding import compute_grounding


def _write_tasks(tasks_dir: Path, person_id: str, tasks: list) -> None:
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f"{person_id}_tasks.json").write_text(
        json.dumps(tasks), encoding="utf-8"
    )


class TestComputeGrounding:
    def test_all_grounded(self, tmp_path):
        _write_tasks(
            tmp_path,
            "p001",
            [
                {"label": "yoga", "ontology_uri": "https://ex.org/task/yoga"},
                {"label": "walk", "ontology_uri": "https://ex.org/task/walk"},
            ],
        )
        out = compute_grounding(tmp_path, ["p001"])
        assert out["grounded"] == 2
        assert out["total"] == 2
        assert out["ratio"] == pytest.approx(1.0)
        assert out["per_person"]["p001"]["ratio"] == pytest.approx(1.0)

    def test_reads_weekly_wrapper_flattening_the_union(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "p001_tasks.json").write_text(
            json.dumps(
                {
                    "weeks": [
                        [{"label": "a", "ontology_uri": "https://ex.org/task/a"}],
                        [
                            {"label": "b", "ontology_uri": "https://ex.org/task/b"},
                            {"label": "c", "ontology_uri": "https://ex.org/task/c"},
                        ],
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = compute_grounding(tmp_path, ["p001"])
        assert out["total"] == 3
        assert out["grounded"] == 3

    def test_skips_non_list_non_wrapper_file(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "p001_tasks.json").write_text(json.dumps(42), encoding="utf-8")
        out = compute_grounding(tmp_path, ["p001"])
        assert "p001" not in out["per_person"]

    def test_none_grounded(self, tmp_path):
        _write_tasks(
            tmp_path,
            "p001",
            [{"label": "x", "ontology_uri": None}, {"label": "y"}],
        )
        out = compute_grounding(tmp_path, ["p001"])
        assert out["grounded"] == 0
        assert out["total"] == 2
        assert out["ratio"] == pytest.approx(0.0)

    def test_partial_grounded(self, tmp_path):
        _write_tasks(
            tmp_path,
            "p001",
            [
                {"label": "yoga", "ontology_uri": "https://ex.org/task/yoga"},
                {"label": "snack", "ontology_uri": None},
                {"label": "walk", "ontology_uri": "https://ex.org/task/walk"},
                {"label": "fruit"},
            ],
        )
        out = compute_grounding(tmp_path, ["p001"])
        assert out["grounded"] == 2
        assert out["total"] == 4
        assert out["ratio"] == pytest.approx(0.5)

    def test_aggregates_across_persons(self, tmp_path):
        _write_tasks(
            tmp_path,
            "p001",
            [{"ontology_uri": "https://ex.org/task/a"}],
        )
        _write_tasks(
            tmp_path,
            "p002",
            [{"ontology_uri": None}, {"ontology_uri": "https://ex.org/task/b"}],
        )
        out = compute_grounding(tmp_path, ["p001", "p002"])
        assert out["grounded"] == 2
        assert out["total"] == 3
        assert out["ratio"] == pytest.approx(2 / 3)

    def test_missing_file_skipped_silently(self, tmp_path):
        """Persons without a tasks file just don't contribute to the count."""
        _write_tasks(
            tmp_path,
            "p001",
            [{"ontology_uri": "https://ex.org/task/a"}],
        )
        out = compute_grounding(tmp_path, ["p001", "missing_person"])
        assert out["grounded"] == 1
        assert out["total"] == 1
        assert "missing_person" not in out["per_person"]

    def test_empty_person_list_returns_zero_ratio(self, tmp_path):
        out = compute_grounding(tmp_path, [])
        assert out == {
            "per_person": {},
            "grounded": 0,
            "verified": None,
            "total": 0,
            "ratio": 0.0,
            "verified_ratio": None,
            "expected_per_person": None,
            "persons_short_fetched": 0,
            "persons_with_unverified": None,
        }

    def test_malformed_json_skipped(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "p001_tasks.json").write_text("{not json", encoding="utf-8")
        out = compute_grounding(tmp_path, ["p001"])
        assert out["total"] == 0
        assert "p001" not in out["per_person"]

    def test_non_list_payload_skipped(self, tmp_path):
        """A tasks file that does not contain a JSON list is ignored."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "p001_tasks.json").write_text(
            json.dumps({"not": "a list"}), encoding="utf-8"
        )
        out = compute_grounding(tmp_path, ["p001"])
        assert out["total"] == 0

    def test_empty_task_list_returns_zero_ratio(self, tmp_path):
        """A person with zero tasks contributes 0/0 = ratio 0.0 (no division)."""
        _write_tasks(tmp_path, "p001", [])
        out = compute_grounding(tmp_path, ["p001"])
        assert out["per_person"]["p001"] == {
            "grounded": 0,
            "verified": None,
            "total": 0,
            "ratio": 0.0,
            "verified_ratio": None,
            "expected": None,
            "short_fetched": False,
            "unverified_uris": [],
        }

    def test_short_fetched_increments_when_total_below_expected(self, tmp_path):
        """`persons_short_fetched` counts every person whose generated
        task list ended up below the expected per-person target; the
        signal upstream consumers use to flag GraphRAG fetch failures.
        """
        _write_tasks(
            tmp_path,
            "p001",
            [{"ontology_uri": "https://ex.org/task/x"}],  # only 1 task
        )
        out = compute_grounding(tmp_path, ["p001"], expected_total=5)
        assert out["persons_short_fetched"] == 1
        assert out["per_person"]["p001"]["short_fetched"] is True
        assert out["per_person"]["p001"]["expected"] == 5

    def test_non_dict_task_entries_ignored(self, tmp_path):
        """A list whose entries are not dicts is iterated but each entry is
        treated as ungrounded; no crash."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "p001_tasks.json").write_text(
            json.dumps(["not-a-dict", 42, {"ontology_uri": "https://ex.org/x"}]),
            encoding="utf-8",
        )
        out = compute_grounding(tmp_path, ["p001"])
        assert out["grounded"] == 1
        assert out["total"] == 3


class TestComputeGroundingWithValidator:
    """`uri_validator` is the eval-time safety net: every URI in the
    persisted tasks/*.json is re-checked against the live ontology, so
    URIs that no longer resolve (or that snuck through under an older
    bridge) are surfaced separately from the presence-only count."""

    def _write(self, tasks_dir: Path, person_id: str, uris: list[str | None]) -> None:
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / f"{person_id}_tasks.json").write_text(
            json.dumps(
                [{"label": f"t{i}", "ontology_uri": u} for i, u in enumerate(uris)]
            ),
            encoding="utf-8",
        )

    def test_all_verified(self, tmp_path):
        self._write(tmp_path, "p001", ["uri-a", "uri-b"])
        out = compute_grounding(tmp_path, ["p001"], uri_validator=lambda _: True)
        assert out["grounded"] == 2
        assert out["verified"] == 2
        assert out["verified_ratio"] == 1.0
        assert out["persons_with_unverified"] == 0
        per = out["per_person"]["p001"]
        assert per["verified"] == 2
        assert per["verified_ratio"] == 1.0
        assert per["unverified_uris"] == []

    def test_some_unverified(self, tmp_path):
        self._write(tmp_path, "p001", ["good", "bad"])
        out = compute_grounding(
            tmp_path,
            ["p001"],
            uri_validator=lambda u: u != "bad",
        )
        assert out["grounded"] == 2
        assert out["verified"] == 1
        assert out["verified_ratio"] == 0.5
        assert out["persons_with_unverified"] == 1
        assert out["per_person"]["p001"]["unverified_uris"] == ["bad"]

    def test_all_unverified(self, tmp_path):
        self._write(tmp_path, "p001", ["bad-1", "bad-2"])
        out = compute_grounding(tmp_path, ["p001"], uri_validator=lambda _: False)
        assert out["grounded"] == 2
        assert out["verified"] == 0
        assert out["verified_ratio"] == 0.0
        assert out["persons_with_unverified"] == 1
        assert set(out["per_person"]["p001"]["unverified_uris"]) == {
            "bad-1",
            "bad-2",
        }

    def test_validator_skipped_when_uri_missing(self, tmp_path):
        """A None `ontology_uri` is dropped before the validator runs
        ; it counts as ungrounded, never as unverified."""
        self._write(tmp_path, "p001", [None, "good"])
        out = compute_grounding(tmp_path, ["p001"], uri_validator=lambda _: True)
        assert out["grounded"] == 1
        assert out["verified"] == 1
        assert out["per_person"]["p001"]["unverified_uris"] == []

    def test_no_validator_keeps_verified_none(self, tmp_path):
        self._write(tmp_path, "p001", ["good"])
        out = compute_grounding(tmp_path, ["p001"])  # no validator
        assert out["verified"] is None
        assert out["verified_ratio"] is None
        assert out["persons_with_unverified"] is None
        assert out["per_person"]["p001"]["verified"] is None
        assert out["per_person"]["p001"]["verified_ratio"] is None

    def test_aggregates_across_personas(self, tmp_path):
        self._write(tmp_path, "p001", ["good", "bad"])
        self._write(tmp_path, "p002", ["good"])
        out = compute_grounding(
            tmp_path,
            ["p001", "p002"],
            uri_validator=lambda u: u == "good",
        )
        assert out["grounded"] == 3
        assert out["verified"] == 2
        assert out["verified_ratio"] == pytest.approx(2 / 3)
        assert out["persons_with_unverified"] == 1
