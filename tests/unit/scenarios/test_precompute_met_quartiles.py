"""Unit tests for src.scripts.scenarios.precompute_met_quartiles."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scripts.scenarios.precompute_met_quartiles import (
    HUMAN_ACTIVITIES_PREFIX,
    _percentile,
    compute_quartiles,
    fetch_met_values,
    main,
    write_quartiles_file,
)

# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_single_value_returns_itself(self):
        assert _percentile([3.0], 50.0) == pytest.approx(3.0)
        assert _percentile([3.0], 0.0) == pytest.approx(3.0)
        assert _percentile([3.0], 100.0) == pytest.approx(3.0)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            _percentile([], 50.0)

    def test_median_of_odd_length(self):
        # [1,2,3,4,5]: median is 3
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0) == pytest.approx(3.0)

    def test_median_of_even_length_interpolates(self):
        # [1,2,3,4]: median is 2.5
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)

    def test_percentile_0_returns_min(self):
        assert _percentile([5.0, 10.0, 15.0], 0.0) == pytest.approx(5.0)

    def test_percentile_100_returns_max(self):
        assert _percentile([5.0, 10.0, 15.0], 100.0) == pytest.approx(15.0)

    def test_quartile_q1(self):
        # On [1..9], 25th pct via linear interp is 1 + 0.25*8 = 3.0 (rank 2)
        result = _percentile(list(range(1, 10)), 25.0)
        assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# compute_quartiles
# ---------------------------------------------------------------------------


class TestComputeQuartiles:
    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            compute_quartiles([])

    def test_single_value(self):
        result = compute_quartiles([4.0])
        assert result["q1"] == pytest.approx(4.0)
        assert result["q2"] == pytest.approx(4.0)
        assert result["q3"] == pytest.approx(4.0)
        assert result["max_met"] == pytest.approx(4.0)
        assert result["n"] == 1

    def test_simple_distribution(self):
        # [1, 2, 3, 4, 5, 6, 7, 8]
        # q1 = pct 25 over 8 vals (rank 1.75 to linear 2 + 0.75*(3-2) = 2.75)
        # q2 = pct 50 (rank 3.5 to 4.5)
        # q3 = pct 75 (rank 5.25 to 6.25)
        result = compute_quartiles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert result["q1"] == pytest.approx(2.75)
        assert result["q2"] == pytest.approx(4.5)
        assert result["q3"] == pytest.approx(6.25)
        assert result["max_met"] == pytest.approx(8.0)
        assert result["n"] == 8

    def test_unsorted_input_handled(self):
        result = compute_quartiles([5.0, 1.0, 3.0, 2.0, 4.0])
        # sorted: [1,2,3,4,5]; q1 rank 1, q2 rank 2, q3 rank 3
        assert result["q1"] == pytest.approx(2.0)
        assert result["q2"] == pytest.approx(3.0)
        assert result["q3"] == pytest.approx(4.0)
        assert result["max_met"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# fetch_met_values
# ---------------------------------------------------------------------------


def _mock_driver_returning(records):
    """Build a mock Neo4j driver whose session.run returns *records*."""
    driver = MagicMock()
    session = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=session)
    cm.__exit__ = MagicMock(return_value=False)
    driver.session = MagicMock(return_value=cm)
    session.run = MagicMock(return_value=iter(records))
    return driver, session


class TestFetchMetValues:
    def test_returns_floats(self):
        records = [{"met": 1.5}, {"met": 3.0}, {"met": 8.0}]
        driver, _ = _mock_driver_returning(records)
        result = fetch_met_values(driver)
        assert result == [1.5, 3.0, 8.0]

    def test_skips_none_records(self):
        records = [{"met": 1.5}, {"met": None}, {"met": 3.0}]
        driver, _ = _mock_driver_returning(records)
        result = fetch_met_values(driver)
        assert result == [1.5, 3.0]

    def test_uses_supplied_prefix(self):
        records = [{"met": 2.0}]
        driver, session = _mock_driver_returning(records)
        fetch_met_values(driver, prefix="https://example.org/foo/")
        kwargs = session.run.call_args.kwargs
        assert kwargs["prefix"] == "https://example.org/foo/"

    def test_default_prefix(self):
        records = []
        driver, session = _mock_driver_returning(records)
        fetch_met_values(driver)
        kwargs = session.run.call_args.kwargs
        assert kwargs["prefix"] == HUMAN_ACTIVITIES_PREFIX


# ---------------------------------------------------------------------------
# write_quartiles_file
# ---------------------------------------------------------------------------


class TestWriteQuartilesFile:
    def test_writes_payload_with_prefix_and_ts(self, tmp_path: Path):
        out = tmp_path / "x.json"
        write_quartiles_file(
            {"q1": 1.0, "q2": 2.0, "q3": 3.0, "max_met": 8.0, "n": 4},
            output_path=out,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["q1"] == pytest.approx(1.0)
        assert data["q2"] == pytest.approx(2.0)
        assert data["q3"] == pytest.approx(3.0)
        assert data["max_met"] == pytest.approx(8.0)
        assert data["n"] == 4
        assert data["ontology_prefix"] == HUMAN_ACTIVITIES_PREFIX
        assert "ts" in data and data["ts"]

    def test_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "out.json"
        write_quartiles_file(
            {"q1": 0.0, "q2": 0.0, "q3": 0.0, "max_met": 0.0, "n": 0},
            output_path=nested,
        )
        assert nested.exists()

    def test_custom_prefix_persisted(self, tmp_path: Path):
        out = tmp_path / "x.json"
        write_quartiles_file(
            {"q1": 1.0, "q2": 2.0, "q3": 3.0, "max_met": 4.0, "n": 1},
            output_path=out,
            prefix="https://example.org/",
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["ontology_prefix"] == "https://example.org/"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_values_returns_usage(self, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.json"
        driver, _ = _mock_driver_returning([])
        monkeypatch.setattr(
            "src.graphrag.neo4j_client.make_driver",
            lambda *a, **kw: driver,
        )
        rc = main(["--output", str(out)])
        assert rc != 0
        assert not out.exists()

    def test_writes_file_and_returns_ok(self, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.json"
        records = [{"met": float(v)} for v in (1.0, 2.0, 3.0, 4.0)]
        driver, _ = _mock_driver_returning(records)
        monkeypatch.setattr(
            "src.graphrag.neo4j_client.make_driver",
            lambda *a, **kw: driver,
        )
        rc = main(["--output", str(out)])
        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["n"] == 4
        assert data["max_met"] == pytest.approx(4.0)

    def test_custom_prefix_passed_through(self, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.json"
        records = [{"met": 5.0}]
        driver, session = _mock_driver_returning(records)
        monkeypatch.setattr(
            "src.graphrag.neo4j_client.make_driver",
            lambda *a, **kw: driver,
        )
        rc = main(["--output", str(out), "--prefix", "https://example.org/"])
        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["ontology_prefix"] == "https://example.org/"
