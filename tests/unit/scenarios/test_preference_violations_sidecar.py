"""Tests for the preference_violations.jsonl sidecar writer + reader."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import fakeredis

from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    Persona,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.scenarios.domain.calendar import AugmentedCalendar
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.export.json_writer import write_preference_violations_sidecar
from src.scripts.scenarios.metrics.loss import LegStats, compute_l_pref_v2
from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
from src.scripts.scenarios.metrics.preference_constraints import PersonaConstraints
from src.scripts.scenarios.metrics.preference_mapping import PreferenceMapper
from src.scripts.scenarios.metrics.preference_score import (
    PatternViolation,
    read_preference_violations,
)

HEALTH_PREFIX = "https://w3id.org/calendar-bench/health/task/"


def _wm() -> WindowMap:
    return WindowMap.from_config(
        {
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        }
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class TestWriteSidecar:
    def test_writes_leg_and_pattern_rows(self, tmp_path: Path):
        out_path = tmp_path / "p1_pref_violations.jsonl"
        leg_stats = [
            LegStats(
                name="per_occurrence_duration",
                mean_loss=0.2,
                applicable_count=5,
                mape=None,
            ),
            LegStats(
                name="temporal_pattern_trend",
                mean_loss=0.4,
                applicable_count=2,
                mape=0.55,
            ),
            LegStats(
                name="persona_stage_semantic",
                mean_loss=None,
                applicable_count=0,
                mape=None,
            ),
        ]
        pattern_rows = [
            PatternViolation(mode="trend", scale="week", loss=0.4, mape=0.55),
        ]
        result = write_preference_violations_sidecar(
            out_path,
            person_id="p1",
            leg_stats=leg_stats,
            pattern_rows=pattern_rows,
        )
        assert result == out_path
        assert out_path.exists()
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4  # 3 leg rows + 1 pattern row
        parsed = [json.loads(line) for line in lines]
        # First three are legs.
        assert parsed[0]["kind"] == "leg"
        assert parsed[0]["name"] == "per_occurrence_duration"
        assert parsed[0]["mean_loss"] == 0.2
        # Persona-stage row had no signal.
        assert parsed[2]["mean_loss"] is None
        assert parsed[2]["applicable_count"] == 0
        # Last is a pattern row.
        assert parsed[3]["kind"] == "pattern"
        assert parsed[3]["mode"] == "trend"
        assert parsed[3]["mape"] == 0.55

    def test_empty_inputs_writes_empty_file(self, tmp_path: Path):
        out_path = tmp_path / "p1_pref_violations.jsonl"
        write_preference_violations_sidecar(
            out_path, person_id="p1", leg_stats=[], pattern_rows=[]
        )
        # File exists and is empty.
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Reader / aggregator
# ---------------------------------------------------------------------------


def _write_sidecar(path: Path, person_id: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


class TestReadPreferenceViolations:
    def test_empty_input_returns_empty(self, tmp_path: Path):
        assert read_preference_violations([]) == {}

    def test_missing_paths_skipped(self, tmp_path: Path):
        assert read_preference_violations([tmp_path / "absent.jsonl"]) == {}

    def test_aggregates_means_across_personas(self, tmp_path: Path):
        _write_sidecar(
            tmp_path / "p1.jsonl",
            "p1",
            [
                {
                    "kind": "leg",
                    "person_id": "p1",
                    "name": "per_occurrence_duration",
                    "applicable_count": 4,
                    "mean_loss": 0.5,
                    "mape": None,
                }
            ],
        )
        _write_sidecar(
            tmp_path / "p2.jsonl",
            "p2",
            [
                {
                    "kind": "leg",
                    "person_id": "p2",
                    "name": "per_occurrence_duration",
                    "applicable_count": 6,
                    "mean_loss": 0.1,
                    "mape": None,
                }
            ],
        )
        out = read_preference_violations([tmp_path / "p1.jsonl", tmp_path / "p2.jsonl"])
        assert "per_occurrence_duration" in out
        leg = out["per_occurrence_duration"]
        # Total applicable = 10, weighted-mean loss = (4*0.5 + 6*0.1)/10 = 0.26
        assert leg["applicable_tasks"] == 10
        assert leg["mean_loss"] == 0.26

    def test_aggregates_mape_for_pattern_legs(self, tmp_path: Path):
        _write_sidecar(
            tmp_path / "p1.jsonl",
            "p1",
            [
                {
                    "kind": "leg",
                    "person_id": "p1",
                    "name": "temporal_pattern_trend",
                    "applicable_count": 2,
                    "mean_loss": 0.4,
                    "mape": 0.5,
                }
            ],
        )
        out = read_preference_violations([tmp_path / "p1.jsonl"])
        leg = out["temporal_pattern_trend"]
        assert leg["mape"] == 0.5

    def test_no_signal_row_registered_but_not_averaged(self, tmp_path: Path):
        """A row with `mean_loss: null` registers the leg but doesn't
        push a numeric average."""
        _write_sidecar(
            tmp_path / "p1.jsonl",
            "p1",
            [
                {
                    "kind": "leg",
                    "person_id": "p1",
                    "name": "persona_stage_semantic",
                    "applicable_count": 0,
                    "mean_loss": None,
                    "mape": None,
                }
            ],
        )
        out = read_preference_violations([tmp_path / "p1.jsonl"])
        leg = out["persona_stage_semantic"]
        assert leg["applicable_tasks"] == 0
        assert leg["mean_loss"] is None

    def test_malformed_lines_dropped(self, tmp_path: Path):
        path = tmp_path / "p1.jsonl"
        path.write_text(
            "not-json\n[1, 2, 3]\n"
            '{"kind": "leg", "person_id": "p1", "name": "per_occurrence_duration", "applicable_count": 1, "mean_loss": 0.5}\n',
            encoding="utf-8",
        )
        out = read_preference_violations([path])
        assert out["per_occurrence_duration"]["mean_loss"] == 0.5

    def test_pattern_rows_are_ignored_by_reader(self, tmp_path: Path):
        """Only `kind=leg` rows contribute to the breakdown; pattern
        rows are kept in the sidecar for audit but the per-leg roll-up
        already encoded the MAPE."""
        _write_sidecar(
            tmp_path / "p1.jsonl",
            "p1",
            [
                {
                    "kind": "pattern",
                    "person_id": "p1",
                    "mode": "fix",
                    "scale": None,
                    "loss": 1.0,
                    "mape": None,
                }
            ],
        )
        out = read_preference_violations([tmp_path / "p1.jsonl"])
        assert out == {}

    def test_blank_lines_silently_skipped(self, tmp_path: Path):
        """Blank lines in the sidecar (e.g. from manual edits) must be
        skipped without crashing."""
        path = tmp_path / "p1.jsonl"
        path.write_text(
            '\n\n\n{"kind": "leg", "person_id": "p1", "name": "per_occurrence_duration", "applicable_count": 1, "mean_loss": 0.25}\n\n',
            encoding="utf-8",
        )
        out = read_preference_violations([path])
        assert out["per_occurrence_duration"]["mean_loss"] == 0.25

    def test_row_with_missing_name_skipped(self, tmp_path: Path):
        _write_sidecar(
            tmp_path / "p1.jsonl",
            "p1",
            [
                {
                    "kind": "leg",
                    "person_id": "p1",
                    "applicable_count": 1,
                    "mean_loss": 0.2,
                }
            ],
        )
        assert read_preference_violations([tmp_path / "p1.jsonl"]) == {}


# ---------------------------------------------------------------------------
# Round-trip through compute_l_pref_v2
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_write_then_read_returns_consistent_breakdown(self, tmp_path: Path):
        # Build a tiny scenario, compute v2, write the sidecar, then
        # read it back.  The aggregated breakdown should match the leg
        # stats produced by the aggregator.
        cache = PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))
        mapper = PreferenceMapper(cache=cache)
        walking = EventDefinition(
            name="walking",
            category="sports",
            per_event_duration=DurationRange(min=15, max=60, unit="minutes"),
            total_event_duration=TotalDuration(
                min=15, max=60, scale="day", unit="minutes"
            ),
            total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
            health_task_iri=HEALTH_PREFIX + "schedule-a-walk",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "morning"})
            ],
        )
        ev_cfg = EventConfig(
            categories={"sports": Category(name="sports", events={"walking": walking})}
        )
        persona = Persona(id="p1", occupation_status="parttime", stages=[])
        pc = PersonaConstraints(
            persona=persona,
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=_dt.date(2026, 6, 1),
            mapper=mapper,
        )
        task = RecommendedTask(
            label="schedule-a-walk",
            duration_min=30,
            duration_max=30,
            ontology_uri=HEALTH_PREFIX + "schedule-a-walk",
        )
        sched = ScheduledTask(
            task=task,
            start_minutes=480,
            end_minutes=510,
            is_standalone=True,
            concurrent_with=None,
            date=_dt.date(2026, 6, 1),
        )
        solution = SchedulingSolution(
            person_id="p1",
            augmented_calendar=AugmentedCalendar(person_id="p1"),
            tasks=[task],
            scheduled=[sched],
            unscheduled=[],
        )
        loss, pattern_rows, leg_stats = compute_l_pref_v2(
            solution, persona_constraints=pc, window_map=_wm()
        )
        assert loss is not None
        out_path = tmp_path / "p1_pref_violations.jsonl"
        write_preference_violations_sidecar(
            out_path,
            person_id="p1",
            leg_stats=leg_stats,
            pattern_rows=pattern_rows,
        )
        breakdown = read_preference_violations([out_path])
        # The per_occurrence_duration leg fired with one applicable task.
        assert "per_occurrence_duration" in breakdown
        assert breakdown["per_occurrence_duration"]["applicable_tasks"] == 1
