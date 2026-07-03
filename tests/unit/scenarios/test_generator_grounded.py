"""Integration-style unit tests for `TaskGenerator._generate_grounded`.

The grounded path threads Stage 1 (fetch + verify), Stage 2 (paraphrase),
Stage 3 (three gates), and the telemetry sidecar together.  These tests
inject in-memory fakes for the GraphRAG pipeline, the Neo4j session, the
paraphrase LLM, and the embedder so the full happy + degraded paths can
be exercised without external services.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.scripts.persona.config.schema import JitterConfig
from src.scripts.persona.domain.persona import Person
from src.scripts.scenarios.config.schema import TaskFilters, TaskGenerationConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.task_generation.generator import TaskGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_person() -> Person:
    return Person(
        person_id="p001",
        persona_id="test_persona",
        person_seed=42,
        instance_index=0,
        occupation_status="fulltime",
        stages=[],
        jitter_applied=JitterConfig(),
        event_overrides={},
    )


def _make_config(num_tasks: int = 2) -> TaskGenerationConfig:
    return TaskGenerationConfig(
        method="graphrag_grounded",
        num_tasks=num_tasks,
        prompt_template="health_improvement",
        # Empty domain/difficulty filters to validator skips the
        # branch + level checks, so the in-memory mock session only
        # has to answer `uri_resolves` (and the enrichment fetch).
        # The branch + level paths are exercised separately in the
        # bridge / fetch_loop tests.
        filters=TaskFilters(domains=[], difficulty=[]),
        max_fetch_retries=2,
        paraphrase=True,
        paraphrase_similarity_threshold=0.0,  # always pass gate C in tests
        paraphrase_max_length_delta_pct=1.0,  # generous so length passes
    )


class _FakePipelineResponse:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _FakePipeline:
    """Returns successive answers from a queue."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.queries: list[str] = []

    def search(self, *, query_text: str, **_kw: Any) -> _FakePipelineResponse:
        self.queries.append(query_text)
        return _FakePipelineResponse(self._answers.pop(0))


class _FakeSession:
    """Routes `MATCH (n:Resource ...) RETURN 1` queries against a known set;
    returns minimal property/branch rows for everything else."""

    def __init__(self, known_uris: set[str]) -> None:
        self._known = known_uris

    def run(self, query: str, **kwargs: Any):
        uri = kwargs.get("uri", "")
        if "RETURN 1 AS hit" in query:
            return _Result(_Record({"hit": 1}) if uri in self._known else None)
        if "isDividable" in query:
            return _Result(
                _Record(
                    {
                        "is_dividable": None,
                        "is_concurrent": None,
                        "duration_minutes": None,
                        "display_name": None,
                        "description": None,
                    }
                )
            )
        if "branch" in query:
            return _Result(_Record({"branch": None, "depth": 0}))
        return _Result(None)


class _Record:
    def __init__(self, data: dict) -> None:
        self._data = data

    def data(self) -> dict:
        return dict(self._data)

    def get(self, key: str, default=None):
        return self._data.get(key, default)


class _Result:
    def __init__(self, record: _Record | None) -> None:
        self._record = record

    def single(self) -> _Record | None:
        return self._record


class _FakeAIMessage:
    """LangChain-style response with `content` + `usage_metadata`."""

    def __init__(self, content: str, in_t: int = 100, out_t: int = 30) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": in_t, "output_tokens": out_t}


class _FakeLLM:
    """Returns a fixed AIMessage from `invoke`."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def invoke(self, _prompt: str) -> _FakeAIMessage:
        self.calls += 1
        return _FakeAIMessage(self._content)


class _ConstantEmbedder:
    """Identical vectors to cosine 1.0 to gate C always passes."""

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_URI_A = "https://ex.org/task/a"
_URI_B = "https://ex.org/task/b"


def _payload(uris: list[str]) -> str:
    return json.dumps(
        [
            {
                "label": uri.rsplit("/", 1)[-1],
                "display_name": uri.rsplit("/", 1)[-1].title(),
                "description": "Walk a lot for the day.",
                "ontology_uri": uri,
            }
            for uri in uris
        ]
    )


def _paraphrase_payload(labels: list[str]) -> str:
    return json.dumps(
        [
            {"label": label, "personalized_description": "Walk a lot for today."}
            for label in labels
        ]
    )


class TestGroundedHappyPath:
    def test_two_tasks_in_one_attempt(self, tmp_path):
        pipeline = _FakePipeline([_payload([_URI_A, _URI_B])])
        session = _FakeSession({_URI_A, _URI_B})
        llm = _FakeLLM(_paraphrase_payload(["a", "b"]))

        gen = TaskGenerator(
            config=_make_config(num_tasks=2),
            graphrag_pipeline=pipeline,
            neo4j_session=session,
            paraphrase_llm=llm,
            embedder=_ConstantEmbedder(),
            scenario_id="nutrition_l1",
            tasks_dir=tmp_path,
            provider="openai",
            fetch_model="gpt-4o-mini",
            paraphrase_model="gpt-4o-mini",
            embedder_model="text-embedding-3-small",
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 2
        assert {t.ontology_uri for t in tasks} == {_URI_A, _URI_B}
        # Every task carries the paraphrase (gate C threshold 0 to always passes).
        assert all(t.description == "Walk a lot for today." for t in tasks)

        # Telemetry sidecar landed.
        sidecar = tmp_path / "_telemetry" / "p001.json"
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert loaded["person_id"] == "p001"
        assert loaded["llm_calls"]["paraphrase"]["attempts"] == 1
        assert loaded["llm_calls"]["paraphrase"]["input_tokens"] == 100
        assert loaded["fetch_summary"]["num_tasks_accepted"] == 2
        assert loaded["tasks"][0]["paraphrase"]["used"] == "paraphrase"

    def test_retry_when_first_attempt_only_fabricates(self, tmp_path):
        bogus = "https://ex.org/task/bogus"
        pipeline = _FakePipeline([_payload([bogus]), _payload([_URI_A, _URI_B])])
        session = _FakeSession({_URI_A, _URI_B})
        llm = _FakeLLM(_paraphrase_payload(["a", "b"]))

        gen = TaskGenerator(
            config=_make_config(num_tasks=2),
            graphrag_pipeline=pipeline,
            neo4j_session=session,
            paraphrase_llm=llm,
            embedder=_ConstantEmbedder(),
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 2
        # Two pipeline attempts.
        sidecar = tmp_path / "_telemetry" / "p001.json"
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert loaded["llm_calls"]["fetch"]["attempts"] == 2
        # Fabricated count from attempt 1 surfaced.
        assert loaded["fetch_summary"]["fabricated_dropped_per_attempt"][0] == 1


class TestGroundedShortFetched:
    def test_returns_partial_list_when_max_retries_exhausted(self, tmp_path):
        bogus = "https://ex.org/task/bogus"
        pipeline = _FakePipeline([_payload([bogus]), _payload([bogus])])
        session = _FakeSession(set())  # no URI resolves
        gen = TaskGenerator(
            config=_make_config(num_tasks=3),
            graphrag_pipeline=pipeline,
            neo4j_session=session,
            paraphrase_llm=_FakeLLM("[]"),
            embedder=_ConstantEmbedder(),
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert tasks == []

    def test_partial_list_writes_telemetry(self, tmp_path):
        # First attempt returns one good URI; second attempt returns none.
        bogus = "https://ex.org/task/bogus"
        pipeline = _FakePipeline([_payload([_URI_A, bogus]), _payload([bogus])])
        session = _FakeSession({_URI_A})
        llm = _FakeLLM(_paraphrase_payload(["a"]))
        gen = TaskGenerator(
            config=_make_config(num_tasks=3),
            graphrag_pipeline=pipeline,
            neo4j_session=session,
            paraphrase_llm=llm,
            embedder=_ConstantEmbedder(),
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 1
        loaded = json.loads(
            (tmp_path / "_telemetry" / "p001.json").read_text(encoding="utf-8")
        )
        assert loaded["fetch_summary"]["num_tasks_accepted"] == 1
        assert loaded["fetch_summary"]["num_tasks_requested"] == 3


class TestGroundedDegradedCollaborators:
    def test_no_pipeline_returns_empty(self):
        gen = TaskGenerator(
            config=_make_config(),
            graphrag_pipeline=None,
            neo4j_session=_FakeSession(set()),
        )
        assert gen.generate(_make_person(), CalendarTrace(person_id="p001")) == []

    def test_no_session_returns_empty(self):
        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=_make_config(),
            graphrag_pipeline=pipeline,
            neo4j_session=None,
            neo4j_driver=None,
        )
        assert gen.generate(_make_person(), CalendarTrace(person_id="p001")) == []

    def test_no_paraphrase_llm_skips_stage_2_and_3(self, tmp_path):
        """Without a paraphrase LLM, every task ships with the canonical
        description and no Stage-2/3 telemetry rows are filled."""
        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=None,
            embedder=_ConstantEmbedder(),
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 1
        # Canonical description survives.
        assert tasks[0].description == "Walk a lot for the day."
        loaded = json.loads(
            (tmp_path / "_telemetry" / "p001.json").read_text(encoding="utf-8")
        )
        assert loaded["llm_calls"]["paraphrase"]["attempts"] == 0
        assert loaded["tasks"][0]["paraphrase"]["used"] == "skipped"

    def test_paraphrase_disabled_in_config_skips_stage_2_and_3(self, tmp_path):
        cfg = _make_config(num_tasks=1)
        cfg = cfg.model_copy(update={"paraphrase": False})
        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=cfg,
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=_FakeLLM(
                _paraphrase_payload(["a"])
            ),  # would run if not disabled
            embedder=_ConstantEmbedder(),
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert tasks[0].description == "Walk a lot for the day."

    def test_no_embedder_uses_paraphrase_without_gates(self, tmp_path):
        """Stage 2 runs but Stage 3 is skipped; paraphrases ship as-is
        with `used == "paraphrase"` in telemetry."""
        pipeline = _FakePipeline([_payload([_URI_A])])
        llm = _FakeLLM(_paraphrase_payload(["a"]))
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=llm,
            embedder=None,
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert tasks[0].description == "Walk a lot for today."
        loaded = json.loads(
            (tmp_path / "_telemetry" / "p001.json").read_text(encoding="utf-8")
        )
        assert loaded["tasks"][0]["paraphrase"]["used"] == "paraphrase"
        assert loaded["embeddings"]["calls"] == 0


class TestGroundedSidecarBestEffort:
    def test_unwritable_sidecar_does_not_crash(self, tmp_path):
        """`write_persona_sidecar` raises on a non-directory path -
        the generator catches the OSError and continues."""
        # Block the sidecar parent dir by creating a file in its place.
        bad = tmp_path / "blocker"
        bad.write_text("blocker", encoding="utf-8")
        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=_FakeLLM(_paraphrase_payload(["a"])),
            embedder=_ConstantEmbedder(),
            scenario_id="x",
            tasks_dir=bad,
        )
        # No exception bubbles up.
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 1


class _EnrichingSession:
    """Like `_FakeSession` but returns non-null property + branch
    rows so the enrichment branch (duration, ontology-wins,
    fill-only) is exercised."""

    def __init__(self, known_uris: set[str]) -> None:
        self._known = known_uris

    def run(self, query: str, **kwargs: Any):
        uri = kwargs.get("uri", "")
        if "RETURN 1 AS hit" in query:
            return _Result(_Record({"hit": 1}) if uri in self._known else None)
        if "isDividable" in query:
            return _Result(
                _Record(
                    {
                        "is_dividable": True,
                        "is_concurrent": True,
                        "duration_minutes": 12,
                        "display_name": "Walk Daily 🚶",
                        "description": "Take a 12-minute walk.",
                    }
                )
            )
        if "branch" in query:
            return _Result(_Record({"branch": "https://x", "depth": 1}))
        return _Result(None)


class TestGroundedEnrichmentBranches:
    """Cover the ontology-wins / fill-only / duration-mapping branches in
    `_enrich_dicts_inner` (the inner enrichment path that runs after
    URIs have already passed Stage-1 verification)."""

    def test_llm_supplied_durations_win_but_ontology_owns_booleans(self, tmp_path):
        """Durations stay fill-only (LLM may pick); `is_dividable` is now
        an ontology-wins field so the ontology value overrides the LLM.
        """
        payload = json.dumps(
            [
                {
                    "label": "a",
                    "ontology_uri": _URI_A,
                    "duration_min": 30,
                    "duration_max": 45,
                    "is_dividable": False,  # LLM said False; ontology says True
                }
            ]
        )
        pipeline = _FakePipeline([payload])
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_EnrichingSession({_URI_A}),
            paraphrase_llm=None,
            embedder=None,
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert tasks[0].duration_min == 30
        assert tasks[0].duration_max == 45
        assert tasks[0].is_dividable is True  # ontology overrides LLM

    def test_no_uri_task_skips_enrichment(self, tmp_path):
        """A raw dict with no `ontology_uri` cannot be enriched -
        the inner loop's `if uri:` branch is False.  The task is
        still emitted (Stage 1 already accepted it earlier; but
        without a URI it would have been rejected, so this exercises
        the defensive branch only)."""
        # Two payloads: first attempt returns a no-URI dict (rejected as
        # fabricated by Stage 1), second returns a real one that the
        # bridge enrichment then sees.
        payloads = [
            json.dumps([{"label": "ghost"}]),  # no ontology_uri to fabricated
            json.dumps([{"label": "a", "ontology_uri": _URI_A}]),
        ]
        pipeline = _FakePipeline(payloads)
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_EnrichingSession({_URI_A}),
            paraphrase_llm=None,
            embedder=None,
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 1
        assert tasks[0].ontology_uri == _URI_A

    def test_inner_enrichment_handles_dict_with_uri_but_no_description(self):
        """Defensive branch: a raw dict with `ontology_uri` but no
        `description` skips the citation-strip helper without
        crashing."""
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=_FakePipeline([]),
            neo4j_session=None,
        )
        # Stage-1 verified URI but description was never set by the LLM
        # AND the bridge returned no description (None).  `isinstance`
        # guard skips the strip without error.
        out = gen._enrich_dicts_inner(
            [{"label": "x", "ontology_uri": "https://ex.org/task/x"}],
            session=_FakeSession({"https://ex.org/task/x"}),
        )
        assert out[0].get("description") is None

    def test_inner_enrichment_handles_no_uri_dict_directly(self):
        """Direct call to the inner helper with a no-URI dict exercises
        the `if uri:` False branch."""
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=_FakePipeline([]),
            neo4j_session=None,
        )
        out = gen._enrich_dicts_inner(
            [{"label": "no_uri", "duration_min": 10}], session=None
        )
        assert out == [{"label": "no_uri", "duration_min": 10}]

    def test_non_ontology_wins_key_already_in_raw_is_kept_in_grounded_path(self):
        """An ontology key outside ONTOLOGY_WINS that raw already supplies is not overwritten in the grounded enrichment."""
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=_FakePipeline([]),
            neo4j_session=None,
        )
        # `difficulty_level` is not in ONTOLOGY_WINS and is already set on raw.
        # The ontology props will include `difficulty_level: 0` (no Level token),
        # so the loop falls into the elif and skips the assignment.
        raw = {
            "label": "x",
            "ontology_uri": "https://ex.org/task/x",
            "difficulty_level": "kept-by-llm",
        }
        out = gen._enrich_dicts_inner(
            [raw], session=_EnrichingSession({"https://ex.org/task/x"})
        )
        assert out[0]["difficulty_level"] == "kept-by-llm"

    def test_duration_minutes_fills_min_and_max(self, tmp_path):
        # Raw payload omits duration_min/max so the ontology fills them.
        payload = json.dumps(
            [
                {
                    "label": "a",
                    "ontology_uri": _URI_A,
                }
            ]
        )
        pipeline = _FakePipeline([payload])
        session = _EnrichingSession({_URI_A})
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=session,
            paraphrase_llm=None,
            embedder=None,
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert tasks[0].duration_min == 12
        assert tasks[0].duration_max == 12
        # Display_name comes from the ontology (ontology_wins).
        assert tasks[0].display_name == "Walk Daily 🚶"
        # Description from the ontology too.
        assert tasks[0].description == "Take a 12-minute walk."


class TestGroundedTelemetryOptional:
    def test_no_tasks_dir_skips_sidecar_silently(self):
        """`tasks_dir=None` (typical when the CLI runs without a
        run dir) does NOT write a sidecar; and does not crash."""
        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=None,
            embedder=None,
            scenario_id="x",
            tasks_dir=None,  # ← critical
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 1


class TestGroundedProposedTasksLog:
    """The generator should write a per-persona `tasks_proposed.json`
    under `LLM_DEBUG_DIR/<pid>/` aggregating every URI the LLM
    emitted across all attempts (with verdict)."""

    def test_log_written_when_env_var_set(self, monkeypatch, tmp_path):
        debug_dir = tmp_path / "llm_debug"
        monkeypatch.setenv("LLM_DEBUG_DIR", str(debug_dir))

        pipeline = _FakePipeline([_payload([_URI_A, _URI_B])])
        gen = TaskGenerator(
            config=_make_config(num_tasks=2),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A, _URI_B}),
            paraphrase_llm=None,
            embedder=None,
            scenario_id="nutrition_l1",
            tasks_dir=tmp_path,
        )
        gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        log_path = debug_dir / "p001" / "tasks_proposed.json"
        assert log_path.exists()
        loaded = json.loads(log_path.read_text(encoding="utf-8"))
        assert loaded["accepted_count"] == 2
        assert loaded["scenario_id"] == "nutrition_l1"
        assert loaded["verdict_counts"].get("accepted") == 2

    def test_log_omitted_when_no_env_var(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_DEBUG_DIR", raising=False)
        monkeypatch.delenv("LLM_AGENT_DEBUG_DIR", raising=False)

        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=None,
            embedder=None,
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        # Must not crash even though no debug dir is configured.
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 1


class TestGroundedDispatcher:
    def test_method_routes_to_grounded_path(self, tmp_path):
        """`generate()` dispatches to `_generate_grounded` based on
        the config method literal; the legacy `graphrag` path is NOT
        reached for `graphrag_grounded`."""
        pipeline = _FakePipeline([_payload([_URI_A])])
        # Inject an empty paraphrase LLM so Stage 2 returns canonical;
        # the test only checks dispatch, not Stage 2.
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=None,
            embedder=None,
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        tasks = gen.generate(_make_person(), CalendarTrace(person_id="p001"))
        assert len(tasks) == 1
        # Sidecar exists to grounded path actually ran.
        assert (tmp_path / "_telemetry" / "p001.json").exists()


class TestGroundedProfileFlow:
    """The grounded path surfaces the full characteristics profile."""

    _CHARS = {
        "occupation_status": "fulltime",
        "gender": "female",
        "age": 50,
        "has_kids": True,
    }

    def _rich_person(self) -> Person:
        return Person(
            person_id="p001",
            persona_id="test_persona",
            person_seed=42,
            instance_index=0,
            characteristics=dict(self._CHARS),
            stages=[],
            jitter_applied=JitterConfig(),
            event_overrides={},
        )

    def test_fetch_prompt_carries_full_profile_block(self, tmp_path):
        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=None,
            embedder=None,
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        gen.generate(self._rich_person(), CalendarTrace(person_id="p001"))
        query = pipeline.queries[0]
        assert "occupation_status: fulltime" in query
        assert "gender: female" in query
        assert "age: 50" in query
        assert "has_kids: true" in query

    def test_paraphrase_prompt_carries_full_profile_summary(self, tmp_path):
        captured: dict[str, str] = {}

        class _CapturingLLM:
            def invoke(self, prompt: str) -> _FakeAIMessage:
                captured["prompt"] = prompt
                return _FakeAIMessage(_paraphrase_payload(["a"]))

        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=_make_config(num_tasks=1),
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=_CapturingLLM(),
            embedder=_ConstantEmbedder(),
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        gen.generate(self._rich_person(), CalendarTrace(person_id="p001"))
        assert (
            "occupation_status=fulltime; gender=female; age=50; has_kids=true"
            in captured["prompt"]
        )

    def test_profile_characteristics_filter_applies_to_fetch_prompt(self, tmp_path):
        config = _make_config(num_tasks=1).model_copy(
            update={"profile_characteristics": ["age", "gender"]}
        )
        pipeline = _FakePipeline([_payload([_URI_A])])
        gen = TaskGenerator(
            config=config,
            graphrag_pipeline=pipeline,
            neo4j_session=_FakeSession({_URI_A}),
            paraphrase_llm=None,
            embedder=None,
            scenario_id="x",
            tasks_dir=tmp_path,
        )
        gen.generate(self._rich_person(), CalendarTrace(person_id="p001"))
        query = pipeline.queries[0]
        assert "age: 50" in query
        assert "gender: female" in query
        assert "has_kids" not in query
        assert "occupation_status" not in query
