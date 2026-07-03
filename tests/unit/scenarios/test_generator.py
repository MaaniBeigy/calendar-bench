"""Unit tests for src.scripts.scenarios.task_generation.generator.

Also tests the new extract_raw_task_dicts() function added to parser.py.

Coverage targets (generator.py):
  TaskGenerator.generate; method=manual, method=graphrag, method=template.
  generate; pipeline=None to [].
  generate; empty LLM response to [].
  generate; unparseable LLM response to [].
  _from_manual; tasks present, non-dict items skipped, empty list.
  _build_prompt; existing calendar labels, empty calendar.
  _enrich_dicts; session=None, uri present with session, uri absent,
                  duration_minutes mapped to duration_min/max,
                  LLM value wins over ontology value.

Coverage targets (parser.py; new function):
  extract_raw_task_dicts; empty text, valid text, non-dict items filtered.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.scripts.persona.config.schema import JitterConfig
from src.scripts.persona.domain.persona import Person
from src.scripts.scenarios.config.schema import TaskFilters, TaskGenerationConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.task_generation.generator import TaskGenerator
from src.scripts.scenarios.task_generation.ontology_bridge import PHYSICAL_ACTIVITY_URI
from src.scripts.scenarios.task_generation.parser import extract_raw_task_dicts
from tests.unit.scenarios.conftest import make_event

# ---------------------------------------------------------------------------
# Shared helpers / mocks
# ---------------------------------------------------------------------------


def _make_person(
    occupation_status: str = "fulltime",
    characteristics: dict | None = None,
) -> Person:
    """Minimal Person fixture (no stages, no overrides)."""
    return Person(
        person_id="test_0000",
        persona_id="test_persona",
        person_seed=42,
        instance_index=0,
        occupation_status=occupation_status,
        characteristics=characteristics or {},
        stages=[],
        jitter_applied=JitterConfig(),
        event_overrides={},
    )


def _make_config(
    method: str = "graphrag",
    prompt_template: str = "health_improvement",
    num_tasks: int = 3,
    manual_tasks: list | None = None,
    task_overrides: dict | None = None,
    profile_characteristics: list[str] | None = None,
) -> TaskGenerationConfig:
    return TaskGenerationConfig(
        method=method,
        num_tasks=num_tasks,
        prompt_template=prompt_template,
        filters=TaskFilters(domains=["PhysicalActivityTask"], difficulty=["Level2"]),
        manual_tasks=manual_tasks or [],
        task_overrides=task_overrides or {},
        profile_characteristics=profile_characteristics,
    )


class _MockResponse:
    """Mimics the GraphRAG pipeline response object."""

    def __init__(self, answer: str) -> None:
        self.answer = answer


class _MockPipeline:
    """Minimal GraphRAG pipeline stub that returns a fixed answer string."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.last_query: str | None = None

    def search(self, query_text: str, **_kwargs: Any) -> _MockResponse:
        self.last_query = query_text
        return _MockResponse(self._answer)


# Reuse MockSession infrastructure from test_ontology_bridge.py
class _MockRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class _MockResult:
    def __init__(self, record: _MockRecord | None) -> None:
        self._record = record

    def single(self) -> _MockRecord | None:
        return self._record


class _MockSession:
    """FIFO mock for `Session.run` with URI-resolution short-circuit.

    The generator's enrichment pass runs three Cypher shapes per task:

      1. `uri_resolves`; `MATCH (n:Resource ...) RETURN 1 AS hit`
      2. `fetch_task_properties`; multi-column property fetch
      3. `resolve_task_branch`; class-hierarchy walk

    Steps 2 and 3 are what existing tests parametrise via *records*, so
    we want step 1 to be silent.  When *uri_resolves* is True (default)
    the EXISTS query returns a synthetic hit and does NOT consume a
    record from the queue.  When False, the EXISTS query returns
    `None` (URI fabricated) and the queue is left untouched; the
    generator short-circuits before any property / branch query runs.
    """

    def __init__(self, *records: _MockRecord | None, uri_resolves: bool = True) -> None:
        self._queue = list(records)
        self._idx = 0
        self._uri_resolves = uri_resolves

    def run(self, query: str, **_kwargs: Any) -> _MockResult:
        if "RETURN 1 AS hit" in query:
            return _MockResult(_MockRecord({"hit": 1}) if self._uri_resolves else None)
        record = self._queue[self._idx] if self._idx < len(self._queue) else None
        self._idx += 1
        return _MockResult(record)


def _record(**kwargs: Any) -> _MockRecord:
    return _MockRecord(kwargs)


def _make_generator(
    answer: str = "[]",
    method: str = "graphrag",
    session: Any = None,
    manual_tasks: list | None = None,
    task_overrides: dict | None = None,
    prompt_template: str = "health_improvement",
) -> TaskGenerator:
    config = _make_config(
        method=method,
        manual_tasks=manual_tasks,
        task_overrides=task_overrides,
        prompt_template=prompt_template,
    )
    pipeline = _MockPipeline(answer) if method != "manual" else None
    return TaskGenerator(
        config=config, graphrag_pipeline=pipeline, neo4j_session=session
    )


# ---------------------------------------------------------------------------
# extract_raw_task_dicts (new parser.py function)
# ---------------------------------------------------------------------------


class TestExtractRawTaskDicts:
    def test_empty_text_returns_empty(self):
        assert extract_raw_task_dicts("") == []

    def test_whitespace_only_returns_empty(self):
        assert extract_raw_task_dicts("   ") == []

    def test_valid_array_returns_dicts(self):
        raw = '[{"label": "yoga"}, {"label": "run"}]'
        result = extract_raw_task_dicts(raw)
        assert len(result) == 2
        assert result[0]["label"] == "yoga"

    def test_non_dict_items_filtered(self):
        raw = '[{"label": "yoga"}, "string", 42, null]'
        result = extract_raw_task_dicts(raw)
        assert len(result) == 1
        assert result[0]["label"] == "yoga"

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError):
            extract_raw_task_dicts("not json {{")


class TestExtractParaphrases:
    """Sister parser for the persona-paraphrase response."""

    def test_empty_returns_empty(self):
        from src.scripts.scenarios.task_generation.parser import extract_paraphrases

        assert extract_paraphrases("") == []

    def test_whitespace_returns_empty(self):
        from src.scripts.scenarios.task_generation.parser import extract_paraphrases

        assert extract_paraphrases("   ") == []

    def test_valid_array_returns_dicts(self):
        from src.scripts.scenarios.task_generation.parser import extract_paraphrases

        out = extract_paraphrases(
            '[{"label": "yoga", "personalized_description": "hi"}]'
        )
        assert out == [{"label": "yoga", "personalized_description": "hi"}]

    def test_non_dict_items_filtered(self):
        from src.scripts.scenarios.task_generation.parser import extract_paraphrases

        out = extract_paraphrases('[{"label": "x"}, "noise", 7]')
        assert out == [{"label": "x"}]

    def test_invalid_json_raises(self):
        from src.scripts.scenarios.task_generation.parser import extract_paraphrases

        with pytest.raises(ValueError):
            extract_paraphrases("xx not json")


# ---------------------------------------------------------------------------
# TaskGenerator; graphrag method
# ---------------------------------------------------------------------------


class TestTaskGeneratorGraphrag:
    _PERSON = _make_person("fulltime")
    _TRACE = CalendarTrace(person_id="p001")

    def test_generate_returns_tasks(self):
        gen = _make_generator('[{"label": "yoga"}]')
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 1
        assert tasks[0].label == "yoga"
        assert isinstance(tasks[0], RecommendedTask)

    def test_generate_empty_llm_response_returns_empty(self):
        gen = _make_generator("")
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks == []

    def test_generate_unparseable_llm_response_returns_empty(self):
        gen = _make_generator("definitely not valid json :{")
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks == []

    def test_generate_no_pipeline_returns_empty(self):
        config = _make_config(method="graphrag")
        gen = TaskGenerator(config=config, graphrag_pipeline=None)
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks == []

    def test_generate_response_with_none_answer_returns_empty(self):
        """response.answer being None must not crash; treated as empty."""

        class _NoneAnswerPipeline:
            def search(self, query_text: str, **_kwargs: Any) -> Any:
                class _Resp:
                    answer = None

                return _Resp()

        config = _make_config(method="graphrag")
        gen = TaskGenerator(config=config, graphrag_pipeline=_NoneAnswerPipeline())
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks == []

    def test_generate_prompt_contains_occupation_status(self):
        person = _make_person("student")
        pipeline = _MockPipeline("[]")
        config = _make_config()
        gen = TaskGenerator(config=config, graphrag_pipeline=pipeline)
        gen.generate(person, self._TRACE)
        assert "student" in pipeline.last_query

    def test_generate_prompt_contains_every_characteristic_by_default(self):
        person = _make_person(
            "fulltime",
            characteristics={
                "occupation_status": "fulltime",
                "gender": "female",
                "has_kids": True,
                "age": 50,
                "neuroticism": 0.38884503449169394,
            },
        )
        pipeline = _MockPipeline("[]")
        gen = TaskGenerator(config=_make_config(), graphrag_pipeline=pipeline)
        gen.generate(person, self._TRACE)
        assert "occupation_status: fulltime" in pipeline.last_query
        assert "gender: female" in pipeline.last_query
        assert "has_kids: true" in pipeline.last_query
        assert "age: 50" in pipeline.last_query
        assert "neuroticism: 0.39" in pipeline.last_query

    def test_profile_characteristics_restricts_prompt_profile(self):
        person = _make_person(
            "fulltime",
            characteristics={
                "occupation_status": "fulltime",
                "gender": "female",
                "age": 50,
            },
        )
        pipeline = _MockPipeline("[]")
        config = _make_config(profile_characteristics=["age"])
        gen = TaskGenerator(config=config, graphrag_pipeline=pipeline)
        gen.generate(person, self._TRACE)
        assert "age: 50" in pipeline.last_query
        assert "gender" not in pipeline.last_query

    def test_profile_summary_renders_all_characteristics(self):
        person = _make_person(
            "fulltime",
            characteristics={
                "occupation_status": "fulltime",
                "weekly_workouts": 3,
            },
        )
        gen = TaskGenerator(config=_make_config(), graphrag_pipeline=None)
        summary = gen._profile_summary(person)
        assert summary == "occupation_status=fulltime; weekly_workouts=3"

    def test_generate_prompt_contains_existing_labels(self):
        trace = CalendarTrace(
            person_id="p001",
            events=[make_event("lunch"), make_event("sleep")],
        )
        pipeline = _MockPipeline("[]")
        config = _make_config()
        gen = TaskGenerator(config=config, graphrag_pipeline=pipeline)
        gen.generate(self._PERSON, trace)
        assert "lunch" in pipeline.last_query
        assert "sleep" in pipeline.last_query

    def test_generate_prompt_contains_scenario_description(self):
        pipeline = _MockPipeline("[]")
        config = _make_config()
        gen = TaskGenerator(config=config, graphrag_pipeline=pipeline)
        gen.generate(self._PERSON, self._TRACE, scenario_description="Get fit!")
        assert "Get fit!" in pipeline.last_query

    def test_generate_applies_task_overrides(self):
        raw = '[{"label": "yoga", "intensity": 1}]'
        gen = _make_generator(raw, task_overrides={"yoga": {"intensity": 5}})
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].intensity == 5

    def test_generate_template_method_same_as_graphrag(self):
        """method='template' uses the same pipeline path as 'graphrag'."""
        gen = _make_generator('[{"label": "stretch"}]', method="template")
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 1
        assert tasks[0].label == "stretch"


# ---------------------------------------------------------------------------
# TaskGenerator; manual method
# ---------------------------------------------------------------------------


class TestTaskGeneratorManual:
    _PERSON = _make_person()
    _TRACE = CalendarTrace(person_id="p001")

    def test_generate_returns_manual_tasks(self):
        gen = _make_generator(method="manual", manual_tasks=[{"label": "stretch"}])
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 1
        assert tasks[0].label == "stretch"

    def test_generate_empty_manual_tasks_returns_empty(self):
        gen = _make_generator(method="manual", manual_tasks=[])
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks == []

    def test_generate_multiple_manual_tasks(self):
        gen = _make_generator(
            method="manual",
            manual_tasks=[{"label": "yoga"}, {"label": "stretch"}],
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 2
        labels = {t.label for t in tasks}
        assert labels == {"yoga", "stretch"}

    def test_generate_manual_applies_task_overrides(self):
        gen = _make_generator(
            method="manual",
            manual_tasks=[{"label": "yoga", "intensity": 1}],
            task_overrides={"yoga": {"intensity": 4}},
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].intensity == 4

    def test_generate_manual_ignores_pipeline(self):
        """Manual method must not call the pipeline even if one is set."""
        config = _make_config(method="manual", manual_tasks=[{"label": "walk"}])
        pipeline = _MockPipeline("should_not_be_used")
        gen = TaskGenerator(config=config, graphrag_pipeline=pipeline)
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 1
        assert pipeline.last_query is None  # never called


# ---------------------------------------------------------------------------
# TaskGenerator; _enrich_dicts
# ---------------------------------------------------------------------------


class TestEnrichDicts:
    _PERSON = _make_person()
    _TRACE = CalendarTrace(person_id="p001")

    def _gen_with_session(
        self, answer: str, *records: _MockRecord | None
    ) -> TaskGenerator:
        config = _make_config()
        pipeline = _MockPipeline(answer)
        session = _MockSession(*records)
        return TaskGenerator(
            config=config, graphrag_pipeline=pipeline, neo4j_session=session
        )

    def test_no_session_returns_tasks_without_enrichment(self):
        """session=None: enrichment step skipped; ontology defaults not applied."""
        gen = _make_generator('[{"label": "yoga"}]', session=None)
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 1
        assert tasks[0].is_dividable is False  # parser default, not from ontology

    def test_uri_present_session_enriches_is_dividable(self):
        raw_answer = '[{"label": "yoga", "ontology_uri": "https://ex.org/yoga"}]'
        gen = self._gen_with_session(
            raw_answer,
            _record(is_dividable=True, is_concurrent=None, duration_minutes=None),
            _record(branch=PHYSICAL_ACTIVITY_URI, depth=1),
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].is_dividable is True

    def test_uri_present_session_enriches_is_concurrent(self):
        raw_answer = '[{"label": "walk", "ontology_uri": "https://ex.org/walk"}]'
        gen = self._gen_with_session(
            raw_answer,
            _record(is_dividable=None, is_concurrent=True, duration_minutes=None),
            _record(branch=None, depth=0),
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].is_concurrent is True

    def test_uri_present_duration_minutes_mapped_to_range(self):
        """duration_minutes from ontology fills duration_min and duration_max."""
        raw_answer = '[{"label": "run", "ontology_uri": "https://ex.org/run"}]'
        gen = self._gen_with_session(
            raw_answer,
            _record(is_dividable=None, is_concurrent=None, duration_minutes=45),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].duration_min == 45
        assert tasks[0].duration_max == 45

    def test_llm_duration_wins_over_ontology_duration(self):
        """If LLM provides duration_min/max, ontology's duration_minutes is ignored."""
        raw_answer = (
            '[{"label": "run", "ontology_uri": "https://ex.org/run", '
            '"duration_min": 20, "duration_max": 40}]'
        )
        gen = self._gen_with_session(
            raw_answer,
            _record(is_dividable=None, is_concurrent=None, duration_minutes=60),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].duration_min == 20  # LLM value preserved
        assert tasks[0].duration_max == 40  # LLM value preserved

    def test_ontology_is_dividable_wins_over_llm(self):
        """Ontology `is_dividable` overrides the LLM-supplied value.

        `hb:isDividable` is an ontology fact, not a subjective LLM call;
        the LLM has no reliable way to know it and tends to default
        `false`, which made every dividable task silently slip past the
        L_divide metric. The enricher now treats it as ontology-wins.
        """
        raw_answer = '[{"label": "yoga", "ontology_uri": "https://ex.org/yoga", "is_dividable": false}]'
        gen = self._gen_with_session(
            raw_answer,
            _record(is_dividable=True, is_concurrent=None, duration_minutes=None),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].is_dividable is True  # ontology value wins over LLM

    def test_uri_absent_no_ontology_query(self):
        """Dicts without ontology_uri must not trigger Neo4j calls."""
        raw_answer = '[{"label": "yoga"}]'  # no ontology_uri
        session = _MockSession()  # any call would fail (empty queue)
        config = _make_config()
        pipeline = _MockPipeline(raw_answer)
        gen = TaskGenerator(
            config=config, graphrag_pipeline=pipeline, neo4j_session=session
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 1
        assert session._idx == 0  # run() never called

    def test_multiple_tasks_enriched_independently(self):
        raw_answer = json.dumps(
            [
                {"label": "yoga", "ontology_uri": "https://ex.org/yoga"},
                {"label": "run", "ontology_uri": "https://ex.org/run"},
            ]
        )
        # Each task triggers up to 3 enrichment phases (props, branch,
        # difficulty_level), each of which iterates URI variants until a
        # record returns the expected key.  We use a URI-routed session so
        # the test does not depend on the variant ladder length.
        records = {
            "https://ex.org/yoga": {
                "props": _record(
                    is_dividable=True, is_concurrent=None, duration_minutes=20
                ),
                "branch": _record(branch=PHYSICAL_ACTIVITY_URI, depth=1),
            },
            "https://ex.org/run": {
                "props": _record(
                    is_dividable=False, is_concurrent=None, duration_minutes=30
                ),
            },
        }

        class _MultiQuerySession:
            def __init__(self, mapping):
                self._mapping = mapping

            def run(self, query, **kwargs):
                uri = kwargs.get("uri")
                bucket = self._mapping.get(uri, {})
                if "is_dividable" in query.lower() or "isDividable" in query:
                    rec = bucket.get("props")
                elif "branch.uri IN" in query:
                    rec = bucket.get("branch")
                else:
                    rec = None
                return _MockResult(rec)

        session = _MultiQuerySession(records)
        config = _make_config()
        pipeline = _MockPipeline(raw_answer)
        gen = TaskGenerator(
            config=config, graphrag_pipeline=pipeline, neo4j_session=session
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 2
        yoga = next(t for t in tasks if t.label == "yoga")
        run = next(t for t in tasks if t.label == "run")
        assert yoga.is_dividable is True
        assert yoga.duration_min == 20
        assert run.is_dividable is False
        assert run.duration_min == 30

    # --- ontology-wins rule for display_name + description ----

    def test_ontology_display_name_fills_in_when_llm_omits(self):
        """When the LLM does not supply `display_name`, the ontology
        value flows through verbatim (emoji included)."""
        raw_answer = '[{"label": "me", "ontology_uri": "https://ex.org/me"}]'
        gen = self._gen_with_session(
            raw_answer,
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name="Mindful Eating 🍽️",
                description=None,
            ),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].display_name == "Mindful Eating 🍽️"

    def test_ontology_display_name_overrides_llm_supplied_display_name(self):
        """Ontology-wins rule: an ontology-supplied `display_name`
        overrides the LLM's value, ensuring authored content is never
        paraphrased away."""
        raw_answer = (
            '[{"label": "me", "ontology_uri": "https://ex.org/me", '
            '"display_name": "Eat Mindfully (LLM rewrite)"}]'
        )
        gen = self._gen_with_session(
            raw_answer,
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name="Mindful Eating 🍽️",
                description=None,
            ),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].display_name == "Mindful Eating 🍽️"

    def test_ontology_description_fills_in_when_llm_omits(self):
        raw_answer = '[{"label": "me", "ontology_uri": "https://ex.org/me"}]'
        gen = self._gen_with_session(
            raw_answer,
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name=None,
                description="Eat slowly and notice every bite. 🍽️",
            ),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert "Eat slowly and notice every bite. 🍽️" in tasks[0].description

    def test_ontology_description_overrides_llm_supplied_description(self):
        """Ontology-wins rule: an ontology-supplied `description`
        overrides the LLM's value."""
        raw_answer = (
            '[{"label": "me", "ontology_uri": "https://ex.org/me", '
            '"description": "An LLM rewrite without emojis."}]'
        )
        gen = self._gen_with_session(
            raw_answer,
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name=None,
                description="Eat slowly and notice every bite. 🍽️",
            ),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert "Eat slowly and notice every bite. 🍽️" in tasks[0].description
        assert "LLM rewrite" not in tasks[0].description

    def test_llm_display_name_kept_when_ontology_silent(self):
        """When the ontology has no `display_name`, the LLM-supplied
        value stands."""
        raw_answer = (
            '[{"label": "yoga", "ontology_uri": "https://ex.org/yoga", '
            '"display_name": "Sunrise Yoga"}]'
        )
        gen = self._gen_with_session(
            raw_answer,
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name=None,
                description=None,
            ),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].display_name == "Sunrise Yoga"

    def test_llm_description_kept_when_ontology_silent(self):
        raw_answer = (
            '[{"label": "yoga", "ontology_uri": "https://ex.org/yoga", '
            '"description": "Gentle morning yoga."}]'
        )
        gen = self._gen_with_session(
            raw_answer,
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name=None,
                description=None,
            ),
            None,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert "Gentle morning yoga." in tasks[0].description

    def test_non_ontology_wins_key_already_in_raw_is_kept(self):
        """An ontology key outside ONTOLOGY_WINS that the LLM already supplied is not overwritten."""
        raw_answer = (
            '[{"label": "yoga", "ontology_uri": "https://ex.org/yoga", '
            '"difficulty_level": 1}]'
        )
        # The first record drives fetch_task_properties; second drives
        # fetch_difficulty_level. A token Level3 forces difficulty_level=3
        # from the ontology, but raw already supplies 1.
        gen = self._gen_with_session(
            raw_answer,
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name=None,
                description=None,
            ),
            _record(token="Level3"),
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        # raw's difficulty_level=1 survives because the elif evaluates False
        # (key was already present), proving the loop iterates without overwriting.
        assert tasks[0].difficulty_level == 1


# ---------------------------------------------------------------------------
# TaskGenerator; fabricated-URI handling (LLM hallucination defense)
# ---------------------------------------------------------------------------


class TestFabricatedUriHandling:
    """When the LLM invents a plausible-looking `ontology_uri` that does not
    resolve in the graph, the generator drops the URI and strips the trailing
    `[<uri>]` citation from `description`, then returns the task as-is."""

    _PERSON = _make_person()
    _TRACE = CalendarTrace(person_id="p001")

    def _make_gen(self, raw_answer: str) -> TaskGenerator:
        config = _make_config()
        pipeline = _MockPipeline(raw_answer)
        # uri_resolves=False to every EXISTS query returns "no hit",
        # so any URI the LLM supplies is treated as fabricated.
        session = _MockSession(uri_resolves=False)
        return TaskGenerator(
            config=config, graphrag_pipeline=pipeline, neo4j_session=session
        )

    def test_fabricated_uri_is_stripped_from_task(self):
        raw_answer = (
            '[{"label": "prepare-a-healthy-snack", '
            '"display_name": "Prepare a Healthy Snack", '
            '"description": "Make a snack. '
            '[https://w3id.org/calendar-bench/health/task/prepare-a-healthy-snack]", '
            '"ontology_uri": "https://w3id.org/calendar-bench/health/task/prepare-a-healthy-snack"}]'
        )
        gen = self._make_gen(raw_answer)
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert len(tasks) == 1
        # URI is dropped; it never reaches the downstream JSON/ICS
        assert tasks[0].ontology_uri is None
        # The trailing [URI] citation is removed from the description
        assert "[https://w3id.org" not in tasks[0].description
        assert "Make a snack" in tasks[0].description
        # The label and display_name remain; only the false claim is gone
        assert tasks[0].label == "prepare-a-healthy-snack"
        assert tasks[0].display_name == "Prepare a Healthy Snack"

    def test_fabricated_uri_logs_at_debug_not_warning(self, caplog):
        """Per-URI fabrication notices are DEBUG so the progress bar isn't
        polluted by one line per task.  The aggregate is surfaced via the
        evaluate-time `ontology_grounding` ratio."""
        import logging

        raw_answer = (
            '[{"label": "fake-task", '
            '"ontology_uri": "https://ex.org/task/does-not-exist"}]'
        )
        gen = self._make_gen(raw_answer)
        with caplog.at_level(
            logging.DEBUG, logger="src.scripts.scenarios.task_generation.generator"
        ):
            gen.generate(self._PERSON, self._TRACE)
        debug_records = [
            r
            for r in caplog.records
            if "fabricated ontology_uri" in r.getMessage().lower()
        ]
        assert debug_records, "no fabrication record was emitted"
        for r in debug_records:
            assert r.levelname == "DEBUG"
            assert "fake-task" in r.getMessage()

    def test_fabricated_uri_does_not_run_props_or_branch_queries(self):
        """When the URI doesn't resolve, the bridge short-circuits; no
        property fetch and no branch walk are issued."""
        raw_answer = '[{"label": "ghost", "ontology_uri": "https://ex.org/task/ghost"}]'
        config = _make_config()
        pipeline = _MockPipeline(raw_answer)
        # Empty queue: any non-EXISTS query would return None and increment
        # the index. We assert the index is still 0 after generation.
        session = _MockSession(uri_resolves=False)
        gen = TaskGenerator(
            config=config, graphrag_pipeline=pipeline, neo4j_session=session
        )
        gen.generate(self._PERSON, self._TRACE)
        assert session._idx == 0

    def test_fabricated_uri_drops_uri_and_keeps_other_llm_fields(self):
        """A stripped URI does not clobber the rest of the LLM payload."""
        raw_answer = (
            '[{"label": "ghost", '
            '"ontology_uri": "https://ex.org/task/ghost", '
            '"duration_min": 15, "duration_max": 30}]'
        )
        gen = self._make_gen(raw_answer)
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].ontology_uri is None
        assert tasks[0].duration_min == 15
        assert tasks[0].duration_max == 30

    def test_resolved_uri_unaffected_by_strip_logic(self):
        """A real (resolving) URI continues to enrich + keep its citation."""
        raw_answer = (
            '[{"label": "yoga", '
            '"description": "Yoga is great. [https://ex.org/yoga]", '
            '"ontology_uri": "https://ex.org/yoga"}]'
        )
        config = _make_config()
        pipeline = _MockPipeline(raw_answer)
        session = _MockSession(
            _record(is_dividable=True, is_concurrent=None, duration_minutes=None),
            _record(branch=PHYSICAL_ACTIVITY_URI, depth=1),
            uri_resolves=True,
        )
        gen = TaskGenerator(
            config=config, graphrag_pipeline=pipeline, neo4j_session=session
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].ontology_uri == "https://ex.org/yoga"
        assert "[https://ex.org/yoga]" in tasks[0].description
        assert tasks[0].is_dividable is True


class TestStripUriCitation:
    """Unit tests for the `_strip_uri_citation` helper used by the
    fabricated-URI defense path."""

    def test_strips_trailing_citation(self):
        from src.scripts.scenarios.task_generation.generator import _strip_uri_citation

        out = _strip_uri_citation(
            "Walk daily. [https://ex.org/task/walk]", "https://ex.org/task/walk"
        )
        assert out == "Walk daily."

    def test_strips_with_extra_whitespace(self):
        from src.scripts.scenarios.task_generation.generator import _strip_uri_citation

        out = _strip_uri_citation(
            "Walk daily.   [https://ex.org/task/walk]   ",
            "https://ex.org/task/walk",
        )
        assert out == "Walk daily."

    def test_no_citation_returns_unchanged(self):
        from src.scripts.scenarios.task_generation.generator import _strip_uri_citation

        out = _strip_uri_citation("No citation here.", "https://ex.org/x")
        assert out == "No citation here."

    def test_empty_description_returns_empty(self):
        from src.scripts.scenarios.task_generation.generator import _strip_uri_citation

        assert _strip_uri_citation("", "https://ex.org/x") == ""

    def test_empty_uri_returns_unchanged(self):
        from src.scripts.scenarios.task_generation.generator import _strip_uri_citation

        out = _strip_uri_citation("Walk. [http://a.b]", "")
        assert out == "Walk. [http://a.b]"

    def test_only_strips_matching_uri(self):
        """A non-matching URI in the description must NOT be removed -
        only the trailing citation that equals *uri*."""
        from src.scripts.scenarios.task_generation.generator import _strip_uri_citation

        out = _strip_uri_citation(
            "Walk. [https://other.example/x]",
            "https://ex.org/task/walk",
        )
        assert out == "Walk. [https://other.example/x]"

    def test_special_regex_chars_in_uri_are_escaped(self):
        """URIs with regex-meta chars (`+`, `.`) must still match
        literally; re.escape is applied."""
        from src.scripts.scenarios.task_generation.generator import _strip_uri_citation

        uri = "https://ex.org/path+with.dots"
        out = _strip_uri_citation(f"Body. [{uri}]", uri)
        assert out == "Body."


# ---------------------------------------------------------------------------
# TaskGenerator; neo4j_driver path (per-call session opening)
# ---------------------------------------------------------------------------


class _DriverReturningSession:
    """Context manager that returns a pre-built `_MockSession` and tracks
    whether `__enter__` / `__exit__` were called; proves that the
    driver path opens and closes the session correctly per enrichment."""

    def __init__(self, session: _MockSession) -> None:
        self.session = session
        self.entered = False
        self.exited = False

    def __enter__(self) -> _MockSession:
        self.entered = True
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class _MockDriver:
    """Mimics a neo4j Driver; its `session()` returns a fresh context
    manager each call so concurrent threads cannot share state."""

    def __init__(self, *records: _MockRecord | None) -> None:
        self._records = records
        self.session_calls: list[_DriverReturningSession] = []

    def session(self) -> _DriverReturningSession:
        ctx = _DriverReturningSession(_MockSession(*self._records))
        self.session_calls.append(ctx)
        return ctx


class TestEnrichDictsViaDriver:
    """When `neo4j_driver` is supplied (and `neo4j_session` is not)
    the generator opens a fresh session per `_enrich_dicts` call.
    This is the production wiring used by `cli.py`: drivers are
    thread-safe, sessions are not."""

    _PERSON = _make_person()
    _TRACE = CalendarTrace(person_id="p001")

    def _gen_with_driver(self, answer: str, *records: _MockRecord | None):
        config = _make_config()
        pipeline = _MockPipeline(answer)
        driver = _MockDriver(*records)
        gen = TaskGenerator(
            config=config,
            graphrag_pipeline=pipeline,
            neo4j_driver=driver,
        )
        return gen, driver

    def test_driver_opens_session_for_enrichment(self):
        raw_answer = '[{"label": "yoga", "ontology_uri": "https://ex.org/task/yoga"}]'
        gen, driver = self._gen_with_driver(
            raw_answer,
            _record(
                is_dividable=True,
                is_concurrent=None,
                duration_minutes=None,
                display_name="Sunrise Yoga",
                description=None,
            ),
            _record(branch=PHYSICAL_ACTIVITY_URI, depth=1),
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        # Properties from the ontology flowed through.
        assert tasks[0].is_dividable is True
        assert tasks[0].display_name == "Sunrise Yoga"
        # The driver opened exactly one session for this generate() call,
        # and properly closed it via __exit__.
        assert len(driver.session_calls) == 1
        assert driver.session_calls[0].entered is True
        assert driver.session_calls[0].exited is True

    def test_session_takes_priority_over_driver(self):
        """When both are provided, the explicit session wins; the driver
        is never asked to open a new one."""
        raw_answer = '[{"label": "yoga", "ontology_uri": "https://ex.org/task/yoga"}]'
        explicit_session = _MockSession(
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name="From Explicit Session",
                description=None,
            ),
            None,
        )
        driver = _MockDriver()
        gen = TaskGenerator(
            config=_make_config(),
            graphrag_pipeline=_MockPipeline(raw_answer),
            neo4j_session=explicit_session,
            neo4j_driver=driver,
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        assert tasks[0].display_name == "From Explicit Session"
        assert driver.session_calls == []  # driver never touched

    def test_neither_session_nor_driver_skips_enrichment(self):
        """The pre-existing `no Neo4j configured` short-circuit still
        works after the driver-path addition."""
        raw_answer = '[{"label": "yoga", "ontology_uri": "https://ex.org/task/yoga"}]'
        gen = TaskGenerator(
            config=_make_config(),
            graphrag_pipeline=_MockPipeline(raw_answer),
            # no neo4j_session, no neo4j_driver
        )
        tasks = gen.generate(self._PERSON, self._TRACE)
        # Parser default; no ontology override.
        assert tasks[0].display_name == ""


# ---------------------------------------------------------------------------
# Grouped filters render in both prompt paths (regression)
# ---------------------------------------------------------------------------


class TestGroupedFilterPromptRendering:
    def _grouped_generator(self) -> TaskGenerator:
        cfg = TaskGenerationConfig(
            method="graphrag",
            num_tasks=3,
            prompt_template="health_improvement",
            filters=[
                {"domains": ["NutritionTask"], "difficulty": ["Level1"]},
                {
                    "domains": ["PhysicalActivityTask"],
                    "difficulty": ["Level2", "Level3"],
                },
            ],
        )
        return TaskGenerator(
            config=cfg, graphrag_pipeline=_MockPipeline("[]"), neo4j_session=None
        )

    def test_build_prompt_renders_union_of_grouped_domains(self):
        gen = self._grouped_generator()
        prompt = gen._build_prompt(_make_person(), CalendarTrace(person_id="p001"), "")
        assert "NutritionTask" in prompt
        assert "PhysicalActivityTask" in prompt

    def test_fetch_prompt_renders_union_of_grouped_domains(self):
        gen = self._grouped_generator()
        prompt = gen._render_fetch_prompt(
            _make_person(),
            CalendarTrace(person_id="p001"),
            "",
            needed=2,
            accepted_names=[],
            accepted_uris=[],
            rejected_uris=[],
        )
        assert "NutritionTask" in prompt
        assert "PhysicalActivityTask" in prompt
