"""LLMAugmenter prompt builders honour the ObservationConfig leak gate."""

from __future__ import annotations

import datetime
from collections import defaultdict

import pytest

from src.scripts.scenarios.augmentation.llm_agent import (
    _build_calendar_summary,
    _build_context_summary,
    _build_tasks_list,
    _resolve_event_label_on_date,
)
from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
    ABLATABLE_BLOCKS,
    render,
)
from src.scripts.scenarios.config.schema import ObservationConfig
from src.scripts.scenarios.domain.calendar import CalendarEvent
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.task import RecommendedTask


def _desired(label: str = "tea-time", **over) -> RecommendedTask:
    return RecommendedTask(
        label=label,
        duration_min=over.get("duration_min", 10),
        duration_max=over.get("duration_max", 20),
        intensity=over.get("intensity", 2),
        is_concurrent=over.get("is_concurrent", True),
        is_dividable=over.get("is_dividable", False),
        display_name=over.get("display_name", "Tea Time"),
        description=over.get("description", ""),
    )


def _event() -> CalendarEvent:
    return CalendarEvent(
        label="lunch",
        start_minutes=720,
        end_minutes=765,
        date=datetime.date(2026, 5, 4),
        is_concurrent=True,
        is_dividable=False,
        concurrent_with=["snack"],
        intensity=2,
    )


def test_context_block_is_an_ablatable_block() -> None:
    """`context_block` is ablatable; 13 blocks total."""
    from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
        get_template_source,
    )

    assert "context_block" in ABLATABLE_BLOCKS
    assert len(ABLATABLE_BLOCKS) == 13
    src = get_template_source()
    assert "{% block context_block %}" in src


def test_build_tasks_list_default_renders_today_full_field_set() -> None:
    out = _build_tasks_list([_desired()])
    for needle in (
        "duration_min",
        "duration_max",
        "intensity",
        "concurrent_ok",
        "dividable_ok",
    ):
        assert needle in out


def test_build_tasks_list_with_observation_trims_to_listed_flags() -> None:
    obs = ObservationConfig(task_flags=["duration_min", "duration_max"])
    out = _build_tasks_list([_desired()], observation=obs)
    assert "duration_min" in out
    assert "duration_max" in out
    for hidden in ("intensity", "concurrent_ok", "dividable_ok"):
        assert f"  {hidden}" not in out


def test_build_tasks_list_blind_augmenter_renders_only_title_and_description() -> None:
    obs = ObservationConfig(task_flags=[])
    task = _desired(description="Replace coffee with tea.")
    out = _build_tasks_list([task], observation=obs)
    assert 'task_title        : "Tea Time"' in out
    assert "description" in out
    for hidden in (
        "duration_min",
        "duration_max",
        "intensity",
        "concurrent_ok",
        "dividable_ok",
    ):
        assert f"  {hidden}" not in out


def test_build_calendar_summary_empty_host_flags_matches_today_shape() -> None:
    """Empty host_flags reproduces the historical busy payload byte-for-byte."""
    events = defaultdict(list)
    events[datetime.date(2026, 5, 4)].append(_event())
    week = [datetime.date(2026, 5, 4)]
    baseline = _build_calendar_summary(week, events)
    with_obs = _build_calendar_summary(week, events, observation=ObservationConfig())
    assert baseline == with_obs
    assert "is_concurrent" not in baseline


def test_build_calendar_summary_host_flags_inject_extra_keys() -> None:
    events = defaultdict(list)
    events[datetime.date(2026, 5, 4)].append(_event())
    week = [datetime.date(2026, 5, 4)]
    obs = ObservationConfig(host_flags=["is_concurrent", "is_dividable", "intensity"])
    out = _build_calendar_summary(week, events, observation=obs)
    assert '"is_concurrent":true' in out
    assert '"is_dividable":false' in out
    assert '"intensity":2' in out


def test_build_calendar_summary_host_flags_concurrent_with_list() -> None:
    events = defaultdict(list)
    events[datetime.date(2026, 5, 4)].append(_event())
    week = [datetime.date(2026, 5, 4)]
    obs = ObservationConfig(host_flags=["concurrent_with"])
    out = _build_calendar_summary(week, events, observation=obs)
    assert '"concurrent_with":["snack"]' in out


def _office_event(display_label: str | None = None) -> CalendarEvent:
    return CalendarEvent(
        label="office_work",
        start_minutes=540,
        end_minutes=600,
        date=datetime.date(2026, 5, 4),
        is_concurrent=True,
        display_label=display_label,
    )


def test_build_calendar_summary_uses_display_label() -> None:
    events = defaultdict(list)
    events[datetime.date(2026, 5, 4)].append(_office_event(display_label="standup"))
    out = _build_calendar_summary([datetime.date(2026, 5, 4)], events)
    assert '"label":"Standup"' in out
    assert "Office Work" not in out


def test_build_calendar_summary_falls_back_to_label_without_display() -> None:
    events = defaultdict(list)
    events[datetime.date(2026, 5, 4)].append(_office_event())
    out = _build_calendar_summary([datetime.date(2026, 5, 4)], events)
    assert '"label":"Office Work"' in out


def test_resolve_event_label_matches_display_title_returns_canonical() -> None:
    date = datetime.date(2026, 5, 4)
    events_by_date = {date: [_office_event(display_label="external meeting")]}
    assert (
        _resolve_event_label_on_date("External Meeting", date, events_by_date)
        == "office_work"
    )


def test_resolve_event_label_still_matches_canonical_label() -> None:
    date = datetime.date(2026, 5, 4)
    events_by_date = {date: [_office_event(display_label="standup")]}
    assert (
        _resolve_event_label_on_date("office_work", date, events_by_date)
        == "office_work"
    )


def test_build_context_summary_empty_when_no_contexts_in_observation() -> None:
    ep = ContextEpisode(
        name="happy",
        category="mood_emotion",
        date=datetime.date(2026, 5, 4),
        start_minutes=480,
        end_minutes=540,
    )
    out = _build_context_summary(
        [ep], [datetime.date(2026, 5, 4)], ObservationConfig(), catalog=None
    )
    assert out == ""


def test_build_context_summary_renders_only_opted_in_categories() -> None:
    eps = [
        ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=datetime.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
        ),
        ContextEpisode(
            name="home",
            category="location",
            date=datetime.date(2026, 5, 4),
            start_minutes=540,
            end_minutes=600,
        ),
    ]
    obs = ObservationConfig(contexts=["mood_emotion"])
    out = _build_context_summary(eps, [datetime.date(2026, 5, 4)], obs, catalog=None)
    assert "mood_emotion" in out
    assert "happy" in out
    assert "location" not in out
    assert "home" not in out


def test_build_context_summary_full_detail_adds_uri_and_dimension() -> None:
    eps = [
        ContextEpisode(
            name="energetic",
            category="energy_state",
            date=datetime.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
            ontology_uri="http://example.org/iri",
            dimension="extraversion",
            polarity="high",
        ),
    ]
    obs = ObservationConfig(contexts=["energy_state"], context_detail="full")
    out = _build_context_summary(eps, [datetime.date(2026, 5, 4)], obs, catalog=None)
    assert "http://example.org/iri" in out
    assert "extraversion" in out
    assert "high" in out


def test_build_context_summary_episodes_outside_week_excluded() -> None:
    eps = [
        ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=datetime.date(2026, 5, 10),
            start_minutes=480,
            end_minutes=540,
        ),
    ]
    obs = ObservationConfig(contexts=["mood_emotion"])
    out = _build_context_summary(eps, [datetime.date(2026, 5, 4)], obs, catalog=None)
    assert out == ""


def test_template_renders_with_empty_context_summary_kwarg() -> None:
    """An empty context_summary renders without emitting the CONTEXT heading."""
    out = render(
        {
            "week_start": "Mon 05-04",
            "week_end": "Sun 05-10",
            "wake_start": "06:00",
            "sleep_start": "22:00",
            "num_tasks": 0,
            "calendar_summary": "TIMELINE: []",
            "tasks_list": "",
            "context_summary": "",
        }
    )
    # No `## CONTEXT` heading appears when context_summary is empty.
    assert "## CONTEXT" not in out


def test_llm_augmenter_injects_catalog_label_into_context_block() -> None:
    """When observation.contexts is non-empty, the catalog labels resolve into the rendered block."""
    import datetime

    from src.scripts.persona.config.schema import DailyWindow
    from src.scripts.persona.context.catalog import CatalogEntry, ContextIriCatalog
    from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter
    from src.scripts.scenarios.config.schema import AugmentationConfig, LLMAgentConfig
    from src.scripts.scenarios.domain.calendar import CalendarTrace
    from src.scripts.scenarios.domain.context import ContextEpisode

    catalog = ContextIriCatalog(
        entries=[
            CatalogEntry(
                category="mood_emotion",
                name="feeling_happy",
                iri="http://example.org/iri",
                label="feeling happy",
                source="MFOEM",
                definition=None,
                polarity=None,
                instrument=None,
                theory_mappings=None,
            ),
        ]
    )

    class _NullPipeline:
        def search(self, query_text: str):  # noqa: ANN001
            raise RuntimeError("unused")

    augmenter = LLMAugmenter(
        llm_pipeline=_NullPipeline(),
        daily_window=DailyWindow(),
        context_catalog=catalog,
    )
    cfg = AugmentationConfig(
        method="llm_agent",
        llm_agent=LLMAgentConfig(),
        observation=ObservationConfig(contexts=["mood_emotion"]),
    )
    trace_contexts = [
        ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=datetime.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
            ontology_uri="http://example.org/iri",
        ),
    ]
    prompt = augmenter._build_oneshot_prompt(
        tasks=[],
        week_dates=[datetime.date(2026, 5, 4)],
        events_by_date={},
        scheduled_so_far=None,
        config=cfg,
        contexts=trace_contexts,
    )
    assert "feeling happy" in prompt
    assert "happy" in prompt


def test_template_renders_with_populated_context_summary() -> None:
    summary = "## CONTEXT (mood_emotion)\n  - happy  08:00-09:00"
    out = render(
        {
            "week_start": "Mon 05-04",
            "week_end": "Sun 05-10",
            "wake_start": "06:00",
            "sleep_start": "22:00",
            "num_tasks": 0,
            "calendar_summary": "TIMELINE: []",
            "tasks_list": "",
            "context_summary": summary,
        }
    )
    assert summary in out


# ---------------------------------------------------------------------------
# Per-task context_recommends
# ---------------------------------------------------------------------------


def _desired_with_uri(uri: str | None) -> RecommendedTask:
    return RecommendedTask(
        label="task",
        duration_min=10,
        duration_max=20,
        intensity=2,
        is_concurrent=True,
        is_dividable=False,
        display_name="Task",
        ontology_uri=uri,
    )


def test_context_recommends_renders_intersection_in_observation_order() -> None:
    """The rendered list orders categories by `observation.contexts`, not by the task's own ordering."""
    obs = ObservationConfig(
        contexts=["behaviour_state", "capability_opportunity"],
        task_flags=[
            "duration_min",
            "duration_max",
            "intensity",
            "concurrent_ok",
            "dividable_ok",
            "context_recommends",
        ],
    )
    task = _desired_with_uri("https://w3id.org/calendar-bench/health/task/foo")
    links = {
        task.ontology_uri: [
            "mood_emotion",
            "capability_opportunity",
            "behaviour_state",
            "weather_environment",
        ],
    }
    out = _build_tasks_list([task], observation=obs, context_links_by_uri=links)
    # Only the two observed categories appear; weather/mood are dropped.
    assert "context_recommends: [behaviour_state, capability_opportunity]" in out
    assert "mood_emotion" not in out
    assert "weather_environment" not in out


def test_context_recommends_omitted_when_intersection_empty() -> None:
    """A task whose recommendations don't overlap the observation gets no field."""
    obs = ObservationConfig(
        contexts=["behaviour_state"],
        task_flags=["context_recommends"],
    )
    task = _desired_with_uri("https://w3id.org/calendar-bench/health/task/foo")
    links = {task.ontology_uri: ["mood_emotion", "location"]}
    out = _build_tasks_list([task], observation=obs, context_links_by_uri=links)
    assert "context_recommends" not in out


def test_context_recommends_omitted_when_observation_blind() -> None:
    """Blind augmenter (empty contexts) renders no `context_recommends` field, even with the flag enabled."""
    obs = ObservationConfig(contexts=[], task_flags=["context_recommends"])
    task = _desired_with_uri("https://w3id.org/calendar-bench/health/task/foo")
    links = {task.ontology_uri: ["behaviour_state"]}
    out = _build_tasks_list([task], observation=obs, context_links_by_uri=links)
    assert "context_recommends" not in out


def test_context_recommends_suppressed_when_flag_not_in_task_flags() -> None:
    """Ablating `context_recommends` from task_flags hides the field even when contexts and links overlap."""
    obs = ObservationConfig(
        contexts=["behaviour_state"],
        task_flags=["duration_min", "duration_max"],
    )
    task = _desired_with_uri("https://w3id.org/calendar-bench/health/task/foo")
    links = {task.ontology_uri: ["behaviour_state"]}
    out = _build_tasks_list([task], observation=obs, context_links_by_uri=links)
    assert "context_recommends" not in out


def test_context_recommends_handles_task_without_ontology_uri() -> None:
    """A task with no `ontology_uri` cannot resolve recommendations; field omitted."""
    obs = ObservationConfig(
        contexts=["behaviour_state"],
        task_flags=["context_recommends"],
    )
    task = _desired_with_uri(None)
    links = {"some_other_uri": ["behaviour_state"]}
    out = _build_tasks_list([task], observation=obs, context_links_by_uri=links)
    assert "context_recommends" not in out


def test_context_recommends_default_observation_path_renders_nothing_without_links() -> (
    None
):
    """Calling `_build_tasks_list` without an observation defaults to the legacy flag set and skips context_recommends."""
    task = _desired_with_uri("https://w3id.org/calendar-bench/health/task/foo")
    links = {task.ontology_uri: ["behaviour_state"]}
    out = _build_tasks_list([task], context_links_by_uri=links)
    assert "context_recommends" not in out


def test_self_verify_block_contains_context_overlap_clause() -> None:
    """The SELF-VERIFY block carries the context-overlap check."""
    from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
        get_template_source,
    )

    src = get_template_source()
    assert "context_recommends" in src
    assert "recommended-category" in src
    assert "CONTEXT block" in src


# ---------------------------------------------------------------------------
# LLMAugmenter._get_context_links lazy loader
# ---------------------------------------------------------------------------


def test_get_context_links_loads_from_disk_when_file_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On first call, loads `context_links_by_uri` from the HealthTasks ontology."""
    from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter

    expected = {
        "https://w3id.org/calendar-bench/health/task/foo": {
            "mood_emotion": frozenset({"http://purl.obolibrary.org/obo/MFOEM_000042"})
        }
    }
    monkeypatch.setattr(
        "src.scripts.scenarios.metrics.context_fit.load_context_categories_by_iri",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.scripts.scenarios.metrics.context_fit.load_context_links_by_uri",
        lambda _ttl, _cats: dict(expected),
    )
    # The path check goes through Path.exists; force True so the loader
    # branch fires regardless of the working directory.
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    augmenter = LLMAugmenter(llm_pipeline=object())
    assert augmenter._get_context_links() == expected


def test_get_context_links_returns_empty_when_file_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing HealthTasks ontology resolves to an empty links map; no exception."""
    from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter

    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    augmenter = LLMAugmenter(llm_pipeline=object())
    assert augmenter._get_context_links() == {}


def test_get_context_links_swallows_loader_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader failures degrade to an empty map; the augmenter keeps planning."""
    from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter

    def _raise(_ttl, _cats):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.scripts.scenarios.metrics.context_fit.load_context_categories_by_iri",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.scripts.scenarios.metrics.context_fit.load_context_links_by_uri",
        _raise,
    )
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    augmenter = LLMAugmenter(llm_pipeline=object())
    assert augmenter._get_context_links() == {}


def test_get_context_links_caches_after_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call returns the cached map without re-invoking the loader."""
    from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter

    call_count = {"n": 0}

    def _loader(_ttl, _cats):
        call_count["n"] += 1
        return {"u": {"mood_emotion": frozenset({"iri"})}}

    monkeypatch.setattr(
        "src.scripts.scenarios.metrics.context_fit.load_context_categories_by_iri",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.scripts.scenarios.metrics.context_fit.load_context_links_by_uri",
        _loader,
    )
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    augmenter = LLMAugmenter(llm_pipeline=object())
    first = augmenter._get_context_links()
    second = augmenter._get_context_links()
    assert first == second
    assert call_count["n"] == 1


class TestOneshotPromptContextBlockEndToEnd:
    """Confirm the augment_oneshot prompt reflects observation.contexts."""

    _CONTEXT_HEADING = "## CONTEXT (per-day momentary signals; persona ground truth)"

    def _augmenter(self):
        from src.scripts.persona.config.schema import DailyWindow
        from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter

        class _NullPipeline:
            def search(self, query_text: str):  # noqa: ANN001
                raise RuntimeError("unused")

        return LLMAugmenter(
            llm_pipeline=_NullPipeline(),
            daily_window=DailyWindow(),
        )

    def _cfg(self, observation: ObservationConfig) -> "AugmentationConfig":
        from src.scripts.scenarios.config.schema import (
            AugmentationConfig,
            LLMAgentConfig,
        )

        return AugmentationConfig(
            method="llm_agent",
            llm_agent=LLMAgentConfig(),
            observation=observation,
        )

    def _episode(self, name: str, category: str, *, start: int = 480, end: int = 540):
        return ContextEpisode(
            name=name,
            category=category,
            date=datetime.date(2026, 5, 4),
            start_minutes=start,
            end_minutes=end,
        )

    def test_includes_context_heading_when_observation_has_contexts(self) -> None:
        """A configured observation renders the CONTEXT heading and matching episode rows."""
        augmenter = self._augmenter()
        cfg = self._cfg(ObservationConfig(contexts=["mood_emotion"]))
        prompt = augmenter._build_oneshot_prompt(
            tasks=[],
            week_dates=[datetime.date(2026, 5, 4)],
            events_by_date={},
            scheduled_so_far=None,
            config=cfg,
            contexts=[self._episode("happy", "mood_emotion")],
        )
        assert self._CONTEXT_HEADING in prompt
        assert "# CONTEXT (mood_emotion)" in prompt
        assert "happy" in prompt
        assert "08:00-09:00" in prompt

    def test_omits_context_heading_when_observation_blind(self) -> None:
        """A blind observation never leaks a CONTEXT block even when the trace has episodes."""
        augmenter = self._augmenter()
        cfg = self._cfg(ObservationConfig())
        prompt = augmenter._build_oneshot_prompt(
            tasks=[],
            week_dates=[datetime.date(2026, 5, 4)],
            events_by_date={},
            scheduled_so_far=None,
            config=cfg,
            contexts=[self._episode("happy", "mood_emotion")],
        )
        assert self._CONTEXT_HEADING not in prompt
        assert "# CONTEXT (" not in prompt
        assert "happy" not in prompt

    def test_excludes_episodes_from_non_observed_categories(self) -> None:
        """Episodes from categories outside observation.contexts are dropped."""
        augmenter = self._augmenter()
        cfg = self._cfg(ObservationConfig(contexts=["mood_emotion"]))
        prompt = augmenter._build_oneshot_prompt(
            tasks=[],
            week_dates=[datetime.date(2026, 5, 4)],
            events_by_date={},
            scheduled_so_far=None,
            config=cfg,
            contexts=[
                self._episode("happy", "mood_emotion"),
                self._episode("tired", "energy_state", start=600, end=660),
            ],
        )
        assert "# CONTEXT (mood_emotion)" in prompt
        assert "# CONTEXT (energy_state)" not in prompt
        assert "happy" in prompt
        assert "tired" not in prompt

    def test_multi_scenario_translation_preserves_context_block_in_prompt(
        self,
    ) -> None:
        """A multi-scenario ObservationConfig survives translation and reaches the prompt."""
        from src.scripts.scenarios.cli import _method_cfg_to_scenario_config
        from src.scripts.scenarios.config.schema import (
            AugmentationMethodConfig,
            ExperimentScenariosConfig,
            ScenarioDefinition,
        )

        exp = ExperimentScenariosConfig.model_validate(
            {"experiment_id": "exp_t", "scenarios": [{"id": "s1"}]}
        )
        scenario = ScenarioDefinition(id="s1")
        method = AugmentationMethodConfig(
            method="llm_agent",
            observation=ObservationConfig(contexts=["mood_emotion"]),
        )
        legacy_cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        augmenter = self._augmenter()
        prompt = augmenter._build_oneshot_prompt(
            tasks=[],
            week_dates=[datetime.date(2026, 5, 4)],
            events_by_date={},
            scheduled_so_far=None,
            config=legacy_cfg.augmentation,
            contexts=[self._episode("happy", "mood_emotion")],
        )
        assert self._CONTEXT_HEADING in prompt
        assert "# CONTEXT (mood_emotion)" in prompt
        assert "happy" in prompt

    def test_context_subsections_follow_observation_order(self) -> None:
        """CONTEXT subsections appear in the observation.contexts order, not trace order."""
        augmenter = self._augmenter()
        cfg = self._cfg(
            ObservationConfig(contexts=["behaviour_state", "capability_opportunity"])
        )
        prompt = augmenter._build_oneshot_prompt(
            tasks=[],
            week_dates=[datetime.date(2026, 5, 4)],
            events_by_date={},
            scheduled_so_far=None,
            config=cfg,
            contexts=[
                # Trace order is reversed vs observation order on purpose.
                self._episode(
                    "high_capability",
                    "capability_opportunity",
                    start=600,
                    end=660,
                ),
                self._episode("action_planning", "behaviour_state"),
            ],
        )
        idx_b = prompt.find("# CONTEXT (behaviour_state)")
        idx_c = prompt.find("# CONTEXT (capability_opportunity)")
        assert idx_b != -1, "behaviour_state subsection missing"
        assert idx_c != -1, "capability_opportunity subsection missing"
        assert idx_b < idx_c, (
            "subsections must follow `observation.contexts` order; "
            f"behaviour_state idx={idx_b} should precede capability_opportunity idx={idx_c}"
        )


# ---------------------------------------------------------------------------
# Small helper coverage: _maybe_clip_rescue, _host_flag_suffix, _build_context_summary,
# _get_context_catalog
# ---------------------------------------------------------------------------


def test_maybe_clip_rescue_returns_decision_unchanged_when_host_missing() -> None:
    """No host event for the concurrent label returns the decision as-is."""
    from src.scripts.scenarios.augmentation.llm_agent import (
        _maybe_clip_rescue,
        _ScheduleDecision,
    )
    from src.scripts.scenarios.domain.task import RecommendedTask

    decision = _ScheduleDecision(
        task_index=1,
        task_label="task",
        scheduled=True,
        date=datetime.date(2026, 5, 4),
        start_minutes=480,
        end_minutes=540,
        concurrent_flag=True,
        concurrent_with="nonexistent",
    )
    task = RecommendedTask(
        label="task",
        duration_min=10,
        duration_max=60,
        intensity=2,
        is_concurrent=True,
        is_dividable=False,
        display_name="Task",
    )
    out = _maybe_clip_rescue(decision, task, events_by_date={})
    assert out is decision


def test_host_flag_suffix_intensity_branch_renders_int() -> None:
    """The intensity flag emits the integer event intensity field."""
    from src.scripts.scenarios.augmentation.llm_agent import _host_flag_suffix

    ev = CalendarEvent(
        label="lunch",
        start_minutes=720,
        end_minutes=765,
        date=datetime.date(2026, 5, 4),
        is_concurrent=True,
        is_dividable=False,
        concurrent_with=[],
        intensity=3,
    )
    out = _host_flag_suffix(ev, host_flags=["intensity"])
    assert ',"intensity":3' in out


def test_host_flag_suffix_iterates_multiple_flags() -> None:
    """Multiple flags render all four supported entries on one busy interval."""
    from src.scripts.scenarios.augmentation.llm_agent import _host_flag_suffix

    ev = CalendarEvent(
        label="lunch",
        start_minutes=720,
        end_minutes=765,
        date=datetime.date(2026, 5, 4),
        is_concurrent=True,
        is_dividable=False,
        concurrent_with=["snack"],
        intensity=2,
    )
    out = _host_flag_suffix(
        ev,
        host_flags=["is_concurrent", "is_dividable", "concurrent_with", "intensity"],
    )
    for needle in (
        '"is_concurrent":true',
        '"is_dividable":false',
        '"concurrent_with":["snack"]',
        '"intensity":2',
    ):
        assert needle in out


def test_host_flag_suffix_loops_after_intensity_when_not_last() -> None:
    """When intensity is not the last flag, the loop continues to render remaining flags."""
    from src.scripts.scenarios.augmentation.llm_agent import _host_flag_suffix

    ev = CalendarEvent(
        label="lunch",
        start_minutes=720,
        end_minutes=765,
        date=datetime.date(2026, 5, 4),
        is_concurrent=True,
        is_dividable=False,
        concurrent_with=["snack"],
        intensity=4,
    )
    out = _host_flag_suffix(ev, host_flags=["intensity", "is_concurrent"])
    assert '"intensity":4' in out
    assert '"is_concurrent":true' in out


def test_host_flag_suffix_skips_unrecognized_flag() -> None:
    """An unknown flag name matches no branch and is silently skipped."""
    from src.scripts.scenarios.augmentation.llm_agent import _host_flag_suffix

    ev = CalendarEvent(
        label="lunch",
        start_minutes=720,
        end_minutes=765,
        date=datetime.date(2026, 5, 4),
        is_concurrent=True,
        is_dividable=False,
        concurrent_with=[],
        intensity=1,
    )
    out = _host_flag_suffix(ev, host_flags=["mystery", "is_concurrent"])
    assert '"is_concurrent":true' in out
    assert "mystery" not in out


def test_build_context_summary_skips_categories_with_no_episodes() -> None:
    """An observed category with no episodes on any day in the week is omitted."""
    obs = ObservationConfig(contexts=["mood_emotion", "energy_state"])
    eps = [
        ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=datetime.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
        ),
    ]
    out = _build_context_summary(eps, [datetime.date(2026, 5, 4)], obs, catalog=None)
    assert "# CONTEXT (mood_emotion)" in out
    assert "# CONTEXT (energy_state)" not in out


class _StubCatalogEntry:
    def __init__(self, label: str | None) -> None:
        self.label = label


def test_build_context_summary_skips_label_when_catalog_label_equals_name() -> None:
    """A catalog label equal to the episode name does not produce a parenthesised suffix."""
    catalog = {"http://example.org/iri": _StubCatalogEntry("happy")}
    obs = ObservationConfig(contexts=["mood_emotion"])
    eps = [
        ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=datetime.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
            ontology_uri="http://example.org/iri",
        ),
    ]
    out = _build_context_summary(eps, [datetime.date(2026, 5, 4)], obs, catalog=catalog)
    assert "  - happy  08:00-09:00" in out
    assert "(happy)" not in out


def test_build_context_summary_full_detail_with_only_uri() -> None:
    """A full-detail episode with only a uri renders uri in brackets but no other extras."""
    obs = ObservationConfig(contexts=["mood_emotion"], context_detail="full")
    eps = [
        ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=datetime.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
            ontology_uri="http://example.org/iri",
        ),
    ]
    out = _build_context_summary(eps, [datetime.date(2026, 5, 4)], obs, catalog=None)
    assert "[uri=http://example.org/iri]" in out
    assert "dimension" not in out
    assert "polarity" not in out


def test_build_context_summary_full_detail_with_no_extras_emits_no_brackets() -> None:
    """A full-detail episode with no uri, dimension, or polarity has no bracket suffix."""
    obs = ObservationConfig(contexts=["mood_emotion"], context_detail="full")
    eps = [
        ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=datetime.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
        ),
    ]
    out = _build_context_summary(eps, [datetime.date(2026, 5, 4)], obs, catalog=None)
    assert "  - happy  08:00-09:00" in out
    assert "[" not in out


def test_get_context_catalog_returns_none_when_loader_raises(monkeypatch) -> None:
    """A failing catalog loader leaves the cached catalog as None."""
    from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter

    def _boom():
        raise RuntimeError("catalog boom")

    monkeypatch.setattr("src.scripts.persona.context.catalog.load_catalog", _boom)
    augmenter = LLMAugmenter(llm_pipeline=object())
    assert augmenter._get_context_catalog() is None
    assert augmenter._get_context_catalog() is None
