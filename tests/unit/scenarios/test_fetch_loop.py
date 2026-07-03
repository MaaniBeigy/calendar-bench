"""Unit tests for src.scripts.scenarios.task_generation.fetch_loop."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from src.scripts.scenarios.task_generation.fetch_loop import (
    FetchAttemptStats,
    FetchResult,
    fetch_grounded_unique,
)
from src.scripts.scenarios.task_generation.parser import extract_raw_task_dicts

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _FakePipeline:
    """Returns successive answers from a queue, recording each query.

    Pop the next answer at every `search()`; raise if the queue is
    empty so a misconfigured test fails loudly instead of silently
    looping on stale data.
    """

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.queries: list[str] = []

    def search(self, *, query_text: str, **_kwargs: Any) -> _FakeResponse:
        self.queries.append(query_text)
        if not self._answers:
            raise AssertionError("FakePipeline ran out of answers")
        return _FakeResponse(self._answers.pop(0))


def _make_resolver(known: set[str]):
    def resolves(uri: str) -> bool:
        return uri in known

    return resolves


def _payload(*uris: str) -> str:
    """Build a JSON-array payload from a list of URIs.

    Each URI becomes a fully-formed task dict so the parser is happy.
    """
    return json.dumps(
        [{"label": uri.rsplit("/", 1)[-1], "ontology_uri": uri} for uri in uris]
    )


def _renderer():
    """Default renderer used by tests; captures
    `(needed, names, accepted_uris, rejected_uris)` in the closure so
    assertions can inspect both blacklists the loop forwards."""

    captured: list[tuple[int, list[str], list[str], list[str]]] = []

    def render(
        needed: int,
        names: list[str],
        accepted_uris: list[str],
        rejected_uris: list[str],
    ) -> str:
        captured.append((needed, list(names), list(accepted_uris), list(rejected_uris)))
        return (
            f"PROMPT needed={needed} accepted={len(accepted_uris)} "
            f"rejected={len(rejected_uris)}"
        )

    render.captured = captured  # type: ignore[attr-defined]
    return render


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgValidation:
    def test_max_retries_below_one_raises(self):
        with pytest.raises(ValueError):
            fetch_grounded_unique(
                pipeline=_FakePipeline([]),
                num_tasks=3,
                max_retries=0,
                render_prompt=_renderer(),
                parse_response=extract_raw_task_dicts,
                uri_resolves=lambda _: True,
            )

    def test_num_tasks_below_one_raises(self):
        with pytest.raises(ValueError):
            fetch_grounded_unique(
                pipeline=_FakePipeline([]),
                num_tasks=0,
                max_retries=1,
                render_prompt=_renderer(),
                parse_response=extract_raw_task_dicts,
                uri_resolves=lambda _: True,
            )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_one_attempt_collects_n_grounded_unique_tasks(self):
        uris = [
            "https://ex.org/task/a",
            "https://ex.org/task/b",
            "https://ex.org/task/c",
        ]
        pipeline = _FakePipeline([_payload(*uris)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=3,
            max_retries=3,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver(set(uris)),
        )
        assert isinstance(result, FetchResult)
        assert [d["ontology_uri"] for d in result.accepted] == uris
        assert len(result.attempts) == 1
        assert result.attempts[0].accepted == 3
        assert result.attempts[0].fabricated_dropped == 0
        assert result.attempts[0].duplicate_dropped == 0
        assert not result.short_fetched

    def test_first_attempt_returns_extras_truncates_to_num_tasks(self):
        """LLM returns more than asked; the loop accepts the first N."""
        uris = [
            "https://ex.org/task/a",
            "https://ex.org/task/b",
            "https://ex.org/task/c",
        ]
        pipeline = _FakePipeline([_payload(*uris)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=2,
            max_retries=3,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver(set(uris)),
        )
        assert len(result.accepted) == 2
        assert result.accepted[0]["ontology_uri"] == uris[0]
        # Loop bailed before processing the third candidate.

    def test_returned_dicts_are_shallow_copies(self):
        uris = ["https://ex.org/task/a"]
        original = [{"label": "a", "ontology_uri": uris[0]}]
        pipeline = _FakePipeline([json.dumps(original)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=lambda _: original,
            uri_resolves=_make_resolver(set(uris)),
        )
        result.accepted[0]["new_field"] = "should not appear in original"
        assert "new_field" not in original[0]


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetryBehaviour:
    def test_one_fabrication_then_retry_fills_slot(self):
        good = "https://ex.org/task/good"
        bad = "https://ex.org/task/bad"  # not in resolver
        pipeline = _FakePipeline([_payload(bad), _payload(good)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=3,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({good}),
        )
        assert [d["ontology_uri"] for d in result.accepted] == [good]
        assert len(result.attempts) == 2
        assert result.attempts[0].fabricated_dropped == 1
        assert result.attempts[1].accepted == 1

    def test_duplicate_uri_dropped_then_retry_fills_slot(self):
        a = "https://ex.org/task/a"
        b = "https://ex.org/task/b"
        # First attempt: a + duplicate-of-a; second attempt: b
        pipeline = _FakePipeline([_payload(a, a), _payload(b)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=2,
            max_retries=3,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({a, b}),
        )
        assert [d["ontology_uri"] for d in result.accepted] == [a, b]
        assert result.attempts[0].duplicate_dropped == 1
        assert result.attempts[1].accepted == 1

    def test_blacklist_propagates_to_next_prompt(self):
        a = "https://ex.org/task/a"
        b = "https://ex.org/task/b"
        bad = "https://ex.org/task/bad"
        pipeline = _FakePipeline([_payload(a, bad), _payload(b)])
        renderer = _renderer()
        fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=2,
            max_retries=3,
            render_prompt=renderer,
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({a, b}),
        )
        # First call sees nothing yet; second call sees `a` in the
        # accepted-URIs blacklist AND `bad` in the rejected-URIs
        # blacklist; proves the loop forwards both lists.
        first = renderer.captured[0]  # type: ignore[attr-defined]
        second = renderer.captured[1]  # type: ignore[attr-defined]
        assert first[0] == 2 and first[2] == [] and first[3] == []
        assert second[0] == 1
        assert second[1] == ["a"]  # display_name
        assert second[2] == [a]  # accepted_uris
        assert second[3] == [bad]  # rejected_uris

    def test_rejected_uri_propagates_across_attempts(self):
        """A URI that fails verification in attempt 1 must appear in the
        `rejected_uris` block of attempt 2's prompt; closes the
        infinite-loop hole observed earlier."""
        good = "https://ex.org/task/good"
        bad = "https://ex.org/task/bad"
        pipeline = _FakePipeline([_payload(bad), _payload(good)])
        renderer = _renderer()
        fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=3,
            render_prompt=renderer,
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({good}),
        )
        first = renderer.captured[0]  # type: ignore[attr-defined]
        second = renderer.captured[1]  # type: ignore[attr-defined]
        assert first[3] == []  # nothing rejected yet
        assert second[3] == [bad]  # bad now blacklisted

    def test_duplicate_uri_added_to_rejected_blacklist(self):
        """When the LLM repeats an already-accepted URI, the duplicate
        is also added to `rejected_uris` so the next prompt forbids
        re-emitting it (the LLM is signalling it's stuck)."""
        a = "https://ex.org/task/a"
        b = "https://ex.org/task/b"
        # Attempt 1: accept a + see duplicate-of-a.  Attempt 2: emit b.
        pipeline = _FakePipeline([_payload(a, a), _payload(b)])
        renderer = _renderer()
        fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=2,
            max_retries=3,
            render_prompt=renderer,
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({a, b}),
        )
        second = renderer.captured[1]  # type: ignore[attr-defined]
        assert a in second[2]  # accepted
        assert a in second[3]  # also rejected (LLM repeated itself)

    def test_max_retries_exhausted_returns_short_list(self, caplog):
        bad = "https://ex.org/task/bad"
        # All attempts return only fabrications.
        pipeline = _FakePipeline([_payload(bad)] * 3)
        with caplog.at_level(logging.ERROR):
            result = fetch_grounded_unique(
                pipeline=pipeline,
                num_tasks=2,
                max_retries=3,
                render_prompt=_renderer(),
                parse_response=extract_raw_task_dicts,
                uri_resolves=_make_resolver(set()),
            )
        assert result.short_fetched is True
        assert result.accepted == []
        assert any("only 0/2" in r.getMessage() for r in caplog.records)

    def test_parse_failure_recorded_and_loop_continues(self):
        good = "https://ex.org/task/good"
        bad_payload = "this is not json"
        pipeline = _FakePipeline([bad_payload, _payload(good)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=2,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({good}),
        )
        assert result.attempts[0].parse_failed is True
        assert result.attempts[0].proposed == 0
        assert result.accepted[0]["ontology_uri"] == good

    def test_response_with_no_answer_attribute_is_treated_as_empty(self):
        class _BlankPipeline:
            def search(self, *, query_text: str, **_kw):
                class _R:
                    pass

                return _R()

        result = fetch_grounded_unique(
            pipeline=_BlankPipeline(),
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=lambda _: True,
        )
        assert result.short_fetched is True
        assert result.accepted == []

    def test_dict_with_missing_uri_counts_as_fabricated(self):
        """A dict with no `ontology_uri` key cannot be verified; it
        is treated as fabricated."""
        pipeline = _FakePipeline(
            [
                json.dumps(
                    [{"label": "no_uri"}, {"label": "blank", "ontology_uri": "  "}]
                )
            ]
        )
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=lambda _: True,
        )
        assert result.short_fetched is True
        assert result.attempts[0].fabricated_dropped == 2

    def test_non_dict_entry_in_parser_output_counts_as_fabricated(self):
        """Custom parser may return non-dict entries; loop treats each
        as fabricated without crashing."""
        pipeline = _FakePipeline(["irrelevant"])

        def parse(_text: str) -> list:
            return ["not-a-dict", 42, None]  # type: ignore[list-item]

        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=parse,
            uri_resolves=lambda _: True,
        )
        assert result.short_fetched is True
        assert result.attempts[0].fabricated_dropped == 3


# ---------------------------------------------------------------------------
# Debug writer hook
# ---------------------------------------------------------------------------


class TestDebugWriter:
    def test_writer_called_once_per_attempt(self):
        good = "https://ex.org/task/good"
        pipeline = _FakePipeline([_payload(good)])
        calls = []

        def writer(attempt, prompt, raw, summary):
            calls.append({"attempt": attempt, "summary": list(summary)})

        fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=2,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({good}),
            debug_writer=writer,
        )
        assert len(calls) == 1
        # Summary block carries the per-attempt counters in stable order.
        joined = " ".join(calls[0]["summary"])
        assert "attempt=1" in joined
        assert "accepted=1" in joined
        assert "accumulated=1/1" in joined

    def test_writer_omitted_works_too(self):
        good = "https://ex.org/task/good"
        pipeline = _FakePipeline([_payload(good)])
        # No debug_writer; must not crash.
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({good}),
        )
        assert len(result.accepted) == 1


class TestTopKForwarding:
    """`top_k` widens the GraphRAG retriever's candidate pool so the
    LLM sees more URIs to choose from.  When supplied, the loop
    forwards it via `retriever_config={"top_k": int}`; when omitted,
    `pipeline.search()` is called without that kwarg so the
    library's default is used."""

    class _CapturingPipeline:
        def __init__(self, answer: str) -> None:
            self._answer = answer
            self.calls: list[dict] = []

        def search(self, **kwargs):
            self.calls.append(dict(kwargs))

            class _R:
                pass

            r = _R()
            r.answer = self._answer
            return r

    def test_top_k_forwarded_when_provided(self):
        pipeline = self._CapturingPipeline(_payload("https://ex.org/task/a"))
        fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({"https://ex.org/task/a"}),
            top_k=20,
        )
        assert pipeline.calls[0].get("retriever_config") == {"top_k": 20}

    def test_top_k_omitted_when_none(self):
        pipeline = self._CapturingPipeline(_payload("https://ex.org/task/a"))
        fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({"https://ex.org/task/a"}),
        )
        assert "retriever_config" not in pipeline.calls[0]


class TestVerdictTupleSupport:
    """The validator may now return `(ok, reason)` tuples; the loop
    routes the per-URI count into the right bucket
    (fabricated_dropped / wrong_branch_dropped / wrong_level_dropped)."""

    def test_wrong_branch_routes_to_wrong_branch_bucket(self):
        good = "https://ex.org/task/good"
        off_domain = "https://ex.org/task/off-domain"
        pipeline = _FakePipeline([_payload(off_domain), _payload(good)])

        def resolver(uri: str):
            if uri == off_domain:
                return (False, "wrong_branch")
            if uri == good:
                return (True, "ok")
            return (False, "not_instance")

        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=2,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=resolver,
        )
        assert result.attempts[0].wrong_branch_dropped == 1
        assert result.attempts[0].fabricated_dropped == 0
        assert result.attempts[1].accepted == 1

    def test_wrong_level_routes_to_wrong_level_bucket(self):
        bad = "https://ex.org/task/wrong-level"
        good = "https://ex.org/task/good"
        pipeline = _FakePipeline([_payload(bad), _payload(good)])

        def resolver(uri: str):
            if uri == bad:
                return (False, "wrong_level")
            return (True, "ok")

        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=2,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=resolver,
        )
        assert result.attempts[0].wrong_level_dropped == 1

    def test_legacy_bool_resolver_still_works(self):
        """When the injected resolver returns a plain bool (legacy
        signature), the loop still treats it as ok/not_instance."""
        good = "https://ex.org/task/good"
        pipeline = _FakePipeline([_payload(good)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=lambda u: u == good,  # plain bool
        )
        assert result.accepted[0]["ontology_uri"] == good


class TestProposalsCapture:
    """`FetchAttemptStats.proposals` records every URI the LLM emitted
    in this attempt with its verdict; fed into the per-persona
    `tasks_proposed.json` log."""

    def test_proposals_capture_accepted_label(self):
        good = "https://ex.org/task/good"
        pipeline = _FakePipeline([_payload(good)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({good}),
        )
        assert len(result.attempts[0].proposals) == 1
        entry = result.attempts[0].proposals[0]
        assert entry["uri"] == good
        assert entry["verdict"] == "accepted"
        assert entry["label"] == "good"

    def test_proposals_capture_missing_uri(self):
        pipeline = _FakePipeline([json.dumps([{"label": "x"}])])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=lambda _u: True,
        )
        assert result.attempts[0].proposals[0]["verdict"] == "missing_uri"

    def test_proposals_capture_duplicate_verdict(self):
        a = "https://ex.org/task/a"
        pipeline = _FakePipeline([_payload(a, a)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=2,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({a}),
        )
        verdicts = [p["verdict"] for p in result.attempts[0].proposals]
        assert verdicts == ["accepted", "duplicate"]

    def test_proposals_capture_filter_verdicts(self):
        bad_branch = "https://ex.org/task/off-domain"
        bad_level = "https://ex.org/task/off-level"
        pipeline = _FakePipeline([_payload(bad_branch, bad_level)])

        def resolver(uri: str):
            if uri == bad_branch:
                return (False, "wrong_branch")
            return (False, "wrong_level")

        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=resolver,
        )
        verdicts = [p["verdict"] for p in result.attempts[0].proposals]
        assert verdicts == ["wrong_branch", "wrong_level"]


def test_fetch_attempt_stats_now_has_filter_counters():
    """The dataclass exposes the new wrong_branch / wrong_level /
    branch_unresolved_passes counters with default 0."""
    s = FetchAttemptStats(attempt=1)
    assert s.wrong_branch_dropped == 0
    assert s.wrong_level_dropped == 0
    assert s.branch_unresolved_passes == 0
    assert s.proposals == []


class TestBranchUnresolvedAcceptance:
    """The validator can return `(True, "ok_branch_unresolved")` for
    URIs accepted under graceful-degradation.  The loop must:

      * still ACCEPT the task (count it in `stats.accepted`),
      * bump `stats.branch_unresolved_passes` so the user can spot
        a runaway false-positive,
      * record the verdict as `"accepted_branch_unresolved"` in the
        per-URI proposals log so it's distinguishable from a strict
        accept.
    """

    def test_branch_unresolved_pass_counted_separately(self):
        good = "https://ex.org/task/good"
        pipeline = _FakePipeline([_payload(good)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=lambda _u: (True, "ok_branch_unresolved"),
        )
        assert len(result.accepted) == 1
        assert result.attempts[0].accepted == 1
        assert result.attempts[0].branch_unresolved_passes == 1
        assert (
            result.attempts[0].proposals[0]["verdict"] == "accepted_branch_unresolved"
        )

    def test_strict_ok_does_not_increment_branch_unresolved_counter(self):
        good = "https://ex.org/task/good"
        pipeline = _FakePipeline([_payload(good)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=lambda _u: (True, "ok"),
        )
        assert result.attempts[0].branch_unresolved_passes == 0
        assert result.attempts[0].proposals[0]["verdict"] == "accepted"

    def test_summary_lines_include_branch_unresolved_counter(self):
        good = "https://ex.org/task/good"
        pipeline = _FakePipeline([_payload(good)])
        captured = {}

        def writer(attempt, prompt, raw, summary):
            captured["summary"] = list(summary)

        fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=lambda _u: (True, "ok_branch_unresolved"),
            debug_writer=writer,
        )
        joined = " ".join(captured["summary"])
        assert "branch_unresolved_passes=1" in joined


# ---------------------------------------------------------------------------
# Dataclass plumbing
# ---------------------------------------------------------------------------


def test_fetch_attempt_stats_defaults():
    s = FetchAttemptStats(attempt=1)
    assert s.proposed == s.accepted == 0
    assert s.fabricated_dropped == s.duplicate_dropped == 0
    assert s.parse_failed is False
    assert s.wall_time_seconds == 0.0


def test_fetch_result_defaults():
    r = FetchResult()
    assert r.accepted == []
    assert r.attempts == []
    assert r.short_fetched is False


# ---------------------------------------------------------------------------
# Per-domain floor, cross-week exclusion, branch reporting
# ---------------------------------------------------------------------------


def _branch_resolver(branch_of: dict[str, str]):
    """Resolver returning `(ok, reason, branch)`; unknown URIs are rejected."""

    def resolves(uri: str):
        if uri in branch_of:
            return True, "ok", branch_of[uri]
        return False, "not_instance", None

    return resolves


class TestDomainFloor:
    def test_floor_pulls_in_under_represented_branch(self):
        a1, a2, a3, b1 = "u/a1", "u/a2", "u/a3", "u/b1"
        pipeline = _FakePipeline([_payload(a1, a2, a3, b1)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=3,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_branch_resolver({a1: "A", a2: "A", a3: "A", b1: "B"}),
            domain_floor={"A": 1, "B": 1},
        )
        accepted = [d["ontology_uri"] for d in result.accepted]
        assert b1 in accepted  # the floor pulled B in despite A flooding proposals
        assert len(accepted) == 3
        assert result.branch_counts == {"A": 2, "B": 1}

    def test_without_floor_accepts_in_proposal_order(self):
        a1, a2, a3, b1 = "u/a1", "u/a2", "u/a3", "u/b1"
        pipeline = _FakePipeline([_payload(a1, a2, a3, b1)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=3,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_branch_resolver({a1: "A", a2: "A", a3: "A", b1: "B"}),
        )
        accepted = [d["ontology_uri"] for d in result.accepted]
        assert accepted == [a1, a2, a3]
        assert b1 not in accepted


class TestExcludeUris:
    def test_prior_week_uris_are_treated_as_duplicates(self):
        a, b = "u/a", "u/b"
        pipeline = _FakePipeline([_payload(a, b)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=_make_resolver({a, b}),
            exclude_uris={a},
        )
        assert [d["ontology_uri"] for d in result.accepted] == [b]
        assert result.attempts[0].duplicate_dropped == 1


class TestResolverArities:
    def test_two_tuple_resolver_supported(self):
        a = "u/a"
        pipeline = _FakePipeline([_payload(a)])
        result = fetch_grounded_unique(
            pipeline=pipeline,
            num_tasks=1,
            max_retries=1,
            render_prompt=_renderer(),
            parse_response=extract_raw_task_dicts,
            uri_resolves=lambda uri: (True, "ok"),
        )
        assert [d["ontology_uri"] for d in result.accepted] == [a]
        assert result.branch_counts == {}
