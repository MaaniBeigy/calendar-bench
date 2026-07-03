"""Unit tests for src.scripts.scenarios.task_generation.ontology_bridge."""

from __future__ import annotations

from typing import Any

import pytest

from src.scripts.scenarios.task_generation.ontology_bridge import (
    MENTAL_WELLBEING_URI,
    NUTRITION_URI,
    PHYSICAL_ACTIVITY_URI,
    _uri_lookup_variants,
    enrich_task_from_ontology,
    fetch_task_properties,
    resolve_task_branch,
    uri_resolves,
)

# ---------------------------------------------------------------------------
# Mock Neo4j session helpers
# ---------------------------------------------------------------------------


class _MockRecord:
    """Minimal mock of a neo4j Record."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class _MockResult:
    """Wraps a single record result (or None) to mimic neo4j Result.single()."""

    def __init__(self, record: _MockRecord | None) -> None:
        self._record = record

    def single(self) -> _MockRecord | None:
        return self._record


class _MockSession:
    """Returns pre-defined results for successive run() calls."""

    def __init__(self, *records: _MockRecord | None) -> None:
        self._queue = list(records)
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    def run(self, _query: str, **kwargs: Any) -> _MockResult:
        self.calls.append(dict(kwargs))
        record = self._queue[self._idx] if self._idx < len(self._queue) else None
        self._idx += 1
        return _MockResult(record)


class _UriRoutedSession:
    """Returns the record keyed by the URI passed to `run`; lets a
    test assert which variant from the lookup ladder actually hit."""

    def __init__(self, mapping: dict[str, _MockRecord | None]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []

    def run(self, _query: str, **kwargs: Any) -> _MockResult:
        uri = kwargs.get("uri")
        self.calls.append(uri)
        return _MockResult(self._mapping.get(uri))


def _record(**kwargs: Any) -> _MockRecord:
    return _MockRecord(kwargs)


# ---------------------------------------------------------------------------
# fetch_task_properties
# ---------------------------------------------------------------------------


class TestFetchTaskProperties:
    def test_all_properties_returned(self):
        session = _MockSession(
            _record(is_dividable=True, is_concurrent=False, duration_minutes=30)
        )
        result = fetch_task_properties("https://example.org/task/a", session)
        assert result["is_dividable"] is True
        assert result["is_concurrent"] is False
        assert result["duration_minutes"] == 30

    def test_none_properties_excluded(self):
        """Properties that are None must not appear in the returned dict."""
        session = _MockSession(
            _record(is_dividable=True, is_concurrent=None, duration_minutes=None)
        )
        result = fetch_task_properties("https://example.org/task/b", session)
        assert "is_dividable" in result
        assert "is_concurrent" not in result
        assert "duration_minutes" not in result

    def test_all_none_gives_empty_dict(self):
        session = _MockSession(
            _record(is_dividable=None, is_concurrent=None, duration_minutes=None)
        )
        result = fetch_task_properties("https://example.org/task/c", session)
        assert result == {}

    def test_uri_not_found_gives_empty_dict(self):
        session = _MockSession(None)  # single() returns None
        result = fetch_task_properties("https://example.org/unknown", session)
        assert result == {}

    def test_duration_minutes_integer(self):
        session = _MockSession(
            _record(is_dividable=False, is_concurrent=True, duration_minutes=45)
        )
        result = fetch_task_properties("https://example.org/task/d", session)
        assert result["duration_minutes"] == 45

    def test_display_name_returned_when_present(self):
        """`display_name` flows through verbatim when the ontology has one
        (the Cypher query coalesces `displayName` and `rdfs:label`)."""
        session = _MockSession(
            _record(
                is_dividable=True,
                is_concurrent=None,
                duration_minutes=None,
                display_name="Mindful Eating 🍽️",
                description=None,
            )
        )
        result = fetch_task_properties("https://example.org/task/me", session)
        assert result["display_name"] == "Mindful Eating 🍽️"

    def test_description_returned_when_present(self):
        """`description` (from `rdfs:comment`) flows through verbatim,
        emojis included."""
        session = _MockSession(
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name=None,
                description="Eat slowly and notice every bite. 🍽️",
            )
        )
        result = fetch_task_properties("https://example.org/task/me", session)
        assert result["description"] == "Eat slowly and notice every bite. 🍽️"

    def test_empty_string_display_name_excluded(self):
        """An empty-string `display_name` is not a meaningful authored
        value; it must not appear in the returned dict."""
        session = _MockSession(
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name="",
                description="",
            )
        )
        result = fetch_task_properties("https://example.org/task/empty", session)
        assert "display_name" not in result
        assert "description" not in result

    def test_whitespace_only_display_name_excluded(self):
        """A whitespace-only `display_name` is treated like missing."""
        session = _MockSession(
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name="   ",
                description=None,
            )
        )
        result = fetch_task_properties("https://example.org/task/ws", session)
        assert "display_name" not in result

    def test_emoji_display_name_flows_through_verbatim(self):
        """An emoji-bearing `display_name` (sourced server-side via the
        Cypher `coalesce(displayName, title, label)`) reaches the
        Python layer untouched; emojis and all."""
        session = _MockSession(
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name="Blend a Smoothie 🥤",
                description="What about blending coconut water with frozen fruit.",
            )
        )
        result = fetch_task_properties("https://example.org/task/blend", session)
        assert result["display_name"] == "Blend a Smoothie 🥤"
        assert (
            result["description"]
            == "What about blending coconut water with frozen fruit."
        )

    def test_emoji_description_flows_through_verbatim(self):
        """An emoji-bearing `description` (sourced server-side via the
        Cypher `coalesce(description, comment)`) reaches the Python
        layer untouched."""
        session = _MockSession(
            _record(
                is_dividable=None,
                is_concurrent=None,
                duration_minutes=None,
                display_name=None,
                description="Eat slowly and notice every bite. 🍽️",
            )
        )
        result = fetch_task_properties(
            "https://example.org/task/mindful-eating", session
        )
        assert result["description"] == "Eat slowly and notice every bite. 🍽️"


# ---------------------------------------------------------------------------
# resolve_task_branch
# ---------------------------------------------------------------------------


class TestResolveTaskBranch:
    def test_physical_activity_branch_returned(self):
        session = _MockSession(_record(branch=PHYSICAL_ACTIVITY_URI, depth=1))
        result = resolve_task_branch("https://example.org/task/run", session)
        assert result == PHYSICAL_ACTIVITY_URI

    def test_nutrition_branch_returned(self):
        session = _MockSession(_record(branch=NUTRITION_URI, depth=2))
        result = resolve_task_branch("https://example.org/task/meal", session)
        assert result == NUTRITION_URI

    def test_mental_wellbeing_branch_returned(self):
        session = _MockSession(_record(branch=MENTAL_WELLBEING_URI, depth=1))
        result = resolve_task_branch("https://example.org/task/meditate", session)
        assert result == MENTAL_WELLBEING_URI

    def test_no_result_returns_none(self):
        """When single() returns None there is no branch in the graph."""
        session = _MockSession(None)
        result = resolve_task_branch("https://example.org/unknown", session)
        assert result is None

    def test_result_with_none_branch_returns_none(self):
        """When the record exists but branch is None (OPTIONAL MATCH missed)."""
        session = _MockSession(_record(branch=None, depth=0))
        result = resolve_task_branch("https://example.org/task/ambiguous", session)
        assert result is None


# ---------------------------------------------------------------------------
# default_epochs_for_branch
# ---------------------------------------------------------------------------


# enrich_task_from_ontology
# ---------------------------------------------------------------------------


class TestEnrichTaskFromOntology:
    def test_combines_properties_and_difficulty(self):
        """enrich_task_from_ontology calls fetch_task_properties then fetch_difficulty_level."""
        session = _MockSession(
            _record(is_dividable=True, is_concurrent=None, duration_minutes=20),
            None,  # fetch_difficulty_level returns no Level token
        )
        result = enrich_task_from_ontology("https://example.org/task/run", session)
        assert result["is_dividable"] is True
        assert "is_concurrent" not in result
        assert result["duration_minutes"] == 20
        assert result["difficulty_level"] == 0

    def test_no_properties_returns_only_difficulty_level(self):
        session = _MockSession(
            _record(is_dividable=None, is_concurrent=None, duration_minutes=None),
            None,
        )
        result = enrich_task_from_ontology("https://example.org/task/meditate", session)
        assert result == {"difficulty_level": 0}

    def test_unknown_uri_returns_difficulty_level_only(self):
        session = _MockSession(None, None)
        result = enrich_task_from_ontology("https://example.org/unknown", session)
        assert result == {"difficulty_level": 0}

    def test_difficulty_level_extracted(self):
        session = _MockSession(
            _record(is_dividable=True, is_concurrent=False, duration_minutes=30),
            _record(token="Level3"),
        )
        result = enrich_task_from_ontology("https://example.org/task/cardio", session)
        assert result["difficulty_level"] == 3


# ---------------------------------------------------------------------------
# _uri_lookup_variants
# ---------------------------------------------------------------------------


class TestUriLookupVariants:
    def test_kebab_case_with_task_segment_returns_one_variant(self):
        """A canonical authored URI is the only variant; its three
        normalizations all collapse back to itself."""
        uri = "https://w3id.org/calendar-bench/health/task/blend-a-smoothie"
        assert _uri_lookup_variants(uri) == [uri]

    def test_snake_case_returns_kebab_variant(self):
        uri = "https://w3id.org/calendar-bench/health/task/blend_a_smoothie"
        variants = _uri_lookup_variants(uri)
        assert variants[0] == uri
        assert (
            "https://w3id.org/calendar-bench/health/task/blend-a-smoothie" in variants
        )

    def test_missing_task_segment_returns_task_variant(self):
        uri = "https://w3id.org/calendar-bench/health/blend-a-smoothie"
        variants = _uri_lookup_variants(uri)
        assert variants[0] == uri
        assert (
            "https://w3id.org/calendar-bench/health/task/blend-a-smoothie" in variants
        )

    def test_snake_case_without_task_returns_combined_variant(self):
        """The canonical LLM mistake; snake_case AND missing /task/ -
        is reachable on the fourth variant."""
        uri = "https://w3id.org/calendar-bench/health/blend_a_smoothie"
        variants = _uri_lookup_variants(uri)
        assert variants[0] == uri
        assert (
            "https://w3id.org/calendar-bench/health/task/blend-a-smoothie" in variants
        )

    def test_variants_are_deduplicated(self):
        """A URI with no underscores and a `/task/` segment generates
        duplicate normalization candidates; the helper returns a
        deduplicated list."""
        uri = "https://w3id.org/calendar-bench/health/task/yoga"
        variants = _uri_lookup_variants(uri)
        assert variants == [uri]

    def test_uri_without_slash_returned_as_singleton(self):
        """Pathological input (no path separator) is returned untouched."""
        uri = "blob"
        assert _uri_lookup_variants(uri) == [uri]


# ---------------------------------------------------------------------------
# fetch_task_properties; URI normalization ladder
# ---------------------------------------------------------------------------


_KEBAB_URI = "https://w3id.org/calendar-bench/health/task/blend-a-smoothie"


def _props_record() -> _MockRecord:
    """A populated record returned by the URI-normalized lookup."""
    return _record(
        is_dividable=False,
        is_concurrent=False,
        duration_minutes=30,
        display_name="Blend a Smoothie 🥤",
        description="What about blending coconut water with frozen fruit.",
    )


class TestFetchTaskPropertiesUriNormalization:
    def test_literal_uri_hits_no_extra_queries(self):
        """When the verbatim URI resolves, no further variants are tried."""
        session = _UriRoutedSession({_KEBAB_URI: _props_record()})
        result = fetch_task_properties(_KEBAB_URI, session)
        assert result["display_name"] == "Blend a Smoothie 🥤"
        assert session.calls == [_KEBAB_URI]

    def test_snake_case_misses_then_kebab_hits(self):
        """A snake_case URI under `/task/` falls through to the
        kebab-case variant on the second query."""
        snake_uri = "https://w3id.org/calendar-bench/health/task/blend_a_smoothie"
        session = _UriRoutedSession(
            {
                snake_uri: None,
                _KEBAB_URI: _props_record(),
            }
        )
        result = fetch_task_properties(snake_uri, session)
        assert result["display_name"] == "Blend a Smoothie 🥤"
        assert session.calls == [snake_uri, _KEBAB_URI]

    def test_missing_task_segment_misses_then_task_variant_hits(self):
        """A URI missing `/task/` falls through to the variant with
        `/task/` injected."""
        no_task_uri = "https://w3id.org/calendar-bench/health/blend-a-smoothie"
        session = _UriRoutedSession(
            {
                no_task_uri: None,
                _KEBAB_URI: _props_record(),
            }
        )
        result = fetch_task_properties(no_task_uri, session)
        assert result["display_name"] == "Blend a Smoothie 🥤"
        assert _KEBAB_URI in session.calls

    def test_combined_snake_and_missing_task_segment_hits(self):
        """The canonical LLM mistake (snake_case AND missing `/task/`)
        is rescued by the combined-transformation variant."""
        broken_uri = "https://w3id.org/calendar-bench/health/blend_a_smoothie"
        session = _UriRoutedSession(
            {
                broken_uri: None,
                "https://w3id.org/calendar-bench/health/blend-a-smoothie": None,
                "https://w3id.org/calendar-bench/health/task/blend_a_smoothie": None,
                _KEBAB_URI: _props_record(),
            }
        )
        result = fetch_task_properties(broken_uri, session)
        assert result["display_name"] == "Blend a Smoothie 🥤"
        assert _KEBAB_URI in session.calls

    def test_all_variants_miss_returns_empty_dict(self):
        """When none of the four variants resolves, an empty dict is
        returned; no exception raised."""
        broken_uri = "https://w3id.org/calendar-bench/health/no_such_task"
        session = _UriRoutedSession({})
        result = fetch_task_properties(broken_uri, session)
        assert result == {}

    def test_variant_returning_record_with_only_nulls_keeps_walking(self):
        """A variant whose record exists but has every property null
        gives an empty props dict; the loop continues to the next
        variant looking for real data."""
        snake_uri = "https://w3id.org/calendar-bench/health/task/blend_a_smoothie"
        empty_record = _record(
            is_dividable=None,
            is_concurrent=None,
            duration_minutes=None,
            display_name=None,
            description=None,
        )
        session = _UriRoutedSession(
            {
                snake_uri: empty_record,
                _KEBAB_URI: _props_record(),
            }
        )
        result = fetch_task_properties(snake_uri, session)
        assert result["display_name"] == "Blend a Smoothie 🥤"


class TestResolveTaskBranchUriNormalization:
    def test_snake_case_uri_resolves_via_kebab_variant(self):
        snake_uri = "https://w3id.org/calendar-bench/health/task/run_for_5_minutes"
        kebab_uri = "https://w3id.org/calendar-bench/health/task/run-for-5-minutes"
        session = _UriRoutedSession(
            {
                snake_uri: None,
                kebab_uri: _record(branch=PHYSICAL_ACTIVITY_URI, depth=2),
            }
        )
        assert resolve_task_branch(snake_uri, session) == PHYSICAL_ACTIVITY_URI

    def test_missing_task_segment_resolves_via_injected_variant(self):
        no_task_uri = "https://w3id.org/calendar-bench/health/run-for-5-minutes"
        kebab_uri = "https://w3id.org/calendar-bench/health/task/run-for-5-minutes"
        session = _UriRoutedSession(
            {
                no_task_uri: None,
                kebab_uri: _record(branch=NUTRITION_URI, depth=1),
            }
        )
        assert resolve_task_branch(no_task_uri, session) == NUTRITION_URI

    def test_record_with_null_branch_keeps_walking(self):
        """A variant whose record exists but has a null branch is
        treated as a miss; the loop continues."""
        snake_uri = "https://w3id.org/calendar-bench/health/task/x_y_z"
        kebab_uri = "https://w3id.org/calendar-bench/health/task/x-y-z"
        session = _UriRoutedSession(
            {
                snake_uri: _record(branch=None, depth=0),
                kebab_uri: _record(branch=MENTAL_WELLBEING_URI, depth=1),
            }
        )
        assert resolve_task_branch(snake_uri, session) == MENTAL_WELLBEING_URI

    def test_all_variants_miss_returns_none(self):
        broken_uri = "https://w3id.org/calendar-bench/health/no_such_task"
        session = _UriRoutedSession({})
        assert resolve_task_branch(broken_uri, session) is None


# ---------------------------------------------------------------------------
# uri_resolves; fabrication detection helper
# ---------------------------------------------------------------------------


def _hit_record() -> _MockRecord:
    """Synthetic record returned by the EXISTS query for a present node."""
    return _MockRecord({"hit": 1})


class TestUriResolves:
    def test_returns_true_when_first_variant_hits(self):
        canonical = "https://w3id.org/calendar-bench/health/task/blend-a-smoothie"
        session = _UriRoutedSession({canonical: _hit_record()})
        assert uri_resolves(canonical, session) is True

    def test_returns_true_when_later_variant_hits(self):
        """Snake-case URI from the LLM resolves via the kebab + /task/
        rewrite; not the literal URI itself."""
        snake = "https://w3id.org/calendar-bench/health/blend_a_smoothie"
        kebab = "https://w3id.org/calendar-bench/health/task/blend-a-smoothie"
        session = _UriRoutedSession({snake: None, kebab: _hit_record()})
        assert uri_resolves(snake, session) is True

    def test_returns_false_when_no_variant_hits(self):
        bogus = "https://w3id.org/calendar-bench/health/task/prepare-a-healthy-snack"
        session = _UriRoutedSession({})
        assert uri_resolves(bogus, session) is False

    def test_short_circuits_after_first_hit(self):
        """`uri_resolves` stops issuing further queries once a variant
        matches; keeps Neo4j round-trips at one for the common case."""
        canonical = "https://w3id.org/calendar-bench/health/task/x"
        session = _MockSession(_hit_record(), _hit_record(), _hit_record())
        uri_resolves(canonical, session)
        assert session._idx == 1

    def test_walks_full_ladder_on_miss(self):
        """All variants are tried before returning False."""
        snake = "https://w3id.org/calendar-bench/health/foo_bar"
        # 4 variants to 4 queries when none hit
        session = _MockSession(None, None, None, None)
        assert uri_resolves(snake, session) is False
        assert session._idx == 4

    def test_exists_query_walks_full_ladder_when_no_property_match(self):
        """If a node exists but lacks instance properties, the EXISTS
        query returns no row, so the loop walks the next variant."""
        snake = "https://w3id.org/calendar-bench/health/foo_bar"
        # Three of four variants miss; the last one returns a hit.
        session = _MockSession(None, None, None, _MockRecord({"hit": 1}))
        assert uri_resolves(snake, session) is True
        assert session._idx == 4

    def test_exists_query_filters_to_instance_only_properties(self):
        """The EXISTS query must require an *instance-only* property to
        be present so OWL class nodes (which carry only `rdfs:label`
        + `rdfs:comment`) are rejected.

        This guards against the `health/HealthTask` regression where the LLM picked the top-level class URI and
        the bridge accepted it because the node existed.
        """

        class _CapturingSession:
            """Records every Cypher query passed to `run`."""

            def __init__(self) -> None:
                self.queries: list[str] = []

            def run(self, query: str, **_kwargs):
                self.queries.append(query)
                return _MockResult(None)

        session = _CapturingSession()
        uri_resolves("https://ex.org/whatever", session)
        joined = " ".join(session.queries)
        # The query must check at least one instance-only property; the
        # exact set lives in `_EXISTS_QUERY`.
        assert "n.title" in joined
        assert "n.estimatedDurationMinutes" in joined
        assert "n.isConcurrent" in joined
        assert "n.isDividable" in joined
        assert "n.displayName" in joined
        # And explicitly requires non-null on those properties.
        assert "IS NOT NULL" in joined


# ---------------------------------------------------------------------------
# task_matches_level
# ---------------------------------------------------------------------------


class TestTaskMatchesLevel:
    def test_no_levels_returns_true(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            task_matches_level,
        )

        # When the scenario doesn't filter on difficulty, the helper is
        # a pass-through.
        assert task_matches_level("https://ex.org/task/x", _MockSession(), []) is True
        assert task_matches_level("https://ex.org/task/x", _MockSession(), None) is True

    def test_returns_true_when_ancestor_ends_with_level_token(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            task_matches_level,
        )

        # Mock returns a hit for the first variant.
        session = _MockSession(_MockRecord({"hit": 1}))
        assert task_matches_level("https://ex.org/task/x", session, ["Level1"]) is True

    def test_returns_false_when_no_variant_has_matching_level(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            task_matches_level,
        )

        # Every variant misses (single canonical URI to 1 query).
        session = _MockSession(None)
        assert task_matches_level("https://ex.org/task/x", session, ["Level1"]) is False

    def test_query_carries_level_tokens(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            task_matches_level,
        )

        class _CapturingSession:
            def __init__(self):
                self.calls = []

            def run(self, query, **kwargs):
                self.calls.append({"query": query, "kwargs": dict(kwargs)})
                return _MockResult(None)

        session = _CapturingSession()
        task_matches_level("https://ex.org/task/x", session, ["Level1", "Level2"])
        # The first variant's query must carry the supplied tokens.
        assert session.calls[0]["kwargs"]["tokens"] == ["Level1", "Level2"]
        # And ENDS WITH must appear in the cypher (the match operator).
        assert "ENDS WITH" in session.calls[0]["query"]


# ---------------------------------------------------------------------------
# validate_task_uri
# ---------------------------------------------------------------------------


class TestValidateTaskUri:
    def test_returns_ok_when_no_filters_and_uri_resolves(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            validate_task_uri,
        )

        session = _MockSession(_MockRecord({"hit": 1}))
        ok, reason = validate_task_uri("https://ex.org/task/x", session)
        assert ok is True
        assert reason == "ok"

    def test_returns_not_instance_when_uri_misses(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            validate_task_uri,
        )

        # Every EXISTS variant misses to uri_resolves returns False.
        session = _MockSession(None, None, None, None)
        ok, reason = validate_task_uri("https://ex.org/health/foo_bar", session)
        assert ok is False
        assert reason == "not_instance"

    def test_returns_wrong_branch_when_branch_off_filter(self):
        """`uri_resolves` passes (instance) but `resolve_task_branch`
        returns a URI not in `allowed_branches` to wrong_branch."""
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            MENTAL_WELLBEING_URI,
            NUTRITION_URI,
            PHYSICAL_ACTIVITY_URI,
            validate_task_uri,
        )

        # First call: EXISTS to hit.  Then BRANCH walk returns mental
        # wellbeing; not in the allowed set.
        session = _MockSession(
            _MockRecord({"hit": 1}),
            _MockRecord({"branch": MENTAL_WELLBEING_URI, "depth": 1}),
        )
        ok, reason = validate_task_uri(
            "https://ex.org/task/x",
            session,
            allowed_branches={NUTRITION_URI, PHYSICAL_ACTIVITY_URI},
        )
        assert ok is False
        assert reason == "wrong_branch"

    def test_returns_wrong_level_when_level_unmatched(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            NUTRITION_URI,
            validate_task_uri,
        )

        session = _MockSession(
            _MockRecord({"hit": 1}),
            _MockRecord({"branch": NUTRITION_URI, "depth": 1}),
            None,  # LEVEL query misses on every variant
            None,
            None,
            None,
        )
        ok, reason = validate_task_uri(
            "https://ex.org/task/x",
            session,
            allowed_branches={NUTRITION_URI},
            allowed_levels=["Level1"],
        )
        assert ok is False
        assert reason == "wrong_level"

    def test_returns_ok_when_all_filters_pass(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            NUTRITION_URI,
            validate_task_uri,
        )

        session = _MockSession(
            _MockRecord({"hit": 1}),  # EXISTS
            _MockRecord({"branch": NUTRITION_URI, "depth": 1}),  # BRANCH
            _MockRecord({"hit": 1}),  # LEVEL
        )
        ok, reason = validate_task_uri(
            "https://ex.org/task/x",
            session,
            allowed_branches={NUTRITION_URI},
            allowed_levels=["Level1"],
        )
        assert ok is True
        assert reason == "ok"

    def test_branch_unknown_returns_ok_branch_unresolved(self):
        """Graceful degradation: when `resolve_task_branch` returns
        `None` (no ancestor matches a known branch), the URI is
        ACCEPTED with the new verdict `ok_branch_unresolved` rather
        than rejected as `wrong_branch`.  This avoids locking out
        every task when the live Neo4j graph cannot traverse
        `rdf:type to SUBCLASSOF` chains."""
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            NUTRITION_URI,
            validate_task_uri,
        )

        # EXISTS hits, BRANCH walk returns None, no level filter set.
        session = _MockSession(
            _MockRecord({"hit": 1}),
            None,  # branch query returns no row
        )
        ok, reason = validate_task_uri(
            "https://ex.org/task/x",
            session,
            allowed_branches={NUTRITION_URI},
        )
        assert ok is True
        assert reason == "ok_branch_unresolved"

    def test_branch_unknown_still_runs_level_check(self):
        """Even with the graceful-degradation branch fallback, the
        level filter is applied strictly: a URI whose level
        cannot be matched is rejected."""
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            NUTRITION_URI,
            validate_task_uri,
        )

        session = _MockSession(
            _MockRecord({"hit": 1}),  # EXISTS
            None,  # BRANCH unresolved
            None,  # LEVEL miss across variants
            None,
            None,
            None,
        )
        ok, reason = validate_task_uri(
            "https://ex.org/task/x",
            session,
            allowed_branches={NUTRITION_URI},
            allowed_levels=["Level1"],
        )
        assert ok is False
        assert reason == "wrong_level"

    def test_branch_query_walks_subclassof_via_label(self):
        """The branch query must walk SUBCLASSOF from a class node
        whose URI suffix matches one of the instance's labels -
        because n10s on the live graph is configured with
        `handleRDFTypes = LABELS`: an instance's `rdf:type` class
        is materialised as a Neo4j *label*, NOT as an `RDFTYPE`
        relationship.  The previous `RDFTYPE|SUBCLASSOF` walk found
        zero edges from any instance and silently returned `None`
        for every `resolve_task_branch` call, which then bucketed
        every URI into the `ok_branch_unresolved` graceful path or
        the `wrong_level` rejection."""

        class _CapturingSession:
            def __init__(self):
                self.queries: list[str] = []

            def run(self, query, **kwargs):
                self.queries.append(query)
                return _MockResult(None)

        session = _CapturingSession()
        resolve_task_branch("https://ex.org/task/x", session)
        joined = " ".join(session.queries)
        assert "labels(n) AS _il" in joined
        assert "SUBCLASSOF*0..10" in joined
        assert "split(cls.uri, '/')[-1] IN _il" in joined
        # The broken pattern must NOT be re-introduced.
        assert "RDFTYPE" not in joined

    def test_level_query_walks_subclassof_via_label(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            task_matches_level,
        )

        class _CapturingSession:
            def __init__(self):
                self.queries: list[str] = []

            def run(self, query, **kwargs):
                self.queries.append(query)
                return _MockResult(None)

        session = _CapturingSession()
        task_matches_level("https://ex.org/task/x", session, ["Level1"])
        joined = " ".join(session.queries)
        assert "labels(n) AS _il" in joined
        assert "SUBCLASSOF*0..10" in joined
        assert "split(cls.uri, '/')[-1] IN _il" in joined
        assert "RDFTYPE" not in joined

    def test_branch_uris_by_local_name_covers_three_canonical_branches(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            BRANCH_URIS_BY_LOCAL_NAME,
            MENTAL_WELLBEING_URI,
            NUTRITION_URI,
            PHYSICAL_ACTIVITY_URI,
        )

        assert (
            BRANCH_URIS_BY_LOCAL_NAME["PhysicalActivityTask"] == PHYSICAL_ACTIVITY_URI
        )
        assert BRANCH_URIS_BY_LOCAL_NAME["NutritionTask"] == NUTRITION_URI
        assert BRANCH_URIS_BY_LOCAL_NAME["MentalWellbeingTask"] == MENTAL_WELLBEING_URI


# ---------------------------------------------------------------------------
# Context-dictionary lookups
# ---------------------------------------------------------------------------


class _CtxRecord:
    """Mock Neo4j Record supporting both subscript and .get()."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class _CtxResult:
    """Mock Neo4j Result that is both iterable and supports .single()."""

    def __init__(self, records: list[_CtxRecord]) -> None:
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self) -> _CtxRecord | None:
        return self._records[0] if self._records else None


class _CtxSession:
    """Returns pre-defined Context-query results in order."""

    def __init__(self, *results: list[_CtxRecord]) -> None:
        self._results = [_CtxResult(list(r)) for r in results]
        self._idx = 0
        self.queries: list[str] = []
        self.params: list[dict[str, Any]] = []

    def run(self, query: str, **kwargs: Any) -> _CtxResult:
        self.queries.append(query)
        self.params.append(dict(kwargs))
        out = (
            self._results[self._idx]
            if self._idx < len(self._results)
            else _CtxResult([])
        )
        self._idx += 1
        return out


class TestContextIriExists:
    def test_returns_true_when_iri_present(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            context_iri_exists,
        )

        session = _CtxSession([_CtxRecord({"hit": 1})])
        assert context_iri_exists("http://example.org/x", session) is True
        assert session.params[0] == {"iri": "http://example.org/x"}

    def test_returns_false_when_iri_missing(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            context_iri_exists,
        )

        session = _CtxSession([])
        assert context_iri_exists("http://example.org/missing", session) is False


class TestFetchContextEntry:
    def test_full_entry_round_trips(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            fetch_context_entry,
        )

        row = _CtxRecord(
            {
                "slug": "happiness",
                "source": "MFOEM",
                "label": "happiness",
                "definition": "positive affect",
                "polarity": None,
                "instrument": None,
                "comment": None,
                "category": "mood_emotion",
            }
        )
        session = _CtxSession([row])
        out = fetch_context_entry(
            "http://purl.obolibrary.org/obo/MFOEM_000042", session
        )
        assert out["slug"] == "happiness"
        assert out["category"] == "mood_emotion"
        assert "polarity" not in out
        assert "instrument" not in out

    def test_missing_iri_returns_empty_dict(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            fetch_context_entry,
        )

        session = _CtxSession([])
        assert fetch_context_entry("http://example.org/none", session) == {}


class TestFetchContextCategories:
    def test_returns_frozenset_of_slugs(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            fetch_context_categories,
        )

        session = _CtxSession(
            [
                _CtxRecord({"slug": "mood_emotion"}),
                _CtxRecord({"slug": "stress"}),
                _CtxRecord({"slug": "location"}),
            ]
        )
        cats = fetch_context_categories(session)
        assert cats == frozenset({"mood_emotion", "stress", "location"})

    def test_skips_rows_with_null_slug(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            fetch_context_categories,
        )

        session = _CtxSession(
            [
                _CtxRecord({"slug": "mood_emotion"}),
                _CtxRecord({"slug": None}),
                _CtxRecord({"slug": "stress"}),
            ]
        )
        assert fetch_context_categories(session) == frozenset(
            {"mood_emotion", "stress"}
        )


class TestFetchAllContextEntries:
    def test_emits_one_dict_per_row_dropping_nulls(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            fetch_all_context_entries,
        )

        rows = [
            _CtxRecord(
                {
                    "iri": "http://example.org/a",
                    "slug": "x",
                    "source": "TEST",
                    "label": "X",
                    "definition": "d",
                    "polarity": None,
                    "instrument": None,
                    "comment": None,
                    "category": "cat",
                }
            ),
            _CtxRecord(
                {
                    "iri": "http://example.org/b",
                    "slug": "y",
                    "source": "TEST",
                    "label": "Y",
                    "definition": None,
                    "polarity": "high",
                    "instrument": "BFI",
                    "comment": "c",
                    "category": "cat",
                }
            ),
        ]
        session = _CtxSession(rows)
        out = fetch_all_context_entries(session)
        assert len(out) == 2
        assert out[0]["iri"] == "http://example.org/a"
        assert "polarity" not in out[0]
        assert out[1]["polarity"] == "high"
        assert out[1]["comment"] == "c"

    def test_empty_graph_returns_empty_list(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            fetch_all_context_entries,
        )

        session = _CtxSession([])
        assert fetch_all_context_entries(session) == []


class TestContextCategoryBase:
    def test_constant_matches_ontology_namespace(self):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            CONTEXT_CATEGORY_BASE,
        )

        assert CONTEXT_CATEGORY_BASE == (
            "https://w3id.org/calendar-bench/context/category/"
        )


# ---------------------------------------------------------------------------
# validate_task_uri_groups
# ---------------------------------------------------------------------------


class TestValidateTaskUriGroups:
    def _patch(self, monkeypatch, *, resolves=True, branch=None, level=0):
        import src.scripts.scenarios.task_generation.ontology_bridge as ob

        monkeypatch.setattr(ob, "uri_resolves", lambda uri, s: resolves)
        monkeypatch.setattr(ob, "resolve_task_branch", lambda uri, s: branch)
        monkeypatch.setattr(ob, "fetch_difficulty_level", lambda uri, s: level)

    def test_not_instance_short_circuits(self, monkeypatch):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            validate_task_uri_groups,
        )

        self._patch(monkeypatch, resolves=False)
        assert validate_task_uri_groups("u", None, groups=[]) == (
            False,
            "not_instance",
            None,
        )

    def test_no_filter_accepts(self, monkeypatch):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            validate_task_uri_groups,
        )

        self._patch(monkeypatch, resolves=True)
        assert validate_task_uri_groups("u", None, groups=[]) == (True, "ok", None)

    def test_matches_second_of_two_groups(self, monkeypatch):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            NUTRITION_URI,
            PHYSICAL_ACTIVITY_URI,
            validate_task_uri_groups,
        )

        self._patch(monkeypatch, branch=NUTRITION_URI, level=3)
        groups = [
            ({PHYSICAL_ACTIVITY_URI}, ["Level1"]),
            ({NUTRITION_URI}, ["Level2", "Level3"]),
        ]
        assert validate_task_uri_groups("u", None, groups=groups) == (
            True,
            "ok",
            NUTRITION_URI,
        )

    def test_wrong_branch_when_no_group_domain_matches(self, monkeypatch):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            MENTAL_WELLBEING_URI,
            NUTRITION_URI,
            validate_task_uri_groups,
        )

        self._patch(monkeypatch, branch=MENTAL_WELLBEING_URI, level=2)
        ok, reason, _ = validate_task_uri_groups(
            "u", None, groups=[({NUTRITION_URI}, ["Level2"])]
        )
        assert (ok, reason) == (False, "wrong_branch")

    def test_wrong_level_when_branch_matches_but_level_off(self, monkeypatch):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            NUTRITION_URI,
            validate_task_uri_groups,
        )

        self._patch(monkeypatch, branch=NUTRITION_URI, level=4)
        ok, reason, _ = validate_task_uri_groups(
            "u", None, groups=[({NUTRITION_URI}, ["Level1", "Level2"])]
        )
        assert (ok, reason) == (False, "wrong_level")

    def test_branch_unresolved_accepts_when_level_matches(self, monkeypatch):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            NUTRITION_URI,
            validate_task_uri_groups,
        )

        self._patch(monkeypatch, branch=None, level=2)
        assert validate_task_uri_groups(
            "u", None, groups=[({NUTRITION_URI}, ["Level2"])]
        ) == (True, "ok_branch_unresolved", None)

    def test_group_with_domains_only_accepts_any_level(self, monkeypatch):
        from src.scripts.scenarios.task_generation.ontology_bridge import (
            NUTRITION_URI,
            validate_task_uri_groups,
        )

        self._patch(monkeypatch, branch=NUTRITION_URI, level=0)
        assert validate_task_uri_groups("u", None, groups=[({NUTRITION_URI}, [])]) == (
            True,
            "ok",
            NUTRITION_URI,
        )
