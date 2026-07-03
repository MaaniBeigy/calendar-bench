"""Unit tests for src.graphrag.retrieval_filters."""

from __future__ import annotations

import pytest

from src.graphrag.retrieval_filters import (
    ATTRIBUTE_REGISTRY,
    AttributeFilter,
    _ontology_for_prefix,
    attribute_registry_summary,
    cypher_predicate,
    make_attribute_filter,
    ontology_prefix_summary,
    parse_attribute_filter,
    resolve_branch,
    resolve_level,
    resolve_scope,
)

_HA = "https://w3id.org/calendar-bench/human-activities/"
_HEALTH = "https://w3id.org/calendar-bench/health/"


class TestParseAttributeFilter:
    def test_numeric_with_spaces(self):
        flt = parse_attribute_filter("metValue > 7.3")
        assert flt == AttributeFilter("metValue", ">", 7.3)

    def test_numeric_without_spaces(self):
        flt = parse_attribute_filter("estimatedDurationMinutes<=30")
        assert flt == AttributeFilter("estimatedDurationMinutes", "<=", 30.0)

    def test_boolean_true(self):
        flt = parse_attribute_filter("isConcurrent = true")
        assert flt == AttributeFilter("isConcurrent", "=", True)

    def test_boolean_false_word(self):
        assert parse_attribute_filter("isDividable = no").value is False

    def test_unparseable_expression_raises(self):
        with pytest.raises(ValueError, match="could not parse"):
            parse_attribute_filter("this is not a filter")

    def test_unknown_attribute_raises(self):
        with pytest.raises(ValueError, match="unknown filter attribute"):
            parse_attribute_filter("bogus > 1")

    def test_boolean_with_ordering_op_raises(self):
        with pytest.raises(ValueError, match="only = and <>"):
            parse_attribute_filter("isConcurrent > 1")

    def test_numeric_with_non_numeric_value_raises(self):
        with pytest.raises(ValueError, match="numeric"):
            parse_attribute_filter("metValue > high")


class TestMakeAttributeFilter:
    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="unknown filter operator"):
            make_attribute_filter("metValue", "~=", 3)

    def test_boolean_invalid_token_raises(self):
        with pytest.raises(ValueError, match="boolean"):
            make_attribute_filter("isConcurrent", "=", "maybe")

    def test_accepts_non_string_value(self):
        assert make_attribute_filter("metValue", ">", 7).value == 7.0


class TestCypherPredicate:
    def test_numeric_ordering_uses_tofloat(self):
        out = cypher_predicate(AttributeFilter("metValue", ">", 7.3))
        assert out == "toFloat(head(node.metValue)) > 7.3"

    def test_numeric_equality(self):
        out = cypher_predicate(AttributeFilter("activityCode", "=", 9055.0))
        assert out == "toFloat(head(node.activityCode)) = 9055.0"

    def test_boolean_true(self):
        out = cypher_predicate(AttributeFilter("isConcurrent", "=", True))
        assert out == "head(node.isConcurrent) = true"

    def test_boolean_false_with_not_equal(self):
        out = cypher_predicate(AttributeFilter("isDividable", "<>", False))
        assert out == "head(node.isDividable) <> false"

    def test_rejects_attribute_outside_registry(self):
        with pytest.raises(ValueError, match="unsafe filter attribute"):
            cypher_predicate(AttributeFilter("evilProp", ">", 1.0))

    def test_rejects_non_identifier_name(self):
        with pytest.raises(ValueError, match="unsafe filter attribute"):
            cypher_predicate(AttributeFilter("a b", ">", 1.0))

    def test_rejects_unknown_operator(self):
        with pytest.raises(ValueError, match="unsafe filter operator"):
            cypher_predicate(AttributeFilter("metValue", "??", 1.0))


class TestResolveScope:
    def test_resolves_ontology_key(self):
        assert resolve_scope(["HumanActivities"]) == [_HA]

    def test_passes_raw_http_prefix_through(self):
        assert resolve_scope(["http://example.org/x/"]) == ["http://example.org/x/"]

    def test_unknown_ontology_raises(self):
        with pytest.raises(ValueError, match="unknown ontology"):
            resolve_scope(["NotAnOntology"])

    def test_dedupes_repeated_prefixes(self):
        assert resolve_scope(["HumanActivities", "HumanActivities"]) == [_HA]


class TestResolveBranch:
    def test_resolves_friendly_name(self):
        assert resolve_branch("Nutrition") == _HEALTH + "NutritionTask"

    def test_passes_raw_uri_through(self):
        assert resolve_branch("https://x/y") == "https://x/y"

    def test_unknown_branch_raises(self):
        with pytest.raises(ValueError, match="unknown branch"):
            resolve_branch("Spirituality")


class TestResolveLevel:
    def test_resolves_integer(self):
        assert resolve_level(1) == "Level1"

    def test_passes_level_token_through(self):
        assert resolve_level("Level3") == "Level3"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="unknown level"):
            resolve_level(9)


class TestSummaries:
    def test_attribute_summary_lists_registry(self):
        summary = attribute_registry_summary()
        assert "metValue" in summary
        assert summary.count("\n") == len(ATTRIBUTE_REGISTRY) - 1

    def test_ontology_summary_lists_prefixes(self):
        summary = ontology_prefix_summary()
        assert "HumanActivities" in summary
        assert _HA in summary

    def test_attribute_summary_names_home_ontology(self):
        assert "metValue (numeric, HumanActivities)" in attribute_registry_summary()

    def test_ontology_for_prefix_known_and_unknown(self):
        assert _ontology_for_prefix(_HA) == "HumanActivities"
        assert _ontology_for_prefix("http://unmapped/") == "http://unmapped/"
