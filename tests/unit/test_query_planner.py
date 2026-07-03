"""Unit tests for src.graphrag.query_planner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.graphrag.query_planner import (
    QueryPlan,
    _as_list,
    _consistent_scope,
    _content_of,
    _extract_json,
    _valid_top_k,
    plan_query,
)
from src.graphrag.retrieval_filters import AttributeFilter

_HA = "https://w3id.org/calendar-bench/human-activities/"
_HEALTH = "https://w3id.org/calendar-bench/health/"
_OCHV = "http://sbmi.uth.tmc.edu/ontology/ochv#"


class _FakeLLM:
    """An LLM stub whose `invoke` returns a fixed response body."""

    def __init__(self, body: object) -> None:
        self._body = body
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if isinstance(self._body, Exception):
            raise self._body
        return SimpleNamespace(content=self._body)


def _llm(payload: dict) -> _FakeLLM:
    return _FakeLLM(json.dumps(payload))


class TestPlanQuery:
    def test_human_activities_plan_resolves_and_scopes(self):
        llm = _llm(
            {
                "semantic_text": "running jogging",
                "include_ontologies": ["HumanActivities"],
                "exclude_ontologies": ["OCHV"],
                "attribute_filters": [{"name": "metValue", "op": ">", "value": 7.3}],
                "instance_only": True,
                "top_k": 6,
            }
        )
        plan = plan_query("five activities like running with MET over 7.3", llm=llm)
        assert plan.semantic_text == "running jogging"
        assert plan.include_prefixes == (_HA,)
        assert plan.exclude_prefixes == (_OCHV,)
        assert plan.attribute_filters == (AttributeFilter("metValue", ">", 7.3),)
        assert plan.instance_only is True
        assert plan.top_k == 6

    def test_healthtasks_plan_keeps_branch_and_level(self):
        llm = _llm(
            {
                "semantic_text": "easy nutrition",
                "include_ontologies": ["HealthTasks"],
                "branches": ["Nutrition"],
                "levels": [1, 2],
                "attribute_filters": [
                    {"name": "estimatedDurationMinutes", "op": "<=", "value": 15}
                ],
            }
        )
        plan = plan_query("quick level 1-2 nutrition tasks under 15 min", llm=llm)
        assert plan.include_prefixes == (_HEALTH,)
        assert plan.branches == (_HEALTH + "NutritionTask",)
        assert plan.levels == ("Level1", "Level2")
        assert plan.attribute_filters == (
            AttributeFilter("estimatedDurationMinutes", "<=", 15.0),
        )

    def test_branch_dropped_when_filter_is_human_activities_only(self):
        """Reproduces the empty-result bug: a HealthTasks branch paired with
        a HumanActivities metValue filter can never co-occur, so the branch
        and the stray HealthTasks scope are dropped and the search is
        confined to HumanActivities."""
        llm = _llm(
            {
                "semantic_text": "running jogging",
                "include_ontologies": ["HumanActivities", "HealthTasks"],
                "branches": ["PhysicalActivity"],
                "levels": [1],
                "attribute_filters": [{"name": "metValue", "op": ">=", "value": 7.3}],
            }
        )
        plan = plan_query("human activities like running with MET over 7.3", llm=llm)
        assert plan.include_prefixes == (_HA,)
        assert plan.branches == ()
        assert plan.levels == ()
        assert plan.attribute_filters == (AttributeFilter("metValue", ">=", 7.3),)

    def test_missing_semantic_text_falls_back_to_question(self):
        plan = plan_query("how many steps?", llm=_llm({"top_k": 3}))
        assert plan.semantic_text == "how many steps?"
        assert plan.top_k == 3

    def test_unknown_fields_are_dropped(self):
        llm = _llm(
            {
                "semantic_text": "x",
                "include_ontologies": ["NopeOntology"],
                "branches": ["NotABranch"],
                "levels": [9],
                "attribute_filters": [
                    {"name": "bogus", "op": ">", "value": 1},
                    {"name": "metValue"},
                    "not-a-dict",
                ],
            }
        )
        plan = plan_query("q", llm=llm)
        assert plan.include_prefixes == ()
        assert plan.branches == ()
        assert plan.levels == ()
        assert plan.attribute_filters == ()

    def test_malformed_json_falls_back(self):
        plan = plan_query("q", llm=_FakeLLM("not json at all"))
        assert plan == QueryPlan(semantic_text="q")

    def test_llm_error_falls_back(self):
        plan = plan_query("q", llm=_FakeLLM(RuntimeError("boom")))
        assert plan == QueryPlan(semantic_text="q")

    def test_default_llm_is_built_when_none_supplied(self, monkeypatch):
        fake = _llm({"semantic_text": "gist", "top_k": 4})
        monkeypatch.setattr("src.graphrag.llm.make_llm", lambda: fake)
        plan = plan_query("q")
        assert plan.semantic_text == "gist"
        assert plan.top_k == 4
        assert fake.prompts  # the planner prompt reached the LLM


class TestConsistentScope:
    _BRANCH = (_HEALTH + "NutritionTask",)
    _LEVEL = ("Level1",)
    _MET = (AttributeFilter("metValue", ">", 7.3),)
    _DURATION = (AttributeFilter("estimatedDurationMinutes", "<=", 15.0),)

    def test_no_filters_no_branches_passthrough(self):
        assert _consistent_scope((), (), (), ()) == ((), (), ())

    def test_branch_kept_when_scope_unset(self):
        assert _consistent_scope((), (), self._BRANCH, ()) == ((), self._BRANCH, ())

    def test_branch_dropped_when_scope_is_non_health(self):
        out = _consistent_scope((), (_HA,), self._BRANCH, self._LEVEL)
        assert out == ((_HA,), (), ())

    def test_human_activities_filter_forces_include_and_drops_branch(self):
        out = _consistent_scope(self._MET, (_HA, _HEALTH), self._BRANCH, self._LEVEL)
        assert out == ((_HA,), (), ())

    def test_health_filter_keeps_branch_and_level(self):
        out = _consistent_scope(self._DURATION, (_HEALTH,), self._BRANCH, self._LEVEL)
        assert out == ((_HEALTH,), self._BRANCH, self._LEVEL)


class TestHelpers:
    def test_content_of_reads_content_attribute(self):
        assert _content_of(SimpleNamespace(content="hello")) == "hello"

    def test_content_of_accepts_plain_string(self):
        assert _content_of("plain") == "plain"

    def test_content_of_stringifies_non_string_content(self):
        assert _content_of(SimpleNamespace(content=123)) == "123"

    def test_extract_json_pulls_object_from_noise(self):
        assert _extract_json('prefix {"a": 1} suffix') == {"a": 1}

    def test_extract_json_without_object_raises(self):
        with pytest.raises(ValueError, match="no JSON object"):
            _extract_json("no braces here")

    def test_as_list_coerces(self):
        assert _as_list(None) == []
        assert _as_list([1, 2]) == [1, 2]
        assert _as_list("x") == ["x"]

    def test_valid_top_k_clamps_and_coerces(self):
        assert _valid_top_k(None) is None
        assert _valid_top_k("abc") is None
        assert _valid_top_k(999) == 100
        assert _valid_top_k(0) == 1
        assert _valid_top_k(5) == 5
