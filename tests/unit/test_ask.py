"""Unit tests for src.scripts.ask CLI."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.graphrag.query_planner import QueryPlan
from src.graphrag.retrieval_filters import AttributeFilter

_HA = "https://w3id.org/calendar-bench/human-activities/"


def _mock_rag(answer: str = "ans", items: list | None = None) -> MagicMock:
    response = MagicMock()
    response.answer = answer
    if items is None:
        response.retriever_result = None
    else:
        response.retriever_result = MagicMock()
        response.retriever_result.items = items
    rag = MagicMock()
    rag.search.return_value = response
    return rag


def test_ask_main_prints_answer(monkeypatch, capsys):
    from src.scripts.ask import main

    monkeypatch.setattr(sys, "argv", ["ask", "what is X?"])
    with patch(
        "src.scripts.ask.build_graphrag", return_value=_mock_rag("X is foo.")
    ), patch(
        "src.scripts.ask.plan_query",
        return_value=QueryPlan(semantic_text="what is X?"),
    ):
        rc = main()

    assert rc == 0
    assert "X is foo" in capsys.readouterr().out


def test_ask_main_default_top_k_and_semantic_text(monkeypatch):
    from src.scripts.ask import main

    monkeypatch.setattr(sys, "argv", ["ask", "test query"])
    rag = _mock_rag()
    with patch("src.scripts.ask.build_graphrag", return_value=rag), patch(
        "src.scripts.ask.plan_query",
        return_value=QueryPlan(semantic_text="trimmed gist"),
    ):
        main()

    kwargs = rag.search.call_args.kwargs
    assert kwargs["retriever_config"] == {"top_k": 10}
    assert kwargs["query_text"] == "trimmed gist"


def test_ask_main_top_k_flag_overrides_planner(monkeypatch):
    from src.scripts.ask import main

    monkeypatch.setattr(sys, "argv", ["ask", "--top-k", "3", "q"])
    rag = _mock_rag()
    with patch("src.scripts.ask.build_graphrag", return_value=rag), patch(
        "src.scripts.ask.plan_query",
        return_value=QueryPlan(semantic_text="q", top_k=20),
    ):
        main()

    assert rag.search.call_args.kwargs["retriever_config"] == {"top_k": 3}


def test_ask_main_uses_planner_top_k_when_flag_absent(monkeypatch):
    from src.scripts.ask import main

    monkeypatch.setattr(sys, "argv", ["ask", "q"])
    rag = _mock_rag()
    with patch("src.scripts.ask.build_graphrag", return_value=rag), patch(
        "src.scripts.ask.plan_query",
        return_value=QueryPlan(semantic_text="q", top_k=6),
    ):
        main()

    assert rag.search.call_args.kwargs["retriever_config"] == {"top_k": 6}


def test_ask_main_forwards_planner_filters_to_build_graphrag(monkeypatch):
    from src.scripts.ask import main

    monkeypatch.setattr(sys, "argv", ["ask", "activities like running over 7.3 MET"])
    plan = QueryPlan(
        semantic_text="running jogging",
        include_prefixes=(_HA,),
        attribute_filters=(AttributeFilter("metValue", ">", 7.3),),
        top_k=6,
    )
    with patch(
        "src.scripts.ask.build_graphrag", return_value=_mock_rag()
    ) as build, patch("src.scripts.ask.plan_query", return_value=plan):
        main()

    kwargs = build.call_args.kwargs
    assert kwargs["uri_prefixes"] == [_HA]
    assert kwargs["attribute_filters"] == [AttributeFilter("metValue", ">", 7.3)]


def test_ask_main_no_planner_skips_llm_and_uses_raw_query(monkeypatch):
    from src.scripts.ask import main

    monkeypatch.setattr(sys, "argv", ["ask", "--no-planner", "raw question"])
    rag = _mock_rag()
    with patch("src.scripts.ask.build_graphrag", return_value=rag), patch(
        "src.scripts.ask.plan_query"
    ) as planner:
        main()

    planner.assert_not_called()
    assert rag.search.call_args.kwargs["query_text"] == "raw question"


def test_ask_main_explicit_flags_override_planner(monkeypatch):
    from src.scripts.ask import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ask",
            "--no-planner",
            "--include",
            "HumanActivities",
            "--filter",
            "metValue > 7.3",
            "--level",
            "1",
            "--branch",
            "Nutrition",
            "--instance-only",
            "easy nutrition like running",
        ],
    )
    with patch("src.scripts.ask.build_graphrag", return_value=_mock_rag()) as build:
        main()

    kwargs = build.call_args.kwargs
    assert kwargs["uri_prefixes"] == [_HA]
    assert kwargs["attribute_filters"] == [AttributeFilter("metValue", ">", 7.3)]
    assert kwargs["allowed_levels"] == ["Level1"]
    assert kwargs["allowed_branches"] == [
        "https://w3id.org/calendar-bench/health/NutritionTask"
    ]
    assert kwargs["instance_only"] is True


def test_ask_main_exclude_flag_forwarded(monkeypatch, capsys):
    from src.scripts.ask import main

    monkeypatch.setattr(sys, "argv", ["ask", "--no-planner", "--exclude", "OCHV", "q"])
    with patch("src.scripts.ask.build_graphrag", return_value=_mock_rag()) as build:
        main()

    assert build.call_args.kwargs["exclude_prefixes"] == [
        "http://sbmi.uth.tmc.edu/ontology/ochv#"
    ]
    assert "exclude=" in capsys.readouterr().out


def test_ask_main_bad_filter_flag_errors(monkeypatch):
    from src.scripts.ask import main

    monkeypatch.setattr(
        sys, "argv", ["ask", "--no-planner", "--filter", "bogus > 1", "q"]
    )
    with patch("src.scripts.ask.build_graphrag"):
        with pytest.raises(SystemExit):
            main()


def test_ask_main_show_context_prints_items(monkeypatch, capsys):
    from src.scripts.ask import main

    item = MagicMock()
    item.content = "## context block A\nuri: http://x/A"
    monkeypatch.setattr(sys, "argv", ["ask", "--no-planner", "--show-context", "q"])
    with patch("src.scripts.ask.build_graphrag", return_value=_mock_rag(items=[item])):
        rc = main()

    assert rc == 0
    out = capsys.readouterr().out
    assert "context block A" in out
    assert "Retrieved context" in out
