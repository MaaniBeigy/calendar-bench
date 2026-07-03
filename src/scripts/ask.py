"""Question-answering wrapper around the GraphRAG pipeline.

By default a question first goes through the query planner, which turns
phrases like "MET more than 7.3", "human activities", or "level 1" into
structured retrieval filters (numeric thresholds, ontology scope, difficulty
levels) and a stripped-down text to embed. Explicit flags override any field
the planner would have set, and `--no-planner` skips it entirely for a plain
semantic search.

Examples:
    python -m src.scripts.ask "What physical activities are recommended for hypertension?"
    python -m src.scripts.ask "five activities like running with MET over 7.3"
    python -m src.scripts.ask --no-planner --include HumanActivities \\
        --filter "metValue > 7.3" "activities like running"
    python -m src.scripts.ask --level 1 --branch Nutrition "easy nutrition tasks"
"""

from __future__ import annotations

import argparse
import sys

from src.graphrag.config import LLMSettings
from src.graphrag.pipeline import build_graphrag
from src.graphrag.query_planner import QueryPlan, plan_query
from src.graphrag.retrieval_filters import (
    parse_attribute_filter,
    resolve_branch,
    resolve_level,
    resolve_scope,
)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Ask the calendar-bench GraphRAG pipeline."
    )
    ap.add_argument("query", help="Natural-language question")
    ap.add_argument(
        "--top-k", type=int, default=None, help="ANN top-K (default: 10 or planner)"
    )
    ap.add_argument(
        "--show-context", action="store_true", help="Print retrieved context"
    )
    ap.add_argument(
        "--no-planner",
        action="store_true",
        help="Skip the LLM query planner; embed the question as-is",
    )
    ap.add_argument(
        "--include",
        action="append",
        metavar="ONTOLOGY",
        help="Confine search to this ontology or URI prefix (repeatable)",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        metavar="ONTOLOGY",
        help="Keep this ontology or URI prefix out of the search (repeatable)",
    )
    ap.add_argument(
        "--level",
        action="append",
        type=int,
        metavar="N",
        help="Restrict to HealthTasks difficulty level 1-4 (repeatable)",
    )
    ap.add_argument(
        "--branch",
        action="append",
        metavar="NAME",
        help="Restrict to a HealthTasks branch: Nutrition / PhysicalActivity / "
        "MentalWellbeing (repeatable)",
    )
    ap.add_argument(
        "--filter",
        action="append",
        metavar="EXPR",
        help="Numeric or boolean filter such as 'metValue > 7.3' (repeatable)",
    )
    ap.add_argument(
        "--instance-only",
        action="store_true",
        help="Restrict to authored task instances, not ontology classes",
    )
    return ap


def _resolve_flags(args: argparse.Namespace) -> dict:
    """Turn raw CLI flags into resolved retrieval values, or None when unset."""
    return {
        "include": resolve_scope(args.include) if args.include else None,
        "exclude": resolve_scope(args.exclude) if args.exclude else None,
        "branches": [resolve_branch(b) for b in args.branch] if args.branch else None,
        "levels": [resolve_level(n) for n in args.level] if args.level else None,
        "filters": (
            [parse_attribute_filter(f) for f in args.filter] if args.filter else None
        ),
    }


def _merge(flag: object, planned: object) -> object:
    """Return the explicit flag value when set, else the planner's value."""
    if flag is not None:
        return flag or None
    return list(planned) or None if planned else None


def _describe_plan(
    query_text: str,
    include: list | None,
    exclude: list | None,
    branches: list | None,
    levels: list | None,
    filters: list | None,
    instance_only: bool,
    top_k: int,
) -> str:
    """Render a one-line summary of the resolved retrieval plan."""
    parts = [f"embed={query_text!r}", f"top_k={top_k}"]
    if include:
        parts.append(f"include={include}")
    if exclude:
        parts.append(f"exclude={exclude}")
    if branches:
        parts.append(f"branches={branches}")
    if levels:
        parts.append(f"levels={levels}")
    if filters:
        parts.append(
            "filters=" + ", ".join(f"{f.name}{f.op}{f.value}" for f in filters)
        )
    if instance_only:
        parts.append("instance_only=True")
    return "[plan] " + " ".join(parts)


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()

    try:
        flags = _resolve_flags(args)
    except ValueError as exc:
        ap.error(str(exc))

    plan = (
        QueryPlan(semantic_text=args.query)
        if args.no_planner
        else plan_query(args.query)
    )

    include = _merge(flags["include"], plan.include_prefixes)
    exclude = _merge(flags["exclude"], plan.exclude_prefixes)
    branches = _merge(flags["branches"], plan.branches)
    levels = _merge(flags["levels"], plan.levels)
    filters = _merge(flags["filters"], plan.attribute_filters)
    instance_only = args.instance_only or plan.instance_only
    top_k = args.top_k if args.top_k is not None else (plan.top_k or 10)
    query_text = plan.semantic_text

    s = LLMSettings.from_env()
    print(f"[provider={s.provider}] asking…\n")
    print(
        _describe_plan(
            query_text,
            include,
            exclude,
            branches,
            levels,
            filters,
            instance_only,
            top_k,
        )
        + "\n"
    )

    rag = build_graphrag(
        uri_prefixes=include,
        exclude_prefixes=exclude,
        allowed_branches=branches,
        allowed_levels=levels,
        attribute_filters=filters,
        instance_only=instance_only,
    )
    response = rag.search(
        query_text=query_text,
        retriever_config={"top_k": top_k},
        return_context=args.show_context,
    )

    print("=== Answer ===")
    print(response.answer)

    if args.show_context and getattr(response, "retriever_result", None):
        print("\n=== Retrieved context ===")
        for i, item in enumerate(response.retriever_result.items, 1):
            print(f"\n--- {i} ---")
            print(item.content)

    return 0


if __name__ == "__main__":
    sys.exit(main())
