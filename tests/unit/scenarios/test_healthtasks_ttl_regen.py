"""Tests for src.scripts.generate_healthtasks_ttl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rdflib

from src.scripts.generate_healthtasks_ttl import (
    HA_ACT_PREFIX,
    _activity_curie,
    _camel_split,
    _ttl_escape,
    emit_prelude,
    emit_task,
    emit_taxonomy,
    generate_ttl,
    load_context_iris,
    main,
)

REPO = Path(__file__).resolve().parents[3]
HB_NS = "https://w3id.org/calendar-bench/health/"
HB_TK_NS = "https://w3id.org/calendar-bench/health/task/"
HB_CONTEXT_LINK = rdflib.URIRef(HB_NS + "contextLink")
HB_MATCHED = rdflib.URIRef(HB_NS + "matchedActivity")


@pytest.fixture
def fake_iris_path(tmp_path: Path) -> Path:
    iris = {
        "schema": {"version": "test", "totals": {}},
        "context": {
            "mood_emotion": {
                "happy": {
                    "iri": "http://example.org/mood#happy",
                    "label": "happy",
                    "source": "X",
                },
                "sad": {
                    "iri": "http://example.org/mood#sad",
                    "label": "sad",
                    "source": "X",
                },
            },
            "location": {
                "home": {
                    "iri": "http://example.org/loc#home",
                    "label": "home",
                    "source": "X",
                },
                "workplace": {
                    "iri": "http://example.org/loc#workplace",
                    "label": "workplace",
                    "source": "X",
                },
            },
            "weather_environment": {},
        },
    }
    path = tmp_path / "context_iris.json"
    path.write_text(json.dumps(iris), encoding="utf-8")
    return path


@pytest.fixture
def fake_health_json_path(tmp_path: Path) -> Path:
    data = {
        "Nutrition": {
            "BetterBeverageBalance": {
                "Level1": {
                    "tea-time": {
                        "title": "Tea Time",
                        "description": "Replace one coffee with tea.",
                        "isConcurrent": True,
                        "isDividable": False,
                        "estimatedDuration": 15,
                        "matchedActivity": [HA_ACT_PREFIX + "eating-sitting"],
                        "context_links": {
                            "mood_emotion": ["happy", "sad"],
                            "location": ["home", "workplace"],
                            "weather_environment": [],
                        },
                    },
                },
                "Level2": {},  # empty level, must be skipped from the taxonomy
            },
        },
    }
    path = tmp_path / "HealthTasks_test.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_camel_split_inserts_spaces() -> None:
    assert _camel_split("BetterBeverageBalance") == "Better Beverage Balance"
    assert _camel_split("X") == "X"


def test_ttl_escape_backslash_quote_newline() -> None:
    assert _ttl_escape('a"b') == 'a\\"b'
    assert _ttl_escape("a\\b") == "a\\\\b"
    assert _ttl_escape("a\nb") == "a\\nb"


def test_activity_curie_known_prefix() -> None:
    iri = HA_ACT_PREFIX + "eating-sitting"
    assert _activity_curie(iri) == "ha-act:eating-sitting"


def test_activity_curie_unknown_namespace_falls_back_to_full_iri() -> None:
    assert _activity_curie("http://example.org/foo") == "<http://example.org/foo>"


def test_load_context_iris_flattens_to_slug_iri_map(fake_iris_path: Path) -> None:
    loaded = load_context_iris(fake_iris_path)
    assert loaded["mood_emotion"]["happy"] == "http://example.org/mood#happy"
    assert loaded["weather_environment"] == {}


def test_emit_prelude_declares_all_required_properties() -> None:
    s = emit_prelude("2026.05.19")
    for needle in (
        "@prefix hb:",
        "@prefix hb-tk:",
        "@prefix ha:",
        "@prefix ha-act:",
        "@prefix owl:",
        "@prefix rdf:",
        "@prefix rdfs:",
        "@prefix xsd:",
        "@prefix dcterms:",
        "hb:isConcurrent",
        "hb:isDividable",
        "hb:estimatedDurationMinutes",
        "hb:matchedActivity",
        "hb:contextLink",
        'owl:versionInfo     "2026.05.19"',
    ):
        assert needle in s, f"prelude missing {needle!r}"


def test_emit_taxonomy_skips_empty_levels(fake_health_json_path: Path) -> None:
    data = json.loads(fake_health_json_path.read_text(encoding="utf-8"))
    tax = emit_taxonomy(data)
    assert "hb:NutritionBetterBeverageBalanceLevel1" in tax
    assert "hb:NutritionBetterBeverageBalanceLevel2" not in tax
    assert "hb:NutritionTask" in tax
    assert "hb:NutritionBetterBeverageBalanceTask" in tax
    assert 'rdfs:label "Nutrition - Better Beverage Balance"@en' in tax


def test_emit_task_emits_one_contextlink_line_per_slug(fake_iris_path: Path) -> None:
    iris = load_context_iris(fake_iris_path)
    task = {
        "title": "T",
        "description": "D",
        "isConcurrent": True,
        "isDividable": False,
        "estimatedDuration": 5,
        "matchedActivity": [],
        "context_links": {
            "mood_emotion": ["happy"],
            "location": ["home", "workplace"],
            "weather_environment": [],
        },
    }
    out = emit_task("Nutrition", "BetterBeverageBalance", "Level1", "x", task, iris)
    assert out.count("hb:contextLink") == 3
    assert "hb:matchedActivity" not in out
    assert out.endswith(" .\n")


def test_emit_task_normalises_string_matched_activity(fake_iris_path: Path) -> None:
    iris = load_context_iris(fake_iris_path)
    task = {
        "title": "T",
        "description": "D",
        "isConcurrent": False,
        "isDividable": False,
        "estimatedDuration": 1,
        "matchedActivity": HA_ACT_PREFIX + "eating-sitting",
        "context_links": {},
    }
    out = emit_task("Nutrition", "BetterBeverageBalance", "Level1", "x", task, iris)
    assert "hb:matchedActivity ha-act:eating-sitting" in out
    assert "hb:contextLink" not in out


def test_emit_task_unknown_category_raises(fake_iris_path: Path) -> None:
    iris = load_context_iris(fake_iris_path)
    task = {
        "title": "T",
        "description": "D",
        "isConcurrent": False,
        "isDividable": False,
        "estimatedDuration": 1,
        "matchedActivity": [],
        "context_links": {"unknown_cat": ["x"]},
    }
    with pytest.raises(ValueError, match="not present in context_iris.json"):
        emit_task("N", "B", "L", "tid", task, iris)


def test_emit_task_unknown_slug_raises(fake_iris_path: Path) -> None:
    iris = load_context_iris(fake_iris_path)
    task = {
        "title": "T",
        "description": "D",
        "isConcurrent": False,
        "isDividable": False,
        "estimatedDuration": 1,
        "matchedActivity": [],
        "context_links": {"mood_emotion": ["nonexistent_slug"]},
    }
    with pytest.raises(ValueError, match="not in context_iris.json"):
        emit_task("N", "B", "L", "tid", task, iris)


def test_generate_ttl_one_contextlink_per_unique_task_iri(
    fake_health_json_path: Path,
    fake_iris_path: Path,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "out.ttl"
    n = generate_ttl(fake_health_json_path, fake_iris_path, out_path, "test-version")
    assert n == 1

    data = json.loads(fake_health_json_path.read_text(encoding="utf-8"))
    iris = load_context_iris(fake_iris_path)
    expected_pairs: set[tuple[str, str]] = set()
    for cats in data.values():
        for levels in cats.values():
            for tasks in levels.values():
                for tid, task in tasks.items():
                    for c, slugs in (task.get("context_links") or {}).items():
                        for slug in slugs:
                            expected_pairs.add((tid, iris[c][slug]))

    g = rdflib.Graph()
    g.parse(out_path, format="turtle")
    ctx_triples = list(g.triples((None, HB_CONTEXT_LINK, None)))
    assert len(ctx_triples) == len(expected_pairs)


def test_generate_ttl_roundtrip_preserves_triple_count(
    fake_health_json_path: Path,
    fake_iris_path: Path,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "out.ttl"
    generate_ttl(fake_health_json_path, fake_iris_path, out_path, "test-version")
    g1 = rdflib.Graph()
    g1.parse(out_path, format="turtle")
    serialised = g1.serialize(format="turtle")
    g2 = rdflib.Graph()
    g2.parse(data=serialised, format="turtle")
    assert len(g1) == len(g2)


def test_shipped_05_19_ttl_parses_and_carries_expected_edges() -> None:
    """The committed 05.19 TTL parses cleanly with 162 matchedActivity edges and many contextLinks."""
    g = rdflib.Graph()
    g.parse(
        REPO / "src" / "assets" / "ontologies" / "HealthTasks_2026.05.19.ttl",
        format="turtle",
    )
    n_matched = sum(1 for _ in g.triples((None, HB_MATCHED, None)))
    n_ctx = sum(1 for _ in g.triples((None, HB_CONTEXT_LINK, None)))
    assert n_matched == 162
    assert n_ctx > 0


def test_main_raises_when_json_missing(tmp_path: Path) -> None:
    (tmp_path / "context_iris.json").write_text(
        json.dumps({"schema": {}, "context": {}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        main(["--version", "9999.99.99", "--ontologies-dir", str(tmp_path)])


def test_main_raises_when_iris_missing(tmp_path: Path) -> None:
    (tmp_path / "HealthTasks_only-json.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        main(["--version", "only-json", "--ontologies-dir", str(tmp_path)])


def test_main_writes_ttl_outside_repo_prints_absolute_path(
    tmp_path: Path,
    fake_iris_path: Path,
    fake_health_json_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "context_iris.json").write_text(
        fake_iris_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "HealthTasks_test.json").write_text(
        fake_health_json_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    main(["--version", "test", "--ontologies-dir", str(tmp_path)])
    out_path = tmp_path / "HealthTasks_test.ttl"
    assert out_path.exists()
    captured = capsys.readouterr()
    assert "1 task individuals" in captured.out
    assert str(out_path) in captured.out


def test_main_prints_relative_path_when_output_inside_repo(
    tmp_path: Path,
    fake_iris_path: Path,
    fake_health_json_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import src.scripts.generate_healthtasks_ttl as mod

    monkeypatch.setattr(mod, "REPO", tmp_path)
    (tmp_path / "context_iris.json").write_text(
        fake_iris_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "HealthTasks_test.json").write_text(
        fake_health_json_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    main(["--version", "test", "--ontologies-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "HealthTasks_test.ttl" in out
    assert str(tmp_path) not in out  # printed via relative_to, not absolute
