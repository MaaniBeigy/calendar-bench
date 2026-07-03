"""Unit tests for src.scripts.scenarios.metrics.pricing_update.

Coverage targets:
  _normalize           ; dot / dash / case equivalence.
  _catalog_to_lookup   ; per-million conversion, malformed entries skipped.
  update_pricing_file  ; matched models repriced, unmatched preserved,
                         fetch error leaves the file unchanged, zero-match
                         leaves it unchanged, dry-run writes nothing.
  main                 ; exit codes for success / failure / dry-run.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from src.scripts.scenarios.metrics import pricing_update as pu

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _catalog() -> list[dict]:
    """OpenRouter-shaped catalog: per-token USD price strings."""
    return [
        {
            "id": "openai/gpt-4o-mini",
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "anthropic/claude-opus-4.8",
            "pricing": {"prompt": "0.000005", "completion": "0.000025"},
        },
        {
            "id": "openai/gpt-5-mini",
            "pricing": {"prompt": "0.00000025", "completion": "0.000002"},
        },
        {"id": "no-provider-prefix", "pricing": {"prompt": "1", "completion": "1"}},
        {"id": "openai/broken", "pricing": {"prompt": "abc", "completion": "x"}},
    ]


def _write_pricing(path: Path, table: dict) -> None:
    path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")


@pytest.fixture()
def pricing_file(tmp_path: Path) -> Path:
    path = tmp_path / "llm_pricing.json"
    _write_pricing(
        path,
        {
            "openai": {
                "gpt-4o-mini": {"in": 9.0, "out": 9.0},
                "gpt-5-mini": {"in": 9.0, "out": 9.0},
                "text-embedding-3-small": {"in": 0.02, "out": 0.0},
            },
            "anthropic": {"claude-opus-4-8": {"in": 9.0, "out": 9.0}},
        },
    )
    return path


def _ok_fetch(_url, *, timeout):  # noqa: ANN001 - test stub
    return _catalog()


@pytest.fixture(autouse=True)
def _reset_pricing_cache():
    """Clear the telemetry price cache after each test that may touch it."""
    yield
    import src.scripts.scenarios.metrics.telemetry as tel

    tel._PRICING = None


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_dot_and_dash_compare_equal(self):
        assert pu._normalize("claude-opus-4.8") == pu._normalize("claude-opus-4-8")

    def test_case_insensitive(self):
        assert pu._normalize("GPT-5-Mini") == pu._normalize("gpt-5-mini")

    def test_distinct_models_stay_distinct(self):
        assert pu._normalize("gpt-5-mini") != pu._normalize("gpt-5.4-mini")


# ---------------------------------------------------------------------------
# _catalog_to_lookup
# ---------------------------------------------------------------------------


class TestCatalogToLookup:
    def test_converts_per_token_to_per_million(self):
        lookup = pu._catalog_to_lookup(_catalog())
        assert lookup[("openai", pu._normalize("gpt-4o-mini"))] == {
            "in": 0.15,
            "out": 0.6,
        }

    def test_skips_entries_without_provider_prefix(self):
        lookup = pu._catalog_to_lookup(_catalog())
        assert not any(k for k in lookup if k[1] == pu._normalize("no-provider-prefix"))

    def test_skips_unparseable_prices(self):
        lookup = pu._catalog_to_lookup(_catalog())
        assert ("openai", pu._normalize("broken")) not in lookup

    def test_skips_negative_sentinel_prices(self):
        """A `-1` price is a variable/unavailable sentinel, not a real price."""
        lookup = pu._catalog_to_lookup(
            [{"id": "openai/x", "pricing": {"prompt": "-1", "completion": "-1"}}]
        )
        assert ("openai", pu._normalize("x")) not in lookup


# ---------------------------------------------------------------------------
# update_pricing_file
# ---------------------------------------------------------------------------


class TestUpdatePricingFile:
    def test_matched_models_are_repriced(self, pricing_file):
        result = pu.update_pricing_file(pricing_path=pricing_file, fetch=_ok_fetch)
        assert result.ok
        table = json.loads(pricing_file.read_text(encoding="utf-8"))
        assert table["openai"]["gpt-4o-mini"] == {"in": 0.15, "out": 0.6}
        assert table["openai"]["gpt-5-mini"] == {"in": 0.25, "out": 2.0}
        # Dot-vs-dash normalization matches claude-opus-4.8 to our 4-8 key.
        assert table["anthropic"]["claude-opus-4-8"] == {"in": 5.0, "out": 25.0}

    def test_unmatched_models_keep_their_prices(self, pricing_file):
        result = pu.update_pricing_file(pricing_path=pricing_file, fetch=_ok_fetch)
        table = json.loads(pricing_file.read_text(encoding="utf-8"))
        assert table["openai"]["text-embedding-3-small"] == {"in": 0.02, "out": 0.0}
        assert "openai/text-embedding-3-small" in result.unmatched

    def test_http_error_leaves_file_unchanged(self, pricing_file):
        before = pricing_file.read_bytes()

        def _boom(_url, *, timeout):  # noqa: ANN001 - test stub
            raise urllib.error.HTTPError(_url, 500, "boom", {}, None)

        result = pu.update_pricing_file(pricing_path=pricing_file, fetch=_boom)
        assert result.ok is False
        assert result.error
        assert pricing_file.read_bytes() == before

    def test_network_error_leaves_file_unchanged(self, pricing_file):
        before = pricing_file.read_bytes()

        def _boom(_url, *, timeout):  # noqa: ANN001 - test stub
            raise urllib.error.URLError("no route")

        result = pu.update_pricing_file(pricing_path=pricing_file, fetch=_boom)
        assert result.ok is False
        assert pricing_file.read_bytes() == before

    def test_zero_match_leaves_file_unchanged(self, pricing_file):
        before = pricing_file.read_bytes()

        def _unrelated(_url, *, timeout):  # noqa: ANN001 - test stub
            return [
                {
                    "id": "mistral/mistral-large",
                    "pricing": {"prompt": "0.000002", "completion": "0.000006"},
                }
            ]

        result = pu.update_pricing_file(pricing_path=pricing_file, fetch=_unrelated)
        assert result.ok is False
        assert "no tracked model matched" in (result.error or "")
        assert pricing_file.read_bytes() == before

    def test_dry_run_reports_but_does_not_write(self, pricing_file):
        before = pricing_file.read_bytes()
        result = pu.update_pricing_file(
            pricing_path=pricing_file, fetch=_ok_fetch, dry_run=True
        )
        assert result.ok
        assert result.matched
        assert pricing_file.read_bytes() == before

    def test_missing_file_reports_error(self, tmp_path):
        result = pu.update_pricing_file(
            pricing_path=tmp_path / "absent.json", fetch=_ok_fetch
        )
        assert result.ok is False
        assert "cannot read" in (result.error or "")

    def test_tmp_path_does_not_touch_telemetry_cache(self, pricing_file):
        import src.scripts.scenarios.metrics.telemetry as tel

        tel.load_pricing(force=True)
        sentinel = tel._PRICING
        pu.update_pricing_file(pricing_path=pricing_file, fetch=_ok_fetch)
        # The canonical PRICING_FILE was not the target, so the cache is intact.
        assert tel._PRICING is sentinel

    def test_canonical_write_refreshes_cache(self, tmp_path, monkeypatch):
        """Writing the canonical file re-reads it into the telemetry cache."""
        import src.scripts.scenarios.metrics.telemetry as tel

        canonical = tmp_path / "llm_pricing.json"
        _write_pricing(canonical, {"openai": {"gpt-4o-mini": {"in": 9.0, "out": 9.0}}})
        monkeypatch.setattr(tel, "PRICING_FILE", canonical)
        monkeypatch.setattr(pu, "PRICING_FILE", canonical)
        assert tel.load_pricing(force=True)["openai"]["gpt-4o-mini"]["in"] == 9.0

        result = pu.update_pricing_file(pricing_path=canonical, fetch=_ok_fetch)
        assert result.ok
        # The cache reflects the freshly written prices without a manual reload.
        assert tel.load_pricing()["openai"]["gpt-4o-mini"] == {"in": 0.15, "out": 0.6}


# ---------------------------------------------------------------------------
# _http_fetch_models
# ---------------------------------------------------------------------------


class _StubResponse:
    """Context-manager stub for urllib.request.urlopen."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_StubResponse":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


class TestHttpFetchModels:
    def test_payload_without_data_list_raises(self, monkeypatch):
        body = json.dumps({"object": "list"}).encode("utf-8")
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda _req, timeout=None: _StubResponse(body)
        )
        with pytest.raises(ValueError, match="missing a 'data' list"):
            pu._http_fetch_models(pu.DEFAULT_SOURCE_URL, timeout=1.0)

    def test_returns_data_list_on_well_formed_payload(self, monkeypatch):
        body = json.dumps({"data": _catalog()}).encode("utf-8")
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda _req, timeout=None: _StubResponse(body)
        )
        out = pu._http_fetch_models(pu.DEFAULT_SOURCE_URL, timeout=1.0)
        assert out == _catalog()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _full_cover_fetch(_url, *, timeout):  # noqa: ANN001 - test stub
    """Catalog that prices every model in the `pricing_file` fixture."""
    return _catalog() + [
        {
            "id": "openai/text-embedding-3-small",
            "pricing": {"prompt": "0.00000002", "completion": "0"},
        }
    ]


class TestMain:
    def test_success_exit_zero(self, pricing_file, monkeypatch, capsys):
        monkeypatch.setattr(pu, "_http_fetch_models", _ok_fetch)
        code = pu.main(["--pricing-file", str(pricing_file)])
        assert code == 0
        assert "updated" in capsys.readouterr().out

    def test_no_unmatched_omits_the_unmatched_line(
        self, pricing_file, monkeypatch, capsys
    ):
        monkeypatch.setattr(pu, "_http_fetch_models", _full_cover_fetch)
        code = pu.main(["--pricing-file", str(pricing_file)])
        assert code == 0
        out = capsys.readouterr().out
        assert "updated 4 model(s)" in out
        assert "unmatched" not in out

    def test_dry_run_exit_zero_without_write(self, pricing_file, monkeypatch, capsys):
        before = pricing_file.read_bytes()
        monkeypatch.setattr(pu, "_http_fetch_models", _ok_fetch)
        code = pu.main(["--pricing-file", str(pricing_file), "--dry-run"])
        assert code == 0
        assert "would update" in capsys.readouterr().out
        assert pricing_file.read_bytes() == before

    def test_http_error_exit_one(self, pricing_file, monkeypatch, capsys):
        before = pricing_file.read_bytes()

        def _boom(_url, *, timeout):  # noqa: ANN001 - test stub
            raise urllib.error.HTTPError(_url, 503, "down", {}, None)

        monkeypatch.setattr(pu, "_http_fetch_models", _boom)
        code = pu.main(["--pricing-file", str(pricing_file)])
        assert code == 1
        assert "FAILED" in capsys.readouterr().out
        assert pricing_file.read_bytes() == before
