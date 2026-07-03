"""Live integration test for the pricing updater against OpenRouter.

Hits the real, auth-free OpenRouter models catalog to confirm the
fetch + parse + match pipeline still lines up with the source schema.
Marked `integration` so the unit gate skips it; self-skips when the
network is unreachable.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from src.scripts.scenarios.metrics import pricing_update as pu

pytestmark = pytest.mark.integration


def _fetch_live():
    try:
        return pu._http_fetch_models(pu.DEFAULT_SOURCE_URL, timeout=pu.DEFAULT_TIMEOUT)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"OpenRouter catalog unreachable: {exc}")


def test_live_catalog_has_parseable_prices():
    catalog = _fetch_live()
    assert isinstance(catalog, list) and catalog
    lookup = pu._catalog_to_lookup(catalog)
    assert lookup, "no parseable prices in the live catalog"
    # Every parsed price is a non-negative per-million figure.
    for price in lookup.values():
        assert price["in"] >= 0.0 and price["out"] >= 0.0


def test_live_update_reprices_tracked_models(tmp_path):
    catalog = _fetch_live()
    pricing_file = tmp_path / "llm_pricing.json"
    pricing_file.write_text(
        json.dumps(
            {
                "openai": {"gpt-4o-mini": {"in": 0.0, "out": 0.0}},
                "anthropic": {"claude-opus-4-8": {"in": 0.0, "out": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    result = pu.update_pricing_file(
        pricing_path=pricing_file, fetch=lambda _u, *, timeout: catalog
    )
    assert result.ok, result.error
    assert result.matched, "expected at least one tracked model in the live catalog"
    table = json.loads(pricing_file.read_text(encoding="utf-8"))
    # At least one matched model now carries a positive output price.
    repriced = [
        table[p][m] for entry in result.matched for p, m in [entry.split("/", 1)]
    ]
    assert any(price["out"] > 0.0 for price in repriced)
