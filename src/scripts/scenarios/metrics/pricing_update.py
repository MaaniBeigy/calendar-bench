"""Refresh `config/llm_pricing.json` from a live provider price catalog.

The CLI fetches a structured price catalog (OpenRouter's public, auth-
free models endpoint by default, which lists both OpenAI and Anthropic
models with per-token prices), maps each tracked `(provider, model)` to
its catalog entry, and rewrites the JSON with fresh per-million prices.

The file is rewritten only when the fetch succeeds and at least one
tracked model matches.  Any HTTP, network, or parse failure leaves the
file byte-for-byte unchanged.  Run with::

    python -m src.scripts.scenarios.metrics.pricing_update
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.scripts.scenarios.metrics.telemetry import PRICING_FILE, load_pricing

log = logging.getLogger(__name__)

# OpenRouter mirrors both vendors' official per-token prices in one
# auth-free JSON catalog; it is the only structured source that lists
# OpenAI and Anthropic models together.
DEFAULT_SOURCE_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_TIMEOUT = 30.0
_PER_MILLION = 1_000_000


@dataclass
class PricingUpdateResult:
    """Outcome of one `update_pricing_file` run."""

    ok: bool
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    error: str | None = None


def _normalize(name: str) -> str:
    """Lowercase and drop non-alphanumerics so `4.8` and `4-8` compare equal."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _http_fetch_models(url: str, *, timeout: float) -> list[dict[str, Any]]:
    """GET the catalog and return its `data` list; raise on any HTTP error."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("catalog response missing a 'data' list")
    return data


def _catalog_to_lookup(
    catalog: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Index the catalog as `{(provider, normalized_model): {in, out}}`."""
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for entry in catalog:
        model_id = entry.get("id") or "" if isinstance(entry, dict) else ""
        if "/" not in model_id:
            continue
        provider, _, rest = model_id.partition("/")
        pricing = entry.get("pricing") or {}
        try:
            in_m = round(float(pricing["prompt"]) * _PER_MILLION, 6)
            out_m = round(float(pricing["completion"]) * _PER_MILLION, 6)
        except (KeyError, TypeError, ValueError):
            continue
        # Catalogs use a negative price (e.g. -1) as a sentinel for
        # variable or unavailable pricing; never write those.
        if in_m < 0 or out_m < 0:
            continue
        lookup[(provider, _normalize(rest))] = {"in": in_m, "out": out_m}
    return lookup


def update_pricing_file(
    *,
    pricing_path: Path = PRICING_FILE,
    source_url: str = DEFAULT_SOURCE_URL,
    fetch: Callable[..., list[dict[str, Any]]] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    dry_run: bool = False,
) -> PricingUpdateResult:
    """Rewrite `pricing_path` with fresh in/out prices from the catalog.

    The fetch runs before any write, so a failed fetch returns an
    `ok=False` result with the file untouched.  Models the catalog does
    not cover keep their existing prices and are listed in `unmatched`.
    `fetch` resolves to the live HTTP fetcher at call time when omitted,
    so a monkeypatched `_http_fetch_models` is honoured in tests.
    """
    if fetch is None:
        fetch = _http_fetch_models
    try:
        current = json.loads(Path(pricing_path).read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            raise ValueError("pricing file is not a JSON object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return PricingUpdateResult(ok=False, error=f"cannot read pricing file: {exc}")

    try:
        catalog = fetch(source_url, timeout=timeout)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - any fetch failure must keep the file intact
        log.warning("price fetch from %s failed (%s); file unchanged.", source_url, exc)
        return PricingUpdateResult(ok=False, error=str(exc))

    lookup = _catalog_to_lookup(catalog)
    matched: list[str] = []
    unmatched: list[str] = []
    updated = {provider: dict(models) for provider, models in current.items()}
    for provider, models in updated.items():
        if not isinstance(models, dict):
            continue
        for model in models:
            fresh = lookup.get((provider, _normalize(model)))
            if fresh is None:
                unmatched.append(f"{provider}/{model}")
                continue
            models[model] = fresh
            matched.append(f"{provider}/{model}")

    if not matched:
        return PricingUpdateResult(
            ok=False,
            matched=matched,
            unmatched=unmatched,
            error="no tracked model matched the catalog; file unchanged",
        )

    if not dry_run:
        Path(pricing_path).write_text(
            json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Refresh the telemetry cache only when the canonical file changed.
        if Path(pricing_path).resolve() == Path(PRICING_FILE).resolve():
            load_pricing(force=True)
    return PricingUpdateResult(ok=True, matched=matched, unmatched=unmatched)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the update, and print a one-line summary."""
    parser = argparse.ArgumentParser(
        description="Refresh llm_pricing.json from a provider price catalog."
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--pricing-file", type=Path, default=PRICING_FILE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report the changes without writing the file.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    result = update_pricing_file(
        pricing_path=args.pricing_file,
        source_url=args.source_url,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    if not result.ok:
        print(f"[update-pricing] FAILED: {result.error} (file unchanged)")
        return 1
    verb = "would update" if args.dry_run else "updated"
    print(
        f"[update-pricing] {verb} {len(result.matched)} model(s): "
        f"{', '.join(result.matched)}"
    )
    if result.unmatched:
        print(
            f"[update-pricing] unmatched (left as-is): "
            f"{', '.join(result.unmatched)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
