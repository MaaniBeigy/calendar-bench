"""One-shot script: compute MET-quartile cutoffs from the HumanActivities ontology.

Reads every `head(n.metValue)` from the live Neo4j graph (HumanActivities
prefix) and writes the 25th / 50th / 75th percentile cutoffs plus the
maximum MET value to `output/met_quartiles.json`.  Re-run only when the
HumanActivities ontology version changes; the cutoffs are deterministic
functions of the underlying value distribution.

Output schema::

    {
        "q1": <float>,            # 25th percentile
        "q2": <float>,            # 50th percentile
        "q3": <float>,            # 75th percentile
        "max_met": <float>,       # maximum MET value across all activities
        "n": <int>,               # sample size used for the percentiles
        "ontology_prefix": "https://w3id.org/calendar-bench/human-activities/",
        "ts": "<ISO-8601 UTC timestamp>"
    }

Usage::

    python -m src.scripts.scenarios.precompute_met_quartiles \\
        --output ./output/met_quartiles.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("precompute_met_quartiles")

EXIT_OK = 0
EXIT_USAGE = 2

#: HumanActivities URI prefix; the only namespace that carries `ha:metValue`.
HUMAN_ACTIVITIES_PREFIX: str = "https://w3id.org/calendar-bench/human-activities/"

#: Cypher to pull every `head(metValue)` from HumanActivities.  Wrapping
#: `metValue` in `head(...)` because n10s is configured with
#: `handleMultival: ARRAY` so even single-valued literals are stored as
#: 1-element lists.
_MET_QUERY = """
MATCH (n:Resource)
WHERE n.uri STARTS WITH $prefix
  AND n.metValue IS NOT NULL
WITH head(n.metValue) AS met
WHERE met IS NOT NULL
RETURN toFloat(met) AS met
"""


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolated percentile; matches numpy default `method='linear'`.

    Args:
        sorted_values: ascending list of floats; must be non-empty.
        p: percentile in [0, 100].

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: when `sorted_values` is empty.
    """
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty list")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def fetch_met_values(driver, *, prefix: str = HUMAN_ACTIVITIES_PREFIX) -> list[float]:
    """Pull every HumanActivities MET value from the live Neo4j graph.

    Args:
        driver: open Neo4j driver instance.
        prefix: HumanActivities URI prefix (overridable for tests).

    Returns:
        A list of float MET values (unsorted, duplicates preserved).
    """
    values: list[float] = []
    with driver.session() as session:
        result = session.run(_MET_QUERY, prefix=prefix)
        for record in result:
            met = record.get("met")
            if met is None:
                continue
            values.append(float(met))
    return values


def compute_quartiles(values: list[float]) -> dict[str, float]:
    """Compute Q1 / Q2 / Q3 / max from a list of MET values.

    Args:
        values: raw MET values (any order, may contain duplicates).

    Returns:
        Dict with keys `q1`, `q2`, `q3`, `max_met`, `n`.

    Raises:
        ValueError: when `values` is empty.
    """
    if not values:
        raise ValueError("no MET values to compute quartiles from")
    sorted_values = sorted(values)
    return {
        "q1": _percentile(sorted_values, 25.0),
        "q2": _percentile(sorted_values, 50.0),
        "q3": _percentile(sorted_values, 75.0),
        "max_met": float(sorted_values[-1]),
        "n": len(sorted_values),
    }


def write_quartiles_file(
    payload: dict[str, float],
    *,
    output_path: Path,
    prefix: str = HUMAN_ACTIVITIES_PREFIX,
) -> None:
    """Serialise the quartile payload to `output_path` with a UTC timestamp."""
    record = dict(payload)
    record["ontology_prefix"] = prefix
    record["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute HumanActivities MET-quartile cutoffs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/met_quartiles.json"),
        help="Where to write the quartile JSON (default: ./output/met_quartiles.json).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=HUMAN_ACTIVITIES_PREFIX,
        help=f"HumanActivities URI prefix (default: {HUMAN_ACTIVITIES_PREFIX!r}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    args = _build_argparser().parse_args(argv)

    from src.graphrag.neo4j_client import make_driver

    log.info("Pulling MET values from prefix %s …", args.prefix)
    driver = make_driver()
    try:
        values = fetch_met_values(driver, prefix=args.prefix)
    finally:
        driver.close()

    if not values:
        log.error(
            "No MET values found under prefix %s; is HumanActivities imported?",
            args.prefix,
        )
        return EXIT_USAGE

    payload = compute_quartiles(values)
    write_quartiles_file(payload, output_path=args.output, prefix=args.prefix)
    log.info(
        "Wrote %s  n=%d  q1=%.3f  q2=%.3f  q3=%.3f  max=%.3f",
        args.output,
        payload["n"],
        payload["q1"],
        payload["q2"],
        payload["q3"],
        payload["max_met"],
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
