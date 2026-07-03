"""Stage 3 of the grounded task-generation pipeline; the three gates.

Each (canonical, paraphrase) pair is scored by three independent gates
that must ALL pass for the paraphrase to ship to the augmenter:

  * GATE A; length delta:
      `|len(p) − len(c)| / max(len(c), 1) ≤ length_delta_pct`

  * GATE B; emoji multiset equality (description-only):
      `Counter(emojis(p)) == Counter(emojis(canonical_description))`
      The display_name's emoji is preserved verbatim outside the
      paraphrase loop and is therefore EXCLUDED from this comparison.

  * GATE C; semantic similarity:
      `cosine(emb(canonical_text), emb(paraphrase)) ≥ similarity_threshold`
      where `canonical_text = display_name + ". " + description` so
      the topic anchor is rich.

Any failure to fall back to the canonical description; the
:class:`GateDecision` records the failing gate and every measured
value so the caller can persist a complete audit row in telemetry.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# Common Unicode emoji blocks.  This is a pragmatic subset; the regex
# package's `\p{Emoji}` covers more, but pulling that dependency in
# only buys us regional indicators and rare professional combos.  An
# extractor miss on BOTH sides is harmless (multiset comparison ignores
# it); a one-sided miss could falsely pass Gate B but is unlikely with
# the ranges below.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # symbols, transport, misc, emoticons, supplement
    "\U00002600-\U000027bf"  # dingbats + misc symbols
    "\U0001fa70-\U0001faff"  # extended-a, symbols-and-pictographs-extended-a
    "]",
    flags=re.UNICODE,
)


def _extract_emojis(text: str) -> list[str]:
    """Return all emoji characters in *text*, preserving multiplicity.

    Order is preserved (matches appear in the order they occur), but
    the gate compares `Counter` objects so the consumer is order-free.
    """
    if not text:
        return []
    return _EMOJI_RE.findall(text)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length float vectors.

    Returns `0.0` if either vector is the zero vector or the lengths
    differ; never raises on degenerate input.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class GateDecision:
    """Outcome of running the three gates on a single paraphrase pair.

    Attributes:
        text:           the description that should ship downstream
                        (paraphrase iff every gate passed, else canonical).
        used:           `"paraphrase"` or `"canonical"`.
        length_delta:   measured `|Δlen| / len(c_desc)`.
        similarity:     measured cosine on the rich anchor.
        missing_emojis: emojis present in canonical but missing from
                        paraphrase, as `{emoji: count_diff}`.
        extra_emojis:   emojis present in paraphrase but missing from
                        canonical, as `{emoji: count_diff}`.
        failed_gate:    `"length"` / `"emoji"` / `"similarity"`,
                        or `None` when every gate passed.
    """

    text: str
    used: str
    length_delta: float
    similarity: float
    missing_emojis: dict[str, int] = field(default_factory=dict)
    extra_emojis: dict[str, int] = field(default_factory=dict)
    failed_gate: str | None = None


def gate_paraphrase(
    *,
    canonical_description: str,
    canonical_text: str,
    paraphrase: str,
    embedder: Any,
    length_delta_pct: float,
    similarity_threshold: float,
) -> GateDecision:
    """Run the three gates and return the audit-ready `GateDecision`.

    Args:
        canonical_description: the ontology description string.  Used
            by gates A (length) and B (emoji multiset).
        canonical_text: typically `f"{display_name}. {description}"`;
            used by gate C (similarity) only, so the topic anchor is
            rich.
        paraphrase: the LLM's personalized description.
        embedder: any object exposing `embed_query(text) -> list[float]`.
        length_delta_pct: maximum allowed |Δlen| / len ratio (0–1).
        similarity_threshold: minimum cosine for gate C to pass (0–1).

    Returns:
        `GateDecision` carrying the chosen text, every measured value,
        and the failing gate (if any).  On embedder failure, gate C is
        treated as failed and the canonical description is used (a
        WARNING is logged).
    """
    # GATE A; length delta on the description-only anchor.
    len_c = len(canonical_description)
    len_p = len(paraphrase)
    length_delta = abs(len_p - len_c) / max(len_c, 1)
    length_ok = length_delta <= length_delta_pct

    # GATE B; emoji multiset equality, description-vs-description.
    em_c = Counter(_extract_emojis(canonical_description))
    em_p = Counter(_extract_emojis(paraphrase))
    emoji_ok = em_c == em_p
    missing = em_c - em_p  # in canonical, not in paraphrase
    extra = em_p - em_c  # in paraphrase, not in canonical

    # GATE C; semantic similarity on the rich anchor.
    similarity = 0.0
    sim_ok = False
    try:
        a = embedder.embed_query(canonical_text)
        b = embedder.embed_query(paraphrase)
        similarity = _cosine(a, b)
        sim_ok = similarity >= similarity_threshold
    except Exception as exc:  # noqa: BLE001; embedder errors are diverse
        log.warning(
            "gate_paraphrase: embedder failed (%s); gate C marked as failed "
            "and canonical description will be used.",
            exc,
        )
        sim_ok = False

    if length_ok and emoji_ok and sim_ok:
        return GateDecision(
            text=paraphrase,
            used="paraphrase",
            length_delta=length_delta,
            similarity=similarity,
            missing_emojis=dict(missing),
            extra_emojis=dict(extra),
            failed_gate=None,
        )

    # Determine which gate caught it (in priority order length to emoji to
    # similarity) so the telemetry has a single deterministic label.
    failed = "length" if not length_ok else "emoji" if not emoji_ok else "similarity"
    return GateDecision(
        text=canonical_description,
        used="canonical",
        length_delta=length_delta,
        similarity=similarity,
        missing_emojis=dict(missing),
        extra_emojis=dict(extra),
        failed_gate=failed,
    )
