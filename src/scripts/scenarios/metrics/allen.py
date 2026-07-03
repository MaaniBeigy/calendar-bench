"""Allen interval algebra and selector-driven admissible-set construction.

Backs the consistency leg L_cal of the scheduling loss. For two
eventualities e and e' on the same day, rel(e, e') is the unique
Allen relation between their intervals I(e) = [tau_s, tau_e) and
I(e') = [tau_s', tau_e'), and v(e, e') in [0, 1] is the buffer-aware
violation. Implements the 13 basic Allen relations from Allen (1983)
over half-open integer intervals `[s, e)` (minutes from midnight or any
consistent unit; both intervals must be non-instantaneous: `s_i < e_i`).

Selector grammar
----------------
`AllenPairRule.event_a` / `event_b` are :class:`SelectorPredicate`
instances that name activity classes by `name` literal, `intensity`
bucket, `domain` (HealthTasks branch), or raw `met_min` / `met_max`
range.  :func:`build_admissible_pair` walks every rule whose pair of
selectors matches the `(activity_a, activity_b)` pair and intersects
their admissible relation sets; most-restrictive wins. The result is the
admissible set R_ee', holding exactly the relations every applicable
rule allows.

Fallback chain (no rule matches):
  1. Either side has the other listed in `concurrent_with` to `R_ALL`.
  2. Either side carries `is_concurrent` to `R_ALL`.
  3. Default to `R_SEP`.

Buffer-aware violation
----------------------
`compute_buffer_violation(rel, gap, buffer_minutes)` produces the
per-pair contribution v(e, e') to L_cal: 0.0 when rel(e, e') is in the
admissible set R_ee', 1.0 for a hard touching or overlap violation, and
the near-miss ramp `1 - rho / beta` in [0, 1] when the relation is
`p` / `P` and the gap rho(e, e') sits inside the configured buffer beta.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from src.scripts.persona.config.schema import AllenPairRule, SelectorPredicate
from src.scripts.scenarios.domain.calendar import CalendarEvent
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.ontology_source import infer_source


class AllenRelation(enum.Enum):
    """The 13 basic Allen interval relations between I(e) and I(e').

    Lowercase members are the "forward" direction (I(e) relative to
    I(e')); uppercase members are their converses.  `e` (equals) is
    self-converse.
    """

    p = "p"  # precedes      ; e_i < s_j
    m = "m"  # meets         ; e_i == s_j
    o = "o"  # overlaps      ; s_i < s_j < e_i < e_j
    s = "s"  # starts        ; s_i == s_j, e_i < e_j
    d = "d"  # during        ; s_j < s_i, e_i < e_j
    f = "f"  # finishes      ; s_i > s_j, e_i == e_j
    e = "e"  # equals        ; s_i == s_j, e_i == e_j
    P = "P"  # preceded_by   ; s_i > e_j
    M = "M"  # met_by        ; s_i == e_j
    O = "O"  # overlapped_by ; s_j < s_i < e_j < e_i  # noqa: E741
    S = "S"  # started_by    ; s_i == s_j, e_i > e_j
    D = "D"  # contains      ; s_i < s_j, e_i > e_j
    F = "F"  # finished_by   ; s_i < s_j, e_i == e_j


#: Relations where the two intervals share no time (non-overlapping).
R_SEP: frozenset[AllenRelation] = frozenset(
    {AllenRelation.p, AllenRelation.m, AllenRelation.M, AllenRelation.P}
)

#: Relations where the two intervals share at least one point.
R_MERGE: frozenset[AllenRelation] = frozenset(
    {
        AllenRelation.o,
        AllenRelation.O,
        AllenRelation.s,
        AllenRelation.S,
        AllenRelation.d,
        AllenRelation.D,
        AllenRelation.f,
        AllenRelation.F,
        AllenRelation.e,
    }
)

#: All 13 Allen relations; R_SEP ∪ R_MERGE.
R_ALL: frozenset[AllenRelation] = R_SEP | R_MERGE

#: Buffer-eligible separating relations.  `m` / `M` are touching, not
#: bufferable; `p` / `P` admit the near-miss ramp `1 - rho / beta`.
R_BUFFER: frozenset[AllenRelation] = frozenset({AllenRelation.p, AllenRelation.P})


def compute_allen_relation(
    s_i: int,
    e_i: int,
    s_j: int,
    e_j: int,
) -> AllenRelation:
    """Return the rel(e, e') of I(e) = `[s_i, e_i)` against I(e') = `[s_j, e_j)`.

    Precondition: `s_i < e_i` and `s_j < e_j`.
    """
    if e_i < s_j:
        return AllenRelation.p
    if s_i > e_j:
        return AllenRelation.P
    if e_i == s_j:
        return AllenRelation.m
    if s_i == e_j:
        return AllenRelation.M
    if s_i == s_j:
        if e_i == e_j:
            return AllenRelation.e
        if e_i < e_j:
            return AllenRelation.s
        return AllenRelation.S
    if s_i < s_j:
        if e_i == e_j:
            return AllenRelation.F
        if e_i > e_j:
            return AllenRelation.D
        return AllenRelation.o
    # s_i > s_j
    if e_i == e_j:
        return AllenRelation.f
    if e_i < e_j:
        return AllenRelation.d
    return AllenRelation.O


def is_overlapping(s_i: int, e_i: int, s_j: int, e_j: int) -> bool:
    """Return True iff the two half-open intervals share at least one point.

    Touching endpoints (`m` / `M`) are *not* considered overlapping.
    """
    return compute_allen_relation(s_i, e_i, s_j, e_j) in R_MERGE


# ---------------------------------------------------------------------------
# SelectorMatcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedActivity:
    """Snapshot of an activity's match-able attributes for selector dispatch."""

    label: str
    intensity: int | None = None
    domain: str | None = None
    met: float | None = None
    health_task_uri: str | None = None
    health_task_classes: frozenset[str] = frozenset()


class SelectorMatcher:
    """Test whether a :class:`SelectorPredicate` matches a :class:`ResolvedActivity`.

    The class only carries static configuration (an injected
    `IntensityResolver` for resolved-intensity lookups when the caller
    hands raw activities instead of pre-resolved ones).  `matches` is
    a pure function over its inputs.
    """

    def __init__(
        self,
        resolver: Any | None = None,
        class_closure: dict[str, frozenset[str]] | None = None,
    ) -> None:
        self._resolver = resolver
        self._class_closure: dict[str, frozenset[str]] = dict(class_closure or {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def matches(self, sel: SelectorPredicate, activity: Any) -> bool:
        """Return True iff *activity* satisfies every set field of *sel*."""
        if sel.kind == "context" and not isinstance(activity, ContextEpisode):
            return False
        if sel.kind == "task" and not isinstance(
            activity, (RecommendedTask, ScheduledTask)
        ):
            return False
        if sel.kind == "event" and isinstance(activity, ContextEpisode):
            return False

        resolved = self.resolve(activity)

        if sel.name is not None:
            names = [sel.name] if isinstance(sel.name, str) else list(sel.name)
            if resolved.label not in names:
                return False

        if sel.intensity is not None:
            if resolved.intensity is None or resolved.intensity not in sel.intensity:
                return False

        if sel.domain is not None:
            if resolved.domain is None or resolved.domain != sel.domain:
                return False

        if sel.met_min is not None:
            if resolved.met is None or resolved.met < sel.met_min:
                return False
        if sel.met_max is not None:
            if resolved.met is None or resolved.met > sel.met_max:
                return False

        if sel.health_task_class is not None:
            required = (
                [sel.health_task_class]
                if isinstance(sel.health_task_class, str)
                else list(sel.health_task_class)
            )
            if not all(c in resolved.health_task_classes for c in required):
                return False

        if sel.health_task_uri is not None:
            allowed = (
                [sel.health_task_uri]
                if isinstance(sel.health_task_uri, str)
                else list(sel.health_task_uri)
            )
            if resolved.health_task_uri not in allowed:
                return False

        return True

    def resolve(self, activity: Any) -> ResolvedActivity:
        """Project *activity* onto a :class:`ResolvedActivity` snapshot.

        `RecommendedTask` / `ScheduledTask` / `CalendarEvent` /
        :class:`ResolvedActivity` are all accepted.  When the matcher
        was constructed with an `IntensityResolver` the resolved
        intensity AND raw MET are filled in from its record; otherwise
        only the label and any locally-known fields are populated.
        """
        if isinstance(activity, ResolvedActivity):
            return activity

        if isinstance(activity, ScheduledTask):
            base = activity.task
            label = base.label
        elif isinstance(activity, RecommendedTask):
            base = activity
            label = activity.label
        elif isinstance(activity, CalendarEvent):
            base = None
            label = activity.label
        elif isinstance(activity, ContextEpisode):
            return ResolvedActivity(label=activity.name)
        else:
            base = None
            label = getattr(activity, "label", "")

        intensity: int | None = None
        domain: str | None = None
        met: float | None = None

        if self._resolver is not None:
            try:
                rec = self._resolver.record(activity)
                intensity = rec.intensity if rec.intensity > 0 else None
                met = rec.met
            except Exception:  # pragma: no cover - defensive
                intensity = None
                met = None
        if intensity is None and isinstance(activity, CalendarEvent):
            ev_int = getattr(activity, "intensity", None)
            if ev_int and ev_int > 0:
                # Cap at 4 (event intensities are 1–5; collapse 5 to 4).
                intensity = 4 if ev_int >= 4 else int(ev_int)
        if domain is None and base is not None:
            domain = infer_source(base.ontology_uri)

        health_task_uri: str | None = None
        health_task_classes: frozenset[str] = frozenset()
        if base is not None and base.ontology_uri:
            uri = base.ontology_uri
            if self._class_closure:
                closure = self._class_closure.get(uri)
                if closure is not None:
                    health_task_uri = uri
                    health_task_classes = closure
        return ResolvedActivity(
            label=label,
            intensity=intensity,
            domain=domain,
            met=met,
            health_task_uri=health_task_uri,
            health_task_classes=health_task_classes,
        )


# ---------------------------------------------------------------------------
# RuleSet
# ---------------------------------------------------------------------------


class RuleSet:
    """Thin wrapper around `list[AllenPairRule]` for selector-driven lookup.

    The `build_admissible_pair` API walks every rule and keeps the
    intersection of those that match.  Memoizing on
    `(label_a, label_b, intensity_a, intensity_b)` avoids re-walking
    the rule list every pair when many tasks share a label.
    """

    def __init__(self, rules: list[AllenPairRule]) -> None:
        self._rules: list[AllenPairRule] = list(rules)
        self._cache: dict[
            tuple[str, str, int | None, int | None], frozenset[AllenRelation]
        ] = {}
        self._buffer_cache: dict[
            tuple[str, str, int | None, int | None], int | None
        ] = {}

    @property
    def rules(self) -> list[AllenPairRule]:
        return list(self._rules)

    def build_admissible(
        self,
        a: Any,
        b: Any,
        matcher: SelectorMatcher,
        *,
        fallback: frozenset[AllenRelation] | None = None,
    ) -> frozenset[AllenRelation]:
        """Return R_ee', the intersection of admissible sets across matching rules.

        Direction-aware: a rule fires when its `event_a` matches the
        first activity AND `event_b` matches the second, or when the
        roles are swapped (so the YAML author can write the rule once
        without worrying about ordering).  The same admissible set is
        used in both orientations because Allen relations are symmetric
        with respect to the converse mapping (admissible relations
        already cover both directions where needed).

        When no rule matches, returns the supplied `fallback` (or
        `R_SEP` when omitted).
        """
        ra = matcher.resolve(a)
        rb = matcher.resolve(b)
        cache_key = (ra.label, rb.label, ra.intensity, rb.intensity)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        best: frozenset[AllenRelation] | None = None
        for rule in self._rules:
            forward = matcher.matches(rule.event_a, ra) and matcher.matches(
                rule.event_b, rb
            )
            backward = matcher.matches(rule.event_a, rb) and matcher.matches(
                rule.event_b, ra
            )
            if not (forward or backward):
                continue
            adm = frozenset(AllenRelation(tok) for tok in rule.admissible_relations)
            best = adm if best is None else (best & adm)

        result = (
            best if best is not None else (fallback if fallback is not None else R_SEP)
        )
        self._cache[cache_key] = result
        return result

    def buffer_for(
        self,
        a: Any,
        b: Any,
        matcher: SelectorMatcher,
        *,
        default: int | None,
    ) -> int | None:
        """Return the rule-level `buffer` minutes for the first matching rule.

        When multiple rules match, the smallest non-None `buffer` wins
        (most restrictive).  When no rule has a `buffer` set the
        supplied `default` is returned.
        """
        ra = matcher.resolve(a)
        rb = matcher.resolve(b)
        cache_key = (ra.label, rb.label, ra.intensity, rb.intensity)
        cached = self._buffer_cache.get(cache_key)
        if cached is not None:
            return cached

        best: int | None = None
        for rule in self._rules:
            if rule.buffer is None:
                continue
            forward = matcher.matches(rule.event_a, ra) and matcher.matches(
                rule.event_b, rb
            )
            backward = matcher.matches(rule.event_a, rb) and matcher.matches(
                rule.event_b, ra
            )
            if not (forward or backward):
                continue
            best = rule.buffer if best is None else min(best, rule.buffer)

        out = best if best is not None else default
        self._buffer_cache[cache_key] = out
        return out


# ---------------------------------------------------------------------------
# Public selector-aware admissible-pair builder
# ---------------------------------------------------------------------------


def _fallback_for(a: Any, b: Any) -> frozenset[AllenRelation]:
    """Return the no-rule fallback admissible set."""
    a_label = (
        getattr(a, "label", None)
        or getattr(getattr(a, "task", None), "label", None)
        or ""
    )
    b_label = (
        getattr(b, "label", None)
        or getattr(getattr(b, "task", None), "label", None)
        or ""
    )
    a_concurrent_with = list(getattr(a, "concurrent_with", []) or [])
    if isinstance(a_concurrent_with, str):  # pragma: no cover
        a_concurrent_with = [a_concurrent_with] if a_concurrent_with else []
    b_concurrent_with = list(getattr(b, "concurrent_with", []) or [])
    if isinstance(b_concurrent_with, str):  # pragma: no cover
        b_concurrent_with = [b_concurrent_with] if b_concurrent_with else []
    if (b_label and b_label in a_concurrent_with) or (
        a_label and a_label in b_concurrent_with
    ):
        return R_ALL
    a_concurrent = bool(_get_concurrent_flag(a))
    b_concurrent = bool(_get_concurrent_flag(b))
    if a_concurrent or b_concurrent:
        return R_ALL
    return R_SEP


def _get_concurrent_flag(activity: Any) -> bool:
    """Pull the `is_concurrent` flag from any activity-shaped object."""
    if isinstance(activity, ScheduledTask):
        return bool(activity.task.is_concurrent)
    if hasattr(activity, "is_concurrent"):
        return bool(getattr(activity, "is_concurrent"))
    return False


def build_admissible_pair(
    a: Any,
    b: Any,
    ruleset: RuleSet,
    matcher: SelectorMatcher,
) -> frozenset[AllenRelation]:
    """Return the admissible set R_ee' of Allen relations for the pair *(a, b)*.

    Selector rules in *ruleset* are checked first (intersection across
    matches).  When no rule fires the no-rule fallback chain runs:

    1. Either side names the other in `concurrent_with` to `R_ALL`.
    2. Either side carries `is_concurrent` to `R_ALL`.
    3. Default to `R_SEP`.
    """
    return ruleset.build_admissible(a, b, matcher, fallback=_fallback_for(a, b))


# ---------------------------------------------------------------------------
# Buffer-aware partial-credit violation
# ---------------------------------------------------------------------------


def compute_buffer_violation(
    rel: AllenRelation,
    gap: float,
    buffer_minutes: int | None,
    admissible: frozenset[AllenRelation],
) -> float:
    """Return the per-pair violation v(e, e') in [0, 1] for *(rel, gap)*.

    Here `rel` is rel(e, e'), `gap` is the gap rho(e, e'),
    `buffer_minutes` is the buffer beta, and `admissible` is R_ee':

      * `rel ∈ admissible` to 0.0 (no violation).
      * `rel` is `m` / `M` to 1.0 (touching counted as full violation
        since a buffer of 0 is the offending case).
      * `rel` is `p` / `P` and beta > 0:
        - `gap >= buffer_minutes` to 0.0.
        - `gap < buffer_minutes` to the near-miss ramp
          `1 - rho / beta`.
      * Any overlap variant not in `admissible` to 1.0.

    A `buffer_minutes` of `None` or 0 disables the buffer ramp; `p`/`P`
    relations with positive gaps still count as 0.0 (they are always
    separating, with or without buffer).
    """
    if rel in admissible:
        return 0.0
    if rel in (AllenRelation.m, AllenRelation.M):
        return 1.0
    if rel in R_BUFFER:
        if not buffer_minutes or buffer_minutes <= 0:
            return 0.0
        if gap >= buffer_minutes:
            return 0.0
        return 1.0 - (gap / float(buffer_minutes))
    # Some overlap relation that the rule does not admit.
    return 1.0


# ---------------------------------------------------------------------------
# Backwards-compatible legacy helpers (kept for tests still on the old API).
# ---------------------------------------------------------------------------

LegacyRuleIndex = dict[frozenset[str], frozenset[AllenRelation]]


def _build_rule_index(rules: list[AllenPairRule]) -> LegacyRuleIndex:
    """Legacy: map `frozenset({event_a.name, event_b.name})` to admissible set.

    Only literal-name endpoints are indexed; selector-only rules are
    skipped (they require :class:`RuleSet` matching).  Kept so older
    tests on the legacy API keep passing.
    """
    index: LegacyRuleIndex = {}
    for rule in rules:
        a_name = rule.event_a.name if isinstance(rule.event_a.name, str) else None
        b_name = rule.event_b.name if isinstance(rule.event_b.name, str) else None
        if a_name is None or b_name is None:
            continue
        key: frozenset[str] = frozenset({a_name, b_name})
        index[key] = frozenset(AllenRelation(tok) for tok in rule.admissible_relations)
    return index


def build_admissible_rx(
    task: RecommendedTask,
    event: CalendarEvent,
    rule_index: LegacyRuleIndex,
) -> frozenset[AllenRelation]:
    """Legacy admissible-set for (task, event) by literal-label lookup.

    Priority:
      1. Explicit literal-name rule for (task.label, event.label).
      2. `task.label` in `event.concurrent_with` to `R_ALL`.
      3. `task.is_concurrent` or `event.is_concurrent` to `R_ALL`.
      4. Default to `R_SEP`.
    """
    key = frozenset({task.label, event.label})
    if key in rule_index:
        return rule_index[key]
    if task.label in event.concurrent_with:
        return R_ALL
    if task.is_concurrent or event.is_concurrent:
        return R_ALL
    return R_SEP
