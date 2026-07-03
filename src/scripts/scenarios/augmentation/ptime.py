"""PTIME Choquet-integral calendar augmenter with MCS branch-and-bound.

Scores each candidate placement with a 2-order Choquet integral over
four criteria: `time` (alignment with `environment.time_windows`),
`duration` (fit inside `[duration_min, duration_max]`), `overlap`
(Allen-rule admissibility against base events), and `stability` (base
events unchanged; constant 1.0). The MCS solver iterates the four
criteria by decreasing upper-bound slack, pinning each maximised level
as a local constraint for the next mono-criterion search. A fast
`greedy` decode over the same scored candidates is selectable.

When `PTimeLearningConfig.enabled` is set, an `sklearn.svm.LinearSVC`
is fitted per-person on pairwise (preferred, dispreferred) slot
examples drawn from the calendar; the resulting weight vector `B` is
blended with the elicited `A` as `alpha * A.z + (1 - alpha) * B.z`.

Deterministic given `seed`; equal-score ties break via seeded RNG.
"""

from __future__ import annotations

import datetime
import logging
import random
from collections import defaultdict
from dataclasses import dataclass

from src.scripts.persona.config.schema import AllenPairRule, DailyWindow, WindowRange
from src.scripts.scenarios.augmentation.base import Augmenter, HorizonHint
from src.scripts.scenarios.augmentation.greedy import (
    _base_gaps,
    _get_horizon_dates,
    _get_weekly_chunks,
    _tasks_for_week,
)
from src.scripts.scenarios.config.schema import (
    _PTIME_CRITERIA,
    AugmentationConfig,
    PTimeConfig,
    PTimeLearningConfig,
)
from src.scripts.scenarios.domain.calendar import (
    AugmentedCalendar,
    CalendarEvent,
    CalendarTrace,
)
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import (
    _build_rule_index,
    build_admissible_rx,
    compute_allen_relation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Choquet integral
# ---------------------------------------------------------------------------


def default_importance() -> dict[str, float]:
    """Uninformative `a_i = 1/n` for every criterion."""
    n = len(_PTIME_CRITERIA)
    return {c: 1.0 / n for c in _PTIME_CRITERIA}


def merge_importance(overrides: dict[str, float]) -> dict[str, float]:
    """Default `1/n` weights with overrides applied."""
    base = default_importance()
    base.update(overrides)
    return base


def merge_interaction(overrides: dict[str, float]) -> dict[str, float]:
    """All-zero pair interactions with overrides applied."""
    pairs: dict[str, float] = {}
    for i, a in enumerate(_PTIME_CRITERIA):
        for b in _PTIME_CRITERIA[i + 1 :]:
            pairs["|".join((a, b))] = 0.0
    pairs.update(overrides)
    return pairs


def choquet_2order(
    utilities: dict[str, float],
    importance: dict[str, float],
    interaction: dict[str, float],
) -> float:
    """2-order Choquet integral `F(z) = sum_i a_i z_i + sum_{i<j} a_ij min(z_i, z_j)`.

    Args:
        utilities: `{criterion: u_i}` in `[0, 1]`; missing keys map to 0.0.
        importance: `{criterion: a_i}` in `[0, 1]`.
        interaction: `{"a|b": a_ij}` in `[-1, 1]`, keys alphabetically sorted.

    Returns:
        Aggregated score.
    """
    total = 0.0
    for c in _PTIME_CRITERIA:
        z = float(utilities.get(c, 0.0))
        a = float(importance.get(c, 0.0))
        total += a * z
    for i, a in enumerate(_PTIME_CRITERIA):
        for b in _PTIME_CRITERIA[i + 1 :]:
            za = float(utilities.get(a, 0.0))
            zb = float(utilities.get(b, 0.0))
            key = "|".join(sorted((a, b)))
            coeff = float(interaction.get(key, 0.0))
            total += coeff * min(za, zb)
    return total


# ---------------------------------------------------------------------------
# Per-criterion utility functions
# ---------------------------------------------------------------------------


def utility_time(
    start: int,
    end: int,
    time_windows: dict[str, WindowRange],
) -> float:
    """Fraction of `[start, end)` that lies inside any band of `time_windows`.

    Returns 1.0 when `time_windows` is empty; 0.0 when bands exist but
    the slot misses them all.
    """
    if not time_windows:
        return 1.0
    span = max(1, end - start)
    covered = 0
    for w in time_windows.values():
        lo = max(start, w.start)
        hi = min(end, w.end)
        if hi > lo:
            covered += hi - lo
    return min(1.0, covered / span)


def utility_duration(
    start: int,
    end: int,
    task: RecommendedTask,
    preference: str,
) -> float:
    """Score the slot duration against `[duration_min, duration_max]`.

    `minimum` peaks at `duration_min`, `maximum` peaks at `duration_max`,
    `midpoint` peaks at the band center. Out-of-band durations score 0.0.
    """
    duration = end - start
    lo = task.duration_min
    hi = task.duration_max
    if duration < lo or duration > hi:
        return 0.0
    if hi == lo:
        return 1.0
    norm = (duration - lo) / (hi - lo)
    if preference == "minimum":
        return 1.0 - norm
    if preference == "maximum":
        return norm
    return 1.0 - abs(0.5 - norm) * 2.0


def utility_overlap(
    task: RecommendedTask,
    start: int,
    end: int,
    day: datetime.date,
    events_by_date: dict[datetime.date, list[CalendarEvent]],
    rule_index,
) -> float:
    """Mean admissibility of the slot's Allen relation against same-day events.

    Returns 1.0 when no base event touches the slot. Each touching event
    contributes 1.0 if its relation sits in the admissible set, else 0.0.
    """
    events = events_by_date.get(day, [])
    if not events:
        return 1.0
    hits = 0
    relevant = 0
    for ev in events:
        if ev.end_minutes <= start or ev.start_minutes >= end:
            continue
        relevant += 1
        admissible = build_admissible_rx(task, ev, rule_index)
        rel = compute_allen_relation(start, end, ev.start_minutes, ev.end_minutes)
        if rel in admissible:
            hits += 1
    if relevant == 0:
        return 1.0
    return hits / relevant


def score_candidate(
    task: RecommendedTask,
    start: int,
    end: int,
    day: datetime.date,
    events_by_date: dict[datetime.date, list[CalendarEvent]],
    time_windows: dict[str, WindowRange],
    rule_index,
    importance: dict[str, float],
    interaction: dict[str, float],
    duration_preference: str,
) -> tuple[float, dict[str, float]]:
    """Return `(choquet_value, per_criterion_utility)` for one slot."""
    util = {
        "time": utility_time(start, end, time_windows),
        "duration": utility_duration(start, end, task, duration_preference),
        "overlap": utility_overlap(task, start, end, day, events_by_date, rule_index),
        "stability": 1.0,
    }
    return choquet_2order(util, importance, interaction), util


# ---------------------------------------------------------------------------
# SVM-rank preference learning
# ---------------------------------------------------------------------------


def _event_to_pseudo_task(event: CalendarEvent) -> RecommendedTask:
    """Wrap a base event as a fixed-duration `RecommendedTask`."""
    span = max(1, event.end_minutes - event.start_minutes)
    return RecommendedTask(
        label=event.label,
        duration_min=span,
        duration_max=span,
        intensity=event.intensity,
        is_concurrent=event.is_concurrent,
    )


def _pair_keys() -> list[str]:
    """Alphabetically-sorted key for each criterion pair, in Choquet order."""
    keys: list[str] = []
    for i, a in enumerate(_PTIME_CRITERIA):
        for b in _PTIME_CRITERIA[i + 1 :]:
            keys.append("|".join(sorted((a, b))))
    return keys


def _feature_vector_at(
    pseudo_task: RecommendedTask,
    start: int,
    end: int,
    day: datetime.date,
    events_by_date: dict[datetime.date, list[CalendarEvent]],
    time_windows: dict[str, WindowRange],
    rule_index,
    duration_preference: str,
) -> list[float]:
    """Return the Choquet feature vector (singletons then pairwise mins) at a placement.

    Matches the 2-order integral viewed as a linear sum over the expanded
    feature set, so the SVM can recover both importances and interactions.
    """
    util = {
        "time": utility_time(start, end, time_windows),
        "duration": utility_duration(start, end, pseudo_task, duration_preference),
        "overlap": utility_overlap(
            pseudo_task, start, end, day, events_by_date, rule_index
        ),
        "stability": 1.0,
    }
    feats = [float(util[c]) for c in _PTIME_CRITERIA]
    for i, a in enumerate(_PTIME_CRITERIA):
        for b in _PTIME_CRITERIA[i + 1 :]:
            feats.append(min(float(util[a]), float(util[b])))
    return feats


def _shifted_negative_starts(
    event: CalendarEvent,
    window_start: int,
    window_end: int,
    shift_minutes: int,
    n_samples: int,
    rng: random.Random,
) -> list[int]:
    """Sample shifted alternative start minutes for one event.

    Shifts drawn uniformly from `[-shift, +shift]` minus 0, clipped to the
    daily window and deduplicated. Empty when no alternative fits.
    """
    duration = event.end_minutes - event.start_minutes
    if duration <= 0 or window_end - duration <= window_start:
        return []
    candidates: list[int] = []
    seen: set[int] = {event.start_minutes}
    attempts = 0
    while len(candidates) < n_samples and attempts < n_samples * 6:
        attempts += 1
        delta = rng.randint(-shift_minutes, shift_minutes)
        if delta == 0:
            continue
        s = event.start_minutes + delta
        s = max(window_start, min(window_end - duration, s))
        if s in seen:
            continue
        seen.add(s)
        candidates.append(s)
    return candidates


def extract_pairwise_examples(
    calendar: CalendarTrace,
    events_by_date: dict[datetime.date, list[CalendarEvent]],
    time_windows: dict[str, WindowRange],
    rule_index,
    *,
    window_start: int,
    window_end: int,
    duration_preference: str,
    negative_samples_per_event: int,
    negative_shift_minutes: int,
    rng: random.Random,
) -> list[tuple[list[float], list[float]]]:
    """Build `(positive_z, negative_z)` pairs from the calendar's base events.

    Each base event produces a positive (its actual placement) paired with
    shifted alternatives on the same day. Pairs with identical feature
    vectors carry no signal and are dropped.
    """
    pairs: list[tuple[list[float], list[float]]] = []
    for event in calendar.events:
        pseudo = _event_to_pseudo_task(event)
        pos_z = _feature_vector_at(
            pseudo,
            event.start_minutes,
            event.end_minutes,
            event.date,
            events_by_date,
            time_windows,
            rule_index,
            duration_preference,
        )
        for neg_start in _shifted_negative_starts(
            event,
            window_start,
            window_end,
            negative_shift_minutes,
            negative_samples_per_event,
            rng,
        ):
            neg_end = neg_start + (event.end_minutes - event.start_minutes)
            neg_z = _feature_vector_at(
                pseudo,
                neg_start,
                neg_end,
                event.date,
                events_by_date,
                time_windows,
                rule_index,
                duration_preference,
            )
            if neg_z != pos_z:
                pairs.append((pos_z, neg_z))
    return pairs


def _normalise_importance(weights: list[float]) -> dict[str, float]:
    """Clip importances to non-negative and L1-normalize to sum 1."""
    clipped = [max(0.0, w) for w in weights]
    total = sum(clipped)
    if total <= 0.0:
        return {c: 0.0 for c in _PTIME_CRITERIA}
    return {c: clipped[i] / total for i, c in enumerate(_PTIME_CRITERIA)}


def _rescale_interaction(weights: list[float]) -> dict[str, float]:
    """Rescale interaction weights into `[-1, 1]` by their largest magnitude."""
    keys = _pair_keys()
    peak = max((abs(w) for w in weights), default=0.0)
    if peak <= 0.0:
        return {k: 0.0 for k in keys}
    return {k: weights[i] / peak for i, k in enumerate(keys)}


def fit_learned_coefficients(
    pairs: list[tuple[list[float], list[float]]],
    *,
    regularization: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Fit `sklearn.svm.LinearSVC` on pairwise feature differences.

    Each pair adds two samples: `(positive - negative, +1)` and its
    mirror `(negative - positive, -1)`. The learned weights split into
    non-negative L1-normalized importances and pairwise interactions
    rescaled into `[-1, 1]`, or `None` when no pairs exist.
    """
    if not pairs:
        return None
    import numpy as _np
    from sklearn.svm import LinearSVC

    x: list[list[float]] = []
    y: list[int] = []
    for pos, neg in pairs:
        diff = [p - n for p, n in zip(pos, neg)]
        x.append(diff)
        y.append(1)
        x.append([-d for d in diff])
        y.append(-1)
    arr = _np.asarray(x, dtype=float)
    labels = _np.asarray(y, dtype=int)
    clf = LinearSVC(
        C=regularization,
        fit_intercept=False,
        random_state=seed,
        max_iter=5000,
        dual="auto",
    )
    clf.fit(arr, labels)
    raw = [float(w) for w in clf.coef_[0]]
    n = len(_PTIME_CRITERIA)
    return _normalise_importance(raw[:n]), _rescale_interaction(raw[n:])


def _blend_dicts(
    elicited: dict[str, float],
    learned: dict[str, float] | None,
    alpha: float,
    keys: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Return `alpha * elicited + (1 - alpha) * learned` over `keys`."""
    if learned is None:
        return {k: float(elicited.get(k, 0.0)) for k in keys}
    return {
        k: alpha * float(elicited.get(k, 0.0))
        + (1.0 - alpha) * float(learned.get(k, 0.0))
        for k in keys
    }


def blend_importance(
    elicited: dict[str, float],
    learned: dict[str, float] | None,
    alpha: float,
) -> dict[str, float]:
    """Blend elicited and learned importances; pass-through if `learned is None`."""
    return _blend_dicts(elicited, learned, alpha, _PTIME_CRITERIA)


def blend_interaction(
    elicited: dict[str, float],
    learned: dict[str, float] | None,
    alpha: float,
) -> dict[str, float]:
    """Blend elicited and learned pairwise interactions; pass-through if `None`."""
    return _blend_dicts(elicited, learned, alpha, _pair_keys())


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One scored placement option for a recommended task."""

    start: int
    end: int
    day: datetime.date
    base_gap: tuple[int, int]
    score: float
    utilities: dict[str, float]


def _candidate_durations(task: RecommendedTask) -> list[int]:
    """Distinct durations spanning the band so the duration criterion can trade off."""
    lo = task.duration_min
    hi = max(task.duration_max, lo)
    mid = (lo + hi) // 2
    return sorted({lo, mid, hi})


def _candidate_starts_for_gap(
    base_gap: tuple[int, int],
    duration: int,
    time_windows: dict[str, WindowRange],
    strategy: str,
) -> list[int]:
    """Return distinct candidate start minutes inside `base_gap`."""
    gs, ge = base_gap
    if ge - gs < duration:
        return []
    starts: list[int] = [gs]
    if strategy == "earliest_only":
        return starts
    for w in time_windows.values():
        lo = max(gs, w.start)
        if lo + duration <= ge and lo not in starts:
            starts.append(lo)
    if strategy == "windows_and_center":
        center = max(gs, min(ge - duration, (gs + ge - duration) // 2))
        if center not in starts:
            starts.append(center)
    return sorted(starts)


def enumerate_candidates(
    task: RecommendedTask,
    horizon: list[datetime.date],
    events_by_date: dict[datetime.date, list[CalendarEvent]],
    placed_by_date: dict[datetime.date, list[tuple[int, int]]],
    used_base_gaps: set[tuple[datetime.date, int, int]],
    time_windows: dict[str, WindowRange],
    rule_index,
    *,
    window_start: int,
    window_end: int,
    importance: dict[str, float],
    interaction: dict[str, float],
    duration_preference: str,
    strategy: str,
    allow_reuse: bool,
) -> list[Candidate]:
    """Walk every day's base gaps and emit scored candidates.

    Untouched gaps come first; reused gaps appear only when `allow_reuse`
    is True.
    """
    durations = _candidate_durations(task)
    results: list[Candidate] = []
    for day in horizon:
        placed_today = list(placed_by_date.get(day, []))
        for base in _base_gaps(day, events_by_date, window_start, window_end):
            key = (day, base[0], base[1])
            is_reused = key in used_base_gaps
            if is_reused and not allow_reuse:
                continue
            for duration in durations:
                starts = list(
                    _candidate_starts_for_gap(base, duration, time_windows, strategy)
                )
                # On the reuse pass, anchor extra starts right after each
                # placed task that still leaves room inside this gap.
                for ps, pe in placed_today:
                    if base[0] <= pe <= base[1] - duration and pe not in starts:
                        starts.append(pe)
                starts.sort()
                for start in starts:
                    end = start + duration
                    if end > base[1]:  # pragma: no cover  defensive
                        continue
                    if any(not (end <= s or start >= e) for s, e in placed_today):
                        continue
                    score, util = score_candidate(
                        task,
                        start,
                        end,
                        day,
                        events_by_date,
                        time_windows,
                        rule_index,
                        importance,
                        interaction,
                        duration_preference,
                    )
                    results.append(Candidate(start, end, day, base, score, util))
    return results


def _pick_best(cands: list[Candidate], rng: random.Random) -> Candidate | None:
    """Pick the highest-scoring candidate; ties break on `(day, start)` then `rng`."""
    if not cands:
        return None
    best = max(c.score for c in cands)
    top = [c for c in cands if c.score == best]
    top.sort(key=lambda c: (c.day, c.start))
    earliest_key = (top[0].day, top[0].start)
    tied = [c for c in top if (c.day, c.start) == earliest_key]
    if len(tied) == 1:
        return tied[0]
    return rng.choice(tied)


# ---------------------------------------------------------------------------
# MCS multi-criteria solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variable:
    """One recommended-task instance with its feasible candidate placements."""

    index: int
    task: RecommendedTask
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class Solution:
    """Assignment of variables to candidates; `picks[i]=None` means unscheduled."""

    picks: tuple[Candidate | None, ...]

    def utility_means(self) -> dict[str, float]:
        """Per-criterion mean utility; unscheduled picks contribute 0.0."""
        if not self.picks:
            return {c: 0.0 for c in _PTIME_CRITERIA}
        sums = {c: 0.0 for c in _PTIME_CRITERIA}
        for p in self.picks:
            if p is None:
                continue
            for c in _PTIME_CRITERIA:
                sums[c] += float(p.utilities.get(c, 0.0))
        n = len(self.picks)
        return {c: sums[c] / n for c in _PTIME_CRITERIA}

    def choquet(
        self,
        importance: dict[str, float],
        interaction: dict[str, float],
    ) -> float:
        """Aggregate per-criterion means through the 2-order Choquet integral."""
        return choquet_2order(self.utility_means(), importance, interaction)


def _candidates_conflict(a: Candidate, b: Candidate) -> bool:
    """True when `a` and `b` share overlapping minutes on the same day."""
    if a.day != b.day:
        return False
    return not (a.end <= b.start or b.end <= a.start)


def _build_variables(
    tasks_with_candidates: list[tuple[RecommendedTask, list[Candidate]]],
) -> list[Variable]:
    """Wrap each `(task, candidates)` pair as a `Variable`, smallest domain first."""
    indexed: list[tuple[int, RecommendedTask, list[Candidate]]] = [
        (i, t, c) for i, (t, c) in enumerate(tasks_with_candidates)
    ]
    indexed.sort(key=lambda ent: (len(ent[2]), ent[0]))
    return [
        Variable(index=i, task=t, candidates=tuple(c))
        for i, (_orig, t, c) in enumerate(indexed)
    ]


def _mono_criterion_search(
    variables: list[Variable],
    criterion: str,
    locked_thresholds: dict[str, float],
    incumbent: Solution | None,
    *,
    time_budget_seconds: float,
) -> Solution | None:
    """Branch-and-bound maximising the per-variable mean of `u_c`.

    Locked thresholds bind already-fixed criteria as lower bounds on the
    mean. Within each variable, candidates are tried in descending `u_c`
    order; a final skip option drops the task when no fit satisfies the
    locks. `incumbent` seeds the search so only strict improvements take.
    """
    import time as _time

    n = len(variables)
    if n == 0:  # pragma: no cover  defensive
        return Solution(picks=())
    # Per-variable candidates ranked by `u_c` descending.
    ordered: list[list[Candidate]] = [
        sorted(
            v.candidates,
            key=lambda c: (-float(c.utilities.get(criterion, 0.0)), c.day, c.start),
        )
        for v in variables
    ]
    # Per-variable best `u_c` for the upper-bound prune.
    max_per_var: list[float] = [
        max((float(c.utilities.get(criterion, 0.0)) for c in v.candidates), default=0.0)
        for v in variables
    ]
    eps = 1e-9
    deadline = _time.monotonic() + max(0.01, float(time_budget_seconds))
    best_score = -1.0
    best_picks: list[Candidate | None] | None = None
    if incumbent is not None:
        seed_means = incumbent.utility_means()
        best_score = float(seed_means.get(criterion, 0.0))
        best_picks = list(incumbent.picks) if len(incumbent.picks) == n else None

    chosen: list[Candidate | None] = [None] * n
    partial_sum = 0.0
    # Placed slots used for time-conflict pruning.
    placed_intervals: list[tuple[datetime.date, int, int]] = []

    def _check_locks(picks: list[Candidate | None]) -> bool:
        if not locked_thresholds:
            return True
        sol = Solution(picks=tuple(picks))
        means = sol.utility_means()
        for c, thresh in locked_thresholds.items():
            if means[c] + eps < thresh:
                return False
        return True

    def _dfs(i: int) -> None:
        nonlocal best_score, best_picks, partial_sum
        if _time.monotonic() > deadline:  # pragma: no cover  timing-dependent
            return
        if i == n:
            mean = partial_sum / n
            if mean - best_score > eps and _check_locks(chosen):
                best_score = mean
                best_picks = list(chosen)
            return
        # Upper-bound prune via best-case remaining utilities.
        remaining_best = sum(max_per_var[j] for j in range(i, n))
        upper = (partial_sum + remaining_best) / n
        if upper - best_score <= eps:
            return
        for cand in ordered[i]:
            if any(
                d == cand.day and not (cand.end <= s or e <= cand.start)
                for d, s, e in placed_intervals
            ):
                continue
            u = float(cand.utilities.get(criterion, 0.0))
            chosen[i] = cand
            placed_intervals.append((cand.day, cand.start, cand.end))
            partial_sum += u
            _dfs(i + 1)
            placed_intervals.pop()
            partial_sum -= u
            chosen[i] = None
            if _time.monotonic() > deadline:  # pragma: no cover  timing-dependent
                return
        # Skip path: leave variable unassigned.
        chosen[i] = None
        _dfs(i + 1)

    _dfs(0)
    if best_picks is None:  # pragma: no cover  skip-option always returns a Solution
        return None
    return Solution(picks=tuple(best_picks))


def _choquet_upper_bound(
    variables: list[Variable],
    importance: dict[str, float],
    interaction: dict[str, float],
) -> dict[str, float]:
    """Compute each criterion's optimistic upper bound (max mean over variables)."""
    out: dict[str, float] = {}
    n = max(1, len(variables))
    for c in _PTIME_CRITERIA:
        out[c] = (
            sum(
                max(
                    (float(cand.utilities.get(c, 0.0)) for cand in v.candidates),
                    default=0.0,
                )
                for v in variables
            )
            / n
        )
    return out


def mcs_solve(
    tasks_with_candidates: list[tuple[RecommendedTask, list[Candidate]]],
    importance: dict[str, float],
    interaction: dict[str, float],
    *,
    time_budget_seconds: float = 10.0,
) -> tuple[list[Variable], Solution] | None:
    """Run the MCS branch-and-bound loop over PTIME variables.

    Args:
        tasks_with_candidates: per-task feasible-candidate lists for one
            weekly chunk.
        importance: Choquet `a_i` coefficients.
        interaction: Choquet `a_ij` coefficients.
        time_budget_seconds: total wall-time cap split across criteria.

    Returns:
        `(variables, solution)` with variables in solver-internal order,
        or `None` when every task has an empty candidate list.
    """
    pruned = [(t, c) for t, c in tasks_with_candidates if c]
    if not pruned:
        return None
    variables = _build_variables(pruned)
    # Split the total budget evenly across the four criteria.
    per_criterion_budget = max(0.05, time_budget_seconds / max(1, len(_PTIME_CRITERIA)))
    upper_bounds = _choquet_upper_bound(variables, importance, interaction)
    s_star: Solution | None = None
    locked: dict[str, float] = {}
    pending = list(_PTIME_CRITERIA)
    while pending:
        # Pick the criterion with the largest `upper_bound - achieved`
        # slack so the next search has the most room to improve.
        achieved = s_star.utility_means() if s_star is not None else {}
        pending.sort(key=lambda c: -(upper_bounds[c] - float(achieved.get(c, 0.0))))
        c = pending.pop(0)
        s_new = _mono_criterion_search(
            variables,
            c,
            locked,
            s_star,
            time_budget_seconds=per_criterion_budget,
        )
        if s_new is not None:  # pragma: no branch  mono-criterion always returns
            s_star = s_new
            locked[c] = s_new.utility_means()[c]
    if s_star is None:  # pragma: no cover  defensive
        return None
    return variables, s_star


def _picks_to_scheduled(
    variables: list[Variable],
    sol: Solution,
    person_id: str,
) -> tuple[list[ScheduledTask], list[RecommendedTask]]:
    """Split a `Solution` into `scheduled` and `unscheduled` lists."""
    del person_id  # accepted for caller parity; unused locally.
    scheduled: list[ScheduledTask] = []
    unscheduled: list[RecommendedTask] = []
    for var, pick in zip(variables, sol.picks):
        if pick is None:
            unscheduled.append(var.task)
            continue
        scheduled.append(
            ScheduledTask(
                task=var.task,
                start_minutes=pick.start,
                end_minutes=pick.end,
                is_standalone=True,
                concurrent_with=None,
                date=pick.day,
            )
        )
    return scheduled, unscheduled


# ---------------------------------------------------------------------------
# PTimeAugmenter
# ---------------------------------------------------------------------------


class PTimeAugmenter(Augmenter):
    """Choquet-driven augmenter with optional MCS solver.

    Args:
        time_windows: preferred-epoch bands; drives `utility_time` and
            seeds the `windows` candidate-anchor rule.
        daily_window: hard wake/sleep bound applied to every placement.
        allen_rules: temporal-relation rules used by `utility_overlap`.
        seed: RNG seed for tie-breaking.
    """

    def __init__(
        self,
        time_windows: dict[str, WindowRange] | None = None,
        daily_window: DailyWindow | None = None,
        allen_rules: list[AllenPairRule] | None = None,
        seed: int | None = None,
    ) -> None:
        self._time_windows: dict[str, WindowRange] = time_windows or {}
        self._daily_window: DailyWindow = daily_window or DailyWindow()
        self._allen_rules: list[AllenPairRule] = list(allen_rules or [])
        self._seed = int(seed) if seed is not None else 0
        self._rng = random.Random(self._seed)
        self._rule_index = _build_rule_index(self._allen_rules)

    def augment(
        self,
        calendar: CalendarTrace,
        tasks: list[RecommendedTask],
        config: AugmentationConfig,
        horizon: HorizonHint | None = None,
        *,
        weekly_tasks: list[list[RecommendedTask]] | None = None,
    ) -> SchedulingSolution:
        """Place tasks week-by-week via the configured solver.

        `weekly_tasks` gives each week its own batch; when omitted, `tasks`
        is broadcast to every week.
        """
        ptime_cfg = getattr(config, "ptime", None) or PTimeConfig()
        strategy = ptime_cfg.candidate_strategy
        duration_preference = ptime_cfg.duration_preference
        importance = merge_importance(dict(ptime_cfg.importance))
        interaction = merge_interaction(dict(ptime_cfg.interaction))
        horizon_dates = _get_horizon_dates(calendar, horizon=horizon)
        if not horizon_dates:
            aug_cal = AugmentedCalendar(
                person_id=calendar.person_id,
                base_events=list(calendar.events),
            )
            return SchedulingSolution(
                person_id=calendar.person_id,
                augmented_calendar=aug_cal,
                tasks=list(tasks),
                scheduled=[],
                unscheduled=list(tasks),
            )

        events_by_date: dict[datetime.date, list[CalendarEvent]] = defaultdict(list)
        for e in calendar.events:
            events_by_date[e.date].append(e)
        for d in events_by_date:
            events_by_date[d].sort(key=lambda e: e.start_minutes)

        wake = self._daily_window.wake_minutes
        sleep = self._daily_window.sleep_minutes

        learning_cfg = ptime_cfg.learning
        if learning_cfg.enabled and calendar.events:
            learn_rng = random.Random(learning_cfg.seed)
            pairs = extract_pairwise_examples(
                calendar,
                events_by_date,
                self._time_windows,
                self._rule_index,
                window_start=wake,
                window_end=sleep,
                duration_preference=duration_preference,
                negative_samples_per_event=learning_cfg.negative_samples_per_event,
                negative_shift_minutes=learning_cfg.negative_shift_minutes,
                rng=learn_rng,
            )
            learned = fit_learned_coefficients(
                pairs,
                regularization=learning_cfg.regularization,
                seed=learning_cfg.seed,
            )
            learned_imp, learned_int = learned if learned is not None else (None, None)
            importance = blend_importance(importance, learned_imp, learning_cfg.alpha)
            interaction = blend_interaction(
                interaction, learned_int, learning_cfg.alpha
            )
            logger.debug(
                "ptime: person=%s learned-importance=%s learned-interaction=%s "
                "(alpha=%.2f, pairs=%d)",
                calendar.person_id,
                learned_imp,
                learned_int,
                learning_cfg.alpha,
                len(pairs),
            )

        scheduled: list[ScheduledTask] = []
        unscheduled: list[RecommendedTask] = []
        all_task_instances: list[RecommendedTask] = []

        weekly_chunks = _get_weekly_chunks(horizon_dates)
        weeks = weekly_tasks if weekly_tasks is not None else [tasks]
        repeat = config.repeat_per_week if config is not None else True
        carry: list[RecommendedTask] = list(tasks) if not repeat else []

        for week_index, week_dates in enumerate(weekly_chunks):
            week_tasks = _tasks_for_week(weeks, week_index) if repeat else carry
            all_task_instances.extend(week_tasks) if repeat else None
            if not week_tasks:
                continue
            week_scheduled, week_unplaced = self._solve_week(
                week_tasks,
                week_dates,
                events_by_date,
                strategy=strategy,
                duration_preference=duration_preference,
                importance=importance,
                interaction=interaction,
                wake=wake,
                sleep=sleep,
                solver=ptime_cfg.solver,
                time_budget=ptime_cfg.mcs_time_budget_seconds,
            )
            scheduled.extend(week_scheduled)
            if repeat:
                unscheduled.extend(week_unplaced)
            else:
                carry = week_unplaced

        if not repeat:
            all_task_instances = list(tasks)
            unscheduled = carry

        aug_cal = AugmentedCalendar(
            person_id=calendar.person_id,
            base_events=list(calendar.events),
            scheduled_tasks=scheduled,
        )
        return SchedulingSolution(
            person_id=calendar.person_id,
            augmented_calendar=aug_cal,
            tasks=all_task_instances,
            scheduled=scheduled,
            unscheduled=unscheduled,
        )

    def _solve_week(
        self,
        week_tasks: list[RecommendedTask],
        week_dates: list[datetime.date],
        events_by_date: dict[datetime.date, list[CalendarEvent]],
        *,
        strategy: str,
        duration_preference: str,
        importance: dict[str, float],
        interaction: dict[str, float],
        wake: int,
        sleep: int,
        solver: str,
        time_budget: float,
    ) -> tuple[list[ScheduledTask], list[RecommendedTask]]:
        """Solve one weekly chunk via MCS or greedy decode."""
        # Enumerate candidates once per task against an empty per-week state.
        empty_placed: dict[datetime.date, list[tuple[int, int]]] = defaultdict(list)
        empty_used: set[tuple[datetime.date, int, int]] = set()
        per_task_cands: list[list[Candidate]] = []
        for task in week_tasks:
            untouched = enumerate_candidates(
                task,
                week_dates,
                events_by_date,
                empty_placed,
                empty_used,
                self._time_windows,
                self._rule_index,
                window_start=wake,
                window_end=sleep,
                importance=importance,
                interaction=interaction,
                duration_preference=duration_preference,
                strategy=strategy,
                allow_reuse=True,
            )
            per_task_cands.append(untouched)

        if solver == "greedy":
            return self._greedy_decode(week_tasks, per_task_cands)
        result = mcs_solve(
            list(zip(week_tasks, per_task_cands)),
            importance,
            interaction,
            time_budget_seconds=time_budget,
        )
        if result is None:
            return [], list(week_tasks)
        variables, solution = result
        scheduled, unscheduled = _picks_to_scheduled(variables, solution, person_id="")
        # `mcs_solve` drops empty-domain tasks; surface them here.
        seen_tasks = {id(v.task) for v in variables}
        for t in week_tasks:
            if id(t) not in seen_tasks:
                unscheduled.append(t)
        return scheduled, unscheduled

    def _greedy_decode(
        self,
        week_tasks: list[RecommendedTask],
        per_task_cands: list[list[Candidate]],
    ) -> tuple[list[ScheduledTask], list[RecommendedTask]]:
        """Top-1 forward decode over the Choquet-scored candidates.

        Sequential: pick the highest-scoring untouched candidate per task,
        skipping any that conflicts with already-placed slots.
        """
        placed_intervals: list[tuple[datetime.date, int, int]] = []
        scheduled: list[ScheduledTask] = []
        unscheduled: list[RecommendedTask] = []
        for task, cands in zip(week_tasks, per_task_cands):
            valid = [
                c
                for c in cands
                if not any(
                    c.day == d and not (c.end <= s or e <= c.start)
                    for d, s, e in placed_intervals
                )
            ]
            choice = _pick_best(valid, self._rng)
            if choice is None:
                unscheduled.append(task)
                continue
            scheduled.append(
                ScheduledTask(
                    task=task,
                    start_minutes=choice.start,
                    end_minutes=choice.end,
                    is_standalone=True,
                    concurrent_with=None,
                    date=choice.day,
                )
            )
            placed_intervals.append((choice.day, choice.start, choice.end))
        return scheduled, unscheduled
