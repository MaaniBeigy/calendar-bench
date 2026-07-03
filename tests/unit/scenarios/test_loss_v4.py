"""Unit tests for the v4 scheduling loss components.

Covers:
* compute_l_cov / compute_l_pref / compute_l_spread (sanity, mostly unchanged).
* compute_l_cal; merged (task×event) ∪ (task×task) with selector-driven
  rules and buffer-aware partial credit.
* compute_l_disp; median half-life lag × max(categorical, continuous).
* compute_l_concurrent; inclusion (σ ≥ τ) ∧ ¬{three exclusion sources}.
* compute_l_divide; per-week mean divided fraction.
* SchedulingLoss.compute; assembles all 7 with weights.
"""

from __future__ import annotations

import datetime

import pytest

from src.scripts.persona.config.schema import AllenPairRule, WindowRange
from src.scripts.scenarios.config.schema import LossWeights
from src.scripts.scenarios.domain.calendar import (
    AugmentedCalendar,
    CalendarEvent,
    CalendarTrace,
)
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import (
    R_SEP,
    AllenRelation,
    RuleSet,
    SelectorMatcher,
)
from src.scripts.scenarios.metrics.intensity_resolver import (
    IntensityResolver,
    MetQuartiles,
)
from src.scripts.scenarios.metrics.loss import (
    LossComponents,
    SchedulingLoss,
    _half_life_lag,
    _refers_to,
    compute_l_cal,
    compute_l_concurrent,
    compute_l_cov,
    compute_l_disp,
    compute_l_divide,
    compute_l_spread,
    is_excluded_pair,
)
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

DATE = datetime.date(2026, 5, 4)
DATE2 = datetime.date(2026, 5, 5)

DEFAULT_QUARTILES = MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=16.8)


def _task(label="x", duration_min=30, duration_max=60, **kw) -> RecommendedTask:
    return RecommendedTask(
        label=label,
        duration_min=duration_min,
        duration_max=duration_max,
        **kw,
    )


def _scheduled(task=None, start=480, end=540, date=DATE, **kw) -> ScheduledTask:
    if task is None:
        task = _task()
    return ScheduledTask(
        task=task,
        start_minutes=start,
        end_minutes=end,
        is_standalone=kw.pop("is_standalone", True),
        concurrent_with=kw.pop("concurrent_with", None),
        date=date,
        **kw,
    )


def _event(label="lunch", start=720, end=780, date=DATE, **kw) -> CalendarEvent:
    return CalendarEvent(
        label=label, start_minutes=start, end_minutes=end, date=date, **kw
    )


def _solution(tasks=None, scheduled=None, unscheduled=None, person="p001"):
    tasks = tasks or []
    scheduled = scheduled or []
    unscheduled = unscheduled or []
    return SchedulingSolution(
        person_id=person,
        augmented_calendar=AugmentedCalendar(
            person_id=person, scheduled_tasks=scheduled
        ),
        tasks=tasks,
        scheduled=scheduled,
        unscheduled=unscheduled,
    )


def _trace(events=None, person="p001"):
    return CalendarTrace(person_id=person, events=events or [])


def _matcher(resolver=None):
    return SelectorMatcher(resolver=resolver)


def _ruleset(rules=None):
    return RuleSet(list(rules or []))


def _resolver(quartiles=DEFAULT_QUARTILES):
    return IntensityResolver(quartiles)


def _default_weights() -> LossWeights:
    # Use the 7-field defaults.
    return LossWeights()


# ---------------------------------------------------------------------------
# Half-life helper
# ---------------------------------------------------------------------------


class TestHalfLifeLag:
    def test_zero_gap_is_one(self):
        assert _half_life_lag(0.0, 2.0) == pytest.approx(1.0)

    def test_one_half_life_is_half(self):
        # H = 2 days, gap = 2 days = 2880 min
        assert _half_life_lag(2880.0, 2.0) == pytest.approx(0.5)

    def test_two_half_lives_is_quarter(self):
        assert _half_life_lag(5760.0, 2.0) == pytest.approx(0.25)

    def test_zero_half_life_returns_zero(self):
        assert _half_life_lag(60.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_l_cov / pref / spread; minimal sanity (old behaviour preserved)
# ---------------------------------------------------------------------------


class TestSanity:
    def test_cov_partial(self):
        ts = [_task(label=f"t{i}") for i in range(4)]
        sol = _solution(
            tasks=ts, scheduled=[_scheduled(t) for t in ts[:2]], unscheduled=ts[2:]
        )
        assert compute_l_cov(sol) == pytest.approx(0.5)

    def test_spread_one_per_date(self):
        # 3 tasks on 3 separate dates to unique=3, scheduled=3 to 0.0
        ts = [_task(label=f"t{i}") for i in range(3)]
        scheduled = [
            _scheduled(ts[0], date=DATE),
            _scheduled(ts[1], date=DATE2),
            _scheduled(ts[2], date=DATE2 + datetime.timedelta(days=1)),
        ]
        sol = _solution(tasks=ts, scheduled=scheduled)
        assert compute_l_spread(sol, horizon_days=3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_l_cal; merged pair set + buffer-aware partial credit
# ---------------------------------------------------------------------------


class TestComputeLCal:
    def test_empty_pair_set(self):
        sol = _solution()
        assert compute_l_cal(sol, _trace(), _ruleset(), _matcher()) == 0.0

    def test_separating_pair_no_violation(self):
        t = _task(label="meditation", is_concurrent=False)
        st = _scheduled(t, start=480, end=540, date=DATE)
        sleep = _event(label="sleep", start=1380, end=1860, date=DATE)
        sol = _solution(tasks=[t], scheduled=[st])
        assert compute_l_cal(
            sol, _trace([sleep]), _ruleset(), _matcher()
        ) == pytest.approx(0.0)

    def test_overlap_violation_full_credit(self):
        t = _task(label="meditation", is_concurrent=False)
        st = _scheduled(t, start=1400, end=1500, date=DATE)
        sleep = _event(
            label="sleep", start=1380, end=1860, date=DATE, is_concurrent=False
        )
        sol = _solution(tasks=[t], scheduled=[st])
        # 1 violation / 1 pair = 1.0
        assert compute_l_cal(
            sol, _trace([sleep]), _ruleset(), _matcher()
        ) == pytest.approx(1.0)

    def test_buffer_partial_credit(self):
        # task at 720..780 (lunch slot); event sleep precedes (P).  Configure
        # rule: lunch and intensive tasks separated by [p, P].  Then a meeting
        # event meets/precedes within buffer.
        # Build a literal-vs-literal rule that admits only [p, P].
        rule = AllenPairRule(
            id="r",
            event_a="task",
            event_b="event",
            admissible_relations=["p", "P"],
        )
        rs = _ruleset([rule])
        matcher = _matcher()

        # task sits at 600-660; event at 661-720 (gap 1 minute, buffer 30) to ramp
        # rel(task, event) = p (task ends before event starts), gap=1
        t = _task(label="task", duration_min=60, duration_max=60)
        st = _scheduled(t, start=600, end=660, date=DATE)
        ev = _event(label="event", start=661, end=720, date=DATE)
        sol = _solution(tasks=[t], scheduled=[st])

        # admissible = {p, P}, rel = p to admissible to 0.0 (no violation)
        v = compute_l_cal(sol, _trace([ev]), rs, matcher, buffer_minutes=30)
        assert v == pytest.approx(0.0)

    def test_lunch_meets_walk_rule_violation(self):
        """Recreate the failure case: lunch immediately followed by walk."""
        # Rule: lunch_far_from_intensive; lunch admissible only [p, P]
        rule = AllenPairRule(
            id="lunch_far",
            event_a={"name": "lunch"},
            event_b={"intensity": [2, 3, 4]},
            admissible_relations=["p", "P"],
        )
        rs = _ruleset([rule])
        # Use resolver to give the walk task an intensity of 3.
        walk = _task(
            label="walk", difficulty_level=3, duration_min=60, duration_max=120
        )
        st = _scheduled(walk, start=780, end=840, date=DATE)  # 13:00-14:00
        # lunch ends at 13:00 (st start) to Allen 'm' (meets)
        lunch = _event(label="lunch", start=720, end=780, date=DATE)
        sol = _solution(tasks=[walk], scheduled=[st])
        matcher = _matcher(resolver=_resolver())
        v = compute_l_cal(sol, _trace([lunch]), rs, matcher, buffer_minutes=30)
        # m to not admissible to 1.0 (single pair)
        assert v == pytest.approx(1.0)

    def test_task_task_pair_counted(self):
        t1 = _task(label="a")
        t2 = _task(label="b")
        # Both same day, overlapping
        st1 = _scheduled(t1, start=600, end=700, date=DATE)
        st2 = _scheduled(t2, start=650, end=750, date=DATE)
        sol = _solution(tasks=[t1, t2], scheduled=[st1, st2])
        # No event, no rule, fallback R_SEP.  Overlap to not admissible to 1 violation / 1 pair = 1.0.
        v = compute_l_cal(sol, _trace(), _ruleset(), _matcher())
        assert v == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_l_disp; median × half-life lag × max(cat, continuous)
# ---------------------------------------------------------------------------


class TestComputeLDisp:
    def test_empty_returns_zero(self):
        sol = _solution()
        v = compute_l_disp(sol, _trace(), _resolver(), half_life_days=2.0, max_met=16.8)
        assert v == 0.0

    def test_overlapping_high_intensity_pair_high_score(self):
        """Two scheduled high-intensity tasks overlapping to near-1 score."""
        t = _task(label="hard", difficulty_level=4)
        st1 = _scheduled(t, start=600, end=700, date=DATE)
        # Second instance overlapping (same task, repeat)
        st2 = _scheduled(t, start=620, end=720, date=DATE)
        sol = _solution(tasks=[t, t], scheduled=[st1, st2])
        v = compute_l_disp(sol, _trace(), _resolver(), half_life_days=2.0, max_met=16.8)
        # cat = (4+4)/8 = 1.0; lag = 1.0 (gap=0); max(cat, cont) = 1.0
        assert v == pytest.approx(1.0)

    def test_distant_pair_low_score(self):
        t = _task(label="hard", difficulty_level=4)
        # Events at midnight (start=0, end=10) so the gap is exactly N*1440 - 10.
        # Pick day diff so the lag is small but stable.
        st1 = _scheduled(t, start=0, end=10, date=DATE)
        st2 = _scheduled(
            t,
            start=0,
            end=10,
            date=DATE + datetime.timedelta(days=10),
        )
        sol = _solution(tasks=[t, t], scheduled=[st1, st2])
        v = compute_l_disp(sol, _trace(), _resolver(), half_life_days=2.0, max_met=16.8)
        # gap = 10*1440 - 10 = 14390 min ≈ 9.993 days; lag = 0.5^(9.993/2) ≈ 0.03127
        # cat = 1.0 to score ≈ 0.03127
        assert v == pytest.approx(0.5 ** ((10 * 1440 - 10) / 1440 / 2), abs=1e-6)
        # Sanity: the score is small because the pair is far apart.
        assert v < 0.05

    def test_low_intensity_low_score(self):
        t = _task(label="easy", difficulty_level=1)
        st1 = _scheduled(t, start=600, end=700, date=DATE)
        st2 = _scheduled(t, start=620, end=720, date=DATE)
        sol = _solution(tasks=[t, t], scheduled=[st1, st2])
        v = compute_l_disp(sol, _trace(), _resolver(), half_life_days=2.0, max_met=16.8)
        # cat = (1+1)/8 = 0.25; lag = 1; to 0.25
        assert v == pytest.approx(0.25)

    def test_median_aggregation(self):
        """Three pairs with widely different scores to median wins."""
        t = _task(label="x", difficulty_level=2)
        # Three scheduled tasks, all overlapping for max categorical
        st1 = _scheduled(t, start=0, end=60, date=DATE)
        st2 = _scheduled(t, start=10, end=70, date=DATE)
        st3 = _scheduled(t, start=20, end=80, date=DATE)
        sol = _solution(tasks=[t, t, t], scheduled=[st1, st2, st3])
        # All 3 pairs identical to median = single pair score = (2+2)/8 = 0.5
        v = compute_l_disp(sol, _trace(), _resolver(), half_life_days=2.0, max_met=16.8)
        assert v == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_l_concurrent + is_excluded_pair
# ---------------------------------------------------------------------------


def _semantic_const(score: float) -> SemanticCompatibility:
    return SemanticCompatibility(llm_scorer=lambda a, b: score)


class TestIsExcludedPair:
    def test_event_not_concurrent_excludes(self):
        t = _task(label="cooking", is_concurrent=True)  # task allows; event forbids
        st = _scheduled(t, start=720, end=780, date=DATE)
        ev = _event(label="lunch", is_concurrent=False, date=DATE)
        excluded = is_excluded_pair(
            st,
            ev,
            event_concurrent_map={"lunch": False},
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert excluded is True

    def test_task_not_concurrent_excludes(self):
        t = _task(label="cook_meal", is_concurrent=False)
        st = _scheduled(t, start=720, end=780, date=DATE)
        ev = _event(label="lunch", is_concurrent=False, date=DATE)
        excluded = is_excluded_pair(
            st,
            ev,
            event_concurrent_map={"lunch": False},
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert excluded is True

    def test_both_concurrent_not_excluded(self):
        t = _task(label="podcast", is_concurrent=True)
        st = _scheduled(t, start=720, end=780, date=DATE)
        ev = _event(label="lunch", is_concurrent=True, date=DATE)
        excluded = is_excluded_pair(
            st,
            ev,
            event_concurrent_map={"lunch": True},
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert excluded is False

    def test_pp_only_rule_excludes(self):
        rule = AllenPairRule(
            id="r",
            event_a="task",
            event_b="event",
            admissible_relations=["p", "P"],
        )
        rs = _ruleset([rule])
        t = _task(label="task", is_concurrent=True)
        st = _scheduled(t, start=720, end=780, date=DATE)
        ev = _event(label="event", is_concurrent=True, date=DATE)
        excluded = is_excluded_pair(
            st,
            ev,
            event_concurrent_map={"event": True},
            ruleset=rs,
            matcher=_matcher(),
        )
        assert excluded is True

    def test_event_not_concurrent_alone_is_enough_to_exclude(self):
        """The three exclusion sources are independent (OR).
        An event_config flag of `is_concurrent=false` must trigger
        exclusion even when the *task* is concurrent; the pre-fix
        implementation incorrectly required BOTH sides to be false."""
        t = _task(label="podcast", is_concurrent=True)
        st = _scheduled(t, start=720, end=780, date=DATE)
        ev = _event(label="lunch", is_concurrent=False, date=DATE)
        excluded = is_excluded_pair(
            st,
            ev,
            event_concurrent_map={"lunch": False},
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert excluded is True

    def test_task_not_concurrent_alone_is_enough_to_exclude(self):
        """The RecommendedTask side carrying `is_concurrent=false` must
        trigger exclusion on its own, even if the event side is
        concurrent."""
        t = _task(label="plan_week_meals", is_concurrent=False)
        st = _scheduled(t, start=720, end=780, date=DATE)
        ev = _event(label="lunch", is_concurrent=True, date=DATE)
        excluded = is_excluded_pair(
            st,
            ev,
            event_concurrent_map={"lunch": True},
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert excluded is True

    def test_event_concurrent_map_overrides_event_attribute(self):
        """Catalog map (event_config) takes precedence over the
        realised event's own `is_concurrent` attribute; this is
        how the event_config exclusion gate actually queries the
        catalog rather than the realisation."""
        t = _task(label="podcast", is_concurrent=True)
        st = _scheduled(t, start=720, end=780, date=DATE)
        # Realised event says True but catalog says False.
        ev = _event(label="lunch", is_concurrent=True, date=DATE)
        excluded = is_excluded_pair(
            st,
            ev,
            event_concurrent_map={"lunch": False},
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert excluded is True

    def test_missing_from_catalog_falls_back_to_event_attribute(self):
        """When the event's label isn't in the catalog map, the
        realised event's own `is_concurrent` attribute is consulted
        instead; preserving correctness for events generated outside
        the event_config flow."""
        t = _task(label="podcast", is_concurrent=True)
        st = _scheduled(t, start=720, end=780, date=DATE)
        ev = _event(label="custom_block", is_concurrent=False, date=DATE)
        excluded = is_excluded_pair(
            st,
            ev,
            event_concurrent_map={},  # label not present
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert excluded is True


class TestComputeLConcurrent:
    def test_no_eligible_pairs_returns_none(self):
        # No scheduled tasks to no opportunity to assess to None signal
        # so the report renders `n/a` instead of laundering a stub
        # σ oracle into a perfect `G_merge = 1.0` (follow-up, 2026-05-14).
        sol = _solution()
        v = compute_l_concurrent(
            sol,
            _trace(),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert v is None

    def test_perfect_merge_gives_zero(self):
        t = _task(label="reading", is_concurrent=True)
        ev = _event(label="lunch", is_concurrent=True, date=DATE)
        st = _scheduled(
            t,
            start=720,
            end=780,
            date=DATE,
            is_standalone=False,
            concurrent_with="lunch",
        )
        sol = _solution(tasks=[t], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # σ_actual = σ_best = 0.9 to 1 - 0.9/0.9 = 0.0
        assert v == pytest.approx(0.0)

    def test_missed_merge_gives_one(self):
        t = _task(label="reading", is_concurrent=True)
        ev = _event(label="lunch", is_concurrent=True, date=DATE)
        st = _scheduled(t, start=720, end=780, date=DATE, is_standalone=True)
        sol = _solution(tasks=[t], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        assert v == pytest.approx(1.0)

    def test_judge_scores_actual_label_plus_parent(self):
        """The merge oracle sees the episode's actual label and its parent."""
        seen: list[tuple[str, str]] = []

        class _Recorder:
            def score(self, a: str, b: str) -> float:
                seen.append((a, b))
                return 0.9

        t = _task(label="hydrate", is_concurrent=True)
        ev = _event(
            label="office_work",
            start=540,
            end=600,
            is_concurrent=True,
            date=DATE,
            display_label="standup",
        )
        st = _scheduled(
            t,
            start=540,
            end=570,
            date=DATE,
            is_standalone=False,
            concurrent_with="office_work",
        )
        sol = _solution(tasks=[t], scheduled=[st])
        compute_l_concurrent(
            sol,
            _trace([ev]),
            _Recorder(),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        assert ("hydrate", "standup (office_work)") in seen
        assert all(host != "office_work" for _, host in seen)

    def test_excluded_pair_not_in_phi(self):
        # event has is_concurrent=False to exclusion (i) to pair NOT in Φ
        t = _task(label="cooking", is_concurrent=True)
        ev = _event(label="lunch", is_concurrent=False, date=DATE)
        st = _scheduled(t, start=720, end=780, date=DATE, is_standalone=True)
        sol = _solution(tasks=[t], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # Φ empty to None (no opportunity to assess)
        assert v is None

    def test_below_threshold_not_in_phi(self):
        t = _task(label="reading", is_concurrent=True)
        ev = _event(label="lunch", is_concurrent=True, date=DATE)
        st = _scheduled(t, start=720, end=780, date=DATE, is_standalone=True)
        sol = _solution(tasks=[t], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev]),
            _semantic_const(0.3),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # No σ above threshold to Φ empty to None
        assert v is None

    def test_no_events_returns_none(self):
        # Calendar with no base events to no opportunity to None
        t = _task(label="reading", is_concurrent=True)
        st = _scheduled(t, start=720, end=780, date=DATE, is_standalone=True)
        sol = _solution(tasks=[t], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([]),  # no events
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        assert v is None


# ---------------------------------------------------------------------------
# compute_l_concurrent; algorithm rewrite (2026-05-14)
#
# Three new semantics:
#   1. Weekly Φ scoping: a task scheduled in week W is paired only with
#      events in week W.
#   2. Event-side exclusion (i) applied UPFRONT: non-concurrent events
#      are removed before σ-best picking.
#   3. Per-task σ-best is computed over NON-EXCLUDED events only, fixing
#      the prior bug where a task whose σ-best partner was exclusive got
#      silently dropped from Φ even when a second-best concurrent-friendly
#      host was available.
# ---------------------------------------------------------------------------


def _semantic_pair_scores(
    scores: dict[tuple[str, str], float],
) -> SemanticCompatibility:
    """SemanticCompatibility that returns per-(label_a, label_b) scores."""

    def scorer(a: str, b: str) -> float:
        return scores.get((a, b), scores.get((b, a), 0.0))

    return SemanticCompatibility(llm_scorer=scorer)


class TestComputeLConcurrentRewrite:
    def test_filter_exclusions_first_falls_back_to_second_best(self):
        """A task whose σ-best partner is exclusive (catalog
        `is_concurrent=false`) must now fall back to its σ-best
        concurrent-friendly host; instead of being silently dropped
        from Φ as the previous algorithm did.  Concrete: cardio's top
        match is `walking` (exclusive), but `lunch` (concurrent) is
        the achievable σ-best.  Augmenter placed cardio standalone
        to σ_actual=0, σ_best=0.7 (lunch) to L_merge = 1.0.
        """
        task = _task(label="cardio", is_concurrent=True)
        walking = _event(label="walking", is_concurrent=False, date=DATE)
        lunch = _event(label="lunch", is_concurrent=True, date=DATE)
        st = _scheduled(task, start=600, end=620, date=DATE, is_standalone=True)
        sol = _solution(tasks=[task], scheduled=[st])
        sem = _semantic_pair_scores(
            {("cardio", "walking"): 1.0, ("cardio", "lunch"): 0.7}
        )
        v = compute_l_concurrent(
            sol,
            _trace([walking, lunch]),
            sem,
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # σ_best = 0.7 (lunch; walking is excluded and skipped)
        # σ_actual = 0 (task standalone)
        # L = 1 - 0/0.7 = 1.0
        assert v == pytest.approx(1.0)

    def test_augmenter_credit_for_taking_second_best_host(self):
        """When the augmenter places the cardio task concurrent with
        `lunch` (the achievable σ-best), σ_actual catches the same
        0.7 score and L_merge collapses to 0.  This is the case the
        OLD algorithm couldn't reward at all; `walking` was the
        σ-best across all events, so cardio got dropped from Φ even
        though the augmenter did the right thing with the second-best."""
        task = _task(label="cardio", is_concurrent=True)
        walking = _event(label="walking", is_concurrent=False, date=DATE)
        lunch = _event(label="lunch", is_concurrent=True, date=DATE)
        st = _scheduled(
            task,
            start=720,
            end=750,
            date=DATE,
            is_standalone=False,
            concurrent_with="lunch",
        )
        sol = _solution(tasks=[task], scheduled=[st])
        sem = _semantic_pair_scores(
            {("cardio", "walking"): 1.0, ("cardio", "lunch"): 0.7}
        )
        v = compute_l_concurrent(
            sol,
            _trace([walking, lunch]),
            sem,
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # σ_best = σ_actual = 0.7 to L = 0
        assert v == pytest.approx(0.0)

    def test_weekly_scope_only_pairs_within_same_week(self):
        """A concurrent-friendly event in a DIFFERENT ISO week from
        the task's scheduled date must not contribute to Φ.  Augmenter
        decisions are weekly; opportunities outside the week the task
        actually ran in are not real opportunities."""
        task = _task(label="hydrate", is_concurrent=True)
        # task scheduled on a Monday, event in the FOLLOWING week
        date_w1 = datetime.date(2026, 6, 1)  # Mon, week 23 of 2026
        date_w2 = datetime.date(2026, 6, 8)  # Mon, week 24 of 2026
        assert date_w1.isocalendar().week != date_w2.isocalendar().week
        ev_other_week = _event(label="reading", is_concurrent=True, date=date_w2)
        st = _scheduled(task, start=720, end=730, date=date_w1, is_standalone=True)
        sol = _solution(tasks=[task], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev_other_week]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # The only concurrent-friendly event is in a different week
        # from the only scheduled task to Φ empty to None.
        assert v is None

    def test_weekly_scope_pairs_within_same_week(self):
        """Sanity check the other side of the boundary: the same task
        scheduled in the SAME week as the concurrent-friendly event
        DOES land in Φ."""
        task = _task(label="hydrate", is_concurrent=True)
        date_w1 = datetime.date(2026, 6, 1)
        date_w1_other_day = datetime.date(2026, 6, 3)  # same week
        ev = _event(label="reading", is_concurrent=True, date=date_w1_other_day)
        st = _scheduled(task, start=720, end=730, date=date_w1, is_standalone=True)
        sol = _solution(tasks=[task], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # σ_best=0.9 in-week, σ_actual=0 (standalone, and date doesn't
        # match the event's date even if augmenter had named it).
        # L = 1 - 0/0.9 = 1.0
        assert v == pytest.approx(1.0)

    def test_all_events_exclusive_returns_none_short_circuit(self):
        """When NO event in the calendar has `is_concurrent=True`,
        the function short-circuits to `None` without even iterating
        scheduled tasks; Φ is empty by definition."""
        task = _task(label="hydrate", is_concurrent=True)
        ev = _event(label="walking", is_concurrent=False, date=DATE)
        st = _scheduled(task, start=720, end=730, date=DATE, is_standalone=True)
        sol = _solution(tasks=[task], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        assert v is None

    def test_invalid_concurrent_with_exclusive_host_not_credited(self):
        """Augmenter that places task concurrent with an EXCLUSIVE host
        (catalog `is_concurrent=false`) gets 0 σ_actual; invalid
        merges are not credited even when `concurrent_with` is set.
        The augmenter is still punished against the achievable σ_best."""
        task = _task(label="hydrate", is_concurrent=True)
        walking = _event(label="walking", is_concurrent=False, date=DATE)
        reading = _event(label="reading", is_concurrent=True, date=DATE)
        # Augmenter claimed a merge with the exclusive `walking` host.
        st = _scheduled(
            task,
            start=820,
            end=830,
            date=DATE,
            is_standalone=False,
            concurrent_with="walking",
        )
        sol = _solution(tasks=[task], scheduled=[st])
        sem = _semantic_pair_scores(
            {("hydrate", "walking"): 1.0, ("hydrate", "reading"): 0.8}
        )
        v = compute_l_concurrent(
            sol,
            _trace([walking, reading]),
            sem,
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # σ_best = 0.8 (reading); the invalid `walking` merge contributes
        # nothing to σ_actual.  L = 1 - 0/0.8 = 1.0.
        assert v == pytest.approx(1.0)

    def test_multiple_concurrent_events_per_week_picks_per_task_best(self):
        """Two concurrent-friendly events in the same week: each task
        picks its OWN σ-best, independent of which event ranks higher
        overall.  This guards against accidental event-vs-task ordering
        bugs in the new loop structure."""
        t_hydrate = _task(label="hydrate", is_concurrent=True)
        t_breathe = _task(label="breathe", is_concurrent=True)
        office = _event(
            label="office_work", start=540, end=720, is_concurrent=True, date=DATE
        )
        reading = _event(
            label="reading", start=1100, end=1200, is_concurrent=True, date=DATE
        )
        # Hydrate matches office best; breathe matches reading best.
        sem = _semantic_pair_scores(
            {
                ("hydrate", "office_work"): 0.9,
                ("hydrate", "reading"): 0.6,
                ("breathe", "office_work"): 0.55,
                ("breathe", "reading"): 0.95,
            }
        )
        st_hydrate = _scheduled(
            t_hydrate,
            start=600,
            end=610,
            date=DATE,
            is_standalone=False,
            concurrent_with="office_work",
        )
        st_breathe = _scheduled(
            t_breathe,
            start=1110,
            end=1115,
            date=DATE,
            is_standalone=False,
            concurrent_with="reading",
        )
        sol = _solution(
            tasks=[t_hydrate, t_breathe], scheduled=[st_hydrate, st_breathe]
        )
        v = compute_l_concurrent(
            sol,
            _trace([office, reading]),
            sem,
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # Both tasks placed with their respective σ-best hosts to
        # σ_actual = σ_best for each to L = 0.
        assert v == pytest.approx(0.0)

    def test_task_side_exclusion_drops_task_from_phi(self):
        """When `RecommendedTask.is_concurrent=False` (source (ii)),
        the task is excluded per-pair from Φ even if every event in
        the calendar is concurrent-friendly."""
        # `is_concurrent=False` on the task = HealthTasks ontology said no.
        task = _task(label="cook_meal", is_concurrent=False)
        ev = _event(label="reading", is_concurrent=True, date=DATE)
        st = _scheduled(task, start=720, end=750, date=DATE, is_standalone=True)
        sol = _solution(tasks=[task], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # Task-side exclusion to no pair survives to Φ empty to None.
        assert v is None

    def test_host_too_short_for_task_excluded_from_phi(self):
        """A host shorter than the task's `duration_min` cannot contain it,
        so it is not a placeable opportunity and Φ stays empty."""
        task = _task(
            label="walk", is_concurrent=True, duration_min=90, duration_max=120
        )
        lunch = _event(label="lunch", start=720, end=780, is_concurrent=True, date=DATE)
        st = _scheduled(task, start=720, end=780, date=DATE, is_standalone=True)
        sol = _solution(tasks=[task], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([lunch]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # lunch is 60 min; the 90-min task cannot fit, so Φ is empty.
        assert v is None

    def test_too_short_best_host_falls_back_to_placeable_host(self):
        """σ_best ignores a higher-σ host the task cannot fit inside and
        uses the achievable host, so taking that host earns full credit."""
        task = _task(
            label="meditate", is_concurrent=True, duration_min=60, duration_max=120
        )
        snack = _event(label="snack", start=720, end=740, is_concurrent=True, date=DATE)
        meeting = _event(
            label="meeting", start=600, end=720, is_concurrent=True, date=DATE
        )
        st = _scheduled(
            task,
            start=600,
            end=660,
            date=DATE,
            is_standalone=False,
            concurrent_with="meeting",
        )
        sol = _solution(tasks=[task], scheduled=[st])
        sem = _semantic_pair_scores(
            {("meditate", "snack"): 0.9, ("meditate", "meeting"): 0.7}
        )
        v = compute_l_concurrent(
            sol,
            _trace([snack, meeting]),
            sem,
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # snack (σ 0.9) is 20 min and excluded; σ_best and σ_actual are both
        # 0.7 (meeting), so L = 0.
        assert v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_l_divide
# ---------------------------------------------------------------------------


class TestRefersTo:
    def test_label_match(self):
        d = _task(label="walk")
        st = _scheduled(d)
        assert _refers_to(st, d)

    def test_parent_label_match(self):
        d = _task(label="walk_10000_steps", is_dividable=True)
        st = _scheduled(_task(label="walk_morning"))
        st = ScheduledTask(
            task=_task(label="walk_morning"),
            start_minutes=0,
            end_minutes=10,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
            parent_task_label="walk_10000_steps",
        )
        assert _refers_to(st, d)

    def test_no_match_returns_false(self):
        d = _task(label="walk")
        st = _scheduled(_task(label="run"))
        assert not _refers_to(st, d)


class TestComputeLDivide:
    def test_no_dividable_returns_none(self):
        """When no dividable task exists, L_divide is *not applicable*
        and returns `None` so the weighted aggregator can drop the
        dimension instead of awarding free credit."""
        t = _task(label="a", is_dividable=False)
        sol = _solution(tasks=[t], scheduled=[_scheduled(t)])
        scalar, records = compute_l_divide(sol)
        assert scalar is None
        assert records == []

    def test_no_tasks_returns_none(self):
        sol = _solution()
        scalar, records = compute_l_divide(sol)
        assert scalar is None
        assert records == []

    def test_one_dividable_scheduled_once_gives_one(self):
        t = _task(label="walk", is_dividable=True)
        sol = _solution(tasks=[t], scheduled=[_scheduled(t, date=DATE)])
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert [r.verdict for r in records] == ["not_divided"]

    def test_valid_split_two_pieces_inside_band(self):
        # D=120, two pieces of 60 sum to the band centre; valid.
        t = _task(label="walk", is_dividable=True, duration_min=30, duration_max=120)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=540, date=DATE),
                _scheduled(t, start=900, end=960, date=DATE),
            ],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(0.0)
        assert [r.verdict for r in records] == ["divided_valid"]

    def test_valid_split_three_pieces_sum_within_band(self):
        t = _task(label="walk", is_dividable=True, duration_min=30, duration_max=70)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=500, date=DATE),  # 20
                _scheduled(t, start=720, end=745, date=DATE),  # 25
                _scheduled(t, start=1080, end=1105, date=DATE),  # 25
            ],
        )
        # sum=70 lands inside [59.5, 80.5].
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(0.0)
        assert records[0].verdict == "divided_valid"

    def test_rejects_sum_below_lower_band(self):
        t = _task(label="walk", is_dividable=True, duration_min=10, duration_max=120)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=490, date=DATE),  # 10
                _scheduled(t, start=600, end=615, date=DATE),  # 15
            ],
        )
        # sum=25 << 102 (= 120*0.85).
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert records[0].verdict == "divided_invalid_sum_too_low"

    def test_rejects_sum_above_upper_band(self):
        t = _task(label="walk", is_dividable=True, duration_min=10, duration_max=60)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=520, date=DATE),  # 40
                _scheduled(t, start=600, end=640, date=DATE),  # 40
                _scheduled(t, start=720, end=755, date=DATE),  # 35
            ],
        )
        # sum=115 > 69 (= 60*1.15).
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert records[0].verdict == "divided_invalid_sum_too_high"

    def test_rejects_piece_equal_to_original_duration(self):
        # Three 70-minute pieces for a 70-minute task is a repeat, not a divide.
        t = _task(label="walk", is_dividable=True, duration_min=30, duration_max=70)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=420, end=490, date=DATE),  # 70
                _scheduled(t, start=600, end=670, date=DATE),  # 70
                _scheduled(t, start=900, end=970, date=DATE),  # 70
            ],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert records[0].verdict == "divided_invalid_pieces_too_long"

    def test_rejects_piece_longer_than_original_duration(self):
        t = _task(label="walk", is_dividable=True, duration_min=30, duration_max=60)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=420, end=510, date=DATE),  # 90 > 60
                _scheduled(t, start=600, end=620, date=DATE),  # 20
            ],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert records[0].verdict == "divided_invalid_pieces_too_long"

    def test_parent_label_split_credited(self):
        t = _task(label="walk_10000", is_dividable=True, duration_max=120)
        st1 = ScheduledTask(
            task=_task(label="walk_morning"),
            start_minutes=420,
            end_minutes=480,  # 60
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
            parent_task_label="walk_10000",
        )
        st2 = ScheduledTask(
            task=_task(label="walk_evening"),
            start_minutes=1080,
            end_minutes=1140,  # 60
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
            parent_task_label="walk_10000",
        )
        sol = _solution(tasks=[t], scheduled=[st1, st2])
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(0.0)
        assert records[0].verdict == "divided_valid"

    def test_falls_back_to_plain_label_when_parent_task_label_none(self):
        t = _task(label="walk", is_dividable=True, duration_max=120)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=540, date=DATE),
                _scheduled(t, start=900, end=960, date=DATE),
            ],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(0.0)
        assert records[0].verdict == "divided_valid"

    def test_mean_across_weeks(self):
        # Week 1 has one piece (not_divided, 0/1). Week 2 has two pieces
        # summing to D (divided_valid, 1/1). Mean valid fraction is 0.5.
        t = _task(label="walk", is_dividable=True, duration_min=30, duration_max=120)
        date_w1 = DATE
        date_w2 = DATE + datetime.timedelta(days=7)
        sol = _solution(
            tasks=[t, t],
            scheduled=[
                _scheduled(t, start=480, end=540, date=date_w1),
                _scheduled(t, start=480, end=540, date=date_w2),
                _scheduled(t, start=900, end=960, date=date_w2),
            ],
        )
        scalar, _ = compute_l_divide(sol)
        assert scalar == pytest.approx(0.5)

    def test_unscheduled_dividable_counts_as_not_divided(self):
        t = _task(label="walk", is_dividable=True, duration_max=120)
        t2 = _task(label="run", is_dividable=True, duration_max=120)
        sol = _solution(
            tasks=[t, t2],
            scheduled=[_scheduled(t2, date=DATE)],
            unscheduled=[t],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        verdicts = sorted(r.verdict for r in records)
        assert verdicts == ["not_divided", "not_divided"]

    def test_no_scheduled_returns_none(self):
        """Dividable tasks all unscheduled with no scheduled anchors
        to no week is observable, so `L_divide` is not applicable."""
        t = _task(label="walk", is_dividable=True)
        sol = _solution(tasks=[t], unscheduled=[t])
        scalar, records = compute_l_divide(sol)
        assert scalar is None
        assert records == []

    def test_scheduled_non_dividable_task_skipped(self):
        dividable = _task(label="walk", is_dividable=True, duration_max=120)
        non_dividable = _task(label="log_water", is_dividable=False)
        sol = _solution(
            tasks=[dividable, non_dividable],
            scheduled=[
                _scheduled(non_dividable, date=DATE),
                _scheduled(dividable, date=DATE),
            ],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert [r.verdict for r in records] == ["not_divided"]

    def test_unscheduled_without_dividable_skipped(self):
        dividable = _task(label="walk", is_dividable=True, duration_max=120)
        non_dividable = _task(label="log_water", is_dividable=False)
        sol = _solution(
            tasks=[dividable, non_dividable],
            scheduled=[_scheduled(dividable, date=DATE)],
            unscheduled=[non_dividable],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert [r.verdict for r in records] == ["not_divided"]

    def test_unscheduled_mixed_dividable_only_dividable_anchored(self):
        dividable_a = _task(label="walk_a", is_dividable=True, duration_max=120)
        dividable_b = _task(label="walk_b", is_dividable=True, duration_max=120)
        non_dividable = _task(label="log_water", is_dividable=False)
        sol = _solution(
            tasks=[dividable_a, dividable_b, non_dividable],
            scheduled=[_scheduled(dividable_a, date=DATE)],
            unscheduled=[dividable_b, non_dividable],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert sorted(r.task_label for r in records) == ["walk_a", "walk_b"]

    def test_label_pieces_use_single_instance_budget(self):
        # Pieces of one label in a week score against a single instance,
        # even when `tasks` lists the label more than once. Four 60-minute
        # pieces (sum 240) overshoot the [102, 138] band for a 120-minute
        # task.
        t = _task(label="meals", is_dividable=True, duration_min=30, duration_max=120)
        sol = _solution(
            tasks=[t, t],
            scheduled=[
                _scheduled(t, start=300, end=360, date=DATE),  # 60
                _scheduled(t, start=420, end=480, date=DATE),  # 60
                _scheduled(t, start=540, end=600, date=DATE),  # 60
                _scheduled(t, start=720, end=780, date=DATE),  # 60
            ],
        )
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)
        assert records[0].instances_in_week == 1
        assert records[0].verdict == "divided_invalid_sum_too_high"

    def test_split_across_weeks_scores_one_instance_per_week(self):
        # The evaluate reconstruction lists `tasks` as one copy per
        # scheduled piece. A task split into two D/2 pieces each week must
        # still score against one instance, not as if each piece were its
        # own instance.
        t = _task(label="walk", is_dividable=True, duration_min=20, duration_max=60)
        scheduled = []
        for w in range(8):
            day = DATE + datetime.timedelta(days=7 * w)
            scheduled.append(_scheduled(t, start=480, end=510, date=day))  # 30
            scheduled.append(_scheduled(t, start=900, end=930, date=day))  # 30
        sol = _solution(tasks=[s.task for s in scheduled], scheduled=scheduled)
        scalar, records = compute_l_divide(sol)
        assert scalar == pytest.approx(0.0)
        assert {r.verdict for r in records} == {"divided_valid"}
        assert all(r.instances_in_week == 1 for r in records)

    def test_piece_count_does_not_inflate_instance_count(self):
        # Three pieces in one week are still one instance; sum 60 lands in
        # the [51, 69] band for a 60-minute task.
        t = _task(label="walk", is_dividable=True, duration_min=10, duration_max=60)
        scheduled = [
            _scheduled(t, start=480, end=500, date=DATE),  # 20
            _scheduled(t, start=600, end=620, date=DATE),  # 20
            _scheduled(t, start=720, end=740, date=DATE),  # 20
        ]
        sol = _solution(tasks=[s.task for s in scheduled], scheduled=scheduled)
        scalar, records = compute_l_divide(sol)
        assert records[0].instances_in_week == 1
        assert records[0].verdict == "divided_valid"

    def test_tolerance_zero_requires_exact_sum(self):
        t = _task(label="walk", is_dividable=True, duration_min=30, duration_max=100)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=540, date=DATE),  # 60
                _scheduled(t, start=600, end=640, date=DATE),  # 40
            ],
        )
        # sum=100 with tolerance_pct=0.0 matches exactly; valid.
        scalar_exact, _ = compute_l_divide(sol, tolerance_pct=0.0)
        assert scalar_exact == pytest.approx(0.0)
        # Shave one minute off; sum=99 with zero tolerance is below band.
        sol2 = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=540, date=DATE),  # 60
                _scheduled(t, start=600, end=639, date=DATE),  # 39
            ],
        )
        scalar_strict, records = compute_l_divide(sol2, tolerance_pct=0.0)
        assert scalar_strict == pytest.approx(1.0)
        assert records[0].verdict == "divided_invalid_sum_too_low"

    def test_l_divide_cross_week_does_not_count(self):
        """A task scheduled once per week across 2 weeks is not 'divided'."""
        t = _task(label="walk", is_dividable=True, duration_max=120)
        date_w1 = DATE
        date_w2 = DATE + datetime.timedelta(days=7)
        sol = _solution(
            tasks=[t, t],
            scheduled=[
                _scheduled(t, date=date_w1),
                _scheduled(t, date=date_w2),
            ],
        )
        scalar, _ = compute_l_divide(sol)
        assert scalar == pytest.approx(1.0)

    def test_verdict_record_carries_band_and_piece_metadata(self):
        t = _task(label="walk", is_dividable=True, duration_max=100)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=520, date=DATE),  # 40
                _scheduled(t, start=600, end=660, date=DATE),  # 60
            ],
        )
        _, records = compute_l_divide(sol)
        rec = records[0]
        assert rec.iso_year == DATE.isocalendar().year
        assert rec.iso_week == DATE.isocalendar().week
        assert rec.task_label == "walk"
        assert rec.instances_in_week == 1
        assert rec.original_duration_minutes == 100
        assert rec.piece_durations == (40, 60)
        assert rec.piece_dates == (DATE.isoformat(), DATE.isoformat())
        assert rec.sum_minutes == 100
        assert rec.tolerance_band == (pytest.approx(85.0), pytest.approx(115.0))
        assert rec.verdict == "divided_valid"


# ---------------------------------------------------------------------------
# SchedulingLoss.compute; assembles all 7 components
# ---------------------------------------------------------------------------


class TestSchedulingLoss:
    def test_returns_tuple(self):
        loss_fn = SchedulingLoss(
            weights=_default_weights(),
            semantic=_semantic_const(0.0),
            ruleset=_ruleset(),
            matcher=_matcher(),
            resolver=_resolver(),
        )
        total, comp = loss_fn.compute(_solution(), _trace(), horizon_epochs=7)
        assert isinstance(total, float)
        assert isinstance(comp, LossComponents)

    def test_all_zero_loss_returns_total_zero(self):
        loss_fn = SchedulingLoss(
            weights=_default_weights(),
            semantic=_semantic_const(0.0),
            ruleset=_ruleset(),
            matcher=_matcher(),
            resolver=_resolver(),
        )
        total, comp = loss_fn.compute(_solution(), _trace(), horizon_epochs=7)
        assert total == pytest.approx(0.0)

    def test_components_in_unit_interval(self):
        # Every per-component value should sit in [0, 1] regardless of inputs.
        t = _task(label="a", difficulty_level=4, is_dividable=True)
        st1 = _scheduled(t, start=600, end=700, date=DATE)
        st2 = _scheduled(t, start=620, end=720, date=DATE)
        sol = _solution(tasks=[t, t], scheduled=[st1, st2])
        loss_fn = SchedulingLoss(
            weights=_default_weights(),
            semantic=_semantic_const(0.5),
            ruleset=_ruleset(),
            matcher=_matcher(),
            resolver=_resolver(),
        )
        total, comp = loss_fn.compute(
            sol, _trace([_event(date=DATE)]), horizon_epochs=7
        )
        assert 0.0 <= total <= 1.0
        assert 0.0 <= comp.cov <= 1.0
        assert 0.0 <= comp.cal <= 1.0
        # pref / merge / divide / context_fit are float | None; None is the
        # "leg not applicable" signal, otherwise the leg sits in [0, 1].
        assert comp.pref is None or 0.0 <= comp.pref <= 1.0
        assert 0.0 <= comp.disp <= 1.0
        assert comp.merge is None or 0.0 <= comp.merge <= 1.0
        assert 0.0 <= comp.spread <= 1.0
        assert comp.divide is None or 0.0 <= comp.divide <= 1.0
        assert comp.context_fit is None or 0.0 <= comp.context_fit <= 1.0

    def test_divide_tolerance_threaded_to_compute_l_divide(self):
        """Tightening the tolerance should flip a borderline split to
        invalid, demonstrating the kwarg reaches `compute_l_divide`."""
        t = _task(label="walk", is_dividable=True, duration_min=30, duration_max=100)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=540, date=DATE),  # 60
                _scheduled(t, start=600, end=640, date=DATE),  # 40; sum=100
            ],
        )
        loose = SchedulingLoss(
            weights=_default_weights(),
            semantic=_semantic_const(0.0),
            ruleset=_ruleset(),
            matcher=_matcher(),
            resolver=_resolver(),
            divide_tolerance_pct=0.5,
        )
        strict = SchedulingLoss(
            weights=_default_weights(),
            semantic=_semantic_const(0.0),
            ruleset=_ruleset(),
            matcher=_matcher(),
            resolver=_resolver(),
            divide_tolerance_pct=0.0,
        )
        _, comp_loose = loose.compute(sol, _trace(), horizon_epochs=7)
        _, comp_strict = strict.compute(sol, _trace(), horizon_epochs=7)
        assert comp_loose.divide == pytest.approx(0.0)
        assert comp_strict.divide == pytest.approx(0.0)
        # Now break the sum so loose still passes but strict fails.
        sol2 = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=540, date=DATE),  # 60
                _scheduled(t, start=600, end=625, date=DATE),  # 25; sum=85
            ],
        )
        _, comp_loose2 = loose.compute(sol2, _trace(), horizon_epochs=7)
        _, comp_strict2 = strict.compute(sol2, _trace(), horizon_epochs=7)
        assert comp_loose2.divide == pytest.approx(0.0)
        assert comp_strict2.divide == pytest.approx(1.0)

    def test_divide_verdicts_populated_when_dividable_signal_present(self):
        t = _task(label="walk", is_dividable=True, duration_min=30, duration_max=120)
        sol = _solution(
            tasks=[t],
            scheduled=[
                _scheduled(t, start=480, end=540, date=DATE),
                _scheduled(t, start=900, end=960, date=DATE),
            ],
        )
        loss_fn = SchedulingLoss(
            weights=_default_weights(),
            semantic=_semantic_const(0.0),
            ruleset=_ruleset(),
            matcher=_matcher(),
            resolver=_resolver(),
        )
        _, comp = loss_fn.compute(sol, _trace(), horizon_epochs=7)
        assert len(comp.divide_verdicts) == 1
        assert comp.divide_verdicts[0].verdict == "divided_valid"

    def test_divide_verdicts_empty_when_no_signal(self):
        t = _task(label="walk", is_dividable=False)
        sol = _solution(tasks=[t], scheduled=[_scheduled(t)])
        loss_fn = SchedulingLoss(
            weights=_default_weights(),
            semantic=_semantic_const(0.0),
            ruleset=_ruleset(),
            matcher=_matcher(),
            resolver=_resolver(),
        )
        _, comp = loss_fn.compute(sol, _trace(), horizon_epochs=7)
        assert comp.divide_verdicts == ()
        assert comp.divide is None


# ---------------------------------------------------------------------------
# LossComponents.weighted / .gain
# ---------------------------------------------------------------------------


class TestLossComponentsArithmetic:
    def test_weighted_sum(self):
        w = LossWeights()  # defaults
        c = LossComponents(
            cov=1.0, cal=0.0, pref=0.0, disp=0.0, merge=0.0, spread=0.0, divide=0.0
        )
        assert c.weighted(w) == pytest.approx(w.lambda_cov)

    def test_gain_is_complement(self):
        w = LossWeights()
        c = LossComponents(
            cov=0.0, cal=0.0, pref=0.0, disp=0.0, merge=0.0, spread=0.0, divide=0.0
        )
        assert c.gain(w) == pytest.approx(1.0)

    def test_gain_equals_one_minus_weighted(self):
        w = LossWeights()
        c = LossComponents(
            cov=0.5, cal=0.5, pref=0.5, disp=0.5, merge=0.5, spread=0.5, divide=0.5
        )
        assert c.gain(w) == pytest.approx(1.0 - c.weighted(w))

    def test_none_divide_excludes_from_weighted_sum(self):
        """When `divide` is `None` the remaining 6 components must
        produce the same weighted loss they would on their own; i.e.
        the missing `λ_divide` slot is renormalised away rather than
        contributing zero, so the augmenter never gets a free 0.0 loss
        for a dimension that didn't apply."""
        w = LossWeights()
        c = LossComponents(
            cov=1.0, cal=1.0, pref=1.0, disp=1.0, merge=1.0, spread=1.0, divide=None
        )
        # All six applicable components are at full loss to weighted == 1.0
        # (renormalised against the 0.92 active-weight sum).
        assert c.weighted(w) == pytest.approx(1.0)
        assert c.gain(w) == pytest.approx(0.0)

    def test_none_divide_does_not_inflate_gain(self):
        """Compare two solutions that differ only on the divide field:
        `None` (no signal) must produce the *same* gain as a person
        whose divide column is filled with the *average* of the other
        six losses, not 0; otherwise greedy gets free credit on the
        divide dimension whenever no dividable task was generated."""
        w = LossWeights()
        signal = LossComponents(
            cov=0.0, cal=0.0, pref=0.0, disp=0.0, merge=1.0, spread=1.0, divide=1.0
        )
        no_signal = LossComponents(
            cov=0.0, cal=0.0, pref=0.0, disp=0.0, merge=1.0, spread=1.0, divide=None
        )
        # With the renormalisation, the no-signal gain is computed
        # over six components only.  We assert the *invariant*: the
        # no-signal version must NOT report a smaller loss than the
        # full-signal version that was already failing on divide.
        assert no_signal.weighted(w) <= signal.weighted(w)
        # And it must NOT collapse to "all gain" (1.0) just because
        # divide dropped out; the bad merge/spread still drag it down.
        assert no_signal.gain(w) < 1.0

    def test_all_none_returns_zero_weighted(self):
        """Defensive: a fully `None` row (no signal anywhere) must
        not raise; weighted just returns 0.0 because there is nothing
        to aggregate."""
        w = LossWeights()
        c = LossComponents(
            cov=None,  # type: ignore[arg-type]  # exercise the defensive branch
            cal=None,  # type: ignore[arg-type]
            pref=None,  # type: ignore[arg-type]
            disp=None,  # type: ignore[arg-type]
            merge=None,  # type: ignore[arg-type]
            spread=None,  # type: ignore[arg-type]
            divide=None,
        )
        assert c.weighted(w) == 0.0

    def test_zero_weight_sum_returns_zero(self):
        """When every applicable component has weight 0 (an edge case
        the validator would normally reject), the weighted aggregator
        must short-circuit instead of dividing by zero."""
        # Build a custom LossWeights stand-in via a tiny dataclass -
        # bypassing pydantic validation so we can probe the divide-by-
        # zero guard directly.
        from dataclasses import dataclass

        @dataclass
        class _ZeroWeights:
            lambda_cov: float = 0.0
            lambda_cal: float = 0.0
            lambda_pref: float = 0.0
            lambda_disp: float = 0.0
            lambda_merge: float = 0.0
            lambda_spread: float = 0.0
            lambda_divide: float = 0.0

        c = LossComponents(
            cov=0.5, cal=0.5, pref=0.5, disp=0.5, merge=0.5, spread=0.5, divide=0.5
        )
        assert c.weighted(_ZeroWeights()) == 0.0


# ---------------------------------------------------------------------------
# LossWeights validation
# ---------------------------------------------------------------------------


class TestLossWeightsValidation:
    def test_default_sums_to_one(self):
        w = LossWeights()
        total = (
            w.lambda_cov
            + w.lambda_cal
            + w.lambda_pref
            + w.lambda_disp
            + w.lambda_merge
            + w.lambda_spread
            + w.lambda_divide
        )
        assert total == pytest.approx(1.0)

    def test_non_one_sum_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LossWeights(
                lambda_cov=0.5,
                lambda_cal=0.5,
                lambda_pref=0.5,
                lambda_disp=0.0,
                lambda_merge=0.0,
                lambda_spread=0.0,
                lambda_divide=0.0,
            )


# ---------------------------------------------------------------------------
# Edge cases; coverage uplift
# ---------------------------------------------------------------------------


class TestLossEdgeCases:
    def test_l_disp_with_met_values_uses_continuous_leg(self):
        """When the resolver records carry MET values, the continuous
        leg of `L_disp` contributes; covers the `rec_a.met is not
        None and rec_b.met is not None` branch."""
        from src.scripts.scenarios.metrics.intensity_resolver import IntensityRecord

        # Tiny stub resolver that returns records with MET populated
        # on both sides; this is the cleanest way to exercise the
        # continuous-leg branch without spinning up a real MetLookup.
        class _StubResolver:
            def record(self, activity):
                return IntensityRecord(
                    intensity=2,
                    level_bucket=2,
                    met_bucket=2,
                    met=5.0,
                    matched_uri=None,
                    query_text="",
                )

        t = _task(label="t")
        st = _scheduled(t, start=420, end=480, date=DATE)
        ev = _event(label="e", start=420, end=480, date=DATE)
        sol = _solution(tasks=[t], scheduled=[st])
        trace = _trace([ev])
        v = compute_l_disp(
            sol, trace, _StubResolver(), half_life_days=2.0, max_met=16.8
        )
        # Categorical = (2+2)/8 = 0.5; continuous = (5+5)/(2*16.8)
        # ≈ 0.2976; max wins to 0.5.  lag = 1 (gap = 0).
        assert v == pytest.approx(0.5)

    def test_l_merge_best_score_not_replaced_when_lower(self):
        """Inside the σ-best lookup, an event with a lower score must
        not displace the current best; covers the `sc > best_score`
        False branch."""
        t = _task(label="task", is_concurrent=True)
        st = _scheduled(t, start=0, end=60, date=DATE, concurrent_with="best_ev")
        # Two events with different scores; the second has a lower
        # score than the first.  The σ-best loop must pick the first.
        # Both must be `is_concurrent=True` so the exclusion gate
        # does not skip the pair before σ_best is recorded.
        ev_best = _event(
            label="best_ev", start=0, end=60, date=DATE, is_concurrent=True
        )
        ev_worse = _event(
            label="worse_ev", start=0, end=60, date=DATE, is_concurrent=True
        )
        sol = _solution(tasks=[t], scheduled=[st])
        trace = _trace([ev_best, ev_worse])

        def scorer(a: str, b: str) -> float:
            return {("task", "best_ev"): 0.9, ("task", "worse_ev"): 0.3}[(a, b)]

        semantic = SemanticCompatibility(llm_scorer=scorer)
        v = compute_l_concurrent(
            sol,
            trace,
            semantic,
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # σ_best = 0.9, σ_actual = 0.9 (st.concurrent_with == best_ev's label)
        # to L_merge = 1 − 0.9/0.9 = 0.0.
        assert v == pytest.approx(0.0)

    def test_l_merge_concurrent_with_label_not_in_calendar(self):
        """`concurrent_with` referring to an event that does not
        exist in the calendar contributes 0 to σ_actual; covers the
        `ev is None` False branch."""
        t = _task(label="task", is_concurrent=True)
        st = _scheduled(
            t, start=0, end=60, date=DATE, concurrent_with="non_existent_label"
        )
        # Event must be `is_concurrent=True` so exclusion does not
        # zero σ_best before the lookup branch runs.
        ev = _event(label="lunch", start=0, end=60, date=DATE, is_concurrent=True)
        sol = _solution(tasks=[t], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace([ev]),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
            merge_threshold=0.5,
        )
        # σ_best = 0.9 (lunch is best), σ_actual = 0 (no match) to
        # L_merge = 1 − 0/0.9 = 1.0.
        assert v == pytest.approx(1.0)

    def test_l_merge_all_zero_phi_sum(self):
        """No events to Φ empty to returns None (no opportunity)."""
        t = _task(label="x")
        st = _scheduled(t, start=0, end=60, date=DATE)
        sol = _solution(tasks=[t], scheduled=[st])
        v = compute_l_concurrent(
            sol,
            _trace(),
            _semantic_const(0.9),
            ruleset=_ruleset(),
            matcher=_matcher(),
        )
        assert v is None
