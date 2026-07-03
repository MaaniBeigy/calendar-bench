"""Unit tests for src.scripts.persona.validation.check_ltl."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.persona.config.schema import TemporalRule
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.validation.check_ltl import (
    _compare,
    _latest_end,
    _latest_start,
    check_ltl_rules,
)


def _sched(person_id: str, day_events: list[dict]) -> PersonSchedule:
    days: list[DaySchedule] = []
    for idx, events in enumerate(day_events):
        evs: dict[str, list[EventInstance]] = {}
        for name, lst in events.items():
            evs[name] = [
                EventInstance(event_name=name, start=s, duration=d) for s, d in lst
            ]
        days.append(
            DaySchedule(
                day_index=idx,
                date=_dt.date(2026, 5, 4) + _dt.timedelta(days=idx),
                weekday="Mon",
                events=evs,
                spillovers=[],
            )
        )
    return PersonSchedule(
        person_id=person_id,
        persona_id="x",
        person_seed=1,
        days=days,
    )


def test_no_overlap_flags_overlapping_events():
    rule = TemporalRule(id="r1", formula="G ¬(sleep ∧ work)")
    sched = _sched(
        "x_0000",
        [
            {
                "sleep": [(0, 480)],
                "work": [(400, 120)],
            }
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any(v.rule_id == "r1" and "overlaps" in v.detail for v in out)


def test_no_overlap_silent_when_no_overlap():
    rule = TemporalRule(id="r1", formula="G ¬(sleep ∧ work)")
    sched = _sched(
        "x_0000",
        [
            {
                "sleep": [(0, 400)],
                "work": [(420, 120)],
            }
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []


def test_no_overlap_skips_when_one_side_missing():
    rule = TemporalRule(id="r1", formula="G ¬(sleep ∧ work)")
    sched = _sched("x_0000", [{"sleep": [(0, 400)]}])
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []


def test_min_overlap_flags_when_no_overlap():
    rule = TemporalRule(
        id="r2", formula="G (dinner → F (dinner ∧ reading))", min_fraction=1.0
    )
    sched = _sched(
        "x_0000",
        [
            {
                "dinner": [(1140, 60)],
                "reading": [(900, 60)],  # before dinner
            }
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any(v.rule_id == "r2" and "no overlapping reading" in v.detail for v in out)


def test_min_overlap_silent_when_overlap_exists():
    rule = TemporalRule(
        id="r2", formula="G (dinner → F (dinner ∧ reading))", min_fraction=1.0
    )
    sched = _sched(
        "x_0000",
        [
            {
                "dinner": [(1140, 60)],
                "reading": [(1170, 30)],
            }
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []


def test_implies_future_flags_when_b_starts_before_a_ends():
    rule = TemporalRule(id="r3", formula="G (work → F lunch)")
    sched = _sched(
        "x_0000",
        [
            {
                "work": [(540, 240)],  # ends at 780
                "lunch": [(720, 30)],  # starts at 720, before 780
            }
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any(v.rule_id == "r3" for v in out)


def test_implies_future_flags_when_b_missing():
    rule = TemporalRule(id="r3", formula="G (work → F lunch)")
    sched = _sched("x_0000", [{"work": [(540, 240)]}])
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any("present but no" in v.detail for v in out)


def test_implies_future_silent_when_a_missing():
    rule = TemporalRule(id="r3", formula="G (work → F lunch)")
    sched = _sched("x_0000", [{"lunch": [(720, 30)]}])
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []


def test_implies_future_silent_when_b_after_a():
    rule = TemporalRule(id="r3", formula="G (work → F lunch)")
    sched = _sched(
        "x_0000",
        [
            {
                "work": [(540, 60)],  # ends 600
                "lunch": [(720, 45)],
            }
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []


def test_weekly_count_ge_flags_under_target():
    rule = TemporalRule(id="r4", formula="weekly_count(running, ≥, 3)")
    sched = _sched(
        "x_0000",
        [{"running": [(420, 60)]}, {}, {}, {}, {}, {}, {}],  # 1 of 3
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any(v.rule_id == "r4" for v in out)


def test_weekly_count_ge_silent_when_target_met():
    rule = TemporalRule(id="r4", formula="weekly_count(running, ≥, 1)")
    sched = _sched(
        "x_0000",
        [{"running": [(420, 60)]}, {}, {}, {}, {}, {}, {}],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []


def test_weekly_count_le_operator():
    rule = TemporalRule(id="r4", formula="weekly_count(running, ≤, 1)")
    sched = _sched(
        "x_0000",
        [
            {"running": [(420, 60)]},
            {"running": [(420, 60)]},
            {},
            {},
            {},
            {},
            {},
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any(v.rule_id == "r4" for v in out)


def test_weekly_count_eq_operator():
    rule = TemporalRule(id="r4", formula="weekly_count(running, =, 1)")
    sched = _sched(
        "x_0000",
        [
            {"running": [(420, 60)]},
            {"running": [(420, 60)]},
            {},
            {},
            {},
            {},
            {},
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any(v.rule_id == "r4" for v in out)


def test_applies_to_filter_drops_persons_outside_set():
    rule = TemporalRule(
        id="r4",
        formula="weekly_count(running, ≥, 1)",
        applies_to={"occupation_status": ["fulltime"]},
    )
    sched = _sched("x_0000", [{}])
    # student is not in the applies_to list - rule must not apply.
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []


def test_applies_to_filter_includes_matching_persons():
    rule = TemporalRule(
        id="r4",
        formula="weekly_count(running, ≥, 1)",
        applies_to={"occupation_status": ["student"]},
    )
    sched = _sched("x_0000", [{}, {}, {}, {}, {}, {}, {}])
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any(v.rule_id == "r4" for v in out)


def test_eventually_co_occur_passes_when_overlap_exists_any_day():
    """`F (A AND B)` passes as long as A and B overlap on at least one day."""
    rule = TemporalRule(id="r_co", formula="F (work ∧ lunch)")
    sched = _sched(
        "x_0000",
        [
            {},  # day 0: nothing
            {"work": [(540, 60)], "lunch": [(560, 30)]},  # day 1: overlap
            {},  # day 2: nothing
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []


def test_eventually_co_occur_records_one_violation_when_never_overlap():
    """`F (A AND B)` fires a single violation across the horizon when A and B never meet."""
    rule = TemporalRule(id="r_co", formula="F (work ∧ lunch)")
    sched = _sched(
        "x_0000",
        [
            {"work": [(540, 60)], "lunch": [(720, 30)]},
            {"work": [(540, 60)], "lunch": [(720, 30)]},
        ],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert len(out) == 1
    assert out[0].rule_id == "r_co"
    assert "never co-occur" in out[0].detail


def test_eventually_co_occur_silent_when_one_side_missing():
    """`F (A AND B)` fires when neither side appears AND nothing overlaps."""
    rule = TemporalRule(id="r_co", formula="F (work ∧ lunch)")
    sched = _sched("x_0000", [{}])
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert len(out) == 1


def test_unrecognized_formula_records_violation():
    rule = TemporalRule(id="bad", formula="some unrecognized syntax")
    sched = _sched("x_0000", [{}])
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert len(out) == 1
    assert "unrecognized formula" in out[0].detail


def test_min_overlap_with_mismatched_head_records_violation():
    """The pattern G (A → F (B ∧ C)) where head != lhs is malformed."""
    rule = TemporalRule(id="r5", formula="G (alpha → F (beta ∧ gamma))")
    sched = _sched("x_0000", [{}])
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert any("malformed" in v.detail for v in out)


# -------------------------------------------------------------------------------------
# ----------------------------- internal helpers -------------------------------------
# -------------------------------------------------------------------------------------


def test_latest_end_with_empty_list_returns_none():
    assert _latest_end([]) is None


def test_latest_start_with_empty_list_returns_none():
    assert _latest_start([]) is None


def test_compare_strict_greater_than():
    assert _compare(5, ">", 3) is True
    assert _compare(3, ">", 3) is False


def test_compare_strict_less_than():
    assert _compare(2, "<", 3) is True
    assert _compare(3, "<", 3) is False


def test_compare_unsupported_operator_raises():
    with pytest.raises(ValueError, match="unsupported weekly_count operator"):
        _compare(1, "??", 1)


def test_weekly_count_returns_empty_when_schedule_has_no_days():
    """Public-API path that exercises the empty-schedule short-circuit."""
    rule = TemporalRule(id="r4", formula="weekly_count(running, ≥, 1)")
    sched = PersonSchedule(
        person_id="x_0000",
        persona_id="x",
        person_seed=1,
        days=[],
    )
    out = check_ltl_rules([sched], [rule], {"x_0000": "student"})
    assert out == []
