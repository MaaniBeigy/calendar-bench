"""LTL rule checks: same-type non-overlap, no-overlap, implies-future, weekly-count.

Formula syntax mirrors the legacy parser so existing rules keep working:

- `G ¬(A ∧ B)`         no overlap between A and B on the same day
- `F (A ∧ B)`          A and B must co-occur at least once in the horizon
- `G (A → F (A ∧ B))`  every A has at least one B overlapping it
- `G (A → F B)`        if A appears, B must appear after A's latest end
- `weekly_count(E, op, n)`  the count of E across each rolling 7-day window
  must satisfy `op n` (op in {>=, <=, =, ≥, ≤}).

Atoms may be event names (bare label) or contexts (prefixed
`context:<name>`); both kinds resolve to per-day intervals.

`applies_to` filters which persons a rule applies to. Reserved key
`personas` matches the persona id; any other key matches a person's
characteristic value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.scripts.persona.config.schema import TemporalRule
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule

CONTEXT_PREFIX = "context:"


@dataclass(frozen=True, slots=True)
class _Inst:
    """Day-local interval used by the LTL checker for either atom kind."""

    start: int
    duration: int


def _atom_instances(
    schedule: PersonSchedule, day: DaySchedule, atom: str
) -> list[_Inst]:
    """Return all `_Inst`s for `atom` on this day, events or contexts."""
    if atom.startswith(CONTEXT_PREFIX):
        name = atom[len(CONTEXT_PREFIX) :]
        out: list[_Inst] = []
        for episode in schedule.contexts:
            if episode.name != name or episode.date != day.date:
                continue
            out.append(_Inst(start=episode.start_minutes, duration=episode.duration))
        return out
    events = day.events.get(atom, [])
    return [_Inst(start=e.start, duration=e.duration) for e in events]


def _atom_count_in_window(
    schedule: PersonSchedule, days: list[DaySchedule], atom: str
) -> int:
    """Sum of per-day instances of `atom` across `days`."""
    return sum(len(_atom_instances(schedule, d, atom)) for d in days)


_NO_OVERLAP_RE = re.compile(r"^\s*G\s+¬\(\s*([^\s∧]+)\s*∧\s*([^\s)]+)\s*\)\s*$")
_EVENTUAL_CO_OCCUR_RE = re.compile(r"^\s*F\s*\(\s*([^\s∧]+)\s*∧\s*([^\s)]+)\s*\)\s*$")
_MIN_OVERLAP_RE = re.compile(
    r"^\s*G\s+\(\s*([^\s→]+)\s*→\s*F\s*\(\s*([^\s∧]+)\s*∧\s*([^\s)]+)\s*\)\s*\)\s*$"
)
_IMPLIES_FUTURE_RE = re.compile(r"^\s*G\s+\(\s*([^\s→]+)\s*→\s*F\s+([^\s)]+)\s*\)\s*$")
_WEEKLY_COUNT_RE = re.compile(
    r"^\s*weekly_count\(\s*([^,]+?)\s*,\s*([≥≤=><]=?)\s*,\s*(\d+)\s*\)\s*$"
)


@dataclass(frozen=True, slots=True)
class LTLViolation:
    """One LTL rule violation."""

    person_id: str
    rule_id: str
    day_index: int | None
    detail: str


def _person_matches(person: Person | None, applies_to: dict[str, list]) -> bool:
    """Filter rules that scope themselves to a subset of personas."""
    if not applies_to:
        return True
    if person is None:
        return False
    for key, allowed in applies_to.items():
        if key == "personas":
            if person.persona_id not in allowed:
                return False
            continue
        value = person.characteristics.get(key)
        if value is None or not _value_in(value, allowed):
            return False
    return True


def _value_in(value, allowed: list) -> bool:
    if value in allowed:
        return True
    lowered = str(value).lower()
    return lowered in {str(v).lower() for v in allowed}


def _events_overlap(s1: int, d1: int, s2: int, d2: int) -> bool:
    return not (s1 + d1 <= s2 or s2 + d2 <= s1)


def _check_no_overlap(
    schedule: PersonSchedule, rule: TemporalRule, t1: str, t2: str
) -> list[LTLViolation]:
    out: list[LTLViolation] = []
    for day in schedule.days:
        e1 = _atom_instances(schedule, day, t1)
        e2 = _atom_instances(schedule, day, t2)
        if not e1 or not e2:
            continue
        for i, a in enumerate(e1):
            for j, b in enumerate(e2):
                if _events_overlap(a.start, a.duration, b.start, b.duration):
                    out.append(
                        LTLViolation(
                            person_id=schedule.person_id,
                            rule_id=rule.id,
                            day_index=day.day_index,
                            detail=(
                                f"{t1}[{i}] (start={a.start}, dur={a.duration}) "
                                f"overlaps {t2}[{j}] (start={b.start}, dur={b.duration})"
                            ),
                        )
                    )
    return out


def _check_eventually_co_occur(
    schedule: PersonSchedule, rule: TemporalRule, a_name: str, b_name: str
) -> list[LTLViolation]:
    """Emit one violation when A and B never overlap anywhere in the horizon."""
    for day in schedule.days:
        e_a = _atom_instances(schedule, day, a_name)
        e_b = _atom_instances(schedule, day, b_name)
        for a in e_a:
            for b in e_b:
                if _events_overlap(a.start, a.duration, b.start, b.duration):
                    return []
    return [
        LTLViolation(
            person_id=schedule.person_id,
            rule_id=rule.id,
            day_index=None,
            detail=f"{a_name} and {b_name} never co-occur across the horizon",
        )
    ]


def _check_min_overlap(
    schedule: PersonSchedule, rule: TemporalRule, a_name: str, b_name: str
) -> list[LTLViolation]:
    out: list[LTLViolation] = []
    for day in schedule.days:
        e_a = _atom_instances(schedule, day, a_name)
        e_b = _atom_instances(schedule, day, b_name)
        if not e_a or not e_b:
            continue
        for i, a in enumerate(e_a):
            ok = any(
                _events_overlap(a.start, a.duration, b.start, b.duration) for b in e_b
            )
            if not ok:
                out.append(
                    LTLViolation(
                        person_id=schedule.person_id,
                        rule_id=rule.id,
                        day_index=day.day_index,
                        detail=(
                            f"{a_name}[{i}] (start={a.start}, dur={a.duration}) "
                            f"has no overlapping {b_name} on this day"
                        ),
                    )
                )
    return out


def _latest_end(events: list) -> int | None:
    if not events:
        return None
    return max(ev.start + ev.duration for ev in events)


def _latest_start(events: list) -> int | None:
    if not events:
        return None
    return max(ev.start for ev in events)


def _check_implies_future(
    schedule: PersonSchedule, rule: TemporalRule, a_name: str, b_name: str
) -> list[LTLViolation]:
    out: list[LTLViolation] = []
    for day in schedule.days:
        e_a = _atom_instances(schedule, day, a_name)
        e_b = _atom_instances(schedule, day, b_name)
        if not e_a:
            continue
        if not e_b:
            out.append(
                LTLViolation(
                    person_id=schedule.person_id,
                    rule_id=rule.id,
                    day_index=day.day_index,
                    detail=f"{a_name} present but no {b_name} that day",
                )
            )
            continue
        a_end = _latest_end(e_a)
        b_start = _latest_start(e_b)
        if a_end is not None and b_start is not None and b_start < a_end:
            out.append(
                LTLViolation(
                    person_id=schedule.person_id,
                    rule_id=rule.id,
                    day_index=day.day_index,
                    detail=(
                        f"latest {b_name} start {b_start} precedes latest "
                        f"{a_name} end {a_end}"
                    ),
                )
            )
    return out


def _compare(actual: int, op: str, expected: int) -> bool:
    if op in (">=", "≥"):
        return actual >= expected
    if op in ("<=", "≤"):
        return actual <= expected
    if op == "=":
        return actual == expected
    if op == ">":
        return actual > expected
    if op == "<":
        return actual < expected
    raise ValueError(f"unsupported weekly_count operator: {op!r}")


def _check_weekly_count(
    schedule: PersonSchedule, rule: TemporalRule, event_name: str, op: str, n: int
) -> list[LTLViolation]:
    if not schedule.days:
        return []
    out: list[LTLViolation] = []
    # Slide a 7-day window across the horizon. A horizon shorter than 7 days is
    # checked once with the days available, which is the conservative thing.
    total = len(schedule.days)
    for week_start in range(0, total, 7):
        week = schedule.days[week_start : week_start + 7]
        actual = _atom_count_in_window(schedule, week, event_name)
        if not _compare(actual, op, n):
            out.append(
                LTLViolation(
                    person_id=schedule.person_id,
                    rule_id=rule.id,
                    day_index=schedule.days[week_start].day_index,
                    detail=(
                        f"weekly_count({event_name}) = {actual}, expected "
                        f"{op} {n} (week starting day {week_start})"
                    ),
                )
            )
    return out


def _dispatch_rule(
    schedule: PersonSchedule,
    rule: TemporalRule,
    person: Person | None,
) -> list[LTLViolation]:
    if not _person_matches(person, rule.applies_to):
        return []
    formula = rule.formula.strip()

    m = _NO_OVERLAP_RE.match(formula)
    if m:
        return _check_no_overlap(schedule, rule, m.group(1), m.group(2))

    m = _EVENTUAL_CO_OCCUR_RE.match(formula)
    if m:
        return _check_eventually_co_occur(schedule, rule, m.group(1), m.group(2))

    m = _MIN_OVERLAP_RE.match(formula)
    if m:
        head, lhs, rhs = m.group(1), m.group(2), m.group(3)
        if head != lhs:
            return [
                LTLViolation(
                    person_id=schedule.person_id,
                    rule_id=rule.id,
                    day_index=None,
                    detail=f"malformed min_overlap formula: {formula!r}",
                )
            ]
        return _check_min_overlap(schedule, rule, head, rhs)

    m = _IMPLIES_FUTURE_RE.match(formula)
    if m:
        return _check_implies_future(schedule, rule, m.group(1), m.group(2))

    m = _WEEKLY_COUNT_RE.match(formula)
    if m:
        return _check_weekly_count(
            schedule, rule, m.group(1), m.group(2), int(m.group(3))
        )

    return [
        LTLViolation(
            person_id=schedule.person_id,
            rule_id=rule.id,
            day_index=None,
            detail=f"unrecognized formula shape: {formula!r}",
        )
    ]


def check_ltl_rules(
    schedules: list[PersonSchedule],
    rules: list[TemporalRule],
    persons_by_id: dict[str, Person] | dict[str, str],
) -> list[LTLViolation]:
    """Apply every rule to every (matching) schedule.

    `persons_by_id` maps `person_id` to a `Person` (preferred) or to the
    legacy `occupation_status` string for callers that have not yet been
    updated; the string form is treated as a person with that single
    characteristic populated.
    """
    violations: list[LTLViolation] = []
    for schedule in schedules:
        person = _resolve_person(persons_by_id, schedule.person_id)
        for rule in rules:
            violations.extend(_dispatch_rule(schedule, rule, person))
    return violations


def _resolve_person(persons_by_id: dict, person_id: str) -> Person | None:
    """Promote a legacy `{id: status}` dict to a synthetic `Person` view."""
    candidate = persons_by_id.get(person_id)
    if candidate is None:
        return None
    if isinstance(candidate, Person):
        return candidate
    from src.scripts.persona.config.schema import JitterConfig

    return Person(
        person_id=person_id,
        persona_id=person_id,
        person_seed=0,
        instance_index=0,
        occupation_status=candidate,
        characteristics={"occupation_status": candidate},
        jitter_applied=JitterConfig(),
    )


__all__ = ["LTLViolation", "check_ltl_rules"]
