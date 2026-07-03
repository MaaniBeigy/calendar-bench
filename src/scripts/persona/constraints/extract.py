"""Per-day per-event constraint extraction with seasonality and trend support.

Patterns in `temporal_patterns` accept two related keys that disambiguate
what the pattern adjusts:

- `target: episodes` (default); adjusts `base_count` (episode count).
- `target: duration`; adjusts per-event / total duration bounds.

The `unit` field declares how `amount` is read and is honoured uniformly
by both seasonality and trend:

- `unit: count` (target episodes); `amount` is an absolute episode delta.
- `unit: percent` (target episodes or duration); `amount` is a per-cent of
  the base count or duration.
- `unit: minutes` / `unit: hours` (target duration); `amount` is an absolute
  duration shift (hours convert to minutes).

When `unit` is omitted the legacy reading applies: duration targets are
minutes, episode seasonality is a per-cent, and episode trend is an absolute
count.

Patterns may declare a `scale` for the trend / seasonality axis.  Today
the solver accepts `day | week | month | season | weekday`; `start` /
`end` (on trend) are interpreted as 1-indexed indices on that scale.
Internally the extractor maps `day_idx` to `scale_idx` using simple
bucketing (7 days / week, 30 days / month, 90 days / season) when no
horizon start date is supplied; calendar-accurate arithmetic kicks in when
`horizon_start_date` is passed.

Examples::

    # Trend: gym duration ramps +20 min over weeks 1-4 (+5 min/week)
    - mode: trend
      details:
        scale: week
        unit: minutes
        direction: increasing
        amount: 20
        start: 1
        end: 4

    # Seasonality: 30 % more walking in May/June
    - mode: seasonality
      details:
        target: episodes
        unit: percent
        scale: month
        within: [May, June]
        amount: 30
        direction: increasing
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

from src.scripts.persona.config.schema import EventDefinition, TemporalPattern
from src.scripts.persona.domain.time_windows import WindowMap

# -------------------------------------------------------------------------------------
# ----------------------------------- shared helpers ----------------------------------
# -------------------------------------------------------------------------------------

# Both the long-form ("Monday") and short-form ("Mon") accepted in seasonality details.
_WEEKDAY_INDEX: dict[str, int] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}
_WEEKEND_TOKENS: frozenset[str] = frozenset(
    {"Saturday", "Sunday", "Sat", "Sun", "weekend"}
)

# Month-name to 1-indexed calendar month.
_MONTH_INDEX: dict[str, int] = {
    name: idx
    for idx, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
_MONTH_INDEX.update(
    {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        # "May" already covered (full == short)
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Sept": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
)

# Season-name to 1-indexed meteorological season (spring = MAM, etc.).
# Mapping follows the northern-hemisphere convention.
_SEASON_INDEX: dict[str, int] = {
    "spring": 1,
    "summer": 2,
    "autumn": 3,
    "fall": 3,
    "winter": 4,
}
_SEASON_MONTHS: dict[int, tuple[int, int, int]] = {
    1: (3, 4, 5),  # spring
    2: (6, 7, 8),  # summer
    3: (9, 10, 11),  # autumn
    4: (12, 1, 2),  # winter
}


def _to_minutes(value: int, unit: str) -> int:
    return value * 60 if unit == "hours" else value


def _amount_in_minutes(amount: float, unit: str) -> float:
    """Convert a duration amount expressed in `unit` to minutes."""
    if unit == "hours":
        return amount * 60.0
    return amount


def _resolve_amount_unit(details: dict[str, Any], mode: str) -> str:
    """Return the amount unit (`count` | `percent` | `minutes` | `hours`).

    An explicit `unit` wins. Without it, the legacy reading applies: duration
    targets are minutes, episode seasonality is percent, and episode trend is
    an absolute count.
    """
    unit_raw = details.get("unit")
    if isinstance(unit_raw, str) and unit_raw in ("count", "percent", "minutes", "hours"):
        return unit_raw
    if details.get("target") == "duration":
        return "minutes"
    return "percent" if mode == "seasonality" else "count"


# -------------------------------------------------------------------------------------
# ------------------------------- scale-index derivation ------------------------------
# -------------------------------------------------------------------------------------


def _day_to_calendar_month(day_idx: int, start_date: _dt.date | None) -> int:
    """Return the 1-indexed calendar month for `day_idx` of the horizon.

    When `start_date` is unset, day 0 is treated as the 1st of January
    (a stable, deterministic default for unit tests and configs that do
    not feed a date into the extractor).
    """
    base = start_date if start_date is not None else _dt.date(2000, 1, 1)
    return (base + _dt.timedelta(days=day_idx)).month


def _calendar_month_to_season(month: int) -> int:
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    if month in (9, 10, 11):
        return 3
    return 4  # 12, 1, 2 to winter


def _scale_indices(
    scale: str,
    *,
    day_idx: int,
    total_days: int,
    horizon_start_date: _dt.date | None,
) -> tuple[int, int]:
    """Map `day_idx` to a 0-indexed `(scale_idx, scale_total)` tuple.

    Bucket fallbacks (used when no horizon date is given) approximate week
    (7 days), month (30 days), season (90 days).  When a calendar date is
    supplied we walk the actual calendar.
    """
    if scale == "day":
        return day_idx, max(1, total_days)
    if scale == "week":
        return day_idx // 7, max(1, (total_days + 6) // 7)
    if scale == "month":
        if horizon_start_date is None:
            return day_idx // 30, max(1, (total_days + 29) // 30)
        cur = horizon_start_date + _dt.timedelta(days=day_idx)
        end = horizon_start_date + _dt.timedelta(days=max(0, total_days - 1))
        elapsed = (cur.year - horizon_start_date.year) * 12 + (
            cur.month - horizon_start_date.month
        )
        total_months = (
            (end.year - horizon_start_date.year) * 12
            + (end.month - horizon_start_date.month)
            + 1
        )
        return elapsed, max(1, total_months)
    if scale == "season":
        if horizon_start_date is None:
            return day_idx // 90, max(1, (total_days + 89) // 90)
        cur = horizon_start_date + _dt.timedelta(days=day_idx)
        end = horizon_start_date + _dt.timedelta(days=max(0, total_days - 1))
        cur_season = _calendar_month_to_season(cur.month)
        start_season = _calendar_month_to_season(horizon_start_date.month)
        end_season = _calendar_month_to_season(end.month)
        # Walk seasons inclusively from horizon start to current day.
        season_idx = 0
        s = start_season
        y = horizon_start_date.year
        m = horizon_start_date.month
        while not (y == cur.year and _calendar_month_to_season(m) == cur_season):
            season_idx += 1
            s = (s % 4) + 1
            m = _SEASON_MONTHS[s][0]
            if s == 1:
                y += 1
        # Total seasons in horizon (same walk, until end date).
        total = 1
        s = start_season
        y = horizon_start_date.year
        m = horizon_start_date.month
        while not (y == end.year and _calendar_month_to_season(m) == end_season):
            total += 1
            s = (s % 4) + 1
            m = _SEASON_MONTHS[s][0]
            if s == 1:
                y += 1
        return season_idx, max(1, total)
    # Unknown scales (e.g. `weekday`): day-indexed fall-back.
    return day_idx, max(1, total_days)


# -------------------------------------------------------------------------------------
# --------------------------------- output dataclass ----------------------------------
# -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventDayConstraints:
    """Resolved constraints for one event type on one day.

    `per_event_min` / `per_event_max` and `total_min` / `total_max`
    already incorporate any duration adjustments produced by
    `target: duration` / `unit: minutes|hours` patterns.
    """

    min_count: int
    max_count: int
    base_count: int
    per_event_min: int
    per_event_max: int
    total_min: int
    total_max: int
    allowed_windows: tuple[str, ...]
    window_fraction_constraints: tuple[tuple[str, float], ...]
    event_disabled: bool = False


# -------------------------------------------------------------------------------------
# -------------------------------- pattern application --------------------------------
# -------------------------------------------------------------------------------------


def _within_weekday_indices(within: Any) -> list[int]:
    """Map a within-spec (list or string of weekday names) to weekday indices."""
    if isinstance(within, str):
        return [_WEEKDAY_INDEX[within]] if within in _WEEKDAY_INDEX else []
    if isinstance(within, list):
        return [_WEEKDAY_INDEX[d] for d in within if d in _WEEKDAY_INDEX]
    return []


def _within_month_indices(within: Any) -> list[int]:
    if isinstance(within, str):
        return [_MONTH_INDEX[within]] if within in _MONTH_INDEX else []
    if isinstance(within, list):
        return [_MONTH_INDEX[m] for m in within if m in _MONTH_INDEX]
    return []


def _within_season_indices(within: Any) -> list[int]:
    def _norm(token: str) -> str:
        return token.lower()

    if isinstance(within, str):
        token = _norm(within)
        return [_SEASON_INDEX[token]] if token in _SEASON_INDEX else []
    if isinstance(within, list):
        return [
            _SEASON_INDEX[_norm(s)]
            for s in within
            if isinstance(s, str) and _norm(s) in _SEASON_INDEX
        ]
    return []


# Backwards-compatible alias (older callers and the test suite imported this name).
_within_indices = _within_weekday_indices


def _shift_count(base: float, amount: float, direction: str, is_percent: bool) -> float:
    """Apply a count shift as a percent multiplier or an absolute delta."""
    signed = -amount if direction == "decreasing" else amount
    return base * (1 + signed / 100) if is_percent else base + signed


def _apply_weekday_seasonality(
    base_count: float,
    *,
    within: Any,
    amount: float,
    direction: str,
    day_of_week: int,
    is_percent: bool,
) -> tuple[float, bool]:
    """Return (adjusted_base, event_disabled) for weekday-scoped episode seasonality."""
    allowed = _within_weekday_indices(within)
    if is_percent and amount == 100:
        if day_of_week not in allowed:
            return 0.0, True
        return base_count, False
    if day_of_week in allowed:
        return _shift_count(base_count, amount, direction, is_percent), False
    return base_count, False


def _apply_month_seasonality(
    base_count: float,
    *,
    within: Any,
    amount: float,
    direction: str,
    calendar_month: int,
    is_percent: bool,
) -> tuple[float, bool]:
    """Per-month shift for seasonality patterns scoped to `scale: month`."""
    allowed = _within_month_indices(within)
    if not allowed:
        return base_count, False
    if is_percent and amount == 100:
        if calendar_month not in allowed:
            return 0.0, True
        return base_count, False
    if calendar_month in allowed:
        return _shift_count(base_count, amount, direction, is_percent), False
    return base_count, False


def _apply_season_seasonality(
    base_count: float,
    *,
    within: Any,
    amount: float,
    direction: str,
    calendar_month: int,
    is_percent: bool,
) -> tuple[float, bool]:
    allowed = _within_season_indices(within)
    if not allowed:
        return base_count, False
    cur_season = _calendar_month_to_season(calendar_month)
    if is_percent and amount == 100:
        if cur_season not in allowed:
            return 0.0, True
        return base_count, False
    if cur_season in allowed:
        return _shift_count(base_count, amount, direction, is_percent), False
    return base_count, False


def _is_weekend_day(day_of_week: int) -> bool:
    return day_of_week in (5, 6)


def _classify_window_within(within: Any, day_of_week: int) -> tuple[bool, bool]:
    """Detect Saturday/Sunday/weekend tokens in `within`. Returns (seasonal, skip).

    Behaviour matches the legacy pipeline: a string "weekend" sets `is_skip`
    on weekdays, but a string "Saturday" or "Sunday" never sets `is_skip`,
    only `is_seasonal` when the day matches.
    """
    weekend_now = _is_weekend_day(day_of_week)
    if isinstance(within, str):
        if within == "weekend":
            return (weekend_now, not weekend_now)
        if within in {"Saturday", "Sat"}:
            return day_of_week == 5, False
        if within in {"Sunday", "Sun"}:
            return day_of_week == 6, False
        return False, False
    if isinstance(within, list):
        is_seasonal = False
        is_skip = False
        for token in within:
            if token in _WEEKEND_TOKENS:
                if weekend_now:
                    is_seasonal = True
                else:
                    is_skip = True
        return is_seasonal, is_skip
    return False, False


def _boosted_windows(within: Any, window_map: WindowMap) -> list[str]:
    if isinstance(within, str):
        return [within] if within in window_map else []
    if isinstance(within, list):
        return [w for w in within if w in window_map]
    return []


def _window_fraction_constraints(
    *,
    boosted: list[str],
    all_windows: list[str],
    amount: int,
) -> list[tuple[str, float]]:
    n_allowed = len(all_windows)
    n_boosted = len(boosted)
    if n_allowed == 0:
        return []
    if n_boosted == 0:
        uniform = 1.0 / n_allowed
        return [(w, uniform) for w in all_windows]
    baseline = [w for w in all_windows if w not in boosted]
    n_baseline = n_allowed - n_boosted
    boost = amount / 100.0
    if amount == 100:
        boost_frac = 1.0 / n_boosted
        out = [(w, boost_frac) for w in boosted]
        out.extend((w, 0.0) for w in baseline)
        return out
    baseline_frac = (1.0 - boost) / n_allowed if n_baseline > 0 else 0.0
    boost_frac = baseline_frac + boost / n_boosted
    out = [(w, boost_frac) for w in boosted]
    out.extend((w, baseline_frac) for w in baseline)
    return out


def _apply_trend(
    base_count: float,
    *,
    direction: str,
    amount: float,
    start_idx: int,
    end_idx: int,
    scale_idx: int,
) -> float:
    """Linear ramp on base_count from scale-index `start` to `end` (1-indexed).

    `scale_idx` is 0-indexed within the scale's axis.  Today's behavior
    on `scale: day` matches: `start=1, end=7, day_idx=3` ramps from
    indices 0 to 6 with progress `(3-0)/(6-0) = 0.5`.
    """
    s, e = start_idx - 1, end_idx - 1
    if e <= s:
        return base_count
    if scale_idx < s:
        progress = 0.0
    elif scale_idx > e:
        progress = 1.0
    else:
        progress = (scale_idx - s) / (e - s)
    delta = progress * float(amount)
    if direction == "increasing":
        return base_count + delta
    if direction == "decreasing":
        return base_count - delta
    return base_count


def _trend_duration_delta(
    *,
    direction: str,
    amount: float,
    start_idx: int,
    end_idx: int,
    scale_idx: int,
) -> float:
    """Linear ramp on duration (minutes) from scale-index `start` to `end`.

    `amount` is already in minutes.  Returns the signed minute delta for
    `scale_idx`.
    """
    s, e = start_idx - 1, end_idx - 1
    if e <= s:
        return 0.0
    if scale_idx < s:
        progress = 0.0
    elif scale_idx > e:
        progress = 1.0
    else:
        progress = (scale_idx - s) / (e - s)
    delta = progress * float(amount)
    if direction == "increasing":
        return delta
    if direction == "decreasing":
        return -delta
    return 0.0


def _seasonality_duration_delta(
    *,
    within: Any,
    amount: float,
    direction: str,
    scale: str | None,
    day_of_week: int,
    calendar_month: int,
) -> float:
    """Duration delta (minutes) from a seasonality pattern with target=duration.

    Weekday-scoped (`scale: weekday`): fires when `day_of_week` matches
    `within`.  Month-scoped (`scale: month`): fires when the calendar
    month matches.  Season-scoped (`scale: season`): same shape.
    Weekend-scoped (no scale, weekend tokens in `within`): fires on
    matching Saturday / Sunday.  Window-scoped seasonality (e.g.
    `within: morning`) has no meaningful duration target, so returns 0.
    """
    signed = float(amount) if direction == "increasing" else -float(amount)
    if scale == "weekday":
        allowed = _within_weekday_indices(within)
        return signed if day_of_week in allowed else 0.0
    if scale == "month":
        allowed = _within_month_indices(within)
        return signed if calendar_month in allowed else 0.0
    if scale == "season":
        allowed = _within_season_indices(within)
        return signed if _calendar_month_to_season(calendar_month) in allowed else 0.0
    # Non-weekday/month/season: check weekend/day-name tokens only.
    is_seasonal, _ = _classify_window_within(within, day_of_week)
    return signed if is_seasonal else 0.0


def _trend_progress(start_idx: int, end_idx: int, scale_idx: int) -> float:
    """Linear 0..1 ramp position on a trend axis (1-indexed start/end)."""
    s, e = start_idx - 1, end_idx - 1
    if e <= s or scale_idx < s:
        return 0.0
    if scale_idx > e:
        return 1.0
    return (scale_idx - s) / (e - s)


def _apply_trend_percent(
    base_count: float,
    *,
    direction: str,
    amount: float,
    start_idx: int,
    end_idx: int,
    scale_idx: int,
) -> float:
    """Percent episode-count trend; ramps base_count by amount% across the window."""
    pct = _trend_progress(start_idx, end_idx, scale_idx) * amount / 100.0
    if direction == "increasing":
        return base_count * (1 + pct)
    if direction == "decreasing":
        return base_count * (1 - pct)
    return base_count  # pragma: no cover


def _trend_duration_factor(
    *,
    direction: str,
    amount: float,
    start_idx: int,
    end_idx: int,
    scale_idx: int,
) -> float:
    """Percent duration trend; multiplicative factor ramped across the window."""
    pct = _trend_progress(start_idx, end_idx, scale_idx) * amount / 100.0
    if direction == "increasing":
        return 1.0 + pct
    if direction == "decreasing":
        return 1.0 - pct
    return 1.0  # pragma: no cover


def _seasonality_duration_factor(
    *,
    within: Any,
    amount: float,
    direction: str,
    scale: str | None,
    day_of_week: int,
    calendar_month: int,
) -> float:
    """Percent duration seasonality; multiplicative factor on matching days."""
    signed = -amount / 100.0 if direction == "decreasing" else amount / 100.0
    if scale == "weekday":
        return 1.0 + signed if day_of_week in _within_weekday_indices(within) else 1.0
    if scale == "month":
        return 1.0 + signed if calendar_month in _within_month_indices(within) else 1.0
    if scale == "season":
        cur = _calendar_month_to_season(calendar_month)
        return 1.0 + signed if cur in _within_season_indices(within) else 1.0
    is_seasonal, _ = _classify_window_within(within, day_of_week)
    return 1.0 + signed if is_seasonal else 1.0


def _normalise_fractions(
    fractions: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    total = sum(f for _, f in fractions)
    if total <= 1.0:
        return fractions
    return [(w, f / total) for w, f in fractions]


# -------------------------------------------------------------------------------------
# ------------------------------------- main API --------------------------------------
# -------------------------------------------------------------------------------------


def get_event_constraints(
    event_def: EventDefinition,
    *,
    day_idx: int,
    total_days: int,
    window_map: WindowMap,
    day_of_week: int,
    horizon_start_date: _dt.date | None = None,
) -> EventDayConstraints:
    """Resolve per-event constraints for the given day, applying seasonality and trend.

    Episode counts come from `total_event_episodes`. The returned
    `base_count` is the seasonally-adjusted, trend-shifted target episode
    count clamped to [min_count, max_count].

    Duration bounds (`per_event_min/max`, `total_min/max`) start from the
    catalog's `per_event_duration` / `total_event_duration` fields and are
    then shifted by any `target: duration` (or `unit: minutes | hours`)
    patterns (trend or seasonality).  The shift is applied to both min and
    max, preserving the original range width, then clamped so min ≥ 0 and
    max ≥ min.

    `horizon_start_date` is optional; when provided, month / season
    seasonality patterns use calendar arithmetic.  Without it the extractor
    falls back to 30-day months and 90-day seasons indexed from
    `day_idx`.
    """
    eps = event_def.total_event_episodes
    min_count = eps.min
    max_count = eps.max
    base_count = float(min_count + (max_count - min_count) // 2)
    allowed_windows: list[str] = []
    fraction_constraints: list[tuple[str, float]] = []
    all_window_names = list(window_map.ranges.keys())
    event_disabled = False
    duration_delta = 0.0
    duration_factor = 1.0

    calendar_month = _day_to_calendar_month(day_idx, horizon_start_date)

    for pattern in event_def.temporal_patterns:
        (
            new_base,
            new_disabled,
            new_allowed,
            new_fractions,
            dur_delta,
            dur_factor,
        ) = _apply_pattern(
            pattern,
            base_count=base_count,
            day_idx=day_idx,
            total_days=total_days,
            day_of_week=day_of_week,
            window_map=window_map,
            all_window_names=all_window_names,
            horizon_start_date=horizon_start_date,
            calendar_month=calendar_month,
        )
        base_count = new_base
        if new_disabled:
            event_disabled = True
        allowed_windows.extend(new_allowed)
        fraction_constraints.extend(new_fractions)
        duration_delta += dur_delta
        duration_factor *= dur_factor

    if event_disabled:
        min_count = 0
        max_count = 0
        base_count = 0.0

    clamped = max(min_count, min(max_count, int(round(base_count))))

    fraction_constraints = _normalise_fractions(fraction_constraints)

    # Base duration values before delta application.
    raw_per_min = _to_minutes(
        event_def.per_event_duration.min, event_def.per_event_duration.unit
    )
    raw_per_max = _to_minutes(
        event_def.per_event_duration.max, event_def.per_event_duration.unit
    )
    raw_tot_min = _to_minutes(
        event_def.total_event_duration.min, event_def.total_event_duration.unit
    )
    raw_tot_max = _to_minutes(
        event_def.total_event_duration.max, event_def.total_event_duration.unit
    )

    # Apply duration patterns: multiplicative percent factor, then additive
    # minute delta, from target=duration seasonality / trend.
    per_event_min = max(0, round(raw_per_min * duration_factor + duration_delta))
    per_event_max = max(
        per_event_min, round(raw_per_max * duration_factor + duration_delta)
    )
    total_min = max(0, round(raw_tot_min * duration_factor + duration_delta))
    total_max = max(total_min, round(raw_tot_max * duration_factor + duration_delta))

    return EventDayConstraints(
        min_count=min_count,
        max_count=max_count,
        base_count=clamped,
        per_event_min=per_event_min,
        per_event_max=per_event_max,
        total_min=total_min,
        total_max=total_max,
        allowed_windows=tuple(allowed_windows),
        window_fraction_constraints=tuple(fraction_constraints),
        event_disabled=event_disabled,
    )


def _apply_pattern(
    pattern: TemporalPattern,
    *,
    base_count: float,
    day_idx: int,
    total_days: int,
    day_of_week: int,
    window_map: WindowMap,
    all_window_names: list[str],
    horizon_start_date: _dt.date | None,
    calendar_month: int,
) -> tuple[float, bool, list[str], list[tuple[str, float]], float, float]:
    """Apply a single pattern.

    Returns (new_base_count, event_disabled, allowed_windows_added,
    window_fractions_added, duration_delta_minutes, duration_factor).
    `duration_delta_minutes` carries additive minute shifts; `duration_factor`
    carries multiplicative percent shifts; both default to the identity.
    """
    details = pattern.details
    within = details.get("within")
    amount_raw = float(details.get("amount", 0))
    direction = details.get("direction", "increasing")
    scale = details.get("scale")
    amt_unit = _resolve_amount_unit(details, pattern.mode)
    is_percent = amt_unit == "percent"
    is_duration_target = amt_unit in ("minutes", "hours") or (
        is_percent and details.get("target") == "duration"
    )
    # The placement-fraction seasonality (window-token `within`) reads `amount`
    # as a percent of episode placement.
    amount_percent = int(amount_raw)

    if pattern.mode == "fix":
        added: list[str] = []
        if isinstance(within, list):
            added = [w for w in within if w in window_map]
        elif isinstance(within, str) and within in window_map:
            added = [within]
        return base_count, False, added, [], 0.0, 1.0

    if pattern.mode == "seasonality":
        if is_duration_target:
            if is_percent:
                dur_factor = _seasonality_duration_factor(
                    within=within,
                    amount=amount_raw,
                    direction=direction,
                    scale=scale,
                    day_of_week=day_of_week,
                    calendar_month=calendar_month,
                )
                return base_count, False, [], [], 0.0, dur_factor
            dur_delta = _seasonality_duration_delta(
                within=within,
                amount=_amount_in_minutes(amount_raw, amt_unit),
                direction=direction,
                scale=scale,
                day_of_week=day_of_week,
                calendar_month=calendar_month,
            )
            return base_count, False, [], [], dur_delta, 1.0

        # --- episode-count seasonality (count or percent) ---
        if scale == "weekday":
            new_base, disabled = _apply_weekday_seasonality(
                base_count,
                within=within,
                amount=amount_raw,
                direction=direction,
                day_of_week=day_of_week,
                is_percent=is_percent,
            )
            return new_base, disabled, [], [], 0.0, 1.0
        if scale == "month":
            new_base, disabled = _apply_month_seasonality(
                base_count,
                within=within,
                amount=amount_raw,
                direction=direction,
                calendar_month=calendar_month,
                is_percent=is_percent,
            )
            return new_base, disabled, [], [], 0.0, 1.0
        if scale == "season":
            new_base, disabled = _apply_season_seasonality(
                base_count,
                within=within,
                amount=amount_raw,
                direction=direction,
                calendar_month=calendar_month,
                is_percent=is_percent,
            )
            return new_base, disabled, [], [], 0.0, 1.0
        is_seasonal, is_skip = _classify_window_within(within, day_of_week)
        if is_seasonal:
            return (
                _shift_count(base_count, amount_raw, direction, is_percent),
                False,
                [],
                [],
                0.0,
                1.0,
            )
        if is_skip:
            return base_count, False, [], [], 0.0, 1.0
        boosted = _boosted_windows(within, window_map)
        fractions = _window_fraction_constraints(
            boosted=boosted, all_windows=all_window_names, amount=amount_percent
        )
        return base_count, False, [], fractions, 0.0, 1.0

    if pattern.mode == "trend":
        trend_scale = scale or "day"
        scale_idx, _scale_total = _scale_indices(
            trend_scale,
            day_idx=day_idx,
            total_days=total_days,
            horizon_start_date=horizon_start_date,
        )
        start_idx = int(details.get("start", 1))
        end_idx = int(details.get("end", _scale_total))
        if is_duration_target:
            if is_percent:
                dur_factor = _trend_duration_factor(
                    direction=direction,
                    amount=amount_raw,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    scale_idx=scale_idx,
                )
                return base_count, False, [], [], 0.0, dur_factor
            dur_delta = _trend_duration_delta(
                direction=direction,
                amount=_amount_in_minutes(amount_raw, amt_unit),
                start_idx=start_idx,
                end_idx=end_idx,
                scale_idx=scale_idx,
            )
            return base_count, False, [], [], dur_delta, 1.0

        if is_percent:
            new_base = _apply_trend_percent(
                base_count,
                direction=direction,
                amount=amount_raw,
                start_idx=start_idx,
                end_idx=end_idx,
                scale_idx=scale_idx,
            )
        else:
            new_base = _apply_trend(
                base_count,
                direction=direction,
                amount=amount_raw,
                start_idx=start_idx,
                end_idx=end_idx,
                scale_idx=scale_idx,
            )
        return new_base, False, [], [], 0.0, 1.0

    return (
        base_count,
        False,
        [],
        [],
        0.0,
        1.0,
    )  # pragma: no cover  (pydantic blocks other modes)
