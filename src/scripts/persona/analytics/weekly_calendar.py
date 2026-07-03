"""Weekly calendar PNG renderer for base events, augmented tasks, and context bands."""

from __future__ import annotations

import datetime as _dt
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from calendar_view.calendar import Calendar
from calendar_view.config import style as _cv_style
from calendar_view.core import data
from calendar_view.core.config import CalendarConfig
from calendar_view.core.event import Event, EventStyle
from PIL import Image

from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.schedule import PersonSchedule

log = logging.getLogger(__name__)

DEFAULT_DPI = 300
_DAY_MINUTES = 24 * 60

# Increase the calendar's pixel density and shrink the rounded-corner
# radius so very short events (5 – 15 min augmented tasks) still satisfy
# the library's internal "rounded rectangle height >= 2 * radius"
# invariant.  Without these overrides Pillow raises
# `y1 must be greater than or equal to y0` while drawing the event box.
_cv_style.hour_height = 100
_cv_style.event_radius = 4
_cv_style.event_padding = 6
_cv_style.event_title_margin = 4
# Trim the title / notes fonts so multi-word labels fit a 400-px day
# column on a single line and short padded events still leave room for
# the actual-time annotation.
_cv_style.event_title_font = _cv_style.image_font(22)
_cv_style.event_notes_font = _cv_style.image_font(16)

# RGBA palette for the three event classes. Higher alpha = more solid fill.
BASE_FILL = (200, 200, 200, 190)
BASE_BORDER = (110, 110, 110, 240)
STANDALONE_FILL = (196, 234, 188, 200)
STANDALONE_BORDER = (90, 160, 90, 240)
CONCURRENT_FILL = (170, 190, 240, 200)
CONCURRENT_BORDER = (90, 110, 200, 240)

# Per-category context palette (RGBA, low alpha).
# Unknown categories fall back to CONTEXT_DEFAULT_*. The low alpha keeps
# context bands as background so events draw on top with darker borders.
CONTEXT_DEFAULT_FILL = (220, 220, 235, 90)
CONTEXT_DEFAULT_BORDER = (140, 140, 170, 160)
CATEGORY_PALETTE: dict[
    str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
] = {
    "mood_emotion": ((255, 220, 200, 95), (220, 130, 90, 170)),
    "energy_state": ((255, 240, 190, 95), (210, 170, 60, 170)),
    "physiological": ((230, 250, 220, 95), (130, 180, 90, 170)),
    "stress": ((255, 215, 215, 95), (210, 90, 90, 170)),
    "location": ((220, 235, 255, 95), (110, 140, 210, 170)),
    "social_context": ((240, 215, 245, 95), (170, 100, 200, 170)),
    "weather_environment": ((220, 245, 250, 95), (100, 170, 200, 170)),
    "behaviour_state": ((230, 230, 245, 95), (130, 130, 200, 170)),
    "capability_opportunity": ((225, 240, 230, 95), (110, 170, 140, 170)),
    "goal_intention": ((250, 235, 200, 95), (200, 160, 60, 170)),
    "sdt_regulation": ((240, 225, 215, 95), (180, 130, 90, 170)),
    "trait_state": ((225, 225, 230, 95), (140, 140, 160, 170)),
}

# Floor that keeps `draw_rounded_rectangle` from collapsing.  Setting it
# this low (≥ `2 * event_radius` pixels in clock-minute terms) only
# guards against the library's crash mode; callers should pass a larger
# `min_visible_minutes` to `plot_week_calendar` whenever titles need
# to render inside the event box.
_RENDER_FLOOR_MINUTES = max(
    1, int((_cv_style.event_radius * 2) * 60 / _cv_style.hour_height) + 1
)

# Default minimum visible event duration.  At `hour_height=100` this
# produces a 75-px box that comfortably fits the 22-px title font plus a
# 16-px notes line for the actual-time annotation, so a 5-min augmented
# task still ships its display name and original clock window.
DEFAULT_MIN_VISIBLE_MINUTES = 45
# Backwards-compat alias retained because external callers (and a few
# tests) imported the old constant directly.  Always equal to the
# rendering floor; bumping the visible duration is now a per-call knob.
MIN_VISIBLE_MINUTES = _RENDER_FLOOR_MINUTES


@dataclass(frozen=True)
class AugmentedTaskRecord:
    """One scheduled task ready for rendering on a weekly calendar."""

    label: str
    display_name: str
    date: _dt.date
    start_minutes: int
    end_minutes: int
    is_standalone: bool
    concurrent_with: str | None

    @classmethod
    def from_dict(cls, payload: dict) -> AugmentedTaskRecord:
        return cls(
            label=str(payload["label"]),
            display_name=str(payload.get("display_name") or payload["label"]),
            date=_dt.date.fromisoformat(payload["date"]),
            start_minutes=int(payload["start_minutes"]),
            end_minutes=int(payload["end_minutes"]),
            is_standalone=bool(payload.get("is_standalone", True)),
            concurrent_with=payload.get("concurrent_with"),
        )


def load_augmented_tasks(json_path: Path | str) -> list[AugmentedTaskRecord]:
    """Read `scheduled` tasks from one scenarios augmented JSON.

    Returns an empty list if the file is missing or has no scheduled entries.
    """
    path = Path(json_path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [AugmentedTaskRecord.from_dict(d) for d in payload.get("scheduled", [])]


def load_augmented_tasks_by_person(
    persons_dir: Path | str,
) -> dict[str, list[AugmentedTaskRecord]]:
    """Read every augmented `<person_id>.json` under `persons_dir`.

    Sidecar `*_loss.json` files are skipped.  Returns
    `{person_id: [AugmentedTaskRecord, ...]}` in person-id order, with
    an empty mapping when the directory does not exist.
    """
    out: dict[str, list[AugmentedTaskRecord]] = {}
    base = Path(persons_dir)
    if not base.is_dir():
        return out
    for jp in sorted(base.glob("*.json")):
        if jp.stem.endswith("_loss"):
            continue
        payload = json.loads(jp.read_text(encoding="utf-8"))
        pid = str(payload.get("person_id", jp.stem))
        out[pid] = [
            AugmentedTaskRecord.from_dict(d) for d in payload.get("scheduled", [])
        ]
    return out


@dataclass(frozen=True)
class WeekWindow:
    """One Monday–Sunday slice of the run horizon."""

    week_index: int
    start_date: _dt.date
    end_date: _dt.date


def iso_week_windows(schedule: PersonSchedule) -> list[WeekWindow]:
    """Split the schedule into Monday–Sunday weekly windows.

    The first window's `start_date` is the Monday of the horizon's
    first day (snapping back over any leading weekend); the last
    window's `end_date` is the Sunday of the horizon's last day.  An
    empty schedule returns an empty list.
    """
    if not schedule.days:
        return []
    dates = [d.date for d in schedule.days]
    horizon_start = min(dates)
    horizon_end = max(dates)
    monday = horizon_start - _dt.timedelta(days=horizon_start.weekday())
    cursor = monday
    windows: list[WeekWindow] = []
    idx = 1
    while cursor <= horizon_end:
        sunday = cursor + _dt.timedelta(days=6)
        windows.append(WeekWindow(week_index=idx, start_date=cursor, end_date=sunday))
        cursor = sunday + _dt.timedelta(days=1)
        idx += 1
    return windows


def _minutes_to_hhmm(total: int) -> str:
    """Return `HH:MM` for a minutes-from-midnight value, clamped to `[0, 23:59]`.

    Calendar-view's hour grid spans 00:00 – 23:59 inclusive; values
    outside this range raise inside the library.  Clamping makes the
    renderer robust to spillover events that extend past midnight.
    """
    clamped = max(0, min(total, _DAY_MINUTES - 1))
    return f"{clamped // 60:02d}:{clamped % 60:02d}"


def _visible_end_minutes(start: int, raw_end: int, min_visible: int) -> int:
    """Compute the rendered end-of-event clock so the event box is legible.

    `calendar-view` cannot draw a rounded box whose height is less than
    `2 * event_radius` pixels; the library raises mid-render.  We pad
    the rendered duration to `min_visible` minutes (the larger of the
    caller's choice and the rendering floor) so titles can render in
    short augmented tasks, and we cap the result at 23:59 so the event
    never spills past midnight in the view.
    """
    floor = max(min_visible, _RENDER_FLOOR_MINUTES)
    padded = max(raw_end, start + floor)
    return max(start + 1, min(padded, _DAY_MINUTES - 1))


def _actual_time_note(start: int, raw_end: int, rendered_end: int) -> str | None:
    """Return `actual HH:MM – HH:MM` if the event was visually padded.

    When the renderer expands a 5-minute task to 45 minutes for legibility
    we still want the user to see the truthful clock window.  This note
    is appended to whatever the caller wrote so the slot is obvious.
    """
    real_end = max(start + 1, min(raw_end, _DAY_MINUTES - 1))
    if rendered_end <= real_end:
        return None
    return f"actual {_minutes_to_hhmm(start)} – {_minutes_to_hhmm(real_end)}"


def _compose_notes(*parts: str | None) -> str | None:
    """Join non-empty note fragments with newlines.

    `calendar-view` treats `None` notes as absent; we keep that
    behavior while letting callers stack `with <event>` plus the
    actual-time annotation without sprinkling conditionals everywhere.
    """
    chunks = [p for p in parts if p]
    return "\n".join(chunks) if chunks else None


def _base_event(
    event_name: str,
    day_date: _dt.date,
    start: int,
    duration: int,
    *,
    min_visible: int,
) -> Event:
    raw_end = start + duration
    end_min = _visible_end_minutes(start, raw_end, min_visible)
    notes = _actual_time_note(start, raw_end, end_min)
    return Event(
        title=event_name,
        notes=notes,
        day=day_date,
        start=_minutes_to_hhmm(start),
        end=_minutes_to_hhmm(end_min),
        style=EventStyle(event_border=BASE_BORDER, event_fill=BASE_FILL),
    )


def _augmented_event(task: AugmentedTaskRecord, *, min_visible: int) -> Event:
    end_min = _visible_end_minutes(task.start_minutes, task.end_minutes, min_visible)
    if task.is_standalone:
        border, fill = STANDALONE_BORDER, STANDALONE_FILL
        prefix = None
    else:
        border, fill = CONCURRENT_BORDER, CONCURRENT_FILL
        prefix = f"with {task.concurrent_with}" if task.concurrent_with else None
    notes = _compose_notes(
        prefix,
        _actual_time_note(task.start_minutes, task.end_minutes, end_min),
    )
    return Event(
        title=task.display_name,
        notes=notes,
        day=task.date,
        start=_minutes_to_hhmm(task.start_minutes),
        end=_minutes_to_hhmm(end_min),
        style=EventStyle(event_border=border, event_fill=fill),
    )


def _context_event(episode: ContextEpisode, *, min_visible: int) -> Event:
    """Build one translucent calendar Event for a context episode."""
    fill, border = CATEGORY_PALETTE.get(
        episode.category, (CONTEXT_DEFAULT_FILL, CONTEXT_DEFAULT_BORDER)
    )
    end_min = _visible_end_minutes(
        episode.start_minutes, episode.end_minutes, min_visible
    )
    notes = _compose_notes(
        f"ctx {episode.category}",
        _actual_time_note(episode.start_minutes, episode.end_minutes, end_min),
    )
    return Event(
        title=episode.name,
        notes=notes,
        day=episode.date,
        start=_minutes_to_hhmm(episode.start_minutes),
        end=_minutes_to_hhmm(end_min),
        style=EventStyle(event_border=border, event_fill=fill),
    )


def _resolve_context_filter(
    context_categories: Iterable[str] | None,
) -> frozenset[str] | None:
    """Return None to skip contexts, otherwise a frozenset of categories to render."""
    if context_categories is None:
        return None
    return frozenset(context_categories)


def _events_for_window(
    schedule: PersonSchedule,
    augmented: Iterable[AugmentedTaskRecord],
    window: WeekWindow,
    *,
    min_visible: int,
    context_categories: Iterable[str] | None = None,
) -> list[Event]:
    """Collect every renderable Event that falls inside the window."""
    events: list[Event] = []
    for day in schedule.days:
        if not (window.start_date <= day.date <= window.end_date):
            continue
        for event_name, instances in day.events.items():
            for inst in instances:
                events.append(
                    _base_event(
                        event_name,
                        day.date,
                        inst.start,
                        inst.duration,
                        min_visible=min_visible,
                    )
                )
    for task in augmented:
        if window.start_date <= task.date <= window.end_date:
            events.append(_augmented_event(task, min_visible=min_visible))
    cat_filter = _resolve_context_filter(context_categories)
    if cat_filter:
        for episode in schedule.contexts:
            if episode.category not in cat_filter:
                continue
            if not (window.start_date <= episode.date <= window.end_date):
                continue
            events.append(_context_event(episode, min_visible=min_visible))
    return events


def _save_calendar_with_dpi(calendar: Calendar, target_path: Path, dpi: int) -> None:
    """Render the calendar PNG with embedded `(dpi, dpi)` metadata.

    `Calendar.save` itself does not accept DPI, so we let it write the
    PNG once and then re-save through Pillow with the metadata fields
    populated.  The pixel data is unchanged; only the chunk that
    image viewers read for "pixels per inch" gets stamped.
    """
    calendar.save(str(target_path))
    with Image.open(str(target_path)) as img:
        img.load()
    img.save(str(target_path), "PNG", dpi=(dpi, dpi))


def plot_week_calendar(
    person_id: str,
    window: WeekWindow,
    schedule: PersonSchedule,
    augmented: Iterable[AugmentedTaskRecord],
    out_path: Path | str,
    *,
    dpi: int = DEFAULT_DPI,
    title: str | None = None,
    min_visible_minutes: int = DEFAULT_MIN_VISIBLE_MINUTES,
    context_categories: Iterable[str] | None = None,
) -> Path:
    """Render one week of the schedule plus optional augmented and context overlays to PNG."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = (
            f"{person_id}; week {window.week_index} "
            f"({window.start_date.isoformat()} – {window.end_date.isoformat()})"
        )
    dates_str = f"{window.start_date.isoformat()} - {window.end_date.isoformat()}"

    cfg = CalendarConfig(
        lang="en",
        title=title,
        dates=dates_str,
        show_year=True,
        show_date=True,
        legend=False,
        title_vertical_align="top",
    )
    data.validate_config(cfg)

    events = _events_for_window(
        schedule,
        list(augmented),
        window,
        min_visible=min_visible_minutes,
        context_categories=context_categories,
    )
    data.validate_events(events, cfg)

    calendar = Calendar.build(cfg)
    calendar.add_events(events)
    _save_calendar_with_dpi(calendar, target, dpi=dpi)
    return target


def render_weekly_calendars_for_person(
    schedule: PersonSchedule,
    augmented: list[AugmentedTaskRecord],
    out_dir: Path | str,
    *,
    dpi: int = DEFAULT_DPI,
    title_prefix: str | None = None,
    min_visible_minutes: int = DEFAULT_MIN_VISIBLE_MINUTES,
    context_categories: Iterable[str] | None = None,
) -> list[Path]:
    """Emit one PNG per ISO week of `schedule`'s horizon.

    Files are named `week_<NN>_<YYYY-MM-DD>.png` where `<NN>` is the
    1-based week index and `<YYYY-MM-DD>` is that week's Monday.
    Returns the list of paths in week order; an empty schedule returns
    an empty list.
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for window in iso_week_windows(schedule):
        fname = f"week_{window.week_index:02d}_{window.start_date.isoformat()}.png"
        title = None
        if title_prefix:
            title = (
                f"{title_prefix}; {schedule.person_id}; week {window.week_index} "
                f"({window.start_date.isoformat()} – {window.end_date.isoformat()})"
            )
        written.append(
            plot_week_calendar(
                schedule.person_id,
                window,
                schedule,
                augmented,
                base / fname,
                dpi=dpi,
                title=title,
                min_visible_minutes=min_visible_minutes,
                context_categories=context_categories,
            )
        )
    return written


def render_weekly_calendars(
    schedules: list[PersonSchedule],
    augmented_by_person: dict[str, list[AugmentedTaskRecord]] | None,
    out_dir: Path | str,
    *,
    dpi: int = DEFAULT_DPI,
    title_prefix: str | None = None,
    min_visible_minutes: int = DEFAULT_MIN_VISIBLE_MINUTES,
    context_categories: Iterable[str] | None = None,
) -> dict[str, list[Path]]:
    """Render weekly calendars for every person under out_dir/person_id."""
    root = Path(out_dir)
    out: dict[str, list[Path]] = {}
    for schedule in schedules:
        per_person_dir = root / schedule.person_id
        augmented = (augmented_by_person or {}).get(schedule.person_id, [])
        out[schedule.person_id] = render_weekly_calendars_for_person(
            schedule,
            augmented,
            per_person_dir,
            dpi=dpi,
            title_prefix=title_prefix,
            min_visible_minutes=min_visible_minutes,
            context_categories=context_categories,
        )
    return out


__all__ = [
    "AugmentedTaskRecord",
    "BASE_BORDER",
    "BASE_FILL",
    "CATEGORY_PALETTE",
    "CONCURRENT_BORDER",
    "CONCURRENT_FILL",
    "CONTEXT_DEFAULT_BORDER",
    "CONTEXT_DEFAULT_FILL",
    "DEFAULT_DPI",
    "DEFAULT_MIN_VISIBLE_MINUTES",
    "MIN_VISIBLE_MINUTES",
    "STANDALONE_BORDER",
    "STANDALONE_FILL",
    "WeekWindow",
    "iso_week_windows",
    "load_augmented_tasks",
    "load_augmented_tasks_by_person",
    "plot_week_calendar",
    "render_weekly_calendars",
    "render_weekly_calendars_for_person",
]
