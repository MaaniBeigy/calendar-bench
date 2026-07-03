"""Unit tests for src.scripts.persona.analytics.weekly_calendar."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from src.scripts.persona.analytics.weekly_calendar import (
    BASE_BORDER,
    BASE_FILL,
    CATEGORY_PALETTE,
    CONCURRENT_BORDER,
    CONCURRENT_FILL,
    CONTEXT_DEFAULT_BORDER,
    CONTEXT_DEFAULT_FILL,
    DEFAULT_DPI,
    STANDALONE_BORDER,
    STANDALONE_FILL,
    AugmentedTaskRecord,
    WeekWindow,
    _augmented_event,
    _base_event,
    _context_event,
    _events_for_window,
    _minutes_to_hhmm,
    _resolve_context_filter,
    _save_calendar_with_dpi,
    iso_week_windows,
    load_augmented_tasks,
    load_augmented_tasks_by_person,
    plot_week_calendar,
    render_weekly_calendars,
    render_weekly_calendars_for_person,
)
from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule

_MONDAY = _dt.date(2026, 5, 4)  # Mon
_WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# -------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------


def _day(
    day_index: int,
    events: dict[str, list[tuple[int, int]]],
    *,
    start_date: _dt.date = _MONDAY,
) -> DaySchedule:
    target = start_date + _dt.timedelta(days=day_index)
    return DaySchedule(
        day_index=day_index,
        date=target,
        weekday=_WEEKDAY_LABELS[target.weekday()],
        events={
            name: [
                EventInstance(event_name=name, start=s, duration=d) for s, d in pairs
            ]
            for name, pairs in events.items()
        },
        spillovers=[],
    )


def _schedule(
    person_id: str,
    day_events: list[dict[str, list[tuple[int, int]]]],
    *,
    persona_id: str = "alice",
    start_date: _dt.date = _MONDAY,
    contexts: list[ContextEpisode] | None = None,
) -> PersonSchedule:
    return PersonSchedule(
        person_id=person_id,
        persona_id=persona_id,
        person_seed=42,
        days=[_day(i, ev, start_date=start_date) for i, ev in enumerate(day_events)],
        contexts=list(contexts or []),
    )


def _ctx(
    name: str,
    category: str,
    *,
    day_offset: int = 0,
    start: int = 480,
    end: int = 540,
) -> ContextEpisode:
    return ContextEpisode(
        name=name,
        category=category,
        date=_MONDAY + _dt.timedelta(days=day_offset),
        start_minutes=start,
        end_minutes=end,
    )


def _standalone(label: str, day: _dt.date, start: int, end: int) -> AugmentedTaskRecord:
    return AugmentedTaskRecord(
        label=label,
        display_name=f"{label.title()} 🍎",
        date=day,
        start_minutes=start,
        end_minutes=end,
        is_standalone=True,
        concurrent_with=None,
    )


def _concurrent(
    label: str, day: _dt.date, start: int, end: int, with_label: str
) -> AugmentedTaskRecord:
    return AugmentedTaskRecord(
        label=label,
        display_name=f"{label.title()} ✍️",
        date=day,
        start_minutes=start,
        end_minutes=end,
        is_standalone=False,
        concurrent_with=with_label,
    )


# -------------------------------------------------------------------------------------
# AugmentedTaskRecord
# -------------------------------------------------------------------------------------


class TestAugmentedTaskRecord:
    def test_from_dict_parses_all_fields(self):
        payload = {
            "label": "pack_one_fruit_snack",
            "display_name": "Pack One Fruit Snack 🍉",
            "date": "2026-05-04",
            "start_minutes": 400,
            "end_minutes": 405,
            "is_standalone": True,
            "concurrent_with": None,
        }
        rec = AugmentedTaskRecord.from_dict(payload)
        assert rec.label == "pack_one_fruit_snack"
        assert rec.display_name == "Pack One Fruit Snack 🍉"
        assert rec.date == _dt.date(2026, 5, 4)
        assert rec.start_minutes == 400
        assert rec.end_minutes == 405
        assert rec.is_standalone is True
        assert rec.concurrent_with is None

    def test_from_dict_falls_back_to_label_when_display_name_missing(self):
        rec = AugmentedTaskRecord.from_dict(
            {
                "label": "x",
                "date": "2026-05-04",
                "start_minutes": 0,
                "end_minutes": 5,
            }
        )
        assert rec.display_name == "x"
        assert rec.is_standalone is True
        assert rec.concurrent_with is None

    def test_from_dict_blank_display_name_falls_back_to_label(self):
        """An empty `display_name` value (not just absent) still falls back."""
        rec = AugmentedTaskRecord.from_dict(
            {
                "label": "x",
                "display_name": "",
                "date": "2026-05-04",
                "start_minutes": 0,
                "end_minutes": 5,
            }
        )
        assert rec.display_name == "x"

    def test_from_dict_concurrent_with_value(self):
        rec = AugmentedTaskRecord.from_dict(
            {
                "label": "stretch",
                "date": "2026-05-04",
                "start_minutes": 400,
                "end_minutes": 410,
                "is_standalone": False,
                "concurrent_with": "lunch",
            }
        )
        assert rec.is_standalone is False
        assert rec.concurrent_with == "lunch"


# -------------------------------------------------------------------------------------
# loaders
# -------------------------------------------------------------------------------------


class TestLoadAugmentedTasks:
    def test_returns_empty_list_when_file_missing(self, tmp_path: Path):
        assert load_augmented_tasks(tmp_path / "missing.json") == []

    def test_reads_scheduled_array(self, tmp_path: Path):
        payload = {
            "person_id": "p0",
            "scheduled": [
                {
                    "label": "x",
                    "display_name": "X 🍉",
                    "date": "2026-05-04",
                    "start_minutes": 400,
                    "end_minutes": 405,
                    "is_standalone": True,
                    "concurrent_with": None,
                }
            ],
        }
        file = tmp_path / "p0.json"
        file.write_text(json.dumps(payload), encoding="utf-8")
        records = load_augmented_tasks(file)
        assert len(records) == 1
        assert records[0].label == "x"

    def test_returns_empty_list_when_scheduled_missing(self, tmp_path: Path):
        file = tmp_path / "p0.json"
        file.write_text(json.dumps({"person_id": "p0"}), encoding="utf-8")
        assert load_augmented_tasks(file) == []


class TestLoadAugmentedTasksByPerson:
    def test_returns_empty_when_dir_missing(self, tmp_path: Path):
        assert load_augmented_tasks_by_person(tmp_path / "absent") == {}

    def test_returns_empty_when_dir_is_a_file(self, tmp_path: Path):
        f = tmp_path / "file"
        f.write_text("not a dir", encoding="utf-8")
        assert load_augmented_tasks_by_person(f) == {}

    def test_skips_loss_sidecars_and_orders_by_filename(self, tmp_path: Path):
        for name in ("a.json", "b.json"):
            (tmp_path / name).write_text(
                json.dumps(
                    {
                        "person_id": name.split(".")[0],
                        "scheduled": [
                            {
                                "label": "x",
                                "date": "2026-05-04",
                                "start_minutes": 0,
                                "end_minutes": 5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        (tmp_path / "a_loss.json").write_text(
            json.dumps({"person_id": "a", "components": {}}), encoding="utf-8"
        )
        out = load_augmented_tasks_by_person(tmp_path)
        assert list(out.keys()) == ["a", "b"]
        assert all(len(v) == 1 for v in out.values())

    def test_uses_stem_when_payload_has_no_person_id(self, tmp_path: Path):
        (tmp_path / "ghost.json").write_text(
            json.dumps({"scheduled": []}), encoding="utf-8"
        )
        out = load_augmented_tasks_by_person(tmp_path)
        assert "ghost" in out
        assert out["ghost"] == []


# -------------------------------------------------------------------------------------
# iso_week_windows
# -------------------------------------------------------------------------------------


class TestIsoWeekWindows:
    def test_returns_empty_for_schedule_with_no_days(self):
        sched = PersonSchedule(person_id="p", persona_id="a", person_seed=0, days=[])
        assert iso_week_windows(sched) == []

    def test_single_full_week_from_monday(self):
        sched = _schedule("p", [{} for _ in range(7)])
        windows = iso_week_windows(sched)
        assert len(windows) == 1
        assert windows[0].week_index == 1
        assert windows[0].start_date == _MONDAY
        assert windows[0].end_date == _MONDAY + _dt.timedelta(days=6)

    def test_snaps_to_monday_when_horizon_starts_mid_week(self):
        wed = _MONDAY + _dt.timedelta(days=2)
        sched = _schedule("p", [{} for _ in range(5)], start_date=wed)
        windows = iso_week_windows(sched)
        # 5 days from Wed to ends Sun, all in first week.
        assert len(windows) == 1
        assert windows[0].start_date == _MONDAY  # snapped back
        assert windows[0].end_date == _MONDAY + _dt.timedelta(days=6)

    def test_multiple_weeks(self):
        sched = _schedule("p", [{} for _ in range(15)])
        windows = iso_week_windows(sched)
        assert [w.week_index for w in windows] == [1, 2, 3]
        # All windows are 7 days.
        for w in windows:
            assert (w.end_date - w.start_date).days == 6


# -------------------------------------------------------------------------------------
# _minutes_to_hhmm
# -------------------------------------------------------------------------------------


class TestMinutesToHHMM:
    @pytest.mark.parametrize(
        ("minutes", "expected"),
        [
            (0, "00:00"),
            (60, "01:00"),
            (90, "01:30"),
            (1439, "23:59"),
        ],
    )
    def test_in_range_values(self, minutes: int, expected: str):
        assert _minutes_to_hhmm(minutes) == expected

    def test_negative_clamps_to_zero(self):
        assert _minutes_to_hhmm(-10) == "00:00"

    def test_overflow_clamps_to_2359(self):
        assert _minutes_to_hhmm(2000) == "23:59"


# -------------------------------------------------------------------------------------
# event constructors
# -------------------------------------------------------------------------------------


class TestEventConstructors:
    def test_base_event_uses_base_palette(self):
        ev = _base_event("lunch", _MONDAY, 720, 60, min_visible=5)
        assert ev.title == "lunch"
        assert ev.style.event_border == BASE_BORDER
        assert ev.style.event_fill == BASE_FILL

    def test_base_event_clamps_overflowing_duration(self):
        """A 4-hour event starting at 23:00 must not push the end past 23:59."""
        ev = _base_event("late", _MONDAY, 1380, 240, min_visible=5)
        assert ev.end_time.hour == 23
        assert ev.end_time.minute == 59

    def test_base_event_enforces_positive_duration(self):
        """Zero-duration input is bumped to the minimum visible duration
        so calendar-view's `start < end` and rounded-corner height
        invariants are both preserved."""
        ev = _base_event("ping", _MONDAY, 720, 0, min_visible=5)
        # 720 = 12:00; with min_visible=5, the rendered end rounds up to at
        # least 12:05.
        assert ev.end_time.hour == 12
        assert ev.end_time.minute >= 5

    def test_base_event_pads_short_event_to_min_visible(self):
        """A 5-minute base event with `min_visible=45` renders as a
        45-minute block and stamps the real time range in its notes."""
        ev = _base_event("snack", _MONDAY, 720, 5, min_visible=45)
        # Visual end = 12:45.
        assert ev.end_time.hour == 12
        assert ev.end_time.minute == 45
        # Notes call out the real window.
        assert "actual 12:00" in ev.notes
        assert "12:05" in ev.notes

    def test_base_event_long_enough_skips_actual_time_note(self):
        ev = _base_event("dinner", _MONDAY, 720, 60, min_visible=45)
        # 60 ≥ 45 to no padding to no actual-time note.
        assert ev.notes is None

    def test_augmented_event_standalone_uses_standalone_palette(self):
        task = _standalone("walk", _MONDAY, 400, 415)
        ev = _augmented_event(task, min_visible=5)
        assert ev.style.event_border == STANDALONE_BORDER
        assert ev.style.event_fill == STANDALONE_FILL
        # 15 minutes is below the default 45-min floor in production,
        # but with min_visible=5 nothing was padded so no actual-time note.
        assert ev.notes is None
        assert "Walk" in ev.title

    def test_augmented_event_short_standalone_carries_actual_time(self):
        """A 5-min standalone task padded to 45 min keeps the original
        clock window in its notes so the visual padding is honest."""
        task = _standalone("snack", _MONDAY, 400, 405)
        ev = _augmented_event(task, min_visible=45)
        assert ev.notes is not None
        assert "actual 06:40 – 06:45" in ev.notes

    def test_augmented_event_concurrent_uses_concurrent_palette_with_notes(self):
        task = _concurrent("stretch", _MONDAY, 400, 410, with_label="lunch")
        ev = _augmented_event(task, min_visible=5)
        assert ev.style.event_border == CONCURRENT_BORDER
        assert ev.style.event_fill == CONCURRENT_FILL
        assert ev.notes == "with lunch"

    def test_augmented_event_concurrent_padded_combines_with_and_actual(self):
        """A short concurrent task picks up both the `with <event>`
        prefix and the actual-time line, joined by a newline."""
        task = _concurrent("stretch", _MONDAY, 400, 410, with_label="lunch")
        ev = _augmented_event(task, min_visible=45)
        assert ev.notes is not None
        assert "with lunch" in ev.notes
        assert "actual 06:40 – 06:50" in ev.notes

    def test_augmented_event_concurrent_without_with_label_skips_notes(self):
        task = AugmentedTaskRecord(
            label="x",
            display_name="X",
            date=_MONDAY,
            start_minutes=400,
            end_minutes=410,
            is_standalone=False,
            concurrent_with=None,
        )
        ev = _augmented_event(task, min_visible=5)
        assert ev.notes is None


# -------------------------------------------------------------------------------------
# _events_for_window
# -------------------------------------------------------------------------------------


class TestEventsForWindow:
    def test_filters_base_and_augmented_to_the_window(self):
        sched = _schedule(
            "p",
            [
                {"lunch": [(720, 45)]},  # Mon 2026-05-04
                {"lunch": [(720, 45)]},  # Tue
            ],
        )
        # Window covers the whole week of May 4-10.
        window = WeekWindow(
            week_index=1,
            start_date=_MONDAY,
            end_date=_MONDAY + _dt.timedelta(days=6),
        )
        aug_in = _standalone("walk", _MONDAY, 400, 415)
        aug_out = _standalone("walk", _MONDAY + _dt.timedelta(days=14), 400, 415)
        events = _events_for_window(sched, [aug_in, aug_out], window, min_visible=5)
        # 2 base + 1 augmented (out-of-window is dropped)
        assert len(events) == 3
        titles = {ev.title for ev in events}
        assert "lunch" in titles
        assert any("Walk" in t for t in titles)


# -------------------------------------------------------------------------------------
# _save_calendar_with_dpi + plot_week_calendar
# -------------------------------------------------------------------------------------


def _read_dpi(path: Path) -> tuple[float, float]:
    """Return Pillow's stored DPI for `path` as `(x, y)`.

    PIL stores the PNG's `pHYs` chunk as pixels-per-metre and converts
    back to inches with a 2.54 multiplier, which introduces sub-pixel
    rounding (e.g. `300.0` to `299.9994`).  Tests should compare with
    a small tolerance.
    """
    with Image.open(path) as img:
        return tuple(img.info.get("dpi", (0.0, 0.0)))


class TestSaveAndPlot:
    def test_save_with_dpi_writes_a_png_with_expected_metadata(self, tmp_path: Path):
        sched = _schedule("p", [{"lunch": [(720, 45)]}])
        window = iso_week_windows(sched)[0]
        target = tmp_path / "out.png"
        plot_week_calendar("p", window, sched, [], target, dpi=240)
        assert target.exists()
        assert _read_dpi(target) == pytest.approx((240.0, 240.0), abs=0.1)

    def test_default_dpi_is_300(self, tmp_path: Path):
        sched = _schedule("p", [{"lunch": [(720, 45)]}])
        window = iso_week_windows(sched)[0]
        target = tmp_path / "out.png"
        plot_week_calendar("p", window, sched, [], target)
        assert _read_dpi(target) == pytest.approx(
            (float(DEFAULT_DPI), float(DEFAULT_DPI)), abs=0.1
        )

    def test_creates_missing_parent_directory(self, tmp_path: Path):
        sched = _schedule("p", [{"lunch": [(720, 45)]}])
        window = iso_week_windows(sched)[0]
        target = tmp_path / "deep" / "nested" / "out.png"
        plot_week_calendar("p", window, sched, [], target)
        assert target.exists()

    def test_explicit_title_overrides_default(self, tmp_path: Path):
        sched = _schedule("p", [{"lunch": [(720, 45)]}])
        window = iso_week_windows(sched)[0]
        target = tmp_path / "out.png"
        # Use a sentinel string we can detect; the library will paint it
        # on the image but our test just verifies the call accepts it.
        captured: dict = {}
        original_init = __import__(
            "calendar_view.core.config", fromlist=["CalendarConfig"]
        ).CalendarConfig.__init__

        def spy(self, *a, **kw):  # type: ignore[no-untyped-def]
            captured["title"] = kw.get("title")
            return original_init(self, *a, **kw)

        with patch.object(
            __import__(
                "calendar_view.core.config", fromlist=["CalendarConfig"]
            ).CalendarConfig,
            "__init__",
            spy,
        ):
            plot_week_calendar("p", window, sched, [], target, title="custom-title")
        assert captured["title"] == "custom-title"

    def test_save_helper_called_directly(self, tmp_path: Path):
        """Exercise the lower-level helper to ensure DPI metadata is
        written even when callers drive the Calendar object themselves."""
        from calendar_view.calendar import Calendar
        from calendar_view.core.config import CalendarConfig

        cfg = CalendarConfig(
            lang="en",
            title="t",
            dates=f"{_MONDAY.isoformat()} - {(_MONDAY + _dt.timedelta(days=6)).isoformat()}",
        )
        calendar = Calendar.build(cfg)
        target = tmp_path / "raw.png"
        _save_calendar_with_dpi(calendar, target, dpi=150)
        assert target.exists()
        assert _read_dpi(target) == pytest.approx((150.0, 150.0), abs=0.1)


# -------------------------------------------------------------------------------------
# render_weekly_calendars_for_person
# -------------------------------------------------------------------------------------


class TestRenderWeeklyCalendarsForPerson:
    def test_writes_one_png_per_week(self, tmp_path: Path):
        sched = _schedule("p0", [{"lunch": [(720, 45)]} for _ in range(15)])
        paths = render_weekly_calendars_for_person(sched, [], tmp_path)
        # 15 days starting Mon to 3 weeks.
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_uses_title_prefix_when_provided(self, tmp_path: Path):
        sched = _schedule("p0", [{"lunch": [(720, 45)]}])
        captured: list[str] = []
        from calendar_view.core import config as _cv_config

        original_init = _cv_config.CalendarConfig.__init__

        def spy(self, *a, **kw):  # type: ignore[no-untyped-def]
            captured.append(kw.get("title", ""))
            return original_init(self, *a, **kw)

        with patch.object(_cv_config.CalendarConfig, "__init__", spy):
            render_weekly_calendars_for_person(
                sched, [], tmp_path, title_prefix="greedy"
            )
        assert captured
        assert captured[0].startswith("greedy; p0; week 1")

    def test_empty_schedule_returns_no_paths(self, tmp_path: Path):
        empty = PersonSchedule(
            person_id="ghost", persona_id="g", person_seed=0, days=[]
        )
        assert render_weekly_calendars_for_person(empty, [], tmp_path) == []


# -------------------------------------------------------------------------------------
# render_weekly_calendars
# -------------------------------------------------------------------------------------


class TestRenderWeeklyCalendars:
    def test_renders_per_person_subdirs(self, tmp_path: Path):
        scheds = [
            _schedule("p0", [{"lunch": [(720, 45)]}]),
            _schedule("p1", [{"dinner": [(1140, 30)]}]),
        ]
        out = render_weekly_calendars(scheds, None, tmp_path)
        assert set(out.keys()) == {"p0", "p1"}
        for pid, paths in out.items():
            assert all(p.parent.name == pid for p in paths)
            assert all(p.exists() for p in paths)

    def test_overlays_augmented_only_for_matching_person(self, tmp_path: Path):
        scheds = [_schedule("p0", [{"lunch": [(720, 45)]}])]
        augmented = {
            "p0": [_standalone("walk", _MONDAY, 400, 415)],
            "p_other": [_standalone("noop", _MONDAY, 0, 5)],
        }
        out = render_weekly_calendars(scheds, augmented, tmp_path)
        assert "p0" in out
        assert "p_other" not in out  # only persons in `schedules` are rendered

    def test_propagates_title_prefix(self, tmp_path: Path):
        sched = _schedule("p0", [{"lunch": [(720, 45)]}])
        captured: list[str] = []
        from calendar_view.core import config as _cv_config

        original_init = _cv_config.CalendarConfig.__init__

        def spy(self, *a, **kw):  # type: ignore[no-untyped-def]
            captured.append(kw.get("title", ""))
            return original_init(self, *a, **kw)

        with patch.object(_cv_config.CalendarConfig, "__init__", spy):
            render_weekly_calendars([sched], None, tmp_path, title_prefix="prefix")
        assert all(t.startswith("prefix") for t in captured)


# -------------------------------------------------------------------------------------
# Context-box layer (Fix A): contexts render alongside base + augmented events
# -------------------------------------------------------------------------------------


class TestResolveContextFilter:
    def test_none_returns_none(self) -> None:
        assert _resolve_context_filter(None) is None

    def test_empty_list_returns_empty_frozenset(self) -> None:
        out = _resolve_context_filter([])
        assert out == frozenset()
        assert isinstance(out, frozenset)

    def test_iterable_coerced_to_frozenset(self) -> None:
        out = _resolve_context_filter(["mood_emotion", "energy_state"])
        assert out == frozenset({"mood_emotion", "energy_state"})


class TestContextEvent:
    def test_known_category_uses_palette(self) -> None:
        ep = _ctx("happy", "mood_emotion")
        ev = _context_event(ep, min_visible=5)
        expected_fill, expected_border = CATEGORY_PALETTE["mood_emotion"]
        assert ev.style.event_fill == expected_fill
        assert ev.style.event_border == expected_border
        assert ev.title == "happy"
        assert "ctx mood_emotion" in (ev.notes or "")

    def test_unknown_category_falls_back_to_default_palette(self) -> None:
        ep = _ctx("something", "totally_unknown_category")
        ev = _context_event(ep, min_visible=5)
        assert ev.style.event_fill == CONTEXT_DEFAULT_FILL
        assert ev.style.event_border == CONTEXT_DEFAULT_BORDER

    def test_short_episode_padded_to_min_visible(self) -> None:
        # 5-min context episode with min_visible=45 clamps end up
        # and notes carry the original actual annotation.
        ep = _ctx("happy", "mood_emotion", start=480, end=485)
        ev = _context_event(ep, min_visible=45)
        assert ev.start_time.hour == 8
        assert ev.start_time.minute == 0
        # End should land ~08:45 (start + 45-min padding).
        assert ev.end_time.hour == 8
        assert ev.end_time.minute == 45
        assert "actual" in (ev.notes or "")


class TestEventsForWindowWithContexts:
    def _week_window(self) -> WeekWindow:
        return WeekWindow(
            week_index=1,
            start_date=_MONDAY,
            end_date=_MONDAY + _dt.timedelta(days=6),
        )

    def test_default_drops_contexts(self) -> None:
        sched = _schedule(
            "p",
            [{"lunch": [(720, 45)]}],
            contexts=[_ctx("happy", "mood_emotion")],
        )
        events = _events_for_window(sched, [], self._week_window(), min_visible=5)
        # Only the base "lunch" event renders; the context is invisible.
        assert len(events) == 1
        assert events[0].title == "lunch"

    def test_filter_with_named_categories_includes_only_those(self) -> None:
        sched = _schedule(
            "p",
            [{"lunch": [(720, 45)]}],
            contexts=[
                _ctx("happy", "mood_emotion"),
                _ctx("tired", "energy_state", day_offset=1, start=600, end=660),
            ],
        )
        events = _events_for_window(
            sched,
            [],
            self._week_window(),
            min_visible=5,
            context_categories=["mood_emotion"],
        )
        titles = [ev.title for ev in events]
        assert "lunch" in titles
        assert "happy" in titles
        assert "tired" not in titles

    def test_empty_filter_renders_no_contexts(self) -> None:
        """An explicit empty `context_categories` keeps every context off."""
        sched = _schedule(
            "p",
            [{"lunch": [(720, 45)]}],
            contexts=[_ctx("happy", "mood_emotion")],
        )
        events = _events_for_window(
            sched, [], self._week_window(), min_visible=5, context_categories=[]
        )
        assert all(ev.title != "happy" for ev in events)

    def test_context_outside_window_dropped(self) -> None:
        sched = _schedule(
            "p",
            [{"lunch": [(720, 45)]}],
            contexts=[_ctx("happy", "mood_emotion", day_offset=21)],
        )
        events = _events_for_window(
            sched,
            [],
            self._week_window(),
            min_visible=5,
            context_categories=["mood_emotion"],
        )
        assert all(ev.title != "happy" for ev in events)

    def test_base_and_augmented_and_context_all_combine(self) -> None:
        sched = _schedule(
            "p",
            [{"lunch": [(720, 45)]}],
            contexts=[_ctx("happy", "mood_emotion")],
        )
        aug = _standalone("walk", _MONDAY, 400, 415)
        events = _events_for_window(
            sched,
            [aug],
            self._week_window(),
            min_visible=5,
            context_categories=["mood_emotion"],
        )
        titles = {ev.title for ev in events}
        assert "lunch" in titles
        assert any("Walk" in t for t in titles)
        assert "happy" in titles


class TestPlotWeekCalendarWithContexts:
    def test_calendar_renders_with_context_filter(self, tmp_path: Path) -> None:
        sched = _schedule(
            "p",
            [{"lunch": [(720, 45)]}],
            contexts=[_ctx("happy", "mood_emotion")],
        )
        window = iso_week_windows(sched)[0]
        target = tmp_path / "out.png"
        out = plot_week_calendar(
            "p",
            window,
            sched,
            [],
            target,
            context_categories=["mood_emotion"],
        )
        assert out.exists()


class TestRenderWeeklyCalendarsWithContexts:
    def test_threads_context_categories_through(self, tmp_path: Path) -> None:
        sched = _schedule(
            "p0",
            [{"lunch": [(720, 45)]}],
            contexts=[_ctx("happy", "mood_emotion")],
        )
        out = render_weekly_calendars(
            [sched],
            None,
            tmp_path,
            context_categories=["mood_emotion"],
        )
        assert "p0" in out
        assert all(p.exists() for p in out["p0"])

    def test_per_person_helper_accepts_context_categories(self, tmp_path: Path) -> None:
        sched = _schedule(
            "p0",
            [{"lunch": [(720, 45)]}],
            contexts=[_ctx("happy", "mood_emotion")],
        )
        written = render_weekly_calendars_for_person(
            sched, [], tmp_path, context_categories=["mood_emotion"]
        )
        assert written and all(p.exists() for p in written)
