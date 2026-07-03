"""Per-person JSON and ICS writers carry contexts."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from src.scripts.persona.config.schema import JitterConfig
from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.export.ics_writer import schedule_to_ics_bytes
from src.scripts.persona.export.json_writer import (
    _context_to_dict,
    schedule_to_dict,
    write_person_json,
)

DATE = _dt.date(2026, 5, 4)


def _schedule_with_contexts(contexts):
    return PersonSchedule(
        person_id="p_0001",
        persona_id="p",
        person_seed=0,
        days=[
            DaySchedule(
                day_index=0,
                date=DATE,
                weekday="Mon",
                events={"office_work": [EventInstance("office_work", 540, 60)]},
            )
        ],
        contexts=contexts,
    )


def _person() -> Person:
    return Person(
        person_id="p_0001",
        persona_id="p",
        person_seed=0,
        instance_index=0,
        jitter_applied=JitterConfig(),
    )


def _ep(**overrides) -> ContextEpisode:
    payload: dict = {
        "name": "happy",
        "category": "mood_emotion",
        "date": DATE,
        "start_minutes": 720,
        "end_minutes": 780,
        "ontology_uri": "http://purl.obolibrary.org/obo/MFOEM_000042",
        "dimension": None,
        "polarity": None,
        "instrument": None,
        "theory_mappings": None,
    }
    payload.update(overrides)
    return ContextEpisode(**payload)


def test_context_to_dict_round_trip():
    d = _context_to_dict(_ep())
    assert d["name"] == "happy"
    assert d["category"] == "mood_emotion"
    assert d["date"] == DATE.isoformat()
    assert d["start_minutes"] == 720
    assert d["end_minutes"] == 780
    assert d["ontology_uri"] == "http://purl.obolibrary.org/obo/MFOEM_000042"


def test_schedule_to_dict_includes_contexts_block():
    s = _schedule_with_contexts([_ep()])
    out = schedule_to_dict(_person(), s)
    assert "contexts" in out
    assert len(out["contexts"]) == 1
    assert out["contexts"][0]["name"] == "happy"


def test_schedule_to_dict_empty_contexts_returns_empty_list():
    s = _schedule_with_contexts([])
    out = schedule_to_dict(_person(), s)
    assert out["contexts"] == []


def test_write_person_json_round_trips(tmp_path: Path):
    s = _schedule_with_contexts([_ep()])
    target = write_person_json(_person(), s, tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["contexts"][0]["category"] == "mood_emotion"


def test_ics_bytes_carry_context_summary_and_categories():
    s = _schedule_with_contexts([_ep()])
    data = schedule_to_ics_bytes(s).decode("utf-8").replace("\r\n ", "")
    assert "[ctx:mood_emotion] happy" in data
    assert "CATEGORIES:mood_emotion" in data
    assert "iri=http://purl.obolibrary.org/obo/MFOEM_000042" in data


def test_ics_bytes_no_context_section_when_empty():
    s = _schedule_with_contexts([])
    data = schedule_to_ics_bytes(s).decode("utf-8").replace("\r\n ", "")
    assert "[ctx:" not in data


def test_ics_description_includes_dimension_and_polarity():
    s = _schedule_with_contexts(
        [
            _ep(
                name="extra",
                category="trait_state",
                dimension="extraversion",
                polarity="high",
            )
        ]
    )
    data = schedule_to_ics_bytes(s).decode("utf-8").replace("\r\n ", "")
    assert "dimension=extraversion" in data
    assert "polarity=high" in data


def test_ics_description_skips_iri_when_absent():
    s = _schedule_with_contexts([_ep(ontology_uri=None)])
    data = schedule_to_ics_bytes(s).decode("utf-8").replace("\r\n ", "")
    assert "iri=" not in data


def test_ics_sorts_contexts_by_date_then_start():
    other_date = DATE + _dt.timedelta(days=1)
    s = _schedule_with_contexts(
        [
            _ep(start_minutes=500, end_minutes=560, date=other_date),
            _ep(start_minutes=400, end_minutes=460, date=DATE),
        ]
    )
    data = schedule_to_ics_bytes(s).decode("utf-8").replace("\r\n ", "")
    first_idx = data.find("DTSTART:20260504")
    second_idx = data.find("DTSTART:20260505")
    assert 0 <= first_idx < second_idx
