"""Pydantic schemas for the four YAML configuration files."""

from __future__ import annotations

import datetime as _dt
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -------------------------------------------------------------------------------------
# ---------------------------------- shared primitives --------------------------------
# -------------------------------------------------------------------------------------

# Deprecated alias; new YAML should set `characteristics.occupation_status` directly.
# Kept as a free-form `str` so authors are not locked into the legacy three-value set.
OccupationStatus = str
Weekday = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DurationUnit = Literal["minutes", "hours"]
Scale = Literal["day", "week", "month", "season", "weekday"]
PatternMode = Literal["fix", "seasonality", "trend"]
PatternUnit = Literal["count", "percent", "minutes", "hours"]

_HHMM_RE = re.compile(r"^[0-2]\d:[0-5]\d$")
_NAMED_WINDOW_TOKENS = {
    "early_morning",
    "morning",
    "afternoon",
    "evening",
    "night",
}
_TIME_TOKENS = _NAMED_WINDOW_TOKENS


def _validate_time_token(value: str) -> str:
    """Accept either HH:MM or one of the named windows.

    The schema deliberately stays event-name-agnostic: there are no
    "after_lunch" / "after_dinner" relative anchors built into the
    grammar. If a persona wants something to fire after a specific
    event, they declare an explicit HH:MM time on the staging entry,
    or use a window token.
    """
    value = value.strip()
    if _HHMM_RE.match(value):
        hour = int(value[:2])
        if hour > 23:
            raise ValueError(f"hour out of range: {value!r}")
        return value
    if value in _TIME_TOKENS:
        return value
    raise ValueError(
        f"time must be HH:MM or one of {sorted(_TIME_TOKENS)}; got {value!r}"
    )


TimeToken = Annotated[str, Field(description="HH:MM, named window, or relative anchor")]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------------------------------------------------------------------
# ---------------------------------- environment.yaml ---------------------------------
# -------------------------------------------------------------------------------------


class HorizonConfig(_Frozen):
    start_date: _dt.date
    weeks: int = Field(ge=1, le=520)
    enable_yearly_pass: bool = True
    weekly_template: Literal["locked", "per_week"] = "locked"


class OutputConfig(_Frozen):
    dir: str
    per_person_json: bool = True
    per_person_ics: bool = True
    validation_report: bool = True


class SolverConfig(_Frozen):
    step_minutes: int = Field(default=10, ge=1, le=60)
    max_attempts: int = Field(default=25, ge=1)
    optimize_objective: Literal["maximize_sleep", "none"] = "maximize_sleep"


class ParallelismConfig(_Frozen):
    workers: int = Field(default=0, ge=0, description="0 = os.cpu_count()")
    executor: Literal["thread", "process"] = "process"
    chunk_size: int = Field(default=1, ge=1)


class ComputeConfig(_Frozen):
    """Hardware-acceleration settings for the embedding / retrieval stack.

    `device` picks the compute backend used by the local sentence-
    transformers embedder and any other GPU-aware component reached
    from the persona / scenarios pipelines:

    * `cpu`; run everything on CPU (the safe default for laptops and
      CI where no NVIDIA driver is available).
    * `cuda`; pin to the first available GPU.  Errors loudly at
      embedder construction time if `torch` is not built with CUDA or
      no GPU is visible, so a misconfigured run fails fast instead of
      silently falling back to CPU and producing a 100× slowdown.
    * `auto`; probe for CUDA at runtime; use it when available and
      fall back to CPU otherwise.  The default; lets the same YAML run
      on a laptop and on a GPU host without edits.

    `cuda_device_index` selects which GPU when more than one is
    visible (matches `torch.device("cuda:<n>")`).  Ignored when
    `device` resolves to CPU.
    """

    device: Literal["cpu", "cuda", "auto"] = "auto"
    cuda_device_index: int = Field(default=0, ge=0)


class WindowRange(_Frozen):
    """A [start, end) minute range. In YAML this is written as [start, end]."""

    start: int = Field(ge=0, le=1440)
    end: int = Field(ge=0, le=1440)

    @model_validator(mode="after")
    def _check_order(self) -> WindowRange:
        if self.end <= self.start:
            raise ValueError(
                f"window end must exceed start; got [{self.start}, {self.end})"
            )
        return self


class DailyWindow(_Frozen):
    """Hard waking-hours bound applied uniformly to every day in the horizon.

    Every augmenter MUST keep its placements inside `[wake_minutes, sleep_minutes)`.
    Default is 06:00–22:00 (360–1320), a conservative human-friendly range that
    works for most cohorts.  Override per experiment in `environment.yaml`::

        daily_window:
          wake_minutes: 420       # 07:00
          sleep_minutes: 1380     # 23:00

    The value is purely a scheduling constraint; it does not modify the
    underlying calendar events (which may legally appear outside this window
    for sleep, night-shifts, etc.).
    """

    wake_minutes: int = Field(default=360, ge=0, le=1440)
    sleep_minutes: int = Field(default=1320, ge=0, le=1440)

    @model_validator(mode="after")
    def _check_order(self) -> DailyWindow:
        if self.sleep_minutes <= self.wake_minutes:
            raise ValueError(
                "daily_window.sleep_minutes must exceed wake_minutes; "
                f"got [{self.wake_minutes}, {self.sleep_minutes})"
            )
        return self


class EnvironmentConfig(_Frozen):
    seed: int = Field(ge=0, lt=2**64)
    horizon: HorizonConfig
    output: OutputConfig
    solver: SolverConfig = Field(default_factory=SolverConfig)
    time_windows: dict[str, WindowRange]
    daily_window: DailyWindow = Field(default_factory=DailyWindow)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    # Human-readable label for the whole experiment.  Consumed by the
    # multi-scenario benchmark report (`benchmark_report.md`) to title
    # the document.  `experiment_id` in `scenarios.yaml` stays the
    # machine-readable handle (used for output paths, log lines); this
    # field is purely cosmetic and defaults to `None` so the report
    # falls back to the experiment_id when omitted.
    experiment_name: str | None = Field(default=None, min_length=1)

    @field_validator("time_windows", mode="before")
    @classmethod
    def _coerce_window_lists(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        coerced: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                coerced[k] = {"start": v[0], "end": v[1]}
            else:
                coerced[k] = v
        return coerced

    @model_validator(mode="after")
    def _check_named_windows_present(self) -> EnvironmentConfig:
        missing = _NAMED_WINDOW_TOKENS - set(self.time_windows)
        if missing:
            raise ValueError(
                f"time_windows must define every named window; missing: {sorted(missing)}"
            )
        return self


# -------------------------------------------------------------------------------------
# --------------------------------- persona_config.yaml -------------------------------
# -------------------------------------------------------------------------------------


class JitterConfig(_Frozen):
    time_minutes: int = Field(default=15, ge=0, le=720)
    duration_minutes: int = Field(default=10, ge=0, le=720)


class PersonaCommon(_Frozen):
    default_jitter: JitterConfig = Field(default_factory=JitterConfig)


# -------------------------------------------------------------------------------------
# ---------------------------- characteristic distributions ---------------------------
# -------------------------------------------------------------------------------------


class CategoricalDist(_Frozen):
    """Discrete labels with weights; weights sum to 1.0 within 1e-6."""

    type: Literal["categorical"]
    values: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def _normalised(self) -> CategoricalDist:
        for label, w in self.values.items():
            if w < 0:
                raise ValueError(f"negative weight for {label!r}: {w}")
        total = sum(self.values.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"categorical weights must sum to 1.0; got {total:.8f}")
        return self


class BooleanDist(_Frozen):
    """Two-bucket categorical sugar; samples are Python booleans."""

    type: Literal["boolean"]
    p_true: float = Field(ge=0.0, le=1.0)


class ClipRange(_Frozen):
    """Inclusive bounds for rejection resampling of a numeric draw."""

    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> ClipRange:
        if self.max <= self.min:
            raise ValueError(f"clip.max ({self.max}) must exceed clip.min ({self.min})")
        return self


class ScipyDist(_Frozen):
    """Passthrough for any `scipy.stats` distribution by attribute name."""

    type: str = Field(min_length=1)
    params: dict[str, float] = Field(default_factory=dict)
    clip: ClipRange | None = None
    dtype: Literal["int", "float"] | None = None

    @model_validator(mode="after")
    def _resolves_in_scipy(self) -> ScipyDist:
        if self.type in ("categorical", "boolean"):
            raise ValueError(
                f"type {self.type!r} is reserved; use the matching domain class"
            )
        import scipy.stats as _ss

        dist = getattr(_ss, self.type, None)
        if dist is None:
            raise ValueError(
                f"unknown scipy.stats distribution: {self.type!r}. "
                "See https://docs.scipy.org/doc/scipy/reference/stats.html"
            )
        try:
            dist(**self.params)
        except TypeError as exc:
            raise ValueError(
                f"scipy.stats.{self.type}(**{self.params!r}) rejected: {exc}"
            ) from exc
        return self


CharacteristicDist = CategoricalDist | BooleanDist | ScipyDist


def _coerce_characteristic_dist(value: Any) -> Any:
    """Route a raw dict to the matching distribution model by its `type` tag."""
    if not isinstance(value, dict) or "type" not in value:
        return value
    type_tag = value["type"]
    if type_tag == "categorical":
        return CategoricalDist.model_validate(value)
    if type_tag == "boolean":
        return BooleanDist.model_validate(value)
    return ScipyDist.model_validate(value)


def _matches_legacy_occupation(existing: Any, legacy: str) -> bool:
    """Return True when `existing` is a pinned categorical equal to `legacy`."""
    if isinstance(existing, dict):
        if existing.get("type") != "categorical":
            return False
        values = existing.get("values", {})
        return list(values.keys()) == [legacy] and values.get(legacy) == 1.0
    if isinstance(existing, CategoricalDist):
        return (
            list(existing.values.keys()) == [legacy] and existing.values[legacy] == 1.0
        )
    return False


class PersonaEventStage(_Frozen):
    """One persona-stated entry for a catalog event - generic.

    The pipeline does not care whether this stage describes a meal, a
    sport, an appointment, work, sleep, or weather: a stage is just
    `(catalog event name, optional time pin, optional duration pin,
    calendar trigger)`. The calendar trigger is `days` (weekly cadence)
    or `date` (one-off); at least one must be set.

    Fields:
    * `name` - catalog event this stage targets.
    * `time` - HH:MM or a named time-window token (`morning`, `afternoon`,
      `evening`, `night`, `early_morning`). HH:MM pins the start; a
      window token tells the day model "this event must start inside
      that window". Omit for a free-floating start.
    * `duration_minutes` - persona-stated duration. Omit to let the
      catalog/solver pick a duration in the catalog's
      `per_event_duration` range.
    * `days` - weekly cadence (`[Mon, Tue, Wed, Thu, Fri]` for a
      weekday-only event).
    * `date` - one-off date for a single-day event (e.g. an
      appointment).
    """

    name: str = Field(min_length=1)
    time: TimeToken | None = None
    duration_minutes: int | None = Field(default=None, ge=1, le=24 * 60)
    days: list[Weekday] = Field(default_factory=list)
    date: _dt.date | None = None
    # Optional ontology references; pin the stage to its most-specific
    # class in the HumanActivities / HealthTasks ontologies. The
    # `PreferenceMapper` uses them when present; otherwise it falls back
    # to semantic similarity.
    human_activity_iri: str | None = None
    health_task_iri: str | None = None

    @field_validator("time")
    @classmethod
    def _v_time(cls, v: str | None) -> str | None:
        return _validate_time_token(v) if v is not None else None

    @model_validator(mode="after")
    def _has_calendar_trigger(self) -> PersonaEventStage:
        if not self.days and self.date is None:
            raise ValueError(
                f"stage {self.name!r} must declare `days` (weekly cadence) "
                "or `date` (one-off) to fire"
            )
        return self


class ContextMember(_Frozen):
    """One named context inside a category.

    Reuses the event grammar (duration/episodes/patterns/requires) and
    adds optional `dimension` / `polarity` / `instrument` metadata plus
    a pass-through `theory_mappings` dict.
    """

    per_event_duration: "DurationRange"
    total_event_duration: "TotalDuration"
    total_event_episodes: "EpisodeRange"
    temporal_patterns: list["TemporalPattern"] = Field(default_factory=list)
    requires: dict[str, "RolePredicate"] = Field(default_factory=dict)
    ontology_uri: str | None = None
    dimension: str | None = None
    polarity: Literal["high", "low"] | None = None
    instrument: str | None = None
    theory_mappings: dict[str, Any] | None = None


class ContextCategory(_Frozen):
    """A named group of related context members.

    `mutually_exclusive=True` forbids any two episodes drawn from
    different members of this category from overlapping on the same
    day. `None` means "fall back to the per-category default" (see
    `MUTUALLY_EXCLUSIVE_DEFAULTS` in `context/schema.py`).
    """

    mutually_exclusive: bool | None = None
    members: dict[str, ContextMember] = Field(min_length=1)


class Persona(_Frozen):
    """A persona template. `instances` person records are sampled from it."""

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    instances: int = Field(default=1, ge=1)
    occupation_status: OccupationStatus | None = None
    characteristics: dict[str, CharacteristicDist] = Field(default_factory=dict)
    stages: list[PersonaEventStage] = Field(default_factory=list)
    jitter: JitterConfig | None = Field(
        default=None,
        description="Override common.default_jitter for this persona only",
    )
    event_overrides: dict[str, "EventOverride"] = Field(
        default_factory=dict,
        description=(
            "Per-event partial overrides. Each key is a catalog event name; "
            "fields set on the override replace the global definition for "
            "this persona only."
        ),
    )
    contexts: dict[str, ContextCategory] = Field(
        default_factory=dict,
        description="Per-persona momentary contexts keyed by category name.",
    )

    @model_validator(mode="before")
    @classmethod
    def _desugar_and_coerce(cls, data: Any) -> Any:
        """Desugar legacy `occupation_status:` and coerce characteristic dicts."""
        if not isinstance(data, dict):  # pragma: no cover - defensive
            return data
        chars = data.get("characteristics") or {}
        if not isinstance(chars, dict):  # pragma: no cover - defensive
            return data
        legacy = data.get("occupation_status")
        if legacy is not None:
            existing = chars.get("occupation_status")
            if existing is None:
                chars = dict(chars)
                chars["occupation_status"] = {
                    "type": "categorical",
                    "values": {legacy: 1.0},
                }
            else:
                if not _matches_legacy_occupation(existing, legacy):
                    raise ValueError(
                        f"persona {data.get('id')!r}: legacy `occupation_status: "
                        f"{legacy!r}` disagrees with characteristics block"
                    )
        data = dict(data)
        data["characteristics"] = {
            k: _coerce_characteristic_dist(v) for k, v in chars.items()
        }
        return data


class PersonaConfig(_Frozen):
    common: PersonaCommon = Field(default_factory=PersonaCommon)
    personas: list[Persona] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> PersonaConfig:
        seen: set[str] = set()
        for p in self.personas:
            if p.id in seen:
                raise ValueError(f"duplicate persona id: {p.id!r}")
            seen.add(p.id)
        return self


# -------------------------------------------------------------------------------------
# ---------------------------------- event_config.yaml --------------------------------
# -------------------------------------------------------------------------------------


class DurationRange(_Frozen):
    min: int = Field(ge=0)
    max: int = Field(ge=0)
    unit: DurationUnit = "minutes"

    @model_validator(mode="after")
    def _check_order(self) -> DurationRange:
        if self.max < self.min:
            raise ValueError(f"max ({self.max}) < min ({self.min})")
        return self


class TotalDuration(DurationRange):
    scale: Scale = "day"


class EpisodeRange(_Frozen):
    scale: Scale = "day"
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> EpisodeRange:
        if self.max < self.min:
            raise ValueError(f"episode max ({self.max}) < min ({self.min})")
        return self


class TemporalPattern(_Frozen):
    mode: PatternMode
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_amount_unit(self) -> TemporalPattern:
        """Validate the amount `unit` / `target` pair on seasonality and trend."""
        if self.mode == "fix":
            return self
        unit = self.details.get("unit")
        target = self.details.get("target")
        if unit is not None and unit not in ("count", "percent", "minutes", "hours"):
            raise ValueError(
                f"temporal pattern unit must be count|percent|minutes|hours; got {unit!r}"
            )
        if target is not None and target not in ("episodes", "duration"):
            raise ValueError(
                f"temporal pattern target must be episodes|duration; got {target!r}"
            )
        if target == "episodes" and unit is not None and unit not in ("count", "percent"):
            raise ValueError(f"target: episodes requires unit count|percent; got {unit!r}")
        if target == "duration" and unit is not None and unit not in (
            "minutes",
            "hours",
            "percent",
        ):
            raise ValueError(
                f"target: duration requires unit minutes|hours|percent; got {unit!r}"
            )
        if unit == "percent" and target is None:
            raise ValueError("unit: percent requires an explicit target (episodes|duration)")
        return self


class CalendarVariations(_Frozen):
    task_labels: list[str] = Field(default_factory=list)


class RolePredicate(_Frozen):
    """One `requires:` predicate over a person characteristic."""

    in_: list[str | bool | int | float] | None = Field(default=None, alias="in")
    eq: str | bool | int | float | None = None
    ge: int | float | None = None
    le: int | float | None = None
    gt: int | float | None = None
    lt: int | float | None = None

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _at_least_one(self) -> RolePredicate:
        if all(getattr(self, k) is None for k in ("in_", "eq", "ge", "le", "gt", "lt")):
            raise ValueError(
                "requires-predicate must set one of: in, eq, ge, le, gt, lt"
            )
        return self


class EventDefinition(_Frozen):
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    requires: dict[str, RolePredicate] = Field(default_factory=dict)
    weekdays: list[Weekday] | None = None
    intensity: int | None = Field(default=None, ge=1, le=5)
    is_concurrent: bool = False
    is_dividable: bool = False
    concurrent_with: list[str] = Field(default_factory=list)
    ontology_iri: str | None = None
    # Optional ontology references; pin this event to its most-specific
    # class in the HumanActivities / HealthTasks ontologies. The
    # `PreferenceMapper` uses them when present; otherwise it falls back
    # to semantic similarity.
    human_activity_iri: str | None = None
    health_task_iri: str | None = None
    per_event_duration: DurationRange
    total_event_duration: TotalDuration
    total_event_episodes: EpisodeRange
    temporal_patterns: list[TemporalPattern] = Field(default_factory=list)
    calendar_variations: CalendarVariations | None = None


class EventOverride(_Frozen):
    """Per-persona override for an event in the global catalog.

    Every field is optional; whatever is set replaces the matching field on
    the global `EventDefinition` for this persona. The persona schema lives
    next to the routine fields, so a student can ramp `running` up to four
    days a week while a fulltime worker keeps a different cadence on the
    same catalog event name.
    """

    requires: dict[str, RolePredicate] | None = None
    weekdays: list[Weekday] | None = None
    intensity: int | None = Field(default=None, ge=1, le=5)
    is_concurrent: bool | None = None
    is_dividable: bool | None = None
    concurrent_with: list[str] | None = None
    ontology_iri: str | None = None
    # Optional ontology references (override-tier).  `None` means "fall
    # through to the catalog event's value"; an explicit empty string is
    # rejected by the schema since neither field is whitelisted as empty.
    human_activity_iri: str | None = None
    health_task_iri: str | None = None
    per_event_duration: DurationRange | None = None
    total_event_duration: TotalDuration | None = None
    total_event_episodes: EpisodeRange | None = None
    temporal_patterns: list[TemporalPattern] | None = None
    calendar_variations: CalendarVariations | None = None


class Category(_Frozen):
    name: str = Field(min_length=1)
    events: dict[str, EventDefinition] = Field(min_length=1)


class EventConfig(_Frozen):
    """Top-level event catalog."""

    categories: dict[str, Category] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _inject_dict_keys(cls, data: Any) -> Any:
        """Promote category and event dict keys into name/category fields."""
        if not isinstance(data, dict) or "categories" not in data:
            return data
        cats = data["categories"]
        if not isinstance(cats, dict):
            return data
        for cat_name, cat in cats.items():
            if not isinstance(cat, dict):
                continue
            cat.setdefault("name", cat_name)
            events = cat.get("events")
            if isinstance(events, dict):
                for event_name, event in events.items():
                    if isinstance(event, dict):
                        event.setdefault("name", event_name)
                        event.setdefault("category", cat_name)
        return data

    @model_validator(mode="after")
    def _check_event_name_uniqueness(self) -> EventConfig:
        seen: dict[str, str] = {}
        for cat_name, cat in self.categories.items():
            for event_name in cat.events:
                if event_name in seen:
                    raise ValueError(
                        f"event name {event_name!r} appears in both "
                        f"{seen[event_name]!r} and {cat_name!r}"
                    )
                seen[event_name] = cat_name
        return self


# -------------------------------------------------------------------------------------
# ----------------------------- temporal_relation_rules.yaml --------------------------
# -------------------------------------------------------------------------------------

AllenRelationToken = Literal[
    "p",
    "m",
    "o",
    "s",
    "d",
    "f",
    "e",
    "P",
    "M",
    "O",
    "S",
    "D",
    "F",
]


class TemporalRule(_Frozen):
    id: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    min_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    applies_to: dict[str, list[str | int | bool]] = Field(default_factory=dict)


_SELECTOR_KEYS: frozenset[str] = frozenset(
    {"name", "intensity", "domain", "met_min", "met_max"}
)


class SelectorPredicate(_Frozen):
    """Selector grammar for naming activity classes in `AllenPairRule`.

    A selector matches one endpoint of an Allen rule when **all** set
    fields match the candidate activity's resolved attributes:

      * `kind`; `event` (default) or `context`; selects which side of
        the persona pipeline the `name` lives on.
      * `name`; literal label or list of labels (case-sensitive).
      * `intensity`; list of integers in {1, 2, 3, 4}; matches when
        the resolver-derived intensity is in the list.  Multi-valued
        keys are OR'd within a key.  Event-only.
      * `domain`; HealthTasks branch local name
        (`NutritionTask` / `PhysicalActivityTask` /
        `MentalWellbeingTask`).  Event-only.
      * `met_min` / `met_max`; inclusive raw MET range.  Event-only.

    At least one key must be set; an empty selector is rejected at
    validation time.  Multiple selectors firing for one pair to take the
    intersection of their admissible relation sets (most restrictive
    wins).  See `metrics/allen.py::SelectorMatcher`.
    """

    kind: Literal["event", "task", "context"] = "event"
    name: str | list[str] | None = None
    intensity: list[int] | None = None
    domain: str | None = None
    met_min: float | None = Field(default=None, ge=0.0)
    met_max: float | None = Field(default=None, ge=0.0)
    health_task_class: str | list[str] | None = None
    health_task_uri: str | list[str] | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> SelectorPredicate:
        keys = (
            "name",
            "intensity",
            "domain",
            "met_min",
            "met_max",
            "health_task_class",
            "health_task_uri",
        )
        if all(getattr(self, k) is None for k in keys):
            raise ValueError(
                "SelectorPredicate must set at least one of: " + ", ".join(keys)
            )
        if self.intensity is not None:
            for n in self.intensity:
                if n not in (1, 2, 3, 4):
                    raise ValueError(
                        f"intensity values must be in {{1, 2, 3, 4}}; got {n}"
                    )
        if (
            self.met_min is not None
            and self.met_max is not None
            and self.met_max < self.met_min
        ):
            raise ValueError(
                f"met_max ({self.met_max}) must be >= met_min ({self.met_min})"
            )
        if self.kind == "context":
            for forbidden in (
                "intensity",
                "domain",
                "met_min",
                "met_max",
                "health_task_class",
                "health_task_uri",
            ):
                if getattr(self, forbidden) is not None:
                    raise ValueError(
                        f"selector field {forbidden!r} is event/task-only; "
                        f"drop it when kind: context"
                    )
        return self


class AllenPairRule(_Frozen):
    """Explicit admissible Allen relation set for a pair of activity endpoints.

    Each endpoint is either a literal label string (`"lunch"`) **or** a
    selector predicate dict (`{"intensity": [2, 3, 4]}`).  Literal
    strings are equivalent to `{"name": "<label>"}`; the parser
    promotes them automatically.

    Consumed only by the scenarios evaluation layer (metrics/allen.py).
    Never read by the LTL checker (check_ltl.py) or the Z3 solver.
    Lives under the top-level `allen_pair_rules` key in the YAML file,
    separate from the `rules` list used by the LTL checker.

    Optional `buffer:` overrides the global
    `evaluation.buffer_minutes` for this rule's pair only; useful when
    a particular rule needs a tighter / looser separation than the
    scenario-wide default.
    """

    id: str = Field(min_length=1)
    event_a: SelectorPredicate
    event_b: SelectorPredicate
    admissible_relations: list[AllenRelationToken] = Field(min_length=1)
    buffer: int | None = Field(default=None, ge=0)
    applies_to: dict[str, list[str | int | bool]] = Field(default_factory=dict)

    @field_validator("event_a", "event_b", mode="before")
    @classmethod
    def _coerce_string_to_selector(cls, value: Any) -> Any:
        """Promote a bare label string into `{"name": "<label>"}`.

        Keeps the YAML ergonomic; most rules name a single literal, so
        `event_a: lunch` is sugar for `event_a: {name: lunch}`.
        Lists of labels are also accepted (`event_a: [lunch, dinner]`
        promotes to `{"name": [...]}`).
        """
        if isinstance(value, str):
            return {"name": value}
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return {"name": value}
        return value


class TemporalRelationRules(_Frozen):
    rules: list[TemporalRule] = Field(default_factory=list)
    allen_pair_rules: list[AllenPairRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> TemporalRelationRules:
        seen: set[str] = set()
        for r in self.rules:
            if r.id in seen:
                raise ValueError(f"duplicate temporal-rule id: {r.id!r}")
            seen.add(r.id)
        for r in self.allen_pair_rules:
            if r.id in seen:
                raise ValueError(f"duplicate rule id: {r.id!r}")
            seen.add(r.id)
        return self


# Resolve forward references now that every dependent type exists.
ContextMember.model_rebuild()
ContextCategory.model_rebuild()
Persona.model_rebuild()
