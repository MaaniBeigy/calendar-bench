# CONFIG.md

How to configure an experiment in **calendar-bench**.

An experiment is described by five YAML files. Four of them define the synthetic population and its calendars. The fifth describes the task-scheduling scenarios layered on top.

| File                          | Purpose                                                                  |
| ----------------------------- | ------------------------------------------------------------------------ |
| `environment.yaml`            | Global run settings: dates, output dir, solver, time windows, GPU.       |
| `event_config.yaml`           | Catalog of events (sleep, lunch, gym, ...) with population-wide bounds.  |
| `persona_config.yaml`         | The persona templates and how many people of each you want.              |
| `temporal_relation_rules.yaml`| Cross-event rules (e.g. "no sleep during work") in LTL plus Allen pairs. |
| `scenarios.yaml`              | Task-scheduling scenarios to overlay on the generated calendars.         |

All five live next to each other under `src/experiments/persona/<your_experiment>/`. Working examples live in `src/experiments/persona/example_experiment/`.

---

## 1. `environment.yaml`

Top-level run settings. One file per experiment.

```yaml
seed: 20260503              # RNG seed for instance jitter and sampling
experiment_name: "Working-age adults, 4 weeks"   # optional; cosmetic
horizon:
  start_date: 2026-05-04    # first day of the simulation; must be a Monday
  weeks: 4                  # length of the simulation in weeks, per person
  enable_yearly_pass: true  # pre-pin dated persona stages as fixed obstacles
  weekly_template: locked   # locked | per_week
```

- `seed` is consumed alongside `common.default_jitter` in `persona_config.yaml`. Same seed + same persona file = an identical population.
- `experiment_name` is a human-readable label used to title the multi-scenario benchmark report. `experiment_id` (in `scenarios.yaml`) stays the machine-readable handle for output paths. Defaults to `None`, in which case the report falls back to `experiment_id`.
- `start_date` must be a Monday because the week-aligned planner indexes from Mon=0.
- `enable_yearly_pass`: only matters if any persona in `persona_config.yaml` has a stage with a `date:` field (one-off appointments). When `true`, those stages with both `time` and `duration_minutes` set become fixed obstacles before the day solver runs, so the solver cannot place anything else on top of them. When `false`, dated stages still exist but compete for their slot like normal events.
- `weekly_template`: `locked` reuses the same week-1 placements for every week of the horizon (cheap, deterministic). `per_week` re-solves each week independently so weekday seasonality and per-week trend ramps can shift placements over time. Use `per_week` whenever any event declares `mode: trend` or a weekly `mode: seasonality`.

```yaml
output:
  dir: ./output/example_experiment   # root directory for this run's artifacts
  per_person_json: true        # write <person_id>.json per person
  per_person_ics: true         # write <person_id>.ics per person
  validation_report: true      # write the LTL + Allen violation report
```

The `.ics` files import directly into Google Calendar / Outlook / Apple Calendar. The validation report is what `temporal_relation_rules.yaml` produces output for.

```yaml
solver:
  step_minutes: 10             # Z3 placement grid, in minutes
  max_attempts: 25             # solver retries per person before giving up
  optimize_objective: none     # none | maximize_sleep | maximize_free_time
```

`step_minutes: 10` means every event start and end snaps to a 10-minute boundary. Smaller (e.g. 5) gives finer placements but the Z3 solve takes longer.

`optimize_objective` only ever steers **how long `sleep` lasts each day**. With `maximize_sleep`, each day's Z3 problem is built as a `z3.Optimize` and given one soft goal: maximize the sum of sleep-episode durations on that day. Sleep then stretches toward the top of its `per_event_duration` band (up to 9h in the default catalog) when other constraints allow. With `none`, the cheaper `z3.Solver` is used and sleep takes any feasible duration in its band, often near the minimum. The choice is per-day, not cross-week (there is no "sleep less Monday to free time Tuesday" trade-off), and it affects nothing other than sleep duration.

```yaml
time_windows:                  # named [start, end] ranges, in minutes from midnight
  early_morning: [0, 400]      # 00:00 - 06:40
  morning:       [400, 600]    # 06:40 - 10:00
  afternoon:     [600, 960]    # 10:00 - 16:00
  evening:       [960, 1260]   # 16:00 - 21:00
  night:         [1260, 1440]  # 21:00 - 24:00
```

These names are referenced in two other files:
- `persona_config.yaml` stages: `time: "morning"` instead of `time: "07:30"`.
- `event_config.yaml` temporal patterns: `within: [evening]` to pin where an event may land.

Renaming `morning` here breaks every persona and event that referenced it.

```yaml
daily_window:                  # hard waking-hours bracket for scenario-task placement
  wake_minutes: 360            # 06:00; earliest minute a scenario task may start
  sleep_minutes: 1320          # 22:00; latest minute a scenario task may end
```

Only the scenarios augmentation layer reads this (see `scenarios.yaml`). The persona pipeline itself does not, so the original persona events can still cross these times (e.g. `sleep` from 23:00 to 07:00).

```yaml
parallelism:
  workers: 0                   # 0 = use os.cpu_count(); any positive int caps it
  executor: process            # process | thread. process is the safe default.
  chunk_size: 1                # persons per worker batch
```

`executor: thread` is only useful on GIL-free Python builds; on stock CPython use `process`.

```yaml
compute:
  device: cuda                 # cpu | cuda | auto
  cuda_device_index: 0         # which GPU when more than one is installed
```

`cuda` fails fast if no GPU is visible. `auto` probes and silently falls back to CPU. This setting only changes behavior when `EMBEDDING_PROVIDER=local` is set in `.env`. With the default `EMBEDDING_PROVIDER=openai` embeddings are remote and the GPU stays idle. See README §2a for the full GPU build.

---

## 2. `event_config.yaml`

The catalog of every event the simulator knows about. Each persona later picks a subset of these.

Events are grouped by `category` (sleep, eat, work, sports, ...). The category itself has no behavior beyond grouping.

### A minimal event

```yaml
categories:
  sleep:
    events:
      sleep:
        per_event_duration:    { min: 6, max: 9, unit: hours }   # one occurrence is 6-9 hours long
        total_event_duration:  { scale: day, min: 6, max: 9, unit: hours }  # total per day
        total_event_episodes:  { scale: day, min: 1, max: 1 }    # exactly one per day
        intensity: 1                                              # effort 1 (low) to 5 (high)
        is_concurrent: false                                      # cannot overlap with anything
        is_dividable: false                                       # cannot be split into chunks
        temporal_patterns:
          - { mode: fix, details: { within: [night] } }           # must start inside the `night` window
```

### Duration vs episode bounds

- `per_event_duration` is the size of one occurrence (one "episode").
- `total_event_duration` is the sum of all occurrences across a `scale` of `day` or `week`.
- `total_event_episodes` is how many separate occurrences happen at that same scale.

Example: two coffee breaks of 15 minutes each on the same day would need `total_event_episodes: { scale: day, min: 2, max: 2 }` and `per_event_duration: { min: 15, max: 15, unit: minutes }`.

### Concurrency flags (used only by the scenarios evaluation layer)

```yaml
lunch:
  is_concurrent: true              # can overlap with another event
  is_dividable: false              # cannot be split across a day
  concurrent_with: [reading]       # explicit list of events that may overlap with lunch
```

The Z3 solver ignores these. They feed the "merge" component of the scheduling loss in §5.

### Gating events by who can do them

`requires:` evaluates one predicate per axis against the person's resolved characteristics. All predicates must match (logical AND). Supported predicate keys: `in`, `eq`, `ge`, `le`, `gt`, `lt`. The axis name is free-form; the only requirement is that the persona declares it under `characteristics:` (see § 3).

```yaml
office_work:
  requires:
    occupation_status: { in: [fulltime, parttime] }   # categorical membership
  weekdays: [Mon, Tue, Wed, Thu, Fri]
```

```yaml
expensive_gym:
  requires:
    occupation_status: { eq: fulltime }
    socioeconomic_status: { in: [middle, high] }
    age: { ge: 18, le: 65 }
    has_kids: { eq: false }
```

```yaml
food_bank_visit:
  requires:
    socioeconomic_status: { eq: low }
```

Numeric predicates (`ge` / `le` / `gt` / `lt`) only match `int` or `float` characteristics. `eq:` and `in:` accept strings, booleans, ints, or floats; booleans are compared case-insensitively with their string form, so `eq: true` and `eq: "true"` are equivalent. A person missing the named axis fails the predicate.

### Temporal patterns

Three modes are supported: `fix`, `seasonality`, `trend`.

```yaml
temporal_patterns:
  - { mode: fix, details: { within: [night] } }       # always inside one of these windows
```

```yaml
temporal_patterns:
  - mode: seasonality                                  # softer: weighted preference
    details:
      target: episodes        # episodes | duration
      unit: count             # count | percent   (for target: episodes)
      scale: weekday          # day | week | month | season | weekday
      within: [Mon, Tue, Wed, Thu, Fri]
      direction: increasing   # increasing | decreasing
      amount: 1               # +1 episode on the named days (unit: count)
```

```yaml
temporal_patterns:
  - mode: trend                                        # monotonic ramp over the horizon
    details:
      target: duration        # episodes | duration
      unit: minutes           # minutes | hours | percent   (for target: duration)
      scale: week             # day | week (most common: week)
      direction: increasing
      amount: 60              # +60 min ramped across [start, end]
      start: 1                # first 1-indexed unit (week 1)
      end: 13                 # last  1-indexed unit (week 13)
```

`fix` is a hard constraint; `seasonality` nudges placement on the days/weeks it names; `trend` ramps monotonically over the `[start, end]` window. Both `seasonality` and `trend` read `amount` through an explicit `unit`, applied identically by either mode:

| `target`   | `unit`              | `amount` means                                    |
| ---------- | ------------------- | ------------------------------------------------- |
| `episodes` | `count`             | absolute episode delta (`+1` adds one episode)    |
| `episodes` | `percent`           | percent of the base count (`30` is `x 1.30`)      |
| `duration` | `minutes` / `hours` | absolute duration shift (hours convert to minutes)|
| `duration` | `percent`           | percent of the base duration (`30` is `x 1.30`)   |

The `unit` / `target` pair is validated at load: `target: episodes` takes `count` or `percent`; `target: duration` takes `minutes`, `hours`, or `percent`. Declare `unit` on every seasonality / trend pattern, the same way `per_event_duration` and `total_event_episodes` carry their own units (a pattern that omits it falls back to the legacy reading: duration to minutes, episode seasonality to percent, episode trend to count). A `target: duration` pattern shifts both the per-occurrence and per-day duration bounds; widen the matching `event_overrides` caps to absorb the ramp, otherwise the validator flags stretched placements as out-of-range. `trend` requires `horizon.weekly_template: per_week` to take effect across weeks. The same grammar applies wherever `temporal_patterns` appear: catalog events, persona `event_overrides`, and persona context members.

### Ontology grounding (optional)

```yaml
walking:
  human_activity_iri: https://w3id.org/calendar-bench/human-activities/activity/walking-3-5-mph-mod-pace-firm-surface-walking-for-exercise
  health_task_iri:    https://w3id.org/calendar-bench/health/PhysicalActivityDailyStepsAndWalkingTask
  ontology_iri:       https://w3id.org/calendar-bench/human-activities/activity/walking-3-5-mph-mod-pace-firm-surface-walking-for-exercise   # legacy alias
```

These IRIs pin an event to its most specific class in the HumanActivities / HealthTasks ontologies. The `PreferenceMapper` (used by `L_pref`) consults them across four tiers, in cost order:

1. **Literal IRI** match against the task's `ontology_uri` (weight 1.0).
2. **Cross-ontology bridge** via `hb:matchedActivity` (curator-authored link from each HealthTask to its most representative HumanActivities instance). For events that only carry `human_activity_iri`, the matcher fetches the task's matched activity and either matches the event's IRI directly (weight 1.0) or walks SUBCLASS-OF in HumanActivities space (weight `cross_ancestor_discount × cross_ancestor_decay**hops`).
3. **Ontology ancestor walk** via SUBCLASS-OF inside either ontology (weight `ancestor_discount × ancestor_decay**hops`).
4. **Semantic σ** against the LLM-judge oracle (weight = σ when above threshold).

Pin granularity guidance:

- `health_task_iri` → the most-specific **sub-task class** that captures the event's intent (e.g. `PhysicalActivityCardioAndStaminaTask`, not the broad `PhysicalActivityTask` branch and not a Level class). With the default `ancestor_max_hops=2` every instance under that branch lands at tier 2 in 1 hop (weight 0.42).
- `human_activity_iri` → a **concrete instance** under the `.../activity/` namespace (e.g. `jogging-general-self-selected-pace`), not a class. Instance-level so MET-quartile inheritance and the cross-ontology bridge resolve at tier 1 for ontology-grounded tasks.
- Omit either side when the event has no clean analog in that ontology (e.g. `office_work` has no HealthTask). Forcing a mismatched pin makes the L_pref legs apply unrelated duration / episode bounds to those tasks and reports false violations.

The same two fields apply on every persona `stage` (see § 3): they refine the catalog pin for a single persona when its interpretation of an event is more specific than the catalog default.

The `hb:matchedActivity` link is baked into `HealthTasks_<version>.{ttl,json}` by `precompute_matched_activity.py` + `enrich_healthtasks_with_matched_activity.py`; drift is reported per experiment by `validate_matched_activity_links.py` in the `matched_activity_violations.{txt,json}` sidecar.

### Calendar variations

```yaml
office_work:
  calendar_variations:
    task_labels: [standup, meeting, doing research, external meeting]
```

Each occurrence of `office_work` is given one of these labels, dealt from a per-person shuffled deck so one person's blocks mix all the labels evenly (no one is stuck on a single label) and different people get different sequences. The drawn label is persisted to the per-person JSON and `.ics`, shown to the augmenter as the event title, and scored by the evaluation judge as `"<label> (office_work)"`. The variation is not merely cosmetic: the LLM augmenter sees the label and can fold it into its placement logic, and the judge scores it, so different labels can shift both where tasks land and the merge gain. What stays invariant is the scoring identity: the catalog name (`office_work`) is what rule matching, intensity, and the preference / semantic caches always key on, so the parent type, not the variation, anchors the metrics.

---

## 3. `persona_config.yaml`

Persona templates and the population breakdown.

### Global defaults

```yaml
common:
  default_jitter:
    time_minutes: 15        # +/- 15 min wiggle on start times across instances
    duration_minutes: 10    # +/- 10 min wiggle on durations
```

Jitter is what makes 10 sampled "students" look like 10 different people instead of 10 clones. A single persona can override the default with its own `jitter:` block (same two fields); without it, the persona inherits `common.default_jitter`.

### One persona

```yaml
personas:
  - id: a_student          # any string; appears in output filenames
    instances: 10          # generate 10 persons from this template
    characteristics:       # one entry per axis declared by this persona
      occupation_status: { type: categorical, values: { student: 1.0 } }
    jitter:                # optional per-persona override of common.default_jitter
      time_minutes: 30
      duration_minutes: 15
    stages:                # the flat list of events this persona engages with
      - { name: sleep,    time: "23:00", days: [Mon, Tue, Wed, Thu, Fri, Sat, Sun] }
      - { name: study,    time: "09:00", days: [Mon, Tue, Wed, Thu, Fri] }
      - { name: running,  time: "morning", duration_minutes: 60, days: [Tue, Thu] }
      - { name: dentist,  time: "14:00",   duration_minutes: 30, date: 2026-05-12 }
```

A legacy top-level `occupation_status: <value>` field is still accepted as a one-line shorthand; the loader rewrites it into `characteristics.occupation_status` with a single bucket pinned at weight 1.0. New experiments should declare characteristics directly.

### Stage fields

- `name`: event name from `event_config.yaml`. Required.
- `time`: either an `HH:MM` string or a window name (`morning`, `afternoon`, ...). Optional - if omitted the solver picks any feasible time.
- `duration_minutes`: persona-stated duration. Optional - if omitted, the catalog's `per_event_duration` bound is used.
- `days`: list of weekdays this stage repeats on. **Required** if `date` is not set.
- `date`: one-off ISO date (for an appointment). **Required** if `days` is not set.
- `human_activity_iri` / `health_task_iri`: optional ontology pins. Same semantics as on the event catalog entry; useful when a persona refines a generic catalog event into a more specific class (e.g. tagging `running` as `https://w3id.org/humanactivities#TrailRunning` for a single persona).

There are no hardcoded slots (no `workdays_routine`, no `hobbies`). Every event the persona does is a stage entry.

### Characteristics: per-persona feature distributions

Each persona declares its own characteristic vocabulary under `characteristics:`. Axes are entirely user-defined: pick any name you like, declare a `type`, and provide the matching distribution body. Two personas can declare disjoint axes or the same axis with different distributions; the loader does not enforce a shared vocabulary.

Five distribution shapes ship today:

| `type`            | YAML body                                                                       | Notes                                                                                      |
| ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `categorical`     | `values: { label_a: w_a, label_b: w_b, ... }` (weights sum to 1.0 within 1e-6)  | Labels with weights; sampler shuffles a length-`instances` list with the rounding fix.     |
| `boolean`         | `p_true: <float in [0, 1]>`                                                     | Sugar for a two-bucket categorical with Python `True` / `False` values.                    |
| anything from `scipy.stats` | `params: { ... }` (kwargs forwarded to `scipy.stats.<type>(**params)`) | Add `clip: { min, max }` for rejection resample, `dtype: int|float` to coerce per sample. |

The `type` for the scipy passthrough is any attribute on `scipy.stats` (`norm`, `uniform`, `expon`, `gamma`, `t`, `poisson`, `lognorm`, `beta`, `weibull_min`, `randint`, etc.). The schema rejects unknown names at load time.

```yaml
personas:
  - id: gym_rat
    instances: 10
    characteristics:
      occupation_status:
        type: categorical
        values: { fulltime: 0.80, parttime: 0.20 }
      socioeconomic_status:
        type: categorical
        values: { low: 0.20, middle: 0.50, high: 0.30 }
      age:
        type: norm
        params: { loc: 40, scale: 8 }
        clip: { min: 25, max: 65 }
        dtype: int
      weekly_workouts:
        type: poisson
        params: { mu: 4 }
      has_kids:
        type: boolean
        p_true: 0.30
```

Pinning a single value is a one-bucket distribution: `values: { fulltime: 1.0 }` for a categorical, `p_true: 1.0` for a boolean, `min == max` for a uniform.

### Per-persona overrides

`event_config.yaml` holds population-wide bounds. To tighten them for one persona, use `event_overrides`:

```yaml
event_overrides:
  running:
    intensity: 4
    per_event_duration:   { min: 30, max: 60, unit: minutes }     # students run 30-60 min
    total_event_duration: { scale: day, min: 30, max: 60, unit: minutes }
  visit_family:
    total_event_episodes: { scale: week, min: 0, max: 1 }         # cap to at most one weekly visit
```

This lets two different personas (e.g. `a_student` and `a_fulltime`) see different bounds on the same event name without forking the catalog.

---

### Contexts: per-persona momentary states

Persona authors may declare a `contexts:` block per persona alongside
`stages:` and `event_overrides:`. Each entry is one of the 15 categories
served by the Context ontology imported into Neo4j from
`Context_<version>.ttl` (built from
`src/assets/ontologies/context_iris.json`). Categories contain
named `members` that reuse the event grammar (duration / episodes /
patterns / requires) and add optional `dimension`, `polarity`,
`instrument`, and `theory_mappings` fields:

```yaml
personas:
  - id: f_fulltime
    instances: 6
    characteristics:
      neuroticism: { type: norm, params: { loc: 0.5, scale: 0.18 }, clip: { min: 0, max: 1 } }
    contexts:
      mood_emotion:
        mutually_exclusive: true
        members:
          happy:
            per_event_duration:   { min: 15, max: 90, unit: minutes }
            total_event_duration: { scale: day, min: 30, max: 180, unit: minutes }
            total_event_episodes: { scale: day, min: 0, max: 3 }
            ontology_uri: http://purl.obolibrary.org/obo/MFOEM_000042
          anxious:
            per_event_duration:   { min: 10, max: 60, unit: minutes }
            total_event_duration: { scale: day, min: 0, max: 90, unit: minutes }
            total_event_episodes: { scale: day, min: 0, max: 2 }
            ontology_uri: http://purl.obolibrary.org/obo/MFOEM_000028
            requires:
              neuroticism: { ge: 0.6 }
      trait_state:
        mutually_exclusive: true
        members:
          extraverted_state:
            dimension: extraversion
            polarity: high
            ontology_uri: http://www.ebi.ac.uk/efo/EFO_0004317
            per_event_duration:   { min: 30, max: 240, unit: minutes }
            total_event_duration: { scale: day, min: 0, max: 6, unit: hours }
            total_event_episodes: { scale: day, min: 0, max: 3 }
```

Notes:

- `mutually_exclusive: true` forbids overlap inside the category on the
  same day. When unset, defaults are: `mood_emotion`, `energy_state`,
  `location`, `social_context`, `weather_environment`, `trait_state` to
  `true`; all others to `false`.
- `dimension` groups trait-polarity pairs (e.g. `extraverted_state` and
  `introverted_state` share `extraversion`); same-dimension members are
  forbidden from overlap when the category flag is `true`.
- `ontology_uri` must be present in the Neo4j Context dictionary; the
  loader queries Neo4j at config-load time and rejects unknown IRIs.
  Neo4j must be reachable whenever a persona declares contexts (the
  loader opens an ephemeral driver from `Neo4jSettings.from_env()` when
  the caller does not supply one).
- The placer runs as a second Z3-free greedy pass after the event Z3
  solver. Pass `--no-contexts` to `persona.cli generate` to skip it.

`temporal_relation_rules.yaml` accepts contexts in both LTL atoms and
Allen pair selectors:

```yaml
rules:
  - id: no_sleep_when_anxious
    formula: "G ¬(sleep ∧ context:anxious)"
  - id: tired_after_long_office
    formula: "G (office_work → F context:tired)"

allen_pair_rules:
  - id: anxious_far_from_sleep
    event_a: { kind: context, name: anxious }
    event_b: { name: sleep }
    admissible_relations: [p, P]
    buffer: 60
```

## 4. `temporal_relation_rules.yaml`

Cross-event constraints. Two separate sections live in the same file.

### LTL rules (consumed by the Z3 solver and the LTL checker)

```yaml
rules:
  - id: no_overlap_sleep_work          # any id you like; appears in the violation report
    formula: "G ¬(sleep ∧ office_work)"   # "always not (sleep and office_work)"
```

Supported operators (Unicode or ASCII):

| Symbol | ASCII | Meaning                                  |
| ------ | ----- | ---------------------------------------- |
| `G`    | `G`   | always (globally)                        |
| `F`    | `F`   | eventually (finally)                     |
| `¬`    | `!`   | not                                      |
| `∧`    | `&`   | and                                      |
| `∨`    | `\|`  | or                                       |
| `→`    | `=>`  | implies                                  |
| `≥`    | `>=`  | used inside `weekly_count(...)`          |

```yaml
- id: reading_after_dinner
  formula: "G (dinner → F (dinner ∧ reading))"   # every dinner must eventually overlap with reading
  min_fraction: 1.0                              # must hold for 100% of the time horizon
```

### Counting rules

`weekly_count(event, op, n)` is a built-in. It is not pure LTL.

```yaml
- id: weekly_running_target
  formula: "weekly_count(running, ≥, 1)"     # run at least once per week
  applies_to:
    occupation_status: [student]             # only check this rule against students
```

`applies_to` is optional. If absent, the rule is checked against every persona.

### Rule scoping (`applies_to`)

`applies_to` accepts any axis name a persona declared under `characteristics:`. The reserved key `personas:` filters by `persona_id` directly. All keys must match (logical AND). Values can be strings, ints, or booleans.

```yaml
rules:
  # Restrict by persona id (closed list; typos fail at load time).
  - id: no_overlap_sleep_study
    formula: "G ¬(sleep ∧ study)"
    applies_to:
      personas: [gym_rat, couch_potato]

  # Compose persona id + characteristic axis.
  - id: low_income_food_bank_weekly
    formula: "weekly_count(food_bank_visit, ≥, 1)"
    applies_to:
      personas: [couch_potato]
      socioeconomic_status: [low]

  # Numeric characteristic enumerated as a list of admissible values.
  - id: senior_walk_target
    formula: "weekly_count(walking, ≥, 3)"
    applies_to:
      age: [60, 61, 62, 63, 64, 65]
```

The same `applies_to:` grammar also works on `allen_pair_rules` entries, so selector-driven Allen rules can be scoped the same way.

### Allen-pair rules (consumed only by the scenarios evaluation layer)

These compare ordered pairs of events and accept a set of Allen interval relations.

```yaml
allen_pair_rules:
  - id: sleep_separates_from_running
    event_a: sleep                       # bare string sugar for {name: sleep}
    event_b: running
    admissible_relations: [p, m, M, P]   # precedes, meets, met-by, preceded-by - i.e. no overlap
```

The 13 single-letter codes are the standard Allen relations: `p` (precedes), `m` (meets), `o` (overlaps), `s` (starts), `d` (during), `f` (finishes), `e` (equals), and their inverses `P, M, O, S, D, F`. The Z3 solver and the LTL checker ignore this section.

#### Selector grammar

Each of `event_a` / `event_b` is either a literal label string (auto-promoted to `{name: <label>}`) or a selector predicate. A selector matches a candidate activity when **every** set field matches:

| Field                | Type                | Matches when                                                                                                            |
| -------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `kind`               | `event` / `task` / `context` | restricts the selector to that candidate type; default is lenient `event` matching                                |
| `name`               | str or list[str]    | literal label (or one of a list of labels)                                                                              |
| `intensity`          | list[int] in {1..4} | resolved intensity bucket is in the list (event/task only)                                                              |
| `domain`             | str                 | HealthTasks branch local name (`NutritionTask`, `PhysicalActivityTask`, `MentalWellbeingTask`) (event/task only)        |
| `met_min`            | float               | raw MET ≥ value (event/task only)                                                                                       |
| `met_max`            | float               | raw MET ≤ value (event/task only)                                                                                       |
| `health_task_class`  | str or list[str]    | the candidate's HealthTask SUBCLASSOF closure contains every named class (branch / sub-branch / leaf slug; task only)   |
| `health_task_uri`    | str or list[str]    | exact HealthTask leaf URI match for leaf-pinning ablations (task only)                                                  |

At least one field must be set. When more than one rule matches a pair, their admissible-relation sets are **intersected** (most-restrictive wins). `kind: context` is required to match against context episodes loaded from `persons/<pid>.json`; `health_task_class` walks the HealthTasks SUBCLASSOF chain so a rule named on a branch class matches every leaf under it.

```yaml
# No two high-intensity activities adjacent; insist on a 60-min recovery gap.
- id: intensive_far_from_intensive
  event_a: { intensity: [3, 4] }
  event_b: { intensity: [3, 4] }
  admissible_relations: [p, P]
  buffer: 60                            # per-rule override of evaluation.buffer_minutes
```

```yaml
# Lunch followed only by something at intensity 1 (no walks right after eating).
- id: lunch_far_from_intensive
  event_a: { name: lunch }
  event_b: { intensity: [2, 3, 4] }
  admissible_relations: [p, P]
  buffer: 60
```

`buffer:` is optional; when set it overrides the global `evaluation.buffer_minutes` (see [scenarios.yaml § evaluation](#evaluation-block)) for this rule only. Useful when one pair needs a tighter or looser separation than the scenario-wide default.

---

## 5. `scenarios.yaml`

A scenarios file groups one or more task-scheduling scenarios that run on top of the same persona population. The tasks themselves come from whichever ontology you point retrieval at (HealthTasks, office / productivity, training, etc.); the pipeline is ontology-agnostic. Each scenario can run multiple augmentation methods (greedy, llm_agent, rl) for direct comparison.

### File-level header

```yaml
experiment_id: example_experiment
run_dir: ./output/example_experiment      # where the persona-pipeline output lives
output_base: ./output               # parent dir for augmentation output
scenarios:
  - id: nutrition_l1
    ...
```

Output ends up at `<output_base>/<experiment_id>/<scenario_id>/<method>/`.

### A scenario has three blocks

```yaml
- id: nutrition_l1
  description: "Light nutrition + activity tasks, working-age persona."

  task_generation: { ... }     # how to invent the candidate tasks
  augmentation:    [ ... ]     # one entry per method to compare
```

### `timeframe` (optional)

Pin a scenario to a sub-range of the persona horizon. Omit to run the scenario over the full `environment.horizon.weeks`.

Two forms, picked by `scale`:

```yaml
- id: scenario_first_two_weeks
  timeframe:
    scale: week                # 1-based week index, inclusive
    start: 1
    end:   2
```

```yaml
- id: scenario_last_two_weeks
  timeframe:
    scale: dates               # ISO dates, inclusive
    start: 2026-06-15
    end:   2026-06-28
```

Rules:

- `end` is the last week/day **included** in the window (not "the day after").
- `scale: dates` requires the window length `(end - start + 1)` to be a multiple of 7. Oddly-shaped windows are rejected at schema load.
- `scale: dates` also requires `start` to land on the same weekday as `environment.horizon.start_date` (so the window aligns to whole weeks of the horizon).
- Both forms must fit inside the horizon. Out-of-range bounds raise at runtime with the offending scenario id.
- `num_tasks` keeps its **per-week** meaning. A 2-week scenario with `num_tasks: 20` and `repeat_per_week: true` yields 40 task placements per person.
- When a scenario declares a `timeframe`, the augment step writes the resolved window to `<aug_dir>/timeframe.json` and the evaluate step's `Person Instance Scheduling Gain Report` / `Total Scheduling Gain` reports gain a `Window: <start> .. <end> (<weeks> weeks)` header line + a `window` key in the matching `.json` payload.
- Two scenarios in the same file may declare overlapping timeframes — they remain independent runs, no overlap check is enforced.

### `task_generation`

```yaml
task_generation:
  method: graphrag_grounded     # graphrag_grounded | graphrag | manual | template
  seed: 20260503
  num_tasks: 5                  # tasks PER WEEK. With horizon.weeks=4 you get 20 per person.
  ontologies:
    - HealthTasks               # which ontologies retrieval is restricted to
    - HumanActivities
  prompt_template: health_improvement
  filters:
    domains:                    # ontology classes a candidate task must belong to
      - PhysicalActivityTask
      - NutritionTask
    difficulty:                 # ontology difficulty levels (Level1 = effortless, Level3 = harder)
      - Level1
  profile_characteristics:      # optional; characteristic axes shown to the prompt
    - occupation_status         # omit the key entirely to expose every axis the
    - age                       # person carries (the default); the listed order is
    - weekly_workouts           # the order rendered into the prompt profile block
```

The prompt templates render the person profile from the sampled
`characteristics:` block of `persona_config.yaml` (occupation, age,
personality traits, or any other axis the persona declares). By default
every axis the person carries is exposed; `profile_characteristics`
narrows the profile to the named axes, e.g. for ablating how much
persona detail the task recommender sees. Axes a person does not carry
are skipped silently, so a cross-persona scenario file can name the
union of all axes.

```yaml
  # only read when method == graphrag_grounded:
  max_fetch_retries: 5                       # LLM calls allowed per persona during retrieval
  paraphrase: true                            # run the personalisation + quality-gate stages
  paraphrase_similarity_threshold: 0.80       # min cosine vs canonical description (gate C)
  paraphrase_max_length_delta_pct: 0.10       # paraphrase length within +/- 10% of canonical (gate A)
  fetch_top_k: 20                             # GraphRAG retriever top_k for Stage-1 fetch
```

`fetch_top_k` widens the LLM's candidate pool during retrieval. The neo4j-graphrag default of 5 is too narrow for large sweeps (30 personas once converged on the same 4 URIs); 20 is a good baseline, 40 for larger sweeps.

```yaml
  # only read when method == manual:
  manual_tasks:                               # explicit per-person task list
    - label: "10-minute walk"
      ontology_uri: https://w3id.org/healthtasks#WalkingTask
      difficulty: Level1
  task_overrides:                             # free-form per-task field overrides
    walking: { difficulty: Level2 }
```

Keep `num_tasks` identical across scenarios in the same file. Otherwise the per-scenario loss numbers cannot be compared on equal footing.

### Per-week task generation (weekly ramp)

By default `task_generation` is a single block applied to every week. To ramp the task count or difficulty across the horizon, supply a **list** of blocks, each tied to a `week:`:

```yaml
task_generation:
  - week: 1                       # 1-based horizon week this block covers
    method: graphrag_grounded
    num_tasks: 10
    filters:
      - { domains: [NutritionTask, PhysicalActivityTask], difficulty: [Level1, Level2] }
  - week: 2
    method: graphrag_grounded
    num_tasks: 14
    cross_week_distinct: true     # default; week 2's tasks stay disjoint from week 1's
    filters:
      - { domains: [NutritionTask, PhysicalActivityTask, MentalWellbeingTask], difficulty: [Level1, Level2] }
  - week: 3
    method: graphrag_grounded
    num_tasks: 18
    cross_week_distinct: false    # week 3 may reuse tasks generated for earlier weeks
    filters:
      - { domains: [NutritionTask, MentalWellbeingTask], difficulty: [Level2, Level3] }
      - { domains: [PhysicalActivityTask],               difficulty: [Level1, Level2] }
```

Each block carries the same fields as the single-block form, so `num_tasks`, `filters`, `seed`, and the grounded knobs may all differ per week. A YAML anchor on the first block (`&tg_base` plus `<<: *tg_base` on the rest) keeps the repeated fields in one place. Scenarios that will be compared should share the same ramp (e.g. by pointing at one anchored list) so their per-week counts match.

Block-list rules, enforced at scenario load:

- `week` is 1-based. At most one block may omit `week:`; that block is the default for any horizon week no explicit block claims. A single (non-list) block must NOT set `week:`.
- No two blocks may name the same `week:`.
- Every augmentation method in the scenario must keep `repeat_per_week: true` (the default). A per-week list with `repeat_per_week: false` is rejected, since one-shot carry-forward has no per-week horizon to ramp over.

`cross_week_distinct` (default `true`): when `true`, a week's generated tasks stay disjoint from every task generated for earlier weeks (the running set of prior URIs seeds a do-not-repeat blacklist). Set it `false` on a week to let it reuse earlier tasks. Distinctness is bounded by how many distinct instances the ontology holds and how many the retriever surfaces per persona, so late high-count weeks may need `false` to fill. See PROCEDURES §18 for the worked `healthy_lifestyle_promotion` ramp.

### Grouped filters

`filters` accepts either the single `{domains, difficulty}` block shown earlier or a **list of groups**. A task is kept when it matches some group's domains and that same group's difficulty (groups combine by union). This pairs different difficulty bands with different domains in one week, e.g. Nutrition and MentalWellbeing at Level2-3 alongside PhysicalActivity at Level1-2:

```yaml
filters:
  - { domains: [NutritionTask, MentalWellbeingTask], difficulty: [Level2, Level3] }
  - { domains: [PhysicalActivityTask],               difficulty: [Level1, Level2] }
```

Grouped filters work in the single-block form too; they are independent of the per-week list.

Each domain named across a week's filter groups is guaranteed at least 25% of that week's `num_tasks` (rounded up), so no single domain floods the plan. When the retriever cannot surface enough tasks for a branch to reach its floor, the week ships short and logs the shortfall rather than over-weighting one domain.

A per-week run writes one task list per week, wrapped as `{"weeks": [[...week 1...], [...week 2...], ...]}` under `task_generation/<scenario_id>/tasks/<person_id>_tasks.json`. A bare array (the legacy single-week layout) still loads as a one-week wrapper, so flat scenarios are unchanged.

### Per-stage LLM model overrides (optional)

A scenario can pin the task-generation LLM and the LLM-judge evaluator independently. Either block, when omitted, falls back to whatever the augmenter block uses (or to `.env` defaults).

```yaml
task_generator_model:
  provider: openai          # openai | anthropic | openrouter
  model: gpt-4o-mini
  max_retries: 1

evaluator_model:
  provider: openai
  model: gpt-4.1-mini
  max_retries: 1
```

Useful for axis A/B sweeps like "compare gpt-4o-mini vs gpt-4.1-mini for LLM-judge scoring on identical inputs" without touching `.env`.

### `augmentation` (list, one entry per method)

Each entry is one method (`greedy`, `llm_agent`, `ptime`, or `rl`) with its own settings, loss weights, and optional output. Multiple methods inside the same scenario share the same task list, so their scores are directly comparable.

#### `observation` (optional, per augmentation entry)

Declarative info-leakage gate applied at augment time. Defaults reproduce today's behaviour byte-for-byte; trim or extend a list to control what the augmenter sees:

```yaml
augmentation:
  - method: llm_agent
    observation:
      contexts:        [mood_emotion, energy_state, location]
      context_detail:  summary           # summary | full
      host_flags:      [is_concurrent]   # adds extra keys to busy intervals
      task_flags:      [duration_min, duration_max, intensity]
    llm_agent: { ... }
    loss:      { ... }
```

* `contexts` (list of category slugs from the Neo4j Context ontology, e.g. `mood_emotion`, `energy_state`, ...): when non-empty, the LLM prompt gains a per-day context block per category. Empty list (the default) means the augmenter never sees context episodes. The schema validates names against the live catalog, so an unknown slug fails at scenario load time.
* `context_detail`: `summary` shows `(name, start, end)` per episode; `full` adds `uri`, `dimension`, `polarity` for each entry.
* `host_flags`: extra fields appended to each `busy` interval in the calendar summary. Valid: `is_concurrent`, `is_dividable`, `concurrent_with`, `intensity`. Useful as a "cheat oracle" baseline.
* `task_flags`: which recommended-task fields render in the tasks list. Valid: `concurrent_ok`, `dividable_ok`, `duration_min`, `duration_max`, `intensity`. Trim to stress-test which signals the LLM actually uses.

Greedy parses the block for cross-method consistency but ignores every field (it emits one INFO log per run when the block is non-default).

The evaluator reads the FULL ground-truth context trace when checking which episodes overlap a placement (the numerator), but it scopes the *scored* recommendation set (the denominator) to `recommended ∩ generated ∩ observation.contexts` for methods that declare `contexts` (blind methods pass none and score over every generated recommended category). So `L_context_fit` does depend on `observation.contexts`. A category counts as realized only when an overlapping episode's ontology IRI is one the task links (member-level numerator), and the denominator counts only categories the persona actually generates: a recommended category the person never enters is dropped, not penalized.

```yaml
augmentation:
  - method: greedy
    allow_merge: true
    merge_threshold: 0.65       # min semantic similarity to merge a task into an existing event
    repeat_per_week: true       # apply the same task plan every week of the horizon
    greedy:
      strategy: preference_first   # preference_first | earliest_fit | latest_fit
      retry_on_miss: true
      max_backtrack: 3
    loss: { ... }
```

```yaml
  - method: llm_agent
    allow_merge: true
    merge_threshold: 0.70
    repeat_per_week: true
    llm_agent:
      provider: anthropic              # anthropic | openai | openrouter
      model: claude-sonnet-4-6
      max_retries: 1                   # retries on malformed JSON from the LLM
      prompt_template: augment_oneshot # template under augmentation/prompts/
      prompt_ablation: { ... }         # optional; see below
    loss: { ... }
```

```yaml
  - method: rl
    repeat_per_week: true
    rl:
      policy: dqn                      # dqn | ppo | random
      checkpoint: ./output/exp/rl/dqn.zip  # optional; warm-start + write-back
      train_steps_per_week: 1000       # gradient updates after each week's episode
      learning_rate: 0.0001
      gamma: 0.95
      buffer_size: 10000
      batch_size: 64
      epsilon_start: 1.0               # exploration at week 1
      epsilon_end: 0.05                # exploration after the decay window
      epsilon_decay_weeks: 4           # linear anneal length
      seed: 20260601
      device: cpu                      # auto | cpu | cuda (DQN torch device)
      parallel_per_worker_gb: 1.5      # host RAM budgeted per parallel worker
      parallel_max_retries: 1          # retry a failed person this many times
      parallel_worker_threads: 1       # torch CPU threads per parallel worker
    loss: { ... }
```

The `rl` method is the only learning augmenter; greedy / llm_agent /
ptime stay deterministic per week. Each person gets a fresh DQN seeded
from its `person_id` (so a person trains identically in serial and
parallel runs); the agent trains week-by-week and the per-week
scheduling gain is the terminal reward. `checkpoint` lets the model
warm-start from a prior run and write its final weights back to the
same path; with parallel workers > 1 the file is overwritten by the
last person to finish, so runs that need reproducible checkpoints use
`--workers 1`.

The three `parallel_*` knobs only apply when RL augment is fanned out
with `augment --executor process --workers N` (the per-person DQN runs
are independent, so they parallelize cleanly):

- `parallel_per_worker_gb`: host RAM reserved per worker. The pool caps
  the worker count at `min(--workers, cpu_count - 1, usable_RAM /
  parallel_per_worker_gb)` and logs which limit bound it, so the run
  never oversubscribes the host.
- `parallel_max_retries`: a worker that crashes (e.g. a transient OOM)
  has its person re-run after the round finishes, up to this many extra
  attempts. `0` means one attempt per person.
- `parallel_worker_threads`: torch CPU threads each worker uses. `1`
  keeps N workers from each spawning a full thread pool and thrashing
  the cores.

Parallel RL forces the embedder to CPU inside each worker, so the GPU
is never contended; serial RL (`--workers 1`) honours `device` and the
`COMPUTE_DEVICE` env as before.

```yaml
  - method: ptime
    repeat_per_week: true
    ptime:
      candidate_strategy: windows       # earliest_only | windows | windows_and_center
      duration_preference: minimum      # minimum | maximum | midpoint
      solver: mcs                       # mcs (Berry et al. Fig. 4 B&B) | greedy (top-1)
      mcs_time_budget_seconds: 10.0     # per-week MCS search cap
      importance:                       # Choquet a_i in [0, 1]; missing keys default to 1/n
        time:      0.40
        duration:  0.15
        overlap:   0.40
        stability: 0.05
      interaction:                      # Choquet a_ij in [-1, 1]; key = "a|b" sorted alphabetically
        duration|time: 0.20
        overlap|time:  0.10
      learning:                         # paper Section 5.3 SVM-rank refinement (optional)
        enabled: true
        alpha: 0.6                      # blend; alpha=1.0 elicited-only, alpha=0.0 learned-only
        negative_samples_per_event: 2
        negative_shift_minutes: 120
        regularization: 1.0             # sklearn LinearSVC C
        seed: 20260601
    loss: { ... }
```

`ptime` scores each candidate placement via a 2-order Choquet integral over four scheduling-time criteria:
- `time`: fraction of the slot that lies inside any band of `environment.time_windows`.
- `duration`: how the slot length sits inside `[duration_min, duration_max]` (`minimum` rewards short, `maximum` rewards long, `midpoint` peaks at the centre).
- `overlap`: Allen-rule admissibility against every base event the slot touches (uses `temporal_relation_rules.yaml`).
- `stability`: constant 1.0 (base events stay immutable).

Default coefficients are uninformative (`a_i = 1/n`, `a_ij = 0`); override per criterion via `importance` and `interaction`. Interaction keys must be alphabetically sorted (`duration|time`, not `time|duration`); the schema rejects out-of-order pairs at load time. Positive `a_ij` makes criteria complementary (the slot scores well only when both are satisfied); negative `a_ij` makes them substitutive.

`solver` selects between `mcs` (multi-criteria branch-and-bound, default) and `greedy` (top-1 forward decode) over the same Choquet-scored candidates. The MCS path builds one variable per recommended-task instance per week, iterates the four criteria by decreasing upper-bound slack, and pins each newly maximized criterion as a local constraint while a mono-criterion DFS optimizes the next. Variable ordering is most-constrained-first, value ordering is descending `u_c`, and the remaining-best-utility prune cuts the tree. `mcs_time_budget_seconds` caps each per-week search; on timeout the best assignment so far is kept.

`learning` (optional) refines the importance vector via SVM-rank. When `enabled: true`, an `sklearn.svm.LinearSVC` is fitted per-person on pairwise `(preferred, dispreferred)` slot examples drawn from the calendar: each base `CalendarEvent` is a positive, and `negative_samples_per_event` shifted alternatives within `negative_shift_minutes` on the same day are paired negatives. The fitted `B` vector blends with elicited `A` as `alpha * A.z + (1 - alpha) * B.z`. `alpha=1.0` keeps elicited only; `alpha=0.0` uses learned only. With no base events the blend falls back to elicited.

`candidate_strategy` controls how many anchors are scored per base gap: `earliest_only` (gap start), `windows` (gap start plus one anchor per `time_windows` band that fits inside the gap; default), `windows_and_center` (adds a centred anchor).

`merge_threshold` controls how aggressively a new task can be folded into an existing calendar event instead of taking its own slot. Lower values (0.6) merge more; higher values (0.8) keep tasks separate.

`repeat_per_week`: when `true` (default) every augmenter receives a fresh copy of the task list at the start of each calendar week, so `num_tasks` tasks are targeted every week. When `false` unplaced tasks carry forward (one-shot scheduling across the horizon).

#### Prompt-component ablation (llm_agent only)

`prompt_ablation` expands the `(scenario, method)` pair into one augment + evaluate pass per variant of the named experimental design. The augmenter runs once when omitted (or with `design: single`).

```yaml
llm_agent:
  provider: openai
  model: gpt-4o-mini
  prompt_template: augment_oneshot
  prompt_ablation:
    design: plackett_burman_24      # single | leave_one_out | plackett_burman_12 |
                                    # plackett_burman_24 | full_factorial | custom
    fold: true                       # only meaningful for plackett_burman_12 (promotes to PB-24)
    use_default_placebos: true       # blocks not in `placebos` fall back to the default catalog;
                                     # set false to delete them instead
    placebos:                        # optional per-block replacement text
      hard_constraints: "Follow the user's calendar."
    variants:                        # required only when design == custom
      - { id: v_drop_examples, ablate: [fit_examples, worked_example_standalone] }
      - { id: v_drop_verify,   ablate: [self_verify] }
    seed: 20260518
```

Valid block names (`ablate:` and `placebos:` keys): `default_path`, `fit_examples`, `rule_of_thumb_1`, `rule_of_thumb_2`, `rule_of_thumb_3`, `worked_example_standalone`, `worked_example_concurrent`, `worked_example_waking`, `hard_constraints`, `do_not_drop`, `self_verify`, `split_dividable_tasks`, `context_block`. The schema rejects unknown keys at load time.

`split_dividable_tasks` is the 12th block and instructs the LLM to split `is_dividable=true` tasks into two or more shorter sibling sessions via a per-entry `pieces` array; the augmenter then emits one `ScheduledTask` per piece with `parent_task_label` set so `L_divide` credits the split. `context_block` is the 13th block and is the per-day context summary that renders whenever `observation.contexts` is non-empty; flipping it off blanks the block out (the augmenter still sees today's other prompt scaffolding). PB-12 has 11 testable factors so the 12th and 13th columns are both held at `+1` (always kept) under that design; PB-24 (folded) tests `split_dividable_tasks` but keeps `context_block` co-aliased with it. Use `design: custom` with `ablate: [split_dividable_tasks]` or `ablate: [context_block]` for a targeted before/after comparison.

### `loss` block (per method)

Weights on the eight scheduling-quality components. They MUST sum to `1.0` within `1e-6` or the schema rejects the file. `lambda_context_fit` defaults to `0.0` so legacy seven-component files continue to load.

```yaml
loss:
  lambda_cov:          0.18   # coverage: did we place all requested tasks?
  lambda_cal:          0.18   # calendar fit: rule + buffer violations (task×event + task×task + task×context Allen)
  lambda_pref:         0.13   # preference: deviation from persona's preferred start
  lambda_disp:         0.18   # dispersion: MET / intensity closeness with half-life lag
  lambda_merge:        0.13   # merge: concurrency opportunity cost (reward folding when σ is high)
  lambda_spread:       0.10   # spread: penalise day-clustering across the week
  lambda_divide:       0.10   # divide: reward splitting `is_dividable` tasks into sub-sessions
  lambda_context_fit:  0.00   # context fit: reward placements that overlap a task's recommended context_links
```

A lower total loss is a better schedule. The components are reported individually so you can see *why* a method scored as it did. The defaults above (cov / cal / disp = 0.18; pref / merge = 0.13; spread / divide = 0.10; context_fit = 0.0) are the v4 baseline; tighten or relax per scenario depending on what behaviour you want to reward. `L_context_fit` reads the full ground-truth context trace for the overlap (numerator) check, but scopes the scored recommendation set (denominator) to `recommended ∩ generated ∩ observation.contexts` for methods that declare contexts (blind methods score over all generated recommended categories), so the leg does depend on `observation.contexts`. A category is realized only when an overlapping episode's ontology IRI is one the task links, so placing a task during the wrong member (e.g. `with_family` when it links `alone`) earns nothing; the denominator counts only categories the persona generates, so recommendations for contexts the person never enters do not shrink the score.

### `evaluation` block

Numeric tunables shared by every method inside the scenario. Sits at the scenario level, not inside `augmentation`.

```yaml
evaluation:
  buffer_minutes:                  30     # global default for L_cal's buffer-aware partial credit
                                          # on `p` / `P` violations; per-rule `buffer:` overrides win
  merge_threshold:                 0.65   # minimum σ for a (task, event) pair to enter L_merge
  disp_half_life_days:             2.0    # half-life H in L_disp's time lag: lag(δ) = (1/2)^(δ / H)
  divide_duration_tolerance_pct:   0.15   # multiplicative band around `RecommendedTask.duration_max`
                                          # for L_divide; a `(week, dividable label)` bucket is
                                          # `divided_valid` when sum(pieces) lands inside
                                          # [n * D * (1 - tau), n * D * (1 + tau)]
```

The mapper knobs read from the same evaluation block and default to:

```yaml
evaluation:
  ancestor_max_hops:           2     # SUBCLASS-OF traversal depth inside one ontology
  ancestor_discount:           0.6   # base weight at hop 0
  ancestor_decay:              0.7   # geometric falloff per hop
  cross_ancestor_max_hops:     2     # same shape for the cross-ontology bridge
  cross_ancestor_discount:     0.6
  cross_ancestor_decay:        0.7
  sigma_threshold:             0.70  # minimum σ for tier 3 to count
```

`ancestor_*` controls tier 2 (single-ontology walk); `cross_ancestor_*` controls tier 1.5 (cross-ontology bridge via `hb:matchedActivity`). Tighten the hop budgets if cross-hits over-match in your experiment; loosen if too many pairs fall through to tier 3.

### `output` block (optional, per method)

Auto-derived when omitted; set explicitly to write artifacts to a custom directory.

```yaml
output:
  dir: ./output/example_experiment/custom_run
  write_json:   true     # per-person *_loss.json
  write_ics:    true     # per-person .ics files
  write_report: true     # markdown + JSON benchmark report
```

The default layout is `{output_base}/{experiment_id}/scenarios/{scenario_id}/{method}/`, with task generation under a sibling `task_generation/{scenario_id}/`.

### Running scenarios

```bash
# One scenario, one method (development loop):
docker compose run --rm app python -m src.scripts.scenarios.cli run \
  --scenario    src/experiments/persona/example_experiment/scenarios.yaml \
  --scenario-id nutrition_l1 --method greedy --seed 20260503 --workers 5

# Everything in the file:
docker compose run --rm app python -m src.scripts.scenarios.cli run \
  --scenario src/experiments/persona/example_experiment/scenarios.yaml \
  --seed 20260503 --workers 5
```

Persona generation and scenarios can also run in one shot by passing `--scenario` to `persona.cli generate` (README §"Steps 1 + 2 in a single command").

---

## Quick reference: minimal experiment

Copy `src/experiments/persona/example_experiment/` to a new directory and edit. The five files are wired together by their contents, not their filenames, so you can rename them as long as you pass the right paths on the CLI.

The dependency direction is:

```
event_config.yaml           <-- catalog of events
   ^
   |  references event names
persona_config.yaml         <-- who exists, what they do
   ^
   |  persons generated here
environment.yaml            <-- when / where / how output is written
   ^
   |  feeds into
temporal_relation_rules.yaml<-- cross-event constraints
   ^
   |  validated calendars go into
scenarios.yaml              <-- overlay scenario tasks, evaluate
```
