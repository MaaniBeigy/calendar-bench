# `persona/`: configurable persona-based calendar generation

Solver-driven generation of per-person calendars from typed YAML configs. The
package replaces the ad-hoc scripts under [src/scripts/ics_generation/](../ics_generation/)
with a modular, parallel, validated pipeline.

See [PLAN.md](PLAN.md) for the full design rationale. This README covers how
to run the package and how the pieces fit together.

## What it does

Given four YAML files and a root seed, the package:

1. Loads and validates the config bundle into typed pydantic models.
2. Samples a population: each persona template expands into N person records
   with bounded jitter on times and durations.
3. For every person, runs the planner: per-day event-count allocation
   (persona cadence + seasonality + trend), then a z3 day-model solve, then
   spillover propagation, with an optional yearly-anchor pass that pins
   persona-stated planned activities to their dates.
4. Writes one JSON and one ICS per person plus a run-level `index.json` and
   a snapshot of the resolved configs.
5. Runs the validation suite (event constraints, persona cadence, LTL rules,
   continuity) and emits both `summary_report.txt` and
   `summary_report.json`.

The pipeline is deterministic: same configs and root seed produce
byte-identical output regardless of `--workers` or `--executor`.

## Run

```bash
docker compose run --rm app python -m src.scripts.persona.cli generate \
  --environment src/scripts/persona/config/examples/environment.yaml \
  --personas    src/scripts/persona/config/examples/persona_config.yaml \
  --events      src/scripts/persona/config/examples/event_config.yaml \
  --rules       src/scripts/persona/config/examples/temporal_relation_rules.yaml \
  --out-dir     ./output/persona_run/example_001 \
  --seed        20260503 \
  --workers     0 \
  --executor    process
```

Useful generate flags:

- `--auto-chunk-size`: pick the `chunksize` for `Executor.map` from a
  workload-aware heuristic instead of the YAML default.
- `--validation-workers N`: run the validation suite in parallel; `0` =
  `os.cpu_count()`, `1` = serial.
- `--free-time-report`: also write `free_time_report.txt`, a per-occupation
  / per-person summary of unscheduled minutes.

Re-run validation against an existing run directory without regenerating:

```bash
docker compose run --rm app python -m src.scripts.persona.cli validate \
  ./output/persona_run/example_001 \
  --environment src/scripts/persona/config/examples/environment.yaml \
  --personas    src/scripts/persona/config/examples/persona_config.yaml \
  --events      src/scripts/persona/config/examples/event_config.yaml \
  --rules       src/scripts/persona/config/examples/temporal_relation_rules.yaml \
  --validation-workers 0
```

Sweep `run_pool` across worker counts and chunk sizes to pick a good config
for your workload:

```bash
docker compose run --rm app python -m src.scripts.persona.cli benchmark \
  --environment src/scripts/persona/config/examples/environment.yaml \
  --personas    src/scripts/persona/config/examples/persona_config.yaml \
  --events      src/scripts/persona/config/examples/event_config.yaml \
  --rules       src/scripts/persona/config/examples/temporal_relation_rules.yaml \
  --out-dir     ./output/persona_bench \
  --workers     1 2 4 8 \
  --executor    thread \
  --chunk-sizes 1 2 4
```

The benchmark writes `benchmark_report.{txt,json}` with wall time per row
and a `matches_baseline` flag that confirms output stayed deterministic
across configurations.

Exit code is `0` on a clean run, `1` if the validation report contains any
violations (override with `--allow-violations`), `2` on a config error.

## Output layout

```
<out-dir>/
  used_configs.json          snapshot of the resolved configs
  index.json                 one row per person: id, paths, unsat flag
  persons/
    <persona_id>_<idx>.json  full per-person schedule
  ics_per_person/
    <persona_id>_<idx>.ics   importable calendar
  summary_report.txt         human-readable validation
  summary_report.json        machine-readable validation
```

## Package layout

```
persona/
  cli.py                     argparse entry point
  config/                    YAML loader + pydantic schemas + defaults
  domain/                    runtime types (Person, Catalog, schedules, windows)
  sampling/                  population sampler with bounded jitter
  event_config/              event-config -> indexed Catalog
  constraints/               per-day constraint extractor and window resolver
  solver/                    z3 day model, week template, year anchors, spillover, retry
  planner/                   per-day count allocator and horizon orchestrator
  concurrency/               pool wrapper + workload-aware chunk_size heuristic
  export/                    per-person JSON, ICS, run index
  validation/                event, persona, LTL, continuity checks + report,
                             with a parallel runner that fans out per-person work
  analytics/                 free-time rollup and benchmark sweep helpers
```

## Tests

All new code has unit tests under [tests/unit/persona/](../../../tests/unit/persona/):

```bash
docker compose run --rm app pytest tests/unit/persona -v
```

A single integration test under [tests/integration/persona/](../../../tests/integration/persona/)
runs the full CLI end-to-end on the example configs.
