"""Argparse entry point: `python -m src.scripts.persona.cli ...`.

Four subcommands, all environment-driven (`--environment ...` is the
canonical handle - the run directory is taken from
`environment.output.dir` unless explicitly overridden):

- `generate` runs the full pipeline: load configs, sample population,
  solve every person in parallel, write per-person JSON + ICS, write
  the run index, and emit the validation report. Optional flags add a
  free-time analytics report, render charts, and let validation run in
  its own thread pool.
- `validate` re-runs the validation suite on the population generated
  for an environment. The check is environment-level: every per-person
  JSON under the run directory is validated together.
- `charts` renders per-persona charts (line trend, calendar heatmap,
  single-day Gantt) for an existing run, without regeneration.
  `--kind` selects the family; `--person`, `--date`, `--persona`,
  `--event`, `--metric`, `--aggregate` configure each chart.
- `benchmark` sweeps `run_pool` across (workers, executor, chunk_size)
  configurations on the same workload and prints wall-time +
  matches-baseline.

Exit codes:

- 0 on success (no violations, or `--allow-violations`).
- 1 on validation failures.
- 2 on argument or config errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.scripts.persona.analytics.benchmark import (
    BenchmarkConfig,
    benchmark_pool,
)
from src.scripts.persona.analytics.benchmark import (
    write_report as write_benchmark_report,
)
from src.scripts.persona.analytics.free_time import (
    build_free_time_report,
)
from src.scripts.persona.analytics.free_time import (
    write_report as write_free_time_report,
)
from src.scripts.persona.concurrency.pool import run_pool
from src.scripts.persona.config.loader import Config, ConfigError, load_config
from src.scripts.persona.config.schema import PersonaConfig
from src.scripts.persona.context.planner import plan_contexts
from src.scripts.persona.context.resolver import resolve_for_person
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.export.ics_writer import write_population_ics
from src.scripts.persona.export.index import build_index, write_index
from src.scripts.persona.export.json_writer import write_population_json
from src.scripts.persona.export.task_labels import assign_population_task_labels
from src.scripts.persona.export.timeline_writer import write_person_timeline
from src.scripts.persona.sampling.persona_sampler import sample_population
from src.scripts.persona.validation.parallel import run_validation_parallel
from src.scripts.persona.validation.report import ValidationReport, write_report

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="persona",
        description="Persona-based calendar event generator.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate a population and write outputs.")
    gen.add_argument("--environment", required=True, type=Path)
    gen.add_argument("--personas", required=True, type=Path)
    gen.add_argument("--events", required=True, type=Path)
    gen.add_argument("--rules", required=True, type=Path)
    gen.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to `environment.output.dir`.",
    )
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--workers", type=int, default=None)
    gen.add_argument("--executor", choices=("thread", "process"), default=None)
    gen.add_argument(
        "--auto-chunk-size",
        action="store_true",
        help="Pick chunk_size from a workload-aware heuristic.",
    )
    gen.add_argument(
        "--validation-workers",
        type=int,
        default=None,
        help="Workers for parallel validation; 0 = os.cpu_count(); 1 = serial.",
    )
    gen.add_argument(
        "--free-time-report",
        action="store_true",
        help="Also write free_time_report.txt under the out-dir.",
    )
    gen.add_argument(
        "--free-time-group-by",
        default="occupation_status",
        metavar="AXIS",
        help="Characteristic axis used to group the free-time rollup.",
    )
    gen.add_argument(
        "--no-contexts",
        action="store_true",
        help="Skip the context placer pass even when personas declare contexts.",
    )
    gen.add_argument(
        "--charts",
        nargs="*",
        default=None,
        metavar="KIND",
        help=(
            "Render charts under <out-dir>/charts/. Pass the flag with no "
            "value to render every kind (lines, heatmap, gantt, calendar) "
            "or list specific kinds, e.g. `--charts gantt calendar`."
        ),
    )
    gen.add_argument(
        "--calendar-dpi",
        type=int,
        default=300,
        metavar="DPI",
        help="DPI metadata for the weekly calendar PNG exports (default: 300).",
    )
    gen.add_argument(
        "--calendar-contexts",
        nargs="*",
        default=None,
        metavar="CATEGORY",
        help=(
            "Render context episodes of the named categories as translucent "
            "bands on the weekly calendar (in addition to base + augmented "
            "events). Pass with no value to render every category the "
            "personas declare, or list specific ones, e.g. "
            "`--calendar-contexts mood_emotion energy_state`. Omit the flag "
            "to keep the historical events-only calendar."
        ),
    )
    gen.add_argument(
        "--calendar-min-event-minutes",
        type=int,
        default=45,
        metavar="N",
        help=(
            "Visually pad short events (e.g. 5-min augmented tasks) up to "
            "this many minutes on the weekly calendar so titles stay legible. "
            "The actual clock window is preserved in the event's notes. "
            "Default: 45."
        ),
    )
    gen.add_argument(
        "--allow-violations",
        action="store_true",
        help="Exit 0 even if the validation report contains violations.",
    )
    gen.add_argument(
        "--scenario",
        type=Path,
        default=None,
        metavar="YAML",
        help=(
            "Path to a scenarios.yaml; when provided the scenarios pipeline "
            "(generate-tasks to augment to evaluate) is run automatically "
            "after successful persona generation."
        ),
    )
    gen.add_argument(
        "--scenario-id",
        default=None,
        metavar="ID",
        help=(
            "Restrict the chained scenarios run to one or more scenarios. "
            "Pass a single id (`--scenario-id senior_l1`) or a comma-"
            "separated list (`--scenario-id senior_l1,senior_l2`); the "
            "downstream scenarios CLI receives them as separate values. "
            "Multi-scenario YAML only; ignored when --scenario is absent."
        ),
    )
    gen.add_argument(
        "--scenario-method",
        default=None,
        choices=["greedy", "llm_agent", "rl"],
        metavar="METHOD",
        help=(
            "Restrict the chained scenarios run to one augmentation method "
            "(multi-scenario YAML only; ignored when --scenario is absent)."
        ),
    )
    progress_group = gen.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Force the per-person progress bar on (default: stderr-tty detection).",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=None,
        help="Force the per-person progress bar off.",
    )
    gen.add_argument(
        "--compute-device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        metavar="DEVICE",
        help=(
            "Override environment.compute.device for the embedding / "
            "retrieval stack. 'cuda' fails fast if no GPU is visible; "
            "'auto' probes at runtime."
        ),
    )
    gen.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help=(
            "Hide INFO-level chatter from the chained scenarios pipeline "
            "(only warnings / errors are printed). Useful when piping "
            "output to a file or running long batches."
        ),
    )

    val = sub.add_parser(
        "validate",
        help="Re-run validation against the population generated for an environment.",
    )
    # Validation is environment-level: every per-person JSON under the run
    # directory is checked together. The run directory comes from the
    # environment's `output.dir`; passing `run_dir` positionally only
    # overrides that for ad-hoc runs (e.g. validating an archived copy).
    val.add_argument("run_dir", type=Path, nargs="?", default=None)
    val.add_argument("--environment", required=True, type=Path)
    val.add_argument("--personas", required=True, type=Path)
    val.add_argument("--events", required=True, type=Path)
    val.add_argument("--rules", required=True, type=Path)
    val.add_argument("--validation-workers", type=int, default=None)
    val.add_argument("--allow-violations", action="store_true")

    chrt = sub.add_parser(
        "charts",
        help="Render per-persona charts for an existing run (no regeneration).",
    )
    # Same defaulting rules as `validate`: the run directory is taken
    # from the environment YAML unless overridden.
    chrt.add_argument("run_dir", type=Path, nargs="?", default=None)
    chrt.add_argument("--environment", required=True, type=Path)
    chrt.add_argument(
        "--kind",
        choices=("all", "lines", "gantt", "heatmap", "calendar"),
        default="all",
        help="Which chart family to render (default: all).",
    )
    chrt.add_argument(
        "--augmented-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "(calendar) Directory of augmented per-person JSONs to overlay "
            "on top of the base calendar (e.g. "
            "<scenario>/<method>/augmented/persons/). When omitted only the "
            "base persona events are rendered."
        ),
    )
    chrt.add_argument(
        "--calendar-dpi",
        type=int,
        default=300,
        metavar="DPI",
        help="(calendar) DPI metadata for the weekly calendar PNG exports.",
    )
    chrt.add_argument(
        "--calendar-min-event-minutes",
        type=int,
        default=45,
        metavar="N",
        help=(
            "(calendar) Visually pad short events up to this many minutes so "
            "titles stay legible. The actual clock window is preserved in "
            "the event's notes. Default: 45."
        ),
    )
    chrt.add_argument(
        "--calendar-contexts",
        nargs="*",
        default=None,
        metavar="CATEGORY",
        help=(
            "(calendar/all) Render context episodes of the named categories "
            "as translucent bands. Pass with no value to render every "
            "category the personas declare. Omit the flag to keep the "
            "historical events-only calendar."
        ),
    )
    chrt.add_argument(
        "--persona",
        type=str,
        default=None,
        help=(
            "(lines/heatmap) Filter to one persona_id. Default: every "
            "persona in the run."
        ),
    )
    chrt.add_argument(
        "--person",
        type=str,
        default=None,
        help=("(gantt) person_id to render. Default: the first person of " "the run."),
    )
    chrt.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "(gantt) ISO date YYYY-MM-DD. Default: the first day with "
            "events for that person."
        ),
    )
    chrt.add_argument(
        "--event",
        type=str,
        default=None,
        help=(
            "(heatmap) Filter the heatmap to one event type. Default: "
            "aggregate every event."
        ),
    )
    chrt.add_argument(
        "--metric",
        choices=("count", "duration"),
        default="count",
        help="(heatmap) `count` of episodes or `duration` in minutes.",
    )
    chrt.add_argument(
        "--aggregate",
        choices=("sum", "mean"),
        default="mean",
        help="(heatmap) `mean` or `sum` across the persons of the persona.",
    )

    bench = sub.add_parser(
        "benchmark", help="Sweep run_pool across worker/chunk configurations."
    )
    bench.add_argument("--environment", required=True, type=Path)
    bench.add_argument("--personas", required=True, type=Path)
    bench.add_argument("--events", required=True, type=Path)
    bench.add_argument("--rules", required=True, type=Path)
    bench.add_argument("--out-dir", required=True, type=Path)
    bench.add_argument("--seed", type=int, default=None)
    bench.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="List of worker counts to evaluate. 0 = os.cpu_count().",
    )
    bench.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Executor used for every benchmark row.",
    )
    bench.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="List of explicit chunk sizes to evaluate (ignored under --auto-chunk-size).",
    )
    bench.add_argument(
        "--auto-chunk-size",
        action="store_true",
        help="Use the workload-aware chunk_size heuristic for every row.",
    )

    return parser


def _resolve_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.environment, args.personas, args.events, args.rules)
    if getattr(args, "seed", None) is not None:
        cfg = Config(
            environment=cfg.environment.model_copy(update={"seed": args.seed}),
            persona=cfg.persona,
            event=cfg.event,
            rules=cfg.rules,
        )
    return cfg


def _attach_contexts(
    persons: list[Person],
    schedules: list[PersonSchedule],
    cfg: Config,
) -> list[PersonSchedule]:
    """Run the context placer once per (person, schedule) pair."""
    if not any(p.contexts for p in persons):
        return schedules
    window_map = WindowMap.from_config(cfg.environment.time_windows)
    rules = list(cfg.rules.rules) if cfg.rules and cfg.rules.rules else None
    out: list[PersonSchedule] = []
    for person, schedule in zip(persons, schedules):
        if not person.contexts:
            out.append(schedule)
            continue
        resolved = resolve_for_person(person, person.contexts)
        out.append(
            plan_contexts(person, schedule, resolved, window_map, ltl_rules=rules)
        )
    return out


def _run_validation(
    cfg: Config,
    persons: list,
    schedules: list,
    catalog: Catalog,
    *,
    workers: int | None = None,
) -> ValidationReport:
    """Wrapper that always uses the parallel validator.

    `workers=1` keeps the run serial (no executor overhead). `workers=None`
    or `workers <= 0` resolves to `os.cpu_count()` inside the helper.
    """
    return run_validation_parallel(
        cfg.persona,
        persons,
        schedules,
        catalog,
        cfg.rules.rules,
        start_date=cfg.environment.horizon.start_date,
        workers=workers,
    )


def _write_persona_timelines(
    persons: list[Person],
    schedules: list[PersonSchedule],
    persons_dir: Path,
) -> list[Path]:
    """Emit one `<person_id>_timeline.tsv` per person next to the json files."""
    # Loading the IRI catalog is optional; without it timeline rows fall back to
    # the persona-local slug for `label` and emit empty `source`.
    catalog = None
    if any(s.contexts for s in schedules):
        try:
            from src.scripts.persona.context.catalog import load_catalog

            catalog = load_catalog()
        except Exception:
            catalog = None

    out_paths: list[Path] = []
    for person, schedule in zip(persons, schedules):
        target = persons_dir / f"{schedule.person_id}_timeline.tsv"
        result = write_person_timeline(
            person_id=schedule.person_id,
            persona_id=schedule.persona_id,
            schedule=schedule,
            contexts=schedule.contexts,
            augmented=None,
            catalog=catalog,
            out_path=target,
        )
        out_paths.append(result.path)
    return out_paths


def _write_used_configs(cfg: Config, out_dir: Path) -> Path:
    """Snapshot the resolved configs so a run is fully reproducible."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": cfg.environment.model_dump(mode="json"),
        "persona": cfg.persona.model_dump(mode="json", by_alias=True),
        "event": cfg.event.model_dump(mode="json"),
        "rules": cfg.rules.model_dump(mode="json"),
    }
    target = out_dir / "used_configs.json"
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def _apply_compute_device(args: argparse.Namespace, cfg: Config) -> None:
    """Export the resolved compute device to `COMPUTE_DEVICE`.

    CLI override wins; otherwise `environment.compute.device` from
    the YAML is used.  The result is written to `os.environ` so
    downstream consumers; the local sentence-transformers embedder
    invoked deep inside the chained scenarios pipeline; read the
    same value without further plumbing.
    """
    chosen = getattr(args, "compute_device", None) or cfg.environment.compute.device
    os.environ["COMPUTE_DEVICE"] = chosen


def _resolve_run_dir(cli_value: Path | None, cfg: Config) -> Path:
    """Resolve the run directory used by `generate`/`validate`/`charts`.

    A CLI override wins. Otherwise we use `environment.output.dir` so a
    user pointing only at the environment YAML still gets a sensible
    default.
    """
    if cli_value is not None:
        return cli_value
    return Path(cfg.environment.output.dir)


def _run_scenarios(args: argparse.Namespace) -> int:
    """Invoke the scenarios pipeline after successful persona generation.

    Builds a minimal argv from the generate args and calls
    `scenarios.cli.main` directly (no subprocess).
    """
    from src.scripts.scenarios.cli import main as scenarios_main

    log_level = "WARNING" if getattr(args, "quiet", False) else "INFO"
    scenario_argv = ["--log-level", log_level, "run", "--scenario", str(args.scenario)]
    if getattr(args, "seed", None) is not None:
        scenario_argv += ["--seed", str(args.seed)]
    if getattr(args, "workers", None) is not None:
        scenario_argv += ["--workers", str(args.workers)]
    scenario_id_value = getattr(args, "scenario_id", None)
    if scenario_id_value is not None:
        # `--scenario-id` accepts one or many IDs on the scenarios CLI
        # (`nargs="+"`). The persona CLI exposes the same flag as a
        # comma-separated string so a chained run can target a subset:
        #   --scenario-id A,B   to  --scenario-id A B
        if isinstance(scenario_id_value, str):
            ids = [s.strip() for s in scenario_id_value.split(",") if s.strip()]
        else:
            ids = [str(s) for s in scenario_id_value]
        if ids:
            scenario_argv += ["--scenario-id", *ids]
    if getattr(args, "scenario_method", None) is not None:
        scenario_argv += ["--method", str(args.scenario_method)]
    charts_value = getattr(args, "charts", None)
    if charts_value is not None:
        scenario_argv += ["--charts", *list(charts_value)]
        calendar_dpi = getattr(args, "calendar_dpi", None)
        if calendar_dpi is not None:
            scenario_argv += ["--calendar-dpi", str(calendar_dpi)]
        min_event_minutes = getattr(args, "calendar_min_event_minutes", None)
        if min_event_minutes is not None:
            scenario_argv += ["--calendar-min-event-minutes", str(min_event_minutes)]
    return scenarios_main(scenario_argv)


def _generate(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    _apply_compute_device(args, cfg)
    catalog = Catalog.from_event_config(cfg.event)
    persons = sample_population(cfg.persona, cfg.environment.seed)
    schedules = run_pool(
        persons,
        catalog,
        cfg.environment,
        cfg.rules,
        workers=args.workers,
        executor=args.executor,
        auto_chunk=args.auto_chunk_size,
        show_progress=args.progress,
    )
    if not getattr(args, "no_contexts", False):
        schedules = _attach_contexts(persons, schedules, cfg)

    schedules = assign_population_task_labels(schedules, catalog)

    out_dir: Path = _resolve_run_dir(args.out_dir, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    persons_dir = out_dir / "persons"
    ics_dir = out_dir / "ics_per_person"

    json_paths = (
        write_population_json(persons, schedules, persons_dir)
        if cfg.environment.output.per_person_json
        else []
    )
    ics_paths = (
        write_population_ics(schedules, ics_dir, catalog=catalog)
        if cfg.environment.output.per_person_ics
        else None
    )

    if cfg.environment.output.per_person_json:
        _write_persona_timelines(persons, schedules, persons_dir)

    if cfg.environment.output.per_person_json:
        index = build_index(
            schedules,
            environment=cfg.environment,
            json_paths=json_paths,
            ics_paths=ics_paths,
            run_id=out_dir.name,
        )
        write_index(index, out_dir)

    _write_used_configs(cfg, out_dir)

    if args.free_time_report:
        write_free_time_report(
            build_free_time_report(
                persons, schedules, group_by=args.free_time_group_by
            ),
            out_dir,
        )

    if args.charts is not None:
        # Imported lazily so the matplotlib dependency only loads when charts
        # are actually requested.
        from src.scripts.persona.analytics.charts import (
            ALL_CHART_KINDS,
            render_all_charts,
            variable_heatmap_events,
        )

        selected = set(args.charts) if args.charts else set(ALL_CHART_KINDS)
        try:
            variable_events = variable_heatmap_events(cfg.event, cfg.persona)
            calendar_contexts = _resolve_calendar_contexts(
                getattr(args, "calendar_contexts", None), cfg.persona
            )
            render_all_charts(
                schedules,
                out_dir=out_dir,
                kinds=selected,
                heatmap_events_by_persona=variable_events,
                line_events_by_persona=variable_events,
                event_config=cfg.event,
                persona_config=cfg.persona,
                weekly_calendar_dpi=args.calendar_dpi,
                weekly_calendar_min_event_minutes=args.calendar_min_event_minutes,
                weekly_calendar_context_categories=calendar_contexts,
            )
        except ValueError as exc:
            sys.stderr.write(f"charts: {exc}\n")
            return EXIT_USAGE

    report = _run_validation(
        cfg, persons, schedules, catalog, workers=args.validation_workers
    )
    if cfg.environment.output.validation_report:
        write_report(report, out_dir)

    if report.has_violations and not args.allow_violations:
        return EXIT_VIOLATIONS

    if getattr(args, "scenario", None) is not None:
        return _run_scenarios(args)

    return EXIT_OK


def _load_schedules_from_dir(persons_dir: Path) -> list:
    """Reload schedules from disk by parsing each per-person JSON."""
    # Importing lazily so the CLI does not pull in extra modules at import time.
    from src.scripts.persona.validation.loader import load_schedules

    return load_schedules(persons_dir)


def _validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.environment, args.personas, args.events, args.rules)
    run_dir: Path = _resolve_run_dir(args.run_dir, cfg)
    if not run_dir.is_dir():
        sys.stderr.write(
            f"validate: run directory {run_dir} does not exist - "
            "run `generate` first or pass an explicit run_dir.\n"
        )
        return EXIT_USAGE
    catalog = Catalog.from_event_config(cfg.event)
    persons = sample_population(cfg.persona, cfg.environment.seed)
    schedules = _load_schedules_from_dir(run_dir / "persons")
    if len(schedules) != len(persons):
        sys.stderr.write(
            f"warning: persons sampled ({len(persons)}) does not match "
            f"schedules on disk ({len(schedules)}); validating overlap only.\n"
        )
        n = min(len(schedules), len(persons))
        persons = persons[:n]
        schedules = schedules[:n]
    report = _run_validation(
        cfg, persons, schedules, catalog, workers=args.validation_workers
    )
    write_report(report, run_dir)
    if report.has_violations and not args.allow_violations:
        return EXIT_VIOLATIONS
    return EXIT_OK


def _load_experiment_name_from_run_dir(run_dir: Path) -> str | None:
    """Recover `environment.experiment_name` from `used_configs.json`.

    Returns `None` when the snapshot is missing / unparseable, when the
    `environment` block is absent, or when `experiment_name` itself is
    not set in the YAML. Callers (the benchmark-report writers) fall
    back to the bare `experiment_id` in that case.
    """
    snapshot = run_dir / "used_configs.json"
    if not snapshot.is_file():
        return None
    try:
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        env_block = payload.get("environment") or {}
        name = env_block.get("experiment_name")
        return name if isinstance(name, str) and name.strip() else None
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_calendar_contexts(
    raw: list[str] | None,
    persona_cfg: PersonaConfig | None,
) -> list[str] | None:
    """Resolve the --calendar-contexts argv into a category list or None."""
    if raw is None:
        return None
    if raw:
        return list(raw)
    if persona_cfg is None:
        return None
    cats: set[str] = set()
    for persona in persona_cfg.personas:
        cats.update(persona.contexts.keys())
    return sorted(cats) if cats else None


def _load_event_persona_from_run_dir(
    run_dir: Path,
) -> tuple[Any, Any] | tuple[None, None]:
    """Try to recover `(EventConfig, PersonaConfig)` from `used_configs.json`.

    `generate` writes this snapshot at the end of every run, so the
    standalone `charts` subcommand (and downstream callers like the
    scenarios augment-charts step) can reproduce the same heatmap
    filter without being handed the YAML paths a second time. Returns
    `(None, None)` when the file is missing or unparseable so the
    caller can fall back to the legacy aggregate heatmap.
    """
    snapshot = run_dir / "used_configs.json"
    if not snapshot.is_file():
        return None, None
    try:
        from src.scripts.persona.config.schema import EventConfig, PersonaConfig

        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        event_cfg = EventConfig.model_validate(payload["event"])
        persona_cfg = PersonaConfig.model_validate(payload["persona"])
    except Exception:
        return None, None
    return event_cfg, persona_cfg


def _charts(args: argparse.Namespace) -> int:
    """Render per-persona charts for a previously generated run.

    Only loads the environment YAML (for the run directory default);
    the persona, event, and rules YAMLs are not needed because charts
    read schedules directly from `<run_dir>/persons/*.json`. Imports
    are lazy so matplotlib loads only when this subcommand runs.

    `--kind` selects the chart family. `all` (the default) renders the
    line + heatmap chart for every persona plus one sample Gantt per
    persona. `lines`, `gantt`, `heatmap` render exactly one family,
    optionally narrowed by `--persona`, `--person`, `--date`,
    `--event`, `--metric`, `--aggregate`.
    """
    import datetime as _dt

    from src.scripts.persona.analytics.charts import (
        group_by_persona,
        plot_person_day_gantt,
        plot_persona_heatmap,
        plot_persona_horizon_lines,
        render_all_charts,
        variable_heatmap_events,
    )
    from src.scripts.persona.config.loader import load_environment

    environment = load_environment(args.environment)
    run_dir: Path = (
        args.run_dir if args.run_dir is not None else Path(environment.output.dir)
    )
    persons_dir = run_dir / "persons"
    if not persons_dir.is_dir():
        sys.stderr.write(f"charts: no persons/ directory found under {run_dir}\n")
        return EXIT_USAGE
    schedules = _load_schedules_from_dir(persons_dir)
    if not schedules:
        sys.stderr.write(f"charts: no person JSONs found in {persons_dir}\n")
        return EXIT_USAGE

    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    event_cfg, persona_cfg = _load_event_persona_from_run_dir(run_dir)
    heatmap_events_by_persona = (
        variable_heatmap_events(event_cfg, persona_cfg)
        if event_cfg is not None and persona_cfg is not None
        else None
    )

    if args.kind == "all":
        from src.scripts.persona.analytics.charts import ALL_CHART_KINDS
        from src.scripts.persona.analytics.weekly_calendar import (
            load_augmented_tasks_by_person,
        )

        augmented = (
            load_augmented_tasks_by_person(args.augmented_dir)
            if args.augmented_dir is not None
            else None
        )
        calendar_contexts = _resolve_calendar_contexts(
            getattr(args, "calendar_contexts", None), persona_cfg
        )
        render_all_charts(
            schedules,
            out_dir=run_dir,
            kinds=ALL_CHART_KINDS,
            heatmap_event=args.event,
            heatmap_metric=args.metric,
            heatmap_aggregate=args.aggregate,
            heatmap_events_by_persona=(
                # Honour an explicit `--event` filter (single-event
                # narrow view) over the variable-events default.
                None
                if args.event is not None
                else heatmap_events_by_persona
            ),
            line_events_by_persona=(
                None if args.event is not None else heatmap_events_by_persona
            ),
            event_config=event_cfg,
            persona_config=persona_cfg,
            augmented_by_person=augmented,
            weekly_calendar_dpi=args.calendar_dpi,
            weekly_calendar_min_event_minutes=args.calendar_min_event_minutes,
            weekly_calendar_context_categories=calendar_contexts,
        )
        return EXIT_OK

    if args.kind == "calendar":
        from src.scripts.persona.analytics.weekly_calendar import (
            load_augmented_tasks_by_person,
            render_weekly_calendars,
        )

        augmented = (
            load_augmented_tasks_by_person(args.augmented_dir)
            if args.augmented_dir is not None
            else None
        )
        calendar_contexts = _resolve_calendar_contexts(
            getattr(args, "calendar_contexts", None), persona_cfg
        )
        render_weekly_calendars(
            schedules,
            augmented,
            charts_dir,
            dpi=args.calendar_dpi,
            min_visible_minutes=args.calendar_min_event_minutes,
            context_categories=calendar_contexts,
        )
        return EXIT_OK

    grouped = group_by_persona(schedules)

    if args.kind in ("lines", "heatmap"):
        if args.persona is not None:
            if args.persona not in grouped:
                sys.stderr.write(
                    f"charts: persona {args.persona!r} not found in run "
                    f"(have: {sorted(grouped)})\n"
                )
                return EXIT_USAGE
            targets = {args.persona: grouped[args.persona]}
        else:
            targets = grouped

        if args.kind == "lines":
            # When configs were recovered, switch to the per-event
            # comparison view (one chart per variable event, one line
            # per persona, 95% CI band). Without configs, fall back to
            # the legacy per-persona overlay chart.
            from src.scripts.persona.analytics.charts import (
                duration_unit_for_event,
                plot_per_event_persona_lines,
            )

            use_per_event = (
                heatmap_events_by_persona is not None and args.persona is None
            )
            if use_per_event:
                persona_overrides_by_id: dict[str, dict] = {}
                if persona_cfg is not None:  # pragma: no branch
                    for persona_obj in persona_cfg.personas:
                        persona_overrides_by_id[persona_obj.id] = dict(
                            persona_obj.event_overrides
                        )
                all_events: set[str] = set()
                for evs in heatmap_events_by_persona.values():  # type: ignore[union-attr]
                    all_events.update(evs)
                for event_name in sorted(all_events):
                    schedules_by_persona = {
                        pid: scheds
                        for pid, scheds in targets.items()
                        if event_name
                        in heatmap_events_by_persona.get(pid, set())  # type: ignore[union-attr]
                    }
                    if not schedules_by_persona:
                        continue
                    units = {
                        duration_unit_for_event(
                            event_cfg,
                            event_name,
                            persona_overrides_by_id.get(pid, {}).get(event_name),
                        )
                        for pid in schedules_by_persona
                    }
                    display_unit = (
                        units.pop()
                        if len(units) == 1
                        else duration_unit_for_event(event_cfg, event_name, None)
                    )
                    for metric in ("count", "duration"):
                        plot_per_event_persona_lines(
                            event_name,
                            metric,
                            schedules_by_persona,
                            charts_dir,
                            duration_unit=display_unit,
                        )
            else:
                for pid, sched_list in targets.items():
                    plot_persona_horizon_lines(pid, sched_list, charts_dir)
        else:
            # Per-event heatmap mode: when configs were recovered AND
            # the caller did NOT pin one event via `--event`, render
            # both a count and a duration heatmap for each variable
            # event per persona, with config-anchored color scale.
            from src.scripts.persona.analytics.charts import (
                heatmap_vmax_for_event,
                heatmap_vmin_for_event,
            )

            persona_overrides_by_id: dict[str, dict] = {}
            if persona_cfg is not None:
                for persona_obj in persona_cfg.personas:
                    persona_overrides_by_id[persona_obj.id] = dict(
                        persona_obj.event_overrides
                    )

            use_per_event = heatmap_events_by_persona is not None and args.event is None
            for pid, sched_list in targets.items():
                overrides = persona_overrides_by_id.get(pid, {})
                if use_per_event:
                    event_names = sorted(
                        heatmap_events_by_persona.get(pid, set())  # type: ignore[union-attr]
                    )
                    for event_name in event_names:
                        override = overrides.get(event_name)
                        for metric in ("count", "duration"):
                            vmin = heatmap_vmin_for_event(
                                event_cfg, event_name, metric, override
                            )
                            vmax = heatmap_vmax_for_event(
                                event_cfg, event_name, metric, override
                            )
                            plot_persona_heatmap(
                                pid,
                                sched_list,
                                charts_dir,
                                event_name=event_name,
                                metric=metric,
                                aggregate=args.aggregate,
                                vmin=vmin,
                                vmax=vmax,
                            )
                else:
                    override = (
                        overrides.get(args.event) if args.event is not None else None
                    )
                    vmin = heatmap_vmin_for_event(
                        event_cfg, args.event, args.metric, override
                    )
                    vmax = heatmap_vmax_for_event(
                        event_cfg, args.event, args.metric, override
                    )
                    plot_persona_heatmap(
                        pid,
                        sched_list,
                        charts_dir,
                        event_name=args.event,
                        metric=args.metric,
                        aggregate=args.aggregate,
                        vmin=vmin,
                        vmax=vmax,
                    )
        return EXIT_OK

    # kind == "gantt": pick a person + a date.
    if args.person is not None:
        sched = next((s for s in schedules if s.person_id == args.person), None)
        if sched is None:
            sys.stderr.write(
                f"charts: person {args.person!r} not found in run "
                f"(have: {sorted(s.person_id for s in schedules)})\n"
            )
            return EXIT_USAGE
    else:
        sched = schedules[0]

    if args.date is not None:
        try:
            target_date = _dt.date.fromisoformat(args.date)
        except ValueError:
            sys.stderr.write(
                f"charts: --date {args.date!r} is not a valid YYYY-MM-DD\n"
            )
            return EXIT_USAGE
    else:
        day_with_events = next((d for d in sched.days if d.events), None)
        if day_with_events is None:
            sys.stderr.write(
                f"charts: {sched.person_id} has no day with events to gantt\n"
            )
            return EXIT_USAGE
        target_date = day_with_events.date

    path = plot_person_day_gantt(sched, target_date, charts_dir)
    if path is None:
        sys.stderr.write(
            f"charts: no events on {target_date.isoformat()} for "
            f"{sched.person_id}\n"
        )
        return EXIT_USAGE
    return EXIT_OK


def _benchmark(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    catalog = Catalog.from_event_config(cfg.event)
    persons = sample_population(cfg.persona, cfg.environment.seed)
    if not persons:
        sys.stderr.write("benchmark: persona_config sampled an empty population\n")
        return EXIT_USAGE

    chunk_sizes: list[int | None] = (
        [None] if args.auto_chunk_size else list(args.chunk_sizes)
    )
    configs: list[BenchmarkConfig] = []
    for w in args.workers:
        for c in chunk_sizes:
            configs.append(
                BenchmarkConfig(
                    workers=w,
                    executor=args.executor,
                    chunk_size=c,
                    auto_chunk=args.auto_chunk_size,
                )
            )

    report = benchmark_pool(persons, catalog, cfg.environment, cfg.rules, configs)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_report(report, out_dir)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "generate":
            return _generate(args)
        if args.cmd == "validate":
            return _validate(args)
        if args.cmd == "charts":
            return _charts(args)
        if args.cmd == "benchmark":
            return _benchmark(args)
    except ConfigError as exc:
        sys.stderr.write(f"config error: {exc}\n")
        return EXIT_USAGE
    parser.error(f"unknown command: {args.cmd!r}")  # pragma: no cover
    return EXIT_USAGE  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
