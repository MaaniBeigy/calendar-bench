# Progressive Healthy Lifestyle Challenge (PHLC-2026)

Benchmark results for the `progressive_healthy_lifestyle_promotion` experiment, the main competitive challenge of [CalendarBench](../../README.md).

## The benchmark

Thirty persons sampled from a single full-time-employee persona are scheduled over an **eight-week horizon** on a **progressive ramp**: each week generates its own task batch, the count grows from **10 to 38 tasks per week**, and the difficulty band widens from **Level 1 to Level 4** in **three domains** (nutrition, physical activity, mental wellbeing). Weeks 1 to 5 keep each week's tasks distinct from earlier weeks; weeks 6 to 8 allow reuse. Four scheduler families are compared under identical loss weights (coverage 0.16, consistency 0.16, preference 0.12, intensity 0.16, merging 0.12, spread 0.10, splitting 0.08, context 0.10): a first-come-first-served greedy gap-filler, PTIME, five one-shot LLM planners, and a per-person online DQN.

Experiment configuration (five YAML files): [src/experiments/persona/healthy_lifestyle_promotion](../../src/experiments/persona/healthy_lifestyle_promotion)

- [environment.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/environment.yaml): horizon, time grid, compute device
- [persona_config.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/persona_config.yaml): the persona archetype and sampling
- [event_config.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/event_config.yaml): routine activities
- [temporal_relation_rules.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/temporal_relation_rules.yaml): LTL + Allen rules
- [scenarios.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/scenarios.yaml): the weekly ramp, models, and augmenters

## Weekly scheduling gain

![Average weighted scheduling gain per week for each method in the PHLC-2026 experiment.](cross_augmenter_weekly.png)

## Current SOTA model comparison

Average total scheduling gain over 30 persons (higher is better), sorted; coverage is shown because it dominates the ranking here.

| Method | Total gain $G$ | Coverage $G_{\text{cov}}$ |
|---|---|---|
| First-come-first-served greedy | **0.6462** | 0.9927 |
| SAP one-shot, aug gpt-5-mini | **0.6454** | 0.7304 |
| PTIME | 0.6410 | 0.9933 |
| SAP one-shot, aug gpt-5.4-mini | 0.6033 | 0.5125 |
| SAP one-shot, aug gpt-4.1-mini | 0.5963 | 0.4467 |
| SAP one-shot, aug claude-opus-4-8 | 0.5953 | 0.5379 |
| SAP one-shot, task-gen gpt-4.1-mini | 0.5720 | 0.4316 |
| SAP one-shot, informed gpt-4o-mini | 0.5672 | 0.4229 |
| SAP one-shot, eval gpt-4.1-mini | 0.5638 | 0.4260 |
| SAP one-shot, blind gpt-4o-mini | 0.5621 | 0.4368 |
| Online DQN (RL) | 0.5214 | 0.2808 |

Full per-component breakdown (all eight legs), weekly trajectories, per-person learning distribution, cost, and prompt ablations: [benchmark_report.md](benchmark_report.md).

## Why the heuristics match the SOTA LLM planners

As the weekly task batch grows toward 38, the one-shot LLM planners place a decreasing share of the tasks (42 to 73 percent), whereas the greedy gap-filler and PTIME place about 99 percent. Coverage carries the joint-highest weight and about 59 percent of the run's discriminative signal, so it determines the ranking even though the LLM planners score higher on spread, merging, and context fit. The advantage reverses within the horizon: the LLM planners lead in the lighter early weeks, and the mean of the eight weekly gains places gpt-5-mini first, so the strongest LLM planner is level with the greedy heuristic overall.
