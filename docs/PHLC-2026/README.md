# Progressive Healthy Lifestyle Challenge (PHLC-2026)

![Benchmark: PHLC-2026](https://img.shields.io/badge/CalendarBench-PHLC--2026-8A2BE2?style=flat&logo=googlecharts&logoColor=white)

Benchmark results for the `progressive_healthy_lifestyle_promotion` experiment, the main competitive challenge of [CalendarBench](../../README.md).

## The benchmark

The cohort is **thirty persons** drawn from **three contrasting personas**, each sampled into ten instances. `diligent_anxious_active` is a conscientious, anxious, full-time employee and the only health-oriented archetype, with a full sports routine (gym, running, cycling, swimming) and deliberate planning contexts. `unemployed_family_caregiver` is an unemployed at-home caregiver with the highest fatigue, stress, and time at home in the cohort and no sports routine. `free_spirited_social_student` is a spontaneous, social young adult with a late-shifted, loosely structured routine. Only the first archetype is health-oriented, so recommended health tasks must be fitted into two routines that were not built around them. Each persona and its ten instances are detailed in this [README.md](../../src/experiments/persona/healthy_lifestyle_promotion/README.md).

Every person is simulated over an **eight-week horizon** on a 10-minute placement grid. Task recommendation follows a **progressive ramp**: each week generates its own batch, the count grows from **10 to 38 tasks per week**, and the difficulty band widens from **Level 1 to Level 4** in **three domains** (nutrition, physical activity, mental wellbeing). Weeks 1 to 5 keep each week's tasks distinct from earlier weeks; weeks 6 to 8 allow reuse. Six scheduler families are compared under identical loss weights (coverage 0.16, consistency 0.16, preference 0.12, intensity 0.16, merging 0.12, spread 0.10, splitting 0.08, context 0.10): a first-come-first-served greedy gap-filler, PTIME, five one-shot LLM planners, a per-person online DQN, a multi-agent scheduler that plans one week at a time, and a human coach who placed the recommended tasks into each calendar by hand.

Experiment configuration (five YAML files): [src/experiments/persona/healthy_lifestyle_promotion](../../src/experiments/persona/healthy_lifestyle_promotion)

- [environment.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/environment.yaml): horizon, time grid, compute device
- [persona_config.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/persona_config.yaml): the three persona archetypes and their sampling (see this [README.md](../../src/experiments/persona/healthy_lifestyle_promotion/README.md) for the side-by-side comparison)
- [event_config.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/event_config.yaml): routine activities
- [temporal_relation_rules.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/temporal_relation_rules.yaml): LTL + Allen rules
- [scenarios.yaml](../../src/experiments/persona/healthy_lifestyle_promotion/scenarios.yaml): the weekly ramp, models, and augmenters

## Weekly scheduling gain

![Average weighted scheduling gain per week for each method in the PHLC-2026 experiment.](cross_augmenter_weekly.png)

## Current SOTA model comparison

Average total scheduling gain over 30 persons (higher is better), sorted; coverage is shown because it separates most of the field.

| Method | Total gain $G$ | Coverage $G_{\text{cov}}$ |
|---|---|---|
| Human coach | **0.8533** | 0.9948 |
| Multi-agent scheduler (MAS, gpt-4o-mini) | **0.6519** | 0.7828 |
| First-come-first-served greedy | 0.6462 | 0.9927 |
| SAP one-shot, aug gpt-5-mini | 0.6454 | 0.7304 |
| PTIME | 0.6410 | 0.9933 |
| SAP one-shot, aug gpt-5.4-mini | 0.6033 | 0.5125 |
| SAP one-shot, aug gpt-4.1-mini | 0.5963 | 0.4467 |
| SAP one-shot, aug claude-opus-4-8 | 0.5953 | 0.5379 |
| SAP one-shot, task-gen gpt-4.1-mini | 0.5720 | 0.4316 |
| SAP one-shot, informed gpt-4o-mini | 0.5672 | 0.4229 |
| SAP one-shot, eval gpt-4.1-mini | 0.5638 | 0.4260 |
| SAP one-shot, blind gpt-4o-mini | 0.5621 | 0.4368 |
| Online DQN (RL) | 0.4190 | 0.2863 |

Full per-component breakdown (all eight legs), weekly trajectories, per-person learning distribution, cost, and prompt ablations: [benchmark_report.md](benchmark_report.md).

## The human coach ceiling

The human coach (`human_coach`) is the reference ceiling for the benchmark, a person scheduled the recommended tasks into each calendar by hand. The schedules keep greedy's near-total coverage (0.9948) and lead every quality leg, so total gain of 0.8533 is far from of the next-best 0.6519.

## The multi-agent scheduler

The multi-agent scheduler (`mas`) is the strongest automated entry at 0.6519, ahead of the greedy gap-filler (0.6462), the best one-shot planner (0.6454), and PTIME (0.6410). It reaches the top of the automated augmenters without leading coverage, at 0.7828 it places a smaller share of each batch than greedy (0.9927) and makes that back on the quality legs, leading every automated method on task spread (0.8699), context fit (0.3423), and merging (0.1668). Splitting stays at zero, as it does for every automated method, and intensity dispersion (0.8495) sits below the near-perfect 0.999 the heuristics and one-shot planners reach. Its weekly gain slides from 0.6722 to 0.6223 as the batch grows toward 38 tasks, so its performance degrades under the heaviest load.

## Why the heuristics match the SOTA LLM planners

As the weekly task batch grows toward 38, the one-shot LLM planners place a decreasing share of the tasks (42 to 73 percent), whereas the greedy gap-filler and PTIME place about 99 percent. Coverage carries the joint-highest weight, so it determines the order among the one-shot planners and the heuristics even though those planners lead on the quality legs, as they score higher on task spread and context fit, and they earn merging credit the heuristics never do. That trade only carries them so far, and the mean of the eight weekly gains keeps greedy (0.6245) just ahead of PTIME (0.6185) and gpt-5-mini (0.6182). The multi-agent scheduler is the one automated augmenter that escapes the trade, pairing mid-range coverage with the best quality legs to lead the automated schedulers on both the total and the weekly mean.
