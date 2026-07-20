# Benchmark report: 8-week cohort with online DQN augmenter; 5 survey-collectable context categories observed by every method

_Experiment id: `example_experiment`_

_Generated 2026-07-20T22:22:34+00:00_

_11 `(scenario_id, method)` run(s) aggregated._

## Legend

Compact label conventions used in the tables below:

* **M**: augmenter method. `SAP` = Single-Agent Prompt (one-shot `llm_agent` + `augment_oneshot`); `GRD` = greedy; `PTIME` = PTIME; `RL` = RL.
* **T**, **A**, **E**: Task generator, Augmenter, Evaluator stage models, in pipeline order.
* Stages sharing the same model are grouped: `TAE: gpt-4o-mini` means all three stages use that model; `TE: gpt-4o-mini + A: gpt-4.1-mini` means only the augmenter differs. `env-default` means the stage inherits its model from `.env`.

## Run summary

| Configuration | Task generator | Augmenter | Evaluator | Scenario id |
|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | gpt-4o-mini | gpt-4o-mini | gpt-4o-mini | `blind_oneshot_gpt_4o_mini / llm_agent` |
| Context Aware One-Shot (gpt-4o-mini) | gpt-4o-mini | gpt-4o-mini | gpt-4o-mini | `informed_oneshot_gpt_4o_mini / llm_agent` |
| Context Aware One-Shot (gpt-4.1-mini) | gpt-4o-mini | gpt-4.1-mini | gpt-4o-mini | `informed_oneshot_gpt_4_1_mini / llm_agent` |
| Context Aware One-Shot (gpt-5-mini) | gpt-4o-mini | gpt-5-mini | gpt-4o-mini | `informed_oneshot_gpt_5_mini / llm_agent` |
| Context Aware One-Shot (gpt-5.4-mini) | gpt-4o-mini | gpt-5.4-mini | gpt-4o-mini | `informed_oneshot_gpt_5_4_mini / llm_agent` |
| Context Aware One-Shot (claude-opus-4-8) | gpt-4o-mini | claude-opus-4-8 | gpt-4o-mini | `informed_oneshot_opus_4_8 / llm_agent` |
| Context Aware One-Shot (eval gpt-4.1-mini) | gpt-4o-mini | gpt-4o-mini | gpt-4.1-mini | `informed_eval_gpt_4_1_mini / llm_agent` |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | gpt-4.1-mini | gpt-4o-mini | gpt-4o-mini | `informed_taskgen_gpt_4_1_mini / llm_agent` |
| First-come-first-served greedy | gpt-4o-mini | env-default | gpt-4o-mini | `fcfs_greedy / greedy` |
| PTIME | gpt-4o-mini | env-default | gpt-4o-mini | `ptime_choquet / ptime` |
| DQN RL per person | gpt-4o-mini | env-default | gpt-4o-mini | `dqn_rl_per_person / rl` |

## Scheduling gain

| Configuration | Avg total gain | Scored / Empty | G_cov | G_cal | G_pref | G_disp | G_merge | G_spread | G_divide | G_context |
|---|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.6706 | 8 / 0 | 0.8242 | 0.9641 | 0.8088 | 0.9987 | 0.1176 | 0.8170 | 0.1641 | 0.1868 |
| Context Aware One-Shot (gpt-4o-mini) | 0.6841 | 8 / 0 | 0.8373 | 0.9648 | 0.7819 | 0.9987 | 0.0155 | 0.8527 | 0.1750 | 0.4106 |
| Context Aware One-Shot (gpt-4.1-mini) | 0.7434 | 8 / 0 | 0.7893 | 0.9589 | 0.8431 | 0.9986 | 0.4403 | 0.9018 | 0.2018 | 0.4359 |
| Context Aware One-Shot (gpt-5-mini) | 0.8526 | 8 / 0 | 0.9688 | 0.9542 | 0.7916 | 0.9987 | 0.7691 | 0.9129 | 0.6081 | 0.5794 |
| Context Aware One-Shot (gpt-5.4-mini) | 0.7531 | 8 / 0 | 0.8716 | 0.9638 | 0.7825 | 0.9987 | 0.3186 | 0.9710 | 0.3242 | 0.4449 |
| Context Aware One-Shot (claude-opus-4-8) | 0.8020 | 8 / 0 | 0.9147 | 0.9631 | 0.8274 | 0.9986 | 0.5590 | 0.7946 | 0.6354 | 0.4509 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.6961 | 8 / 0 | 0.8269 | 0.9630 | 0.7954 | 0.9988 | 0.0664 | 0.8393 | 0.2513 | 0.4251 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.6661 | 8 / 0 | 0.8315 | 0.9707 | 0.7467 | 0.9987 | 0.0250 | 0.8438 | 0.0180 | 0.3957 |
| First-come-first-served greedy | 0.6513 | 8 / 0 | 1.0000 | 0.9680 | 0.8311 | 0.9986 | 0.0000 | 0.4018 | 0.0000 | 0.3672 |
| PTIME | 0.6270 | 8 / 0 | 1.0000 | 0.9625 | 0.8049 | 0.9986 | 0.0000 | 0.3661 | 0.0000 | 0.2002 |
| DQN RL per person | 0.6249 | 8 / 0 | 0.8188 | 0.9658 | 0.7292 | 0.9979 | 0.0000 | 0.5835 | 0.0000 | 0.3387 |

## Weekly scheduling gain

_Per-person mean masked scheduling gain per ISO week. The `Δ` column is the learning signal (positive = improves across weeks); only the RL augmenter is expected to trend up._

| Configuration | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 | Wk7 | Wk8 | Δ (last - first) |
|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.6710 | 0.6351 | 0.6427 | 0.6618 | 0.6166 | 0.6363 | 0.6197 | 0.6449 | -0.0260 |
| Context Aware One-Shot (gpt-4o-mini) | 0.6897 | 0.6454 | 0.6523 | 0.6656 | 0.6497 | 0.6349 | 0.6330 | 0.6553 | -0.0344 |
| Context Aware One-Shot (gpt-4.1-mini) | 0.7037 | 0.7108 | 0.7171 | 0.7260 | 0.7112 | 0.7148 | 0.7211 | 0.7021 | -0.0016 |
| Context Aware One-Shot (gpt-5-mini) | 0.8315 | 0.8171 | 0.8189 | 0.8273 | 0.8163 | 0.8156 | 0.8227 | 0.8406 | +0.0091 |
| Context Aware One-Shot (gpt-5.4-mini) | 0.7421 | 0.7040 | 0.7211 | 0.6941 | 0.6990 | 0.7288 | 0.7045 | 0.7266 | -0.0155 |
| Context Aware One-Shot (claude-opus-4-8) | 0.7702 | 0.7457 | 0.7548 | 0.7520 | 0.7709 | 0.7427 | 0.7485 | 0.7625 | -0.0077 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.6947 | 0.6566 | 0.6674 | 0.6598 | 0.6591 | 0.6537 | 0.6583 | 0.6739 | -0.0207 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.6532 | 0.6206 | 0.6553 | 0.6499 | 0.6230 | 0.6376 | 0.6184 | 0.6476 | -0.0056 |
| First-come-first-served greedy | 0.6301 | 0.6100 | 0.6095 | 0.6089 | 0.6087 | 0.6127 | 0.6109 | 0.6600 | +0.0299 |
| PTIME | 0.6088 | 0.5896 | 0.5901 | 0.5903 | 0.5917 | 0.5916 | 0.5898 | 0.6379 | +0.0291 |
| DQN RL per person | 0.1345 | 0.4135 | 0.5854 | 0.5850 | 0.5573 | 0.5101 | 0.5991 | 0.5810 | +0.4465 |

## Per-person learning distribution

_Per-week median weighted gain across persons, IQM at the final week, and the per-person converged/flat/declined split counted by each person's first-to-last gain delta._

| Configuration | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 | Wk7 | Wk8 | IQM (last) | Δ median (last - first) | conv/n | flat/n | decl/n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.6711 | 0.6447 | 0.6512 | 0.6576 | 0.6051 | 0.6407 | 0.6156 | 0.6491 | 0.6478 | -0.0220 | 0/8 | 6/8 | 2/8 |
| Context Aware One-Shot (gpt-4o-mini) | 0.6914 | 0.6518 | 0.6513 | 0.6720 | 0.6433 | 0.6410 | 0.6300 | 0.6460 | 0.6511 | -0.0453 | 0/8 | 7/8 | 1/8 |
| Context Aware One-Shot (gpt-4.1-mini) | 0.6968 | 0.7187 | 0.7344 | 0.7276 | 0.7083 | 0.7189 | 0.7389 | 0.7034 | 0.7058 | +0.0067 | 1/8 | 6/8 | 1/8 |
| Context Aware One-Shot (gpt-5-mini) | 0.8434 | 0.8226 | 0.8261 | 0.8334 | 0.8133 | 0.8107 | 0.8205 | 0.8403 | 0.8391 | -0.0031 | 1/8 | 7/8 | 0/8 |
| Context Aware One-Shot (gpt-5.4-mini) | 0.7475 | 0.7113 | 0.7305 | 0.6965 | 0.6935 | 0.7236 | 0.7114 | 0.7220 | 0.7217 | -0.0255 | 0/8 | 8/8 | 0/8 |
| Context Aware One-Shot (claude-opus-4-8) | 0.7676 | 0.7376 | 0.7531 | 0.7568 | 0.7615 | 0.7389 | 0.7537 | 0.7752 | 0.7671 | +0.0076 | 0/8 | 8/8 | 0/8 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.6826 | 0.6514 | 0.6752 | 0.6566 | 0.6606 | 0.6578 | 0.6508 | 0.6886 | 0.6866 | +0.0060 | 1/8 | 7/8 | 0/8 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.6633 | 0.6225 | 0.6558 | 0.6598 | 0.6346 | 0.6488 | 0.6165 | 0.6500 | 0.6492 | -0.0133 | 0/8 | 8/8 | 0/8 |
| First-come-first-served greedy | 0.6299 | 0.6120 | 0.6118 | 0.6097 | 0.6093 | 0.6158 | 0.6128 | 0.6595 | 0.6597 | +0.0297 | 0/8 | 8/8 | 0/8 |
| PTIME | 0.6091 | 0.5898 | 0.5893 | 0.5913 | 0.5904 | 0.5917 | 0.5907 | 0.6374 | 0.6376 | +0.0282 | 0/8 | 8/8 | 0/8 |
| DQN RL per person | 0.0000 | 0.5261 | 0.5834 | 0.5804 | 0.5570 | 0.5719 | 0.6147 | 0.5960 | 0.5913 | +0.5960 | 8/8 | 0/8 | 0/8 |

## Cross-augmenter weekly trajectory

_Cohort-mean weighted gain per ISO week for every run, with the OLS `slope` over weeks. A positive slope is improvement over time; the RL run should trend up while the non-learning baselines stay near zero._

| Configuration | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 | Wk7 | Wk8 | Δ (last - first) | slope |
|---|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.6710 | 0.6351 | 0.6427 | 0.6618 | 0.6166 | 0.6363 | 0.6197 | 0.6449 | -0.0260 | -0.0039 |
| Context Aware One-Shot (gpt-4o-mini) | 0.6897 | 0.6454 | 0.6523 | 0.6656 | 0.6497 | 0.6349 | 0.6330 | 0.6553 | -0.0344 | -0.0044 |
| Context Aware One-Shot (gpt-4.1-mini) | 0.7037 | 0.7108 | 0.7171 | 0.7260 | 0.7112 | 0.7148 | 0.7211 | 0.7021 | -0.0016 | +0.0002 |
| Context Aware One-Shot (gpt-5-mini) | 0.8315 | 0.8171 | 0.8189 | 0.8273 | 0.8163 | 0.8156 | 0.8227 | 0.8406 | +0.0091 | +0.0008 |
| Context Aware One-Shot (gpt-5.4-mini) | 0.7421 | 0.7040 | 0.7211 | 0.6941 | 0.6990 | 0.7288 | 0.7045 | 0.7266 | -0.0155 | -0.0009 |
| Context Aware One-Shot (claude-opus-4-8) | 0.7702 | 0.7457 | 0.7548 | 0.7520 | 0.7709 | 0.7427 | 0.7485 | 0.7625 | -0.0077 | -0.0007 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.6947 | 0.6566 | 0.6674 | 0.6598 | 0.6591 | 0.6537 | 0.6583 | 0.6739 | -0.0207 | -0.0021 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.6532 | 0.6206 | 0.6553 | 0.6499 | 0.6230 | 0.6376 | 0.6184 | 0.6476 | -0.0056 | -0.0016 |
| First-come-first-served greedy | 0.6301 | 0.6100 | 0.6095 | 0.6089 | 0.6087 | 0.6127 | 0.6109 | 0.6600 | +0.0299 | +0.0027 |
| PTIME | 0.6088 | 0.5896 | 0.5901 | 0.5903 | 0.5917 | 0.5916 | 0.5898 | 0.6379 | +0.0291 | +0.0025 |
| DQN RL per person | 0.1345 | 0.4135 | 0.5854 | 0.5850 | 0.5573 | 0.5101 | 0.5991 | 0.5810 | +0.4465 | +0.0452 |

_The per-week figure is a cohort mean; the per-person distribution above is the unit-of-analysis view, so a flat mean alone does not prove that no individual learned._

## Stage cost & latency

| Configuration | Task generation wall | Task generation tokens | Task generation cost | Augmentation wall | Augmentation tokens | Augmentation cost | Augmentation reasoning (tokens / cost) | Evaluation wall | Evaluation tokens | Evaluation cost | Total cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 1044.7s | 64414 | $0.0216 | 1328.3s | 600073 | $0.1231 | n/a | 18.4s | 0 | $0.0000 | $0.1447 |
| Context Aware One-Shot (gpt-4o-mini) | 1034.9s | 65364 | $0.0220 | 1761.9s | 819143 | $0.1564 | n/a | 16.6s | 0 | $0.0000 | $0.1784 |
| Context Aware One-Shot (gpt-4.1-mini) | 1007.1s | 65408 | $0.0220 | 1624.7s | 822016 | $0.4219 | n/a | 13.9s | 0 | $0.0000 | $0.4440 |
| Context Aware One-Shot (gpt-5-mini) | 987.5s | 65000 | $0.0220 | 9400.5s | 817906 | $0.3330 | n/a | 13.8s | 0 | $0.0000 | $0.3550 |
| Context Aware One-Shot (gpt-5.4-mini) | 1041.0s | 65181 | $0.0220 | 484.8s | 820742 | $0.9026 | n/a | 14.5s | 0 | $0.0000 | $0.9246 |
| Context Aware One-Shot (claude-opus-4-8) | 948.3s | 63248 | $0.0216 | 1150.8s | 814238 | $5.4680 | n/a | 13.4s | 0 | $0.0000 | $5.4896 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 1037.9s | 63193 | $0.0216 | 1713.2s | 818629 | $0.1563 | n/a | 13.0s | 0 | $0.0000 | $0.1779 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 810.5s | 48516 | $0.0503 | 2155.0s | 818840 | $0.1577 | n/a | 12.5s | 0 | $0.0000 | $0.2080 |
| First-come-first-served greedy | 1122.4s | 63996 | $0.0218 | 0.0s | 0 | n/a | n/a | 33.7s | 0 | $0.0000 | $0.0218 |
| PTIME | 1070.9s | 63418 | $0.0216 | 23.5s | 0 | n/a | n/a | 14.5s | 0 | $0.0000 | $0.0216 |
| DQN RL per person | 1078.0s | 63949 | $0.0218 | 3770.6s | 0 | n/a | n/a | 18.3s | 0 | $0.0000 | $0.0218 |

## Ontology grounding

| Configuration | Generated | Grounded | Verified | Persons short-fetched | Persons w/ unverified URI |
|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| Context Aware One-Shot (gpt-4o-mini) | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| Context Aware One-Shot (gpt-4.1-mini) | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| Context Aware One-Shot (gpt-5-mini) | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| Context Aware One-Shot (gpt-5.4-mini) | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| Context Aware One-Shot (claude-opus-4-8) | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| First-come-first-served greedy | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| PTIME | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |
| DQN RL per person | 160 | 160 (100.0%) | 160 (100.0%) | 0 | 0 |

## Preference breakdown

| Configuration | per_occurrence_duration | per_scale_duration | per_scale_episodes | persona_stage_semantic | temporal_pattern_fix | temporal_pattern_seasonality | temporal_pattern_trend |
|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.1867 (n=8530) | 0.2248 (n=1059) | 0.0000 (n=1059) | n/a | 0.4578 (n=765) | 0.3816 (n=589) | 0.4369 (n=312) |
| Context Aware One-Shot (gpt-4o-mini) | 0.2182 (n=7744) | 0.2396 (n=1003) | 0.0000 (n=1003) | n/a | 0.5331 (n=766) | 0.3872 (n=491) | 0.4683 (n=299) |
| Context Aware One-Shot (gpt-4.1-mini) | 0.1496 (n=9130) | 0.1799 (n=1084) | 0.0000 (n=1084) | n/a | 0.5338 (n=798) | 0.3063 (n=596) | 0.4408 (n=278) |
| Context Aware One-Shot (gpt-5-mini) | 0.2048 (n=10634) | 0.2207 (n=1098) | 0.0000 (n=1098) | n/a | 0.5779 (n=850) | 0.4153 (n=610) | 0.3333 (n=296) |
| Context Aware One-Shot (gpt-5.4-mini) | 0.2134 (n=10409) | 0.2544 (n=1083) | 0.0000 (n=1083) | n/a | 0.6588 (n=801) | 0.4594 (n=617) | 0.3854 (n=320) |
| Context Aware One-Shot (claude-opus-4-8) | 0.1620 (n=9753) | 0.1837 (n=1077) | 0.0052 (n=1077) | n/a | 0.5695 (n=789) | 0.3715 (n=581) | 0.3272 (n=312) |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.1984 (n=8921) | 0.2396 (n=1076) | 0.0020 (n=1076) | n/a | 0.5479 (n=780) | 0.3787 (n=636) | 0.4931 (n=288) |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.2467 (n=9147) | 0.3039 (n=1120) | 0.0000 (n=1120) | n/a | 0.6142 (n=1043) | 0.3920 (n=667) | 0.5242 (n=328) |
| First-come-first-served greedy | 0.1629 (n=7040) | 0.1629 (n=880) | 0.0000 (n=880) | n/a | 0.8804 (n=624) | 0.0401 (n=456) | 0.2979 (n=280) |
| PTIME | 0.2024 (n=7360) | 0.1881 (n=920) | 0.0000 (n=920) | n/a | 0.6520 (n=704) | 0.1819 (n=408) | 0.2441 (n=344) |
| DQN RL per person | 0.2685 (n=1386) | 0.2823 (n=266) | 0.0000 (n=266) | n/a | 0.7917 (n=192) | 0.2678 (n=131) | 0.9524 (n=63) |

## Divide breakdown

| Configuration | Buckets | divided_valid | not_divided | divided_invalid_pieces_too_long | divided_invalid_sum_too_low | divided_invalid_sum_too_high |
|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 202 | 35 (17.3%) | 26 (12.9%) | 82 (40.6%) | 27 (13.4%) | 32 (15.8%) |
| Context Aware One-Shot (gpt-4o-mini) | 202 | 38 (18.8%) | 43 (21.3%) | 84 (41.6%) | 17 (8.4%) | 20 (9.9%) |
| Context Aware One-Shot (gpt-4.1-mini) | 205 | 48 (23.4%) | 18 (8.8%) | 61 (29.8%) | 11 (5.4%) | 67 (32.7%) |
| Context Aware One-Shot (gpt-5-mini) | 220 | 132 (60.0%) | 72 (32.7%) | 2 (0.9%) | 0 (0.0%) | 14 (6.4%) |
| Context Aware One-Shot (gpt-5.4-mini) | 138 | 41 (29.7%) | 45 (32.6%) | 8 (5.8%) | 24 (17.4%) | 20 (14.5%) |
| Context Aware One-Shot (claude-opus-4-8) | 136 | 86 (63.2%) | 11 (8.1%) | 3 (2.2%) | 4 (2.9%) | 32 (23.5%) |
| Context Aware One-Shot (eval gpt-4.1-mini) | 188 | 47 (25.0%) | 22 (11.7%) | 51 (27.1%) | 37 (19.7%) | 31 (16.5%) |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 376 | 7 (1.9%) | 164 (43.6%) | 126 (33.5%) | 51 (13.6%) | 28 (7.4%) |
| First-come-first-served greedy | 248 | 0 (0.0%) | 248 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| PTIME | 208 | 0 (0.0%) | 208 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DQN RL per person | 88 | 0 (0.0%) | 88 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## Pairwise A/B

_Baseline: `Context Blind One-Shot (gpt-4o-mini)`. Each Δ is `(other - baseline)`; positive = other does better, negative = other does worse._

### `Context Aware One-Shot (gpt-4o-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: no model changes)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (gpt-4o-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.6841 | +0.0135 |
| G_cov | 0.8242 | 0.8373 | +0.0131 |
| G_cal | 0.9641 | 0.9648 | +0.0007 |
| G_pref | 0.8088 | 0.7819 | -0.0269 |
| G_disp | 0.9987 | 0.9987 | -0.0000 |
| G_merge | 0.1176 | 0.0155 | -0.1021 |
| G_spread | 0.8170 | 0.8527 | +0.0357 |
| G_divide | 0.1641 | 0.1750 | +0.0109 |
| G_context | 0.1868 | 0.4106 | +0.2238 |
| Total cost | $0.1447 | $0.1784 | +0.0337 |
| Total wall time | 2391.3s | 2813.3s | +421.9742 |

### `Context Aware One-Shot (gpt-4.1-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (gpt-4.1-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.7434 | +0.0728 |
| G_cov | 0.8242 | 0.7893 | -0.0349 |
| G_cal | 0.9641 | 0.9589 | -0.0052 |
| G_pref | 0.8088 | 0.8431 | +0.0342 |
| G_disp | 0.9987 | 0.9986 | -0.0001 |
| G_merge | 0.1176 | 0.4403 | +0.3227 |
| G_spread | 0.8170 | 0.9018 | +0.0848 |
| G_divide | 0.1641 | 0.2018 | +0.0378 |
| G_context | 0.1868 | 0.4359 | +0.2491 |
| Total cost | $0.1447 | $0.4440 | +0.2993 |
| Total wall time | 2391.3s | 2645.7s | +254.3466 |

### `Context Aware One-Shot (gpt-5-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (gpt-5-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.8526 | +0.1821 |
| G_cov | 0.8242 | 0.9688 | +0.1446 |
| G_cal | 0.9641 | 0.9542 | -0.0099 |
| G_pref | 0.8088 | 0.7916 | -0.0173 |
| G_disp | 0.9987 | 0.9987 | +0.0000 |
| G_merge | 0.1176 | 0.7691 | +0.6516 |
| G_spread | 0.8170 | 0.9129 | +0.0960 |
| G_divide | 0.1641 | 0.6081 | +0.4440 |
| G_context | 0.1868 | 0.5794 | +0.3926 |
| Total cost | $0.1447 | $0.3550 | +0.2103 |
| Total wall time | 2391.3s | 10401.9s | +8010.5368 |

### `Context Aware One-Shot (gpt-5.4-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (gpt-5.4-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.7531 | +0.0825 |
| G_cov | 0.8242 | 0.8716 | +0.0474 |
| G_cal | 0.9641 | 0.9638 | -0.0004 |
| G_pref | 0.8088 | 0.7825 | -0.0264 |
| G_disp | 0.9987 | 0.9987 | +0.0000 |
| G_merge | 0.1176 | 0.3186 | +0.2010 |
| G_spread | 0.8170 | 0.9710 | +0.1540 |
| G_divide | 0.1641 | 0.3242 | +0.1602 |
| G_context | 0.1868 | 0.4449 | +0.2581 |
| Total cost | $0.1447 | $0.9246 | +0.7799 |
| Total wall time | 2391.3s | 1540.3s | -851.0018 |

### `Context Aware One-Shot (claude-opus-4-8)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (claude-opus-4-8) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.8020 | +0.1314 |
| G_cov | 0.8242 | 0.9147 | +0.0905 |
| G_cal | 0.9641 | 0.9631 | -0.0010 |
| G_pref | 0.8088 | 0.8274 | +0.0186 |
| G_disp | 0.9987 | 0.9986 | -0.0001 |
| G_merge | 0.1176 | 0.5590 | +0.4415 |
| G_spread | 0.8170 | 0.7946 | -0.0223 |
| G_divide | 0.1641 | 0.6354 | +0.4714 |
| G_context | 0.1868 | 0.4509 | +0.2641 |
| Total cost | $0.1447 | $5.4896 | +5.3449 |
| Total wall time | 2391.3s | 2112.6s | -278.7843 |

### `Context Aware One-Shot (eval gpt-4.1-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: evaluator)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (eval gpt-4.1-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.6961 | +0.0255 |
| G_cov | 0.8242 | 0.8269 | +0.0026 |
| G_cal | 0.9641 | 0.9630 | -0.0011 |
| G_pref | 0.8088 | 0.7954 | -0.0135 |
| G_disp | 0.9987 | 0.9988 | +0.0001 |
| G_merge | 0.1176 | 0.0664 | -0.0512 |
| G_spread | 0.8170 | 0.8393 | +0.0223 |
| G_divide | 0.1641 | 0.2513 | +0.0872 |
| G_context | 0.1868 | 0.4251 | +0.2383 |
| Total cost | $0.1447 | $0.1779 | +0.0331 |
| Total wall time | 2391.3s | 2764.1s | +372.7318 |

### `Context Aware One-Shot (task-gen gpt-4.1-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: task_generator)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (task-gen gpt-4.1-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.6661 | -0.0045 |
| G_cov | 0.8242 | 0.8315 | +0.0072 |
| G_cal | 0.9641 | 0.9707 | +0.0066 |
| G_pref | 0.8088 | 0.7467 | -0.0621 |
| G_disp | 0.9987 | 0.9987 | +0.0000 |
| G_merge | 0.1176 | 0.0250 | -0.0925 |
| G_spread | 0.8170 | 0.8438 | +0.0268 |
| G_divide | 0.1641 | 0.0180 | -0.1461 |
| G_context | 0.1868 | 0.3957 | +0.2088 |
| Total cost | $0.1447 | $0.2080 | +0.0633 |
| Total wall time | 2391.3s | 2978.0s | +586.6268 |

### `First-come-first-served greedy`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | First-come-first-served greedy | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.6513 | -0.0193 |
| G_cov | 0.8242 | 1.0000 | +0.1758 |
| G_cal | 0.9641 | 0.9680 | +0.0039 |
| G_pref | 0.8088 | 0.8311 | +0.0222 |
| G_disp | 0.9987 | 0.9986 | -0.0001 |
| G_merge | 0.1176 | 0.0000 | -0.1176 |
| G_spread | 0.8170 | 0.4018 | -0.4152 |
| G_divide | 0.1641 | 0.0000 | -0.1641 |
| G_context | 0.1868 | 0.3672 | +0.1804 |
| Total cost | $0.1447 | $0.0218 | -0.1229 |
| Total wall time | 2391.3s | 1156.1s | -1235.2560 |

### `PTIME`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | PTIME | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.6270 | -0.0436 |
| G_cov | 0.8242 | 1.0000 | +0.1758 |
| G_cal | 0.9641 | 0.9625 | -0.0016 |
| G_pref | 0.8088 | 0.8049 | -0.0039 |
| G_disp | 0.9987 | 0.9986 | -0.0001 |
| G_merge | 0.1176 | 0.0000 | -0.1176 |
| G_spread | 0.8170 | 0.3661 | -0.4509 |
| G_divide | 0.1641 | 0.0000 | -0.1641 |
| G_context | 0.1868 | 0.2002 | +0.0134 |
| Total cost | $0.1447 | $0.0216 | -0.1231 |
| Total wall time | 2391.3s | 1108.9s | -1282.4282 |

### `DQN RL per person`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | DQN RL per person | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.6706 | 0.6249 | -0.0457 |
| G_cov | 0.8242 | 0.8188 | -0.0054 |
| G_cal | 0.9641 | 0.9658 | +0.0017 |
| G_pref | 0.8088 | 0.7292 | -0.0796 |
| G_disp | 0.9987 | 0.9979 | -0.0008 |
| G_merge | 0.1176 | 0.0000 | -0.1176 |
| G_spread | 0.8170 | 0.5835 | -0.2335 |
| G_divide | 0.1641 | 0.0000 | -0.1641 |
| G_context | 0.1868 | 0.3387 | +0.1519 |
| Total cost | $0.1447 | $0.0218 | -0.1229 |
| Total wall time | 2391.3s | 4866.9s | +2475.5130 |

## Prompt component ablation

_OLS main effects of each `augment_oneshot` block on the scheduling-gain responses. `beta(y)` is the additive change in `y` per unit change of the block column (`+1` = block kept, `-1` = block ablated); positive `beta` means the block HELPS the response on average._

### Design summary

| Scenario / method | Design | Variants evaluated |
|---|---|---|
| Context Blind PB-16 ablation | plackett_burman_16 | 16 |
| Context Aware PB-16 ablation | plackett_burman_16 | 16 |

### Per-block main effects

#### `Context Blind PB-16 ablation`

| Block | beta(total) | beta(G_cov) | beta(G_cal) | beta(G_pref) | beta(G_disp) | beta(G_merge) | beta(G_spread) | beta(G_divide) | beta(G_context) |
|---|---|---|---|---|---|---|---|---|---|
| `default_path` | -0.0036 ± 0.0001 | -0.0112 ± 0.0044 | -0.0004 ± 0.0005 | +0.0049 ± 0.0037 | +0.0000 ± 0.0000 | +0.0074 ± 0.0067 | -0.0293 ± 0.0020 | -0.0030 ± 0.0030 | -0.0010 ± 0.0013 |
| `fit_examples` | -0.0052 ± 0.0001 | -0.0179 ± 0.0044 | -0.0012 ± 0.0005 | +0.0025 ± 0.0037 | -0.0000 ± 0.0000 | -0.0040 ± 0.0067 | -0.0081 ± 0.0020 | -0.0151 ± 0.0030 | +0.0001 ± 0.0013 |
| `rule_of_thumb_1` | +0.0007 ± 0.0001 | +0.0047 ± 0.0044 | +0.0001 ± 0.0005 | +0.0024 ± 0.0037 | -0.0000 ± 0.0000 | -0.0010 ± 0.0067 | -0.0020 ± 0.0020 | +0.0012 ± 0.0030 | -0.0013 ± 0.0013 |
| `rule_of_thumb_2` | -0.0013 ± 0.0001 | -0.0034 ± 0.0044 | -0.0001 ± 0.0005 | -0.0002 ± 0.0037 | +0.0000 ± 0.0000 | -0.0065 ± 0.0067 | -0.0067 ± 0.0020 | +0.0103 ± 0.0030 | -0.0007 ± 0.0013 |
| `rule_of_thumb_3` | -0.0024 ± 0.0001 | -0.0060 ± 0.0044 | -0.0002 ± 0.0005 | +0.0002 ± 0.0037 | -0.0000 ± 0.0000 | +0.0057 ± 0.0067 | -0.0162 ± 0.0020 | -0.0056 ± 0.0030 | -0.0002 ± 0.0013 |
| `worked_example_standalone` | -0.0007 ± 0.0001 | -0.0107 ± 0.0044 | -0.0003 ± 0.0005 | -0.0048 ± 0.0037 | -0.0000 ± 0.0000 | +0.0092 ± 0.0067 | +0.0095 ± 0.0020 | -0.0062 ± 0.0030 | +0.0007 ± 0.0013 |
| `worked_example_concurrent` | +0.0004 ± 0.0001 | +0.0034 ± 0.0044 | -0.0001 ± 0.0005 | -0.0010 ± 0.0037 | -0.0000 ± 0.0000 | -0.0035 ± 0.0067 | +0.0039 ± 0.0020 | -0.0009 ± 0.0030 | +0.0013 ± 0.0013 |
| `worked_example_waking` | -0.0005 ± 0.0001 | +0.0013 ± 0.0044 | -0.0001 ± 0.0005 | +0.0032 ± 0.0037 | -0.0000 ± 0.0000 | -0.0036 ± 0.0067 | -0.0025 ± 0.0020 | -0.0024 ± 0.0030 | -0.0021 ± 0.0013 |
| `hard_constraints` | +0.0024 ± 0.0001 | +0.0009 ± 0.0044 | +0.0003 ± 0.0005 | -0.0027 ± 0.0037 | +0.0000 ± 0.0000 | +0.0036 ± 0.0067 | +0.0170 ± 0.0020 | +0.0054 ± 0.0030 | -0.0008 ± 0.0013 |
| `do_not_drop` | +0.0046 ± 0.0001 | +0.0181 ± 0.0044 | -0.0001 ± 0.0005 | +0.0065 ± 0.0037 | -0.0000 ± 0.0000 | +0.0002 ± 0.0067 | +0.0081 ± 0.0020 | +0.0019 ± 0.0030 | -0.0002 ± 0.0013 |
| `self_verify` | -0.0000 ± 0.0001 | +0.0070 ± 0.0044 | -0.0001 ± 0.0005 | -0.0037 ± 0.0037 | +0.0000 ± 0.0000 | -0.0028 ± 0.0067 | -0.0020 ± 0.0020 | -0.0018 ± 0.0030 | +0.0001 ± 0.0013 |
| `split_dividable_tasks` | +0.0070 ± 0.0001 | +0.0071 ± 0.0044 | -0.0001 ± 0.0005 | +0.0208 ± 0.0037 | -0.0000 ± 0.0000 | +0.0113 ± 0.0067 | +0.0089 ± 0.0020 | +0.0132 ± 0.0030 | +0.0007 ± 0.0013 |
| `context_block` | +0.0024 ± 0.0001 | +0.0066 ± 0.0044 | +0.0002 ± 0.0005 | +0.0012 ± 0.0037 | +0.0000 ± 0.0000 | +0.0033 ± 0.0067 | +0.0089 ± 0.0020 | +0.0006 ± 0.0030 | -0.0017 ± 0.0013 |

#### `Context Aware PB-16 ablation`

| Block | beta(total) | beta(G_cov) | beta(G_cal) | beta(G_pref) | beta(G_disp) | beta(G_merge) | beta(G_spread) | beta(G_divide) | beta(G_context) |
|---|---|---|---|---|---|---|---|---|---|
| `default_path` | -0.0039 ± 0.0006 | -0.0090 ± 0.0089 | +0.0002 ± 0.0002 | +0.0005 ± 0.0027 | -0.0000 ± 0.0000 | -0.0035 ± 0.0044 | -0.0280 ± 0.0059 | +0.0124 ± 0.0126 | -0.0030 ± 0.0008 |
| `fit_examples` | -0.0023 ± 0.0006 | -0.0138 ± 0.0089 | -0.0007 ± 0.0002 | +0.0049 ± 0.0027 | -0.0000 ± 0.0000 | +0.0059 ± 0.0044 | -0.0077 ± 0.0059 | -0.0099 ± 0.0126 | +0.0030 ± 0.0008 |
| `rule_of_thumb_1` | +0.0003 ± 0.0006 | +0.0065 ± 0.0089 | +0.0001 ± 0.0002 | -0.0007 ± 0.0027 | +0.0000 ± 0.0000 | -0.0051 ± 0.0044 | -0.0063 ± 0.0059 | -0.0005 ± 0.0126 | +0.0060 ± 0.0008 |
| `rule_of_thumb_2` | -0.0008 ± 0.0006 | -0.0023 ± 0.0089 | +0.0001 ± 0.0002 | -0.0038 ± 0.0027 | +0.0000 ± 0.0000 | +0.0023 ± 0.0044 | -0.0004 ± 0.0059 | -0.0010 ± 0.0126 | -0.0018 ± 0.0008 |
| `rule_of_thumb_3` | +0.0015 ± 0.0006 | +0.0066 ± 0.0089 | -0.0003 ± 0.0002 | -0.0008 ± 0.0027 | +0.0000 ± 0.0000 | +0.0123 ± 0.0044 | -0.0074 ± 0.0059 | -0.0033 ± 0.0126 | +0.0013 ± 0.0008 |
| `worked_example_standalone` | -0.0013 ± 0.0006 | -0.0080 ± 0.0089 | -0.0002 ± 0.0002 | +0.0002 ± 0.0027 | -0.0000 ± 0.0000 | +0.0000 ± 0.0044 | -0.0015 ± 0.0059 | -0.0037 ± 0.0126 | +0.0043 ± 0.0008 |
| `worked_example_concurrent` | +0.0021 ± 0.0006 | +0.0041 ± 0.0089 | -0.0002 ± 0.0002 | -0.0025 ± 0.0027 | -0.0000 ± 0.0000 | -0.0001 ± 0.0044 | +0.0004 ± 0.0059 | +0.0135 ± 0.0126 | +0.0065 ± 0.0008 |
| `worked_example_waking` | +0.0001 ± 0.0006 | +0.0012 ± 0.0089 | +0.0002 ± 0.0002 | +0.0012 ± 0.0027 | -0.0000 ± 0.0000 | -0.0022 ± 0.0044 | +0.0074 ± 0.0059 | -0.0075 ± 0.0126 | -0.0012 ± 0.0008 |
| `hard_constraints` | +0.0024 ± 0.0006 | +0.0039 ± 0.0089 | +0.0006 ± 0.0002 | -0.0039 ± 0.0027 | +0.0000 ± 0.0000 | +0.0009 ± 0.0044 | +0.0160 ± 0.0059 | +0.0124 ± 0.0126 | -0.0055 ± 0.0008 |
| `do_not_drop` | +0.0048 ± 0.0006 | +0.0118 ± 0.0089 | -0.0002 ± 0.0002 | +0.0079 ± 0.0027 | -0.0000 ± 0.0000 | -0.0011 ± 0.0044 | +0.0102 ± 0.0059 | +0.0181 ± 0.0126 | -0.0037 ± 0.0008 |
| `self_verify` | +0.0010 ± 0.0006 | +0.0038 ± 0.0089 | -0.0000 ± 0.0002 | -0.0011 ± 0.0027 | +0.0000 ± 0.0000 | +0.0000 ± 0.0044 | +0.0032 ± 0.0059 | +0.0054 ± 0.0126 | -0.0020 ± 0.0008 |
| `split_dividable_tasks` | +0.0021 ± 0.0006 | +0.0021 ± 0.0089 | +0.0009 ± 0.0002 | +0.0102 ± 0.0027 | +0.0000 ± 0.0000 | -0.0015 ± 0.0044 | +0.0001 ± 0.0059 | +0.0111 ± 0.0126 | -0.0037 ± 0.0008 |
| `context_block` | -0.0036 ± 0.0006 | -0.0071 ± 0.0089 | +0.0001 ± 0.0002 | -0.0125 ± 0.0027 | +0.0000 ± 0.0000 | -0.0096 ± 0.0044 | +0.0038 ± 0.0059 | -0.0081 ± 0.0126 | +0.0044 ± 0.0008 |

### Per-block cost effects

#### `Context Blind PB-16 ablation`

| Block | beta(augment tokens) | beta(augment wall, s) | beta(augment USD) |
|---|---|---|---|
| `default_path` | +8392.8750 | +18.9142 | +0.0012 |
| `fit_examples` | +12486.1250 | -13.8472 | +0.0017 |
| `rule_of_thumb_1` | +2140.7500 | -154.3839 | +0.0004 |
| `rule_of_thumb_2` | +3816.8750 | +198.9919 | +0.0006 |
| `rule_of_thumb_3` | +10763.7500 | +6.2837 | +0.0015 |
| `worked_example_standalone` | +1367.7500 | +11.7795 | -0.0000 |
| `worked_example_concurrent` | +2893.1250 | +1.0905 | +0.0005 |
| `worked_example_waking` | +1352.0000 | +56.2785 | +0.0003 |
| `hard_constraints` | +4476.3750 | +39.7436 | +0.0006 |
| `do_not_drop` | +3794.3750 | +25.5876 | +0.0009 |
| `self_verify` | +6984.7500 | -86.1227 | +0.0008 |
| `split_dividable_tasks` | +21255.8750 | -230.9313 | +0.0043 |
| `context_block` | -1150.0000 | +10.5198 | -0.0002 |

#### `Context Aware PB-16 ablation`

| Block | beta(augment tokens) | beta(augment wall, s) | beta(augment USD) |
|---|---|---|---|
| `default_path` | +8977.6875 | +232.4657 | +0.0016 |
| `fit_examples` | +12410.1875 | -12.0021 | +0.0017 |
| `rule_of_thumb_1` | +2143.8125 | -163.7699 | +0.0004 |
| `rule_of_thumb_2` | +3786.6875 | -247.7065 | +0.0005 |
| `rule_of_thumb_3` | +10738.8125 | -81.2074 | +0.0015 |
| `worked_example_standalone` | +1423.3125 | -67.0214 | +0.0000 |
| `worked_example_concurrent` | +3151.1875 | +43.4229 | +0.0006 |
| `worked_example_waking` | +1294.9375 | +20.4114 | +0.0003 |
| `hard_constraints` | +4735.0625 | +30.5973 | +0.0007 |
| `do_not_drop` | +3639.0625 | -85.8276 | +0.0008 |
| `self_verify` | +7292.9375 | -57.6152 | +0.0010 |
| `split_dividable_tasks` | +20885.0625 | +152.7889 | +0.0041 |
| `context_block` | +96259.4375 | +147.2712 | +0.0142 |

<details><summary>Variant detail: <code>Context Blind PB-16 ablation</code></summary>

| Variant id | Ablated blocks | Avg total gain | Augment tokens | Augment wall (s) | Augment USD |
|---|---|---|---|---|---|
| `0000000000000` | (baseline) | 0.6742 | 601361 | 1843.4s | $0.1239 |
| `1010101010101` | context_block, default_path, hard_constraints, rule_of_thumb_1, rule_of_thumb_3, self_verify, worked_example_concurrent | 0.6748 | 533027 | 2097.3s | $0.1148 |
| `0110011001100` | do_not_drop, fit_examples, rule_of_thumb_1, self_verify, worked_example_concurrent, worked_example_standalone | 0.6751 | 541858 | 2200.9s | $0.1152 |
| `1100110011001` | context_block, default_path, do_not_drop, fit_examples, hard_constraints, rule_of_thumb_3, worked_example_standalone | 0.6795 | 520260 | 1645.1s | $0.1121 |
| `0001111000011` | context_block, rule_of_thumb_2, rule_of_thumb_3, split_dividable_tasks, worked_example_concurrent, worked_example_standalone | 0.6637 | 523297 | 1773.6s | $0.1107 |
| `1011010010110` | default_path, hard_constraints, rule_of_thumb_1, rule_of_thumb_2, self_verify, split_dividable_tasks, worked_example_standalone | 0.6654 | 503652 | 2247.0s | $0.1077 |
| `0111100001111` | context_block, do_not_drop, fit_examples, rule_of_thumb_1, rule_of_thumb_2, rule_of_thumb_3, self_verify, split_dividable_tasks | 0.6625 | 481176 | 2331.2s | $0.1041 |
| `1101001011010` | default_path, do_not_drop, fit_examples, hard_constraints, rule_of_thumb_2, split_dividable_tasks, worked_example_concurrent | 0.6661 | 487799 | 1690.3s | $0.1049 |
| `0000000111111` | context_block, do_not_drop, hard_constraints, self_verify, split_dividable_tasks, worked_example_waking | 0.6430 | 527765 | 2138.9s | $0.1106 |
| `1010101101010` | default_path, do_not_drop, rule_of_thumb_1, rule_of_thumb_3, split_dividable_tasks, worked_example_concurrent, worked_example_waking | 0.6619 | 499337 | 2397.3s | $0.1052 |
| `0110011110011` | context_block, fit_examples, hard_constraints, rule_of_thumb_1, split_dividable_tasks, worked_example_concurrent, worked_example_standalone, worked_example_waking | 0.6613 | 511717 | 2402.9s | $0.1089 |
| `1100110100110` | default_path, fit_examples, rule_of_thumb_3, self_verify, split_dividable_tasks, worked_example_standalone, worked_example_waking | 0.6855 | 476824 | 2244.7s | $0.1046 |
| `0001111111100` | do_not_drop, hard_constraints, rule_of_thumb_2, rule_of_thumb_3, self_verify, worked_example_concurrent, worked_example_standalone, worked_example_waking | 0.6691 | 530463 | 1336.1s | $0.1138 |
| `1011010101001` | context_block, default_path, do_not_drop, rule_of_thumb_1, rule_of_thumb_2, worked_example_standalone, worked_example_waking | 0.6713 | 562601 | 1434.1s | $0.1180 |
| `0111100110000` | fit_examples, hard_constraints, rule_of_thumb_1, rule_of_thumb_2, rule_of_thumb_3, worked_example_waking | 0.6874 | 531120 | 1503.0s | $0.1138 |
| `1101001100101` | context_block, default_path, fit_examples, rule_of_thumb_2, self_verify, worked_example_concurrent, worked_example_waking | 0.6900 | 530971 | 1471.4s | $0.1136 |

</details>

<details><summary>Variant detail: <code>Context Aware PB-16 ablation</code></summary>

| Variant id | Ablated blocks | Avg total gain | Augment tokens | Augment wall (s) | Augment USD |
|---|---|---|---|---|---|
| `0000000000000` | (baseline) | 0.6913 | 818465 | 2144.7s | $0.1561 |
| `1010101010101` | context_block, default_path, hard_constraints, rule_of_thumb_1, rule_of_thumb_3, self_verify, worked_example_concurrent | 0.6904 | 552162 | 1836.5s | $0.1162 |
| `0110011001100` | do_not_drop, fit_examples, rule_of_thumb_1, self_verify, worked_example_concurrent, worked_example_standalone | 0.6797 | 758636 | 2604.6s | $0.1472 |
| `1100110011001` | context_block, default_path, do_not_drop, fit_examples, hard_constraints, rule_of_thumb_3, worked_example_standalone | 0.6950 | 542095 | 1596.2s | $0.1152 |
| `0001111000011` | context_block, rule_of_thumb_2, rule_of_thumb_3, split_dividable_tasks, worked_example_concurrent, worked_example_standalone | 0.6891 | 546268 | 2023.8s | $0.1144 |
| `1011010010110` | default_path, hard_constraints, rule_of_thumb_1, rule_of_thumb_2, self_verify, split_dividable_tasks, worked_example_standalone | 0.6907 | 719973 | 2165.3s | $0.1394 |
| `0111100001111` | context_block, do_not_drop, fit_examples, rule_of_thumb_1, rule_of_thumb_2, rule_of_thumb_3, self_verify, split_dividable_tasks | 0.6855 | 504153 | 2840.8s | $0.1078 |
| `1101001011010` | default_path, do_not_drop, fit_examples, hard_constraints, rule_of_thumb_2, split_dividable_tasks, worked_example_concurrent | 0.6814 | 703590 | 1911.3s | $0.1363 |
| `0000000111111` | context_block, do_not_drop, hard_constraints, self_verify, split_dividable_tasks, worked_example_waking | 0.6753 | 550544 | 1503.7s | $0.1142 |
| `1010101101010` | default_path, do_not_drop, rule_of_thumb_1, rule_of_thumb_3, split_dividable_tasks, worked_example_concurrent, worked_example_waking | 0.6763 | 716801 | 1688.2s | $0.1376 |
| `0110011110011` | context_block, fit_examples, hard_constraints, rule_of_thumb_1, split_dividable_tasks, worked_example_concurrent, worked_example_standalone, worked_example_waking | 0.6918 | 533859 | 1841.3s | $0.1121 |
| `1100110100110` | default_path, fit_examples, rule_of_thumb_3, self_verify, split_dividable_tasks, worked_example_standalone, worked_example_waking | 0.6956 | 692714 | 1763.2s | $0.1361 |
| `0001111111100` | do_not_drop, hard_constraints, rule_of_thumb_2, rule_of_thumb_3, self_verify, worked_example_concurrent, worked_example_standalone, worked_example_waking | 0.6718 | 746341 | 3034.5s | $0.1452 |
| `1011010101001` | context_block, default_path, do_not_drop, rule_of_thumb_1, rule_of_thumb_2, worked_example_standalone, worked_example_waking | 0.6990 | 583710 | 2467.1s | $0.1206 |
| `0111100110000` | fit_examples, hard_constraints, rule_of_thumb_1, rule_of_thumb_2, rule_of_thumb_3, worked_example_waking | 0.6865 | 748538 | 2826.3s | $0.1461 |
| `1101001100101` | context_block, default_path, fit_examples, rule_of_thumb_2, self_verify, worked_example_concurrent, worked_example_waking | 0.7049 | 552116 | 1672.3s | $0.1163 |

</details>
