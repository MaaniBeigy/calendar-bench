# Benchmark report: Progressive Healthy Lifestyle Challenge (PHLC)-2026.06

_Experiment id: `progressive_healthy_lifestyle_promotion`_

_Generated 2026-07-12T20:32:00+00:00_

_12 `(scenario_id, method)` run(s) aggregated._

## Legend

Compact label conventions used in the tables below:

* **M**: augmenter method. `SAP` = Single-Agent Prompt (one-shot `llm_agent` + `augment_oneshot`); `GRD` = greedy; `PTIME` = PTIME; `RL` = RL; `HUMAN` = Human coach (handcrafted schedules).
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
| Human coach | gpt-4o-mini | env-default | gpt-4o-mini | `human_coach / human_coach` |
| DQN RL per person | gpt-4o-mini | env-default | gpt-4o-mini | `dqn_rl_per_person / rl` |

## Scheduling gain

| Configuration | Avg total gain | Scored / Empty | G_cov | G_cal | G_pref | G_disp | G_merge | G_spread | G_divide | G_context |
|---|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.5621 | 30 / 0 | 0.4368 | 0.9853 | 0.7912 | 0.9992 | 0.0040 | 0.6029 | 0.0000 | 0.1902 |
| Context Aware One-Shot (gpt-4o-mini) | 0.5672 | 30 / 0 | 0.4229 | 0.9889 | 0.7573 | 0.9993 | 0.0033 | 0.6000 | 0.0000 | 0.3013 |
| Context Aware One-Shot (gpt-4.1-mini) | 0.5963 | 30 / 0 | 0.4467 | 0.9780 | 0.8074 | 0.9990 | 0.1245 | 0.6964 | 0.0000 | 0.2700 |
| Context Aware One-Shot (gpt-5-mini) | 0.6454 | 30 / 0 | 0.7304 | 0.9629 | 0.7890 | 0.9990 | 0.1021 | 0.7405 | 0.0000 | 0.3368 |
| Context Aware One-Shot (gpt-5.4-mini) | 0.6033 | 30 / 0 | 0.5125 | 0.9782 | 0.7810 | 0.9991 | 0.0707 | 0.7561 | 0.0000 | 0.2709 |
| Context Aware One-Shot (claude-opus-4-8) | 0.5953 | 30 / 0 | 0.5379 | 0.9759 | 0.8227 | 0.9990 | 0.0183 | 0.6744 | 0.0000 | 0.2493 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.5638 | 30 / 0 | 0.4260 | 0.9873 | 0.7550 | 0.9993 | 0.0042 | 0.5714 | 0.0000 | 0.2955 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.5720 | 30 / 0 | 0.4316 | 0.9877 | 0.7322 | 0.9992 | 0.0219 | 0.6077 | 0.0000 | 0.3381 |
| First-come-first-served greedy | 0.6462 | 30 / 0 | 0.9927 | 0.9763 | 0.7964 | 0.9990 | 0.0000 | 0.5304 | 0.0000 | 0.2275 |
| PTIME | 0.6410 | 30 / 0 | 0.9933 | 0.9756 | 0.8083 | 0.9990 | 0.0000 | 0.4911 | 0.0000 | 0.2003 |
| Human coach | 0.8533 | 30 / 0 | 0.9948 | 0.9704 | 0.7705 | 0.9990 | 0.4316 | 0.9839 | 0.9991 | 0.5649 |
| DQN RL per person | 0.5214 | 30 / 0 | 0.2808 | 0.9887 | 0.7173 | 0.9986 | 0.0000 | 0.5907 | 0.0000 | 0.1338 |

## Weekly scheduling gain

_Per-person mean masked scheduling gain per ISO week. The `Δ` column is the learning signal (positive = improves across weeks); only the RL augmenter is expected to trend up._

| Configuration | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 | Wk7 | Wk8 | Δ (last - first) |
|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.5405 | 0.5550 | 0.5669 | 0.5311 | 0.5347 | 0.5341 | 0.5146 | 0.5258 | -0.0147 |
| Context Aware One-Shot (gpt-4o-mini) | 0.5904 | 0.5701 | 0.5606 | 0.5753 | 0.5044 | 0.5342 | 0.5249 | 0.5120 | -0.0783 |
| Context Aware One-Shot (gpt-4.1-mini) | 0.6067 | 0.5739 | 0.5851 | 0.5822 | 0.5570 | 0.5461 | 0.5605 | 0.5497 | -0.0570 |
| Context Aware One-Shot (gpt-5-mini) | 0.6211 | 0.6390 | 0.6388 | 0.6145 | 0.6155 | 0.6244 | 0.5699 | 0.6221 | +0.0010 |
| Context Aware One-Shot (gpt-5.4-mini) | 0.5860 | 0.5752 | 0.5973 | 0.5984 | 0.5493 | 0.5826 | 0.5860 | 0.5766 | -0.0093 |
| Context Aware One-Shot (claude-opus-4-8) | 0.5574 | 0.5662 | 0.6048 | 0.5834 | 0.5615 | 0.5833 | 0.5823 | 0.5742 | +0.0169 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.5681 | 0.5786 | 0.5686 | 0.5009 | 0.5557 | 0.5358 | 0.5301 | 0.5121 | -0.0560 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.6030 | 0.5902 | 0.5922 | 0.5723 | 0.5274 | 0.5322 | 0.5249 | 0.5070 | -0.0960 |
| First-come-first-served greedy | 0.6044 | 0.6083 | 0.6221 | 0.5970 | 0.6216 | 0.6360 | 0.6469 | 0.6594 | +0.0550 |
| PTIME | 0.5962 | 0.6034 | 0.6099 | 0.5918 | 0.6136 | 0.6304 | 0.6435 | 0.6591 | +0.0629 |
| Human coach | 0.8068 | 0.8415 | 0.8109 | 0.8101 | 0.8481 | 0.8235 | 0.8316 | 0.8261 | +0.0193 |
| DQN RL per person | 0.0161 | 0.5575 | 0.5398 | 0.5566 | 0.2040 | 0.5029 | 0.4933 | 0.4861 | +0.4700 |

## Per-person learning distribution

_Per-week median weighted gain across persons, IQM at the final week, and the per-person converged/flat/declined split counted by each person's first-to-last gain delta._

| Configuration | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 | Wk7 | Wk8 | IQM (last) | Δ median (last - first) | conv/n | flat/n | decl/n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.5567 | 0.5482 | 0.5603 | 0.5491 | 0.5431 | 0.5402 | 0.5122 | 0.5252 | 0.5285 | -0.0315 | 1/30 | 23/30 | 6/30 |
| Context Aware One-Shot (gpt-4o-mini) | 0.5970 | 0.5709 | 0.5671 | 0.5752 | 0.5390 | 0.5378 | 0.5258 | 0.5170 | 0.5157 | -0.0799 | 0/30 | 11/30 | 19/30 |
| Context Aware One-Shot (gpt-4.1-mini) | 0.5794 | 0.5742 | 0.5846 | 0.5927 | 0.5610 | 0.5448 | 0.5626 | 0.5563 | 0.5530 | -0.0231 | 1/30 | 21/30 | 8/30 |
| Context Aware One-Shot (gpt-5-mini) | 0.6083 | 0.6352 | 0.6421 | 0.6205 | 0.6174 | 0.6260 | 0.6154 | 0.6206 | 0.6217 | +0.0122 | 1/30 | 20/30 | 9/30 |
| Context Aware One-Shot (gpt-5.4-mini) | 0.5772 | 0.5793 | 0.5992 | 0.5936 | 0.5659 | 0.5862 | 0.5933 | 0.5761 | 0.5763 | -0.0011 | 4/30 | 23/30 | 3/30 |
| Context Aware One-Shot (claude-opus-4-8) | 0.5523 | 0.5681 | 0.6011 | 0.5793 | 0.5627 | 0.5808 | 0.5823 | 0.5865 | 0.5821 | +0.0342 | 6/30 | 24/30 | 0/30 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.5668 | 0.5889 | 0.5692 | 0.5447 | 0.5589 | 0.5356 | 0.5295 | 0.5210 | 0.5154 | -0.0458 | 0/30 | 17/30 | 13/30 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.6061 | 0.5909 | 0.5995 | 0.5703 | 0.5478 | 0.5344 | 0.5279 | 0.5075 | 0.5071 | -0.0986 | 0/30 | 5/30 | 25/30 |
| First-come-first-served greedy | 0.6032 | 0.6091 | 0.6252 | 0.5960 | 0.6205 | 0.6397 | 0.6444 | 0.6592 | 0.6591 | +0.0560 | 15/30 | 15/30 | 0/30 |
| PTIME | 0.5961 | 0.6033 | 0.6092 | 0.5946 | 0.6117 | 0.6327 | 0.6452 | 0.6589 | 0.6593 | +0.0628 | 14/30 | 16/30 | 0/30 |
| Human coach | 0.8057 | 0.8398 | 0.8072 | 0.7934 | 0.8694 | 0.8256 | 0.8292 | 0.8265 | 0.8260 | +0.0207 | 0/30 | 30/30 | 0/30 |
| DQN RL per person | 0.0000 | 0.5606 | 0.5232 | 0.5652 | 0.0000 | 0.4961 | 0.4921 | 0.4878 | 0.4904 | +0.4878 | 30/30 | 0/30 | 0/30 |

## Cross-augmenter weekly trajectory

_Cohort-mean weighted gain per ISO week for every run, with the OLS `slope` over weeks. A positive slope is improvement over time; the RL run should trend up while the non-learning baselines stay near zero._

| Configuration | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 | Wk7 | Wk8 | Δ (last - first) | slope |
|---|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.5405 | 0.5550 | 0.5669 | 0.5311 | 0.5347 | 0.5341 | 0.5146 | 0.5258 | -0.0147 | -0.0048 |
| Context Aware One-Shot (gpt-4o-mini) | 0.5904 | 0.5701 | 0.5606 | 0.5753 | 0.5044 | 0.5342 | 0.5249 | 0.5120 | -0.0783 | -0.0110 |
| Context Aware One-Shot (gpt-4.1-mini) | 0.6067 | 0.5739 | 0.5851 | 0.5822 | 0.5570 | 0.5461 | 0.5605 | 0.5497 | -0.0570 | -0.0072 |
| Context Aware One-Shot (gpt-5-mini) | 0.6211 | 0.6390 | 0.6388 | 0.6145 | 0.6155 | 0.6244 | 0.5699 | 0.6221 | +0.0010 | -0.0045 |
| Context Aware One-Shot (gpt-5.4-mini) | 0.5860 | 0.5752 | 0.5973 | 0.5984 | 0.5493 | 0.5826 | 0.5860 | 0.5766 | -0.0093 | -0.0012 |
| Context Aware One-Shot (claude-opus-4-8) | 0.5574 | 0.5662 | 0.6048 | 0.5834 | 0.5615 | 0.5833 | 0.5823 | 0.5742 | +0.0169 | +0.0013 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.5681 | 0.5786 | 0.5686 | 0.5009 | 0.5557 | 0.5358 | 0.5301 | 0.5121 | -0.0560 | -0.0081 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.6030 | 0.5902 | 0.5922 | 0.5723 | 0.5274 | 0.5322 | 0.5249 | 0.5070 | -0.0960 | -0.0146 |
| First-come-first-served greedy | 0.6044 | 0.6083 | 0.6221 | 0.5970 | 0.6216 | 0.6360 | 0.6469 | 0.6594 | +0.0550 | +0.0077 |
| PTIME | 0.5962 | 0.6034 | 0.6099 | 0.5918 | 0.6136 | 0.6304 | 0.6435 | 0.6591 | +0.0629 | +0.0086 |
| Human coach | 0.8068 | 0.8415 | 0.8109 | 0.8101 | 0.8481 | 0.8235 | 0.8316 | 0.8261 | +0.0193 | +0.0019 |
| DQN RL per person | 0.0161 | 0.5575 | 0.5398 | 0.5566 | 0.2040 | 0.5029 | 0.4933 | 0.4861 | +0.4700 | +0.0298 |

_The per-week figure is a cohort mean; the per-person distribution above is the unit-of-analysis view, so a flat mean alone does not prove that no individual learned._

## Stage cost & latency

| Configuration | Task generation wall | Task generation tokens | Task generation cost | Augmentation wall | Augmentation tokens | Augmentation cost | Augmentation reasoning (tokens / cost) | Evaluation wall | Evaluation tokens | Evaluation cost | Total cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 18574.7s | 643334 | $0.2027 | 6312.0s | 3353805 | $0.6887 | n/a | 52.6s | 0 | $0.0000 | $0.8914 |
| Context Aware One-Shot (gpt-4o-mini) | 23226.2s | 692523 | $0.2101 | 6112.4s | 4092060 | $0.7951 | n/a | 51.8s | 0 | $0.0000 | $1.0053 |
| Context Aware One-Shot (gpt-4.1-mini) | 18354.1s | 685893 | $0.2123 | 7334.7s | 4134142 | $2.1877 | n/a | 49.9s | 0 | $0.0000 | $2.4000 |
| Context Aware One-Shot (gpt-5-mini) | 18230.7s | 682826 | $0.2125 | 37633.8s | 6864686 | $7.2989 | 2873664 / $5.7473 | 56.5s | 0 | $0.0000 | $7.5115 |
| Context Aware One-Shot (gpt-5.4-mini) | 17322.8s | 673066 | $0.2109 | 1852.6s | 4065256 | $4.5192 | n/a | 51.8s | 0 | $0.0000 | $4.7301 |
| Context Aware One-Shot (claude-opus-4-8) | 18169.4s | 667043 | $0.2071 | 4691.4s | 5758651 | $38.3954 | n/a | 51.5s | 0 | $0.0000 | $38.6025 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 17426.0s | 652204 | $0.2049 | 8150.7s | 4066637 | $0.7855 | n/a | 51.1s | 0 | $0.0000 | $0.9904 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 17982.2s | 656319 | $0.5397 | 9701.3s | 4091016 | $0.7923 | n/a | 51.0s | 0 | $0.0000 | $1.3320 |
| First-come-first-served greedy | 16695.5s | 548046 | $0.1693 | 0.8s | 0 | n/a | n/a | 66.5s | 0 | $0.0000 | $0.1693 |
| PTIME | 19934.2s | 691614 | $0.2113 | 13.9s | 0 | n/a | n/a | 65.9s | 0 | $0.0000 | $0.2113 |
| Human coach | 0.0s | 0 | n/a | 3.6s | 0 | n/a | n/a | 72.8s | 0 | $0.0000 | $0.0000 |
| DQN RL per person | 19680.1s | 674228 | $0.2082 | 17771.2s | 0 | n/a | n/a | 40.3s | 0 | $0.0000 | $0.2082 |

## Ontology grounding

| Configuration | Generated | Grounded | Verified | Persons short-fetched | Persons w/ unverified URI |
|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 4850 | 4850 (100.0%) | 4850 (100.0%) | 30 | 0 |
| Context Aware One-Shot (gpt-4o-mini) | 4842 | 4842 (100.0%) | 4842 (100.0%) | 30 | 0 |
| Context Aware One-Shot (gpt-4.1-mini) | 4842 | 4842 (100.0%) | 4842 (100.0%) | 30 | 0 |
| Context Aware One-Shot (gpt-5-mini) | 4748 | 4748 (100.0%) | 4748 (100.0%) | 30 | 0 |
| Context Aware One-Shot (gpt-5.4-mini) | 4738 | 4738 (100.0%) | 4738 (100.0%) | 30 | 0 |
| Context Aware One-Shot (claude-opus-4-8) | 4788 | 4788 (100.0%) | 4788 (100.0%) | 30 | 0 |
| Context Aware One-Shot (eval gpt-4.1-mini) | 4765 | 4765 (100.0%) | 4765 (100.0%) | 30 | 0 |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 4925 | 4925 (100.0%) | 4925 (100.0%) | 30 | 0 |
| First-come-first-served greedy | 4842 | 4842 (100.0%) | 4842 (100.0%) | 30 | 0 |
| PTIME | 4775 | 4775 (100.0%) | 4775 (100.0%) | 30 | 0 |
| Human coach | 4842 | 4842 (100.0%) | 4842 (100.0%) | 30 | 0 |
| DQN RL per person | 4838 | 4838 (100.0%) | 4838 (100.0%) | 30 | 0 |

## Preference breakdown

| Configuration | per_occurrence_duration | per_scale_duration | per_scale_episodes | persona_stage_semantic | temporal_pattern_fix | temporal_pattern_seasonality | temporal_pattern_trend |
|---|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 0.2148 (n=4581) | 0.2278 (n=2258) | 0.0242 (n=2258) | n/a | 0.4144 (n=2838) | 0.2140 (n=1143) | 0.9900 (n=10) |
| Context Aware One-Shot (gpt-4o-mini) | 0.2653 (n=4518) | 0.2674 (n=2044) | 0.0272 (n=2044) | n/a | 0.4176 (n=2972) | 0.2742 (n=1161) | 0.8071 (n=10) |
| Context Aware One-Shot (gpt-4.1-mini) | 0.1771 (n=5043) | 0.1801 (n=2350) | 0.0231 (n=2350) | n/a | 0.4995 (n=3079) | 0.1979 (n=1308) | 0.8100 (n=10) |
| Context Aware One-Shot (gpt-5-mini) | 0.2136 (n=5542) | 0.1863 (n=2944) | 0.0246 (n=2944) | n/a | 0.5153 (n=3944) | 0.1788 (n=1221) | 0.9250 (n=8) |
| Context Aware One-Shot (gpt-5.4-mini) | 0.2272 (n=4685) | 0.2157 (n=2481) | 0.0283 (n=2481) | n/a | 0.4900 (n=3014) | 0.1930 (n=1122) | 0.9929 (n=14) |
| Context Aware One-Shot (claude-opus-4-8) | 0.1550 (n=4397) | 0.1602 (n=2445) | 0.0247 (n=2445) | n/a | 0.5109 (n=2788) | 0.1174 (n=1044) | 0.9200 (n=10) |
| Context Aware One-Shot (eval gpt-4.1-mini) | 0.2677 (n=4338) | 0.2789 (n=2093) | 0.0294 (n=2093) | n/a | 0.4214 (n=2797) | 0.2709 (n=1203) | 1.0000 (n=11) |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 0.3013 (n=6360) | 0.2852 (n=2582) | 0.0250 (n=2582) | n/a | 0.4408 (n=3253) | 0.3011 (n=1563) | 0.7801 (n=9) |
| First-come-first-served greedy | 0.1883 (n=7593) | 0.1613 (n=3684) | 0.0269 (n=3684) | n/a | 0.6144 (n=5102) | 0.0913 (n=1493) | 0.8714 (n=15) |
| PTIME | 0.1692 (n=7369) | 0.1546 (n=3641) | 0.0265 (n=3641) | n/a | 0.5990 (n=4980) | 0.0861 (n=1516) | 0.8944 (n=18) |
| Human coach | 0.2072 (n=14350) | 0.2175 (n=4864) | 0.0210 (n=4864) | n/a | 0.6276 (n=6411) | 0.2984 (n=2305) | 0.8714 (n=15) |
| DQN RL per person | 0.2754 (n=1449) | 0.2714 (n=993) | 0.0205 (n=993) | n/a | 0.7597 (n=1165) | 0.2762 (n=634) | 1.0000 (n=1) |

## Divide breakdown

| Configuration | Buckets | divided_valid | not_divided | divided_invalid_pieces_too_long | divided_invalid_sum_too_low | divided_invalid_sum_too_high |
|---|---|---|---|---|---|---|
| Context Blind One-Shot (gpt-4o-mini) | 1026 | 0 (0.0%) | 691 (67.3%) | 335 (32.7%) | 0 (0.0%) | 0 (0.0%) |
| Context Aware One-Shot (gpt-4o-mini) | 1038 | 0 (0.0%) | 734 (70.7%) | 304 (29.3%) | 0 (0.0%) | 0 (0.0%) |
| Context Aware One-Shot (gpt-4.1-mini) | 1058 | 0 (0.0%) | 703 (66.4%) | 355 (33.6%) | 0 (0.0%) | 0 (0.0%) |
| Context Aware One-Shot (gpt-5-mini) | 1206 | 0 (0.0%) | 1191 (98.8%) | 15 (1.2%) | 0 (0.0%) | 0 (0.0%) |
| Context Aware One-Shot (gpt-5.4-mini) | 954 | 0 (0.0%) | 865 (90.7%) | 89 (9.3%) | 0 (0.0%) | 0 (0.0%) |
| Context Aware One-Shot (claude-opus-4-8) | 1046 | 0 (0.0%) | 960 (91.8%) | 86 (8.2%) | 0 (0.0%) | 0 (0.0%) |
| Context Aware One-Shot (eval gpt-4.1-mini) | 1087 | 0 (0.0%) | 794 (73.0%) | 293 (27.0%) | 0 (0.0%) | 0 (0.0%) |
| Context Aware One-Shot (task-gen gpt-4.1-mini) | 1131 | 0 (0.0%) | 836 (73.9%) | 295 (26.1%) | 0 (0.0%) | 0 (0.0%) |
| First-come-first-served greedy | 1358 | 0 (0.0%) | 1358 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| PTIME | 1357 | 0 (0.0%) | 1357 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Human coach | 1361 | 1359 (99.9%) | 2 (0.1%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DQN RL per person | 991 | 0 (0.0%) | 991 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## Pairwise A/B

_Baseline: `Context Blind One-Shot (gpt-4o-mini)`. Each Δ is `(other - baseline)`; positive = other does better, negative = other does worse._

### `Context Aware One-Shot (gpt-4o-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: no model changes)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (gpt-4o-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.5672 | +0.0050 |
| G_cov | 0.4368 | 0.4229 | -0.0139 |
| G_cal | 0.9853 | 0.9889 | +0.0036 |
| G_pref | 0.7912 | 0.7573 | -0.0340 |
| G_disp | 0.9992 | 0.9993 | +0.0001 |
| G_merge | 0.0040 | 0.0033 | -0.0008 |
| G_spread | 0.6029 | 0.6000 | -0.0029 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.3013 | +0.1112 |
| Total cost | $0.8914 | $1.0053 | +0.1138 |
| Total wall time | 24939.4s | 29390.4s | +4451.0693 |

### `Context Aware One-Shot (gpt-4.1-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (gpt-4.1-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.5963 | +0.0341 |
| G_cov | 0.4368 | 0.4467 | +0.0099 |
| G_cal | 0.9853 | 0.9780 | -0.0073 |
| G_pref | 0.7912 | 0.8074 | +0.0161 |
| G_disp | 0.9992 | 0.9990 | -0.0002 |
| G_merge | 0.0040 | 0.1245 | +0.1205 |
| G_spread | 0.6029 | 0.6964 | +0.0936 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.2700 | +0.0799 |
| Total cost | $0.8914 | $2.4000 | +1.5086 |
| Total wall time | 24939.4s | 25738.7s | +799.3322 |

### `Context Aware One-Shot (gpt-5-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (gpt-5-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.6454 | +0.0833 |
| G_cov | 0.4368 | 0.7304 | +0.2936 |
| G_cal | 0.9853 | 0.9629 | -0.0224 |
| G_pref | 0.7912 | 0.7890 | -0.0023 |
| G_disp | 0.9992 | 0.9990 | -0.0001 |
| G_merge | 0.0040 | 0.1021 | +0.0981 |
| G_spread | 0.6029 | 0.7405 | +0.1376 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.3368 | +0.1466 |
| Total cost | $0.8914 | $7.5115 | +6.6200 |
| Total wall time | 24939.4s | 55921.0s | +30981.6396 |

### `Context Aware One-Shot (gpt-5.4-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (gpt-5.4-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.6033 | +0.0411 |
| G_cov | 0.4368 | 0.5125 | +0.0757 |
| G_cal | 0.9853 | 0.9782 | -0.0072 |
| G_pref | 0.7912 | 0.7810 | -0.0102 |
| G_disp | 0.9992 | 0.9991 | -0.0001 |
| G_merge | 0.0040 | 0.0707 | +0.0666 |
| G_spread | 0.6029 | 0.7561 | +0.1532 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.2709 | +0.0808 |
| Total cost | $0.8914 | $4.7301 | +3.8387 |
| Total wall time | 24939.4s | 19227.2s | -5712.1219 |

### `Context Aware One-Shot (claude-opus-4-8)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (claude-opus-4-8) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.5953 | +0.0332 |
| G_cov | 0.4368 | 0.5379 | +0.1010 |
| G_cal | 0.9853 | 0.9759 | -0.0094 |
| G_pref | 0.7912 | 0.8227 | +0.0315 |
| G_disp | 0.9992 | 0.9990 | -0.0002 |
| G_merge | 0.0040 | 0.0183 | +0.0143 |
| G_spread | 0.6029 | 0.6744 | +0.0715 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.2493 | +0.0592 |
| Total cost | $0.8914 | $38.6025 | +37.7111 |
| Total wall time | 24939.4s | 22912.3s | -2027.0412 |

### `Context Aware One-Shot (eval gpt-4.1-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: evaluator)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (eval gpt-4.1-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.5638 | +0.0017 |
| G_cov | 0.4368 | 0.4260 | -0.0108 |
| G_cal | 0.9853 | 0.9873 | +0.0020 |
| G_pref | 0.7912 | 0.7550 | -0.0363 |
| G_disp | 0.9992 | 0.9993 | +0.0001 |
| G_merge | 0.0040 | 0.0042 | +0.0001 |
| G_spread | 0.6029 | 0.5714 | -0.0314 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.2955 | +0.1054 |
| Total cost | $0.8914 | $0.9904 | +0.0989 |
| Total wall time | 24939.4s | 25627.8s | +688.3885 |

### `Context Aware One-Shot (task-gen gpt-4.1-mini)`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: task_generator)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Context Aware One-Shot (task-gen gpt-4.1-mini) | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.5720 | +0.0099 |
| G_cov | 0.4368 | 0.4316 | -0.0052 |
| G_cal | 0.9853 | 0.9877 | +0.0024 |
| G_pref | 0.7912 | 0.7322 | -0.0590 |
| G_disp | 0.9992 | 0.9992 | +0.0000 |
| G_merge | 0.0040 | 0.0219 | +0.0178 |
| G_spread | 0.6029 | 0.6077 | +0.0049 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.3381 | +0.1479 |
| Total cost | $0.8914 | $1.3320 | +0.4405 |
| Total wall time | 24939.4s | 27734.5s | +2795.1808 |

### `First-come-first-served greedy`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | First-come-first-served greedy | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.6462 | +0.0841 |
| G_cov | 0.4368 | 0.9927 | +0.5559 |
| G_cal | 0.9853 | 0.9763 | -0.0090 |
| G_pref | 0.7912 | 0.7964 | +0.0052 |
| G_disp | 0.9992 | 0.9990 | -0.0001 |
| G_merge | 0.0040 | 0.0000 | -0.0040 |
| G_spread | 0.6029 | 0.5304 | -0.0725 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.2275 | +0.0373 |
| Total cost | $0.8914 | $0.1693 | -0.7221 |
| Total wall time | 24939.4s | 16762.7s | -8176.6195 |

### `PTIME`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | PTIME | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.6410 | +0.0789 |
| G_cov | 0.4368 | 0.9933 | +0.5565 |
| G_cal | 0.9853 | 0.9756 | -0.0097 |
| G_pref | 0.7912 | 0.8083 | +0.0171 |
| G_disp | 0.9992 | 0.9990 | -0.0001 |
| G_merge | 0.0040 | 0.0000 | -0.0040 |
| G_spread | 0.6029 | 0.4911 | -0.1118 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.2003 | +0.0101 |
| Total cost | $0.8914 | $0.2113 | -0.6801 |
| Total wall time | 24939.4s | 20014.0s | -4925.3532 |

### `Human coach`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | Human coach | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.8533 | +0.2912 |
| G_cov | 0.4368 | 0.9948 | +0.5580 |
| G_cal | 0.9853 | 0.9704 | -0.0149 |
| G_pref | 0.7912 | 0.7705 | -0.0207 |
| G_disp | 0.9992 | 0.9990 | -0.0002 |
| G_merge | 0.0040 | 0.4316 | +0.4275 |
| G_spread | 0.6029 | 0.9839 | +0.3811 |
| G_divide | 0.0000 | 0.9991 | +0.9991 |
| G_context | 0.1902 | 0.5649 | +0.3747 |
| Total cost | $0.8914 | $0.0000 | -0.8914 |
| Total wall time | 24939.4s | 76.5s | -24862.8972 |

### `DQN RL per person`  vs  `Context Blind One-Shot (gpt-4o-mini)`  (axis: augmenter)

| Metric | Context Blind One-Shot (gpt-4o-mini) | DQN RL per person | Δ (other - baseline) |
|---|---|---|---|
| Avg total gain | 0.5621 | 0.5214 | -0.0407 |
| G_cov | 0.4368 | 0.2808 | -0.1560 |
| G_cal | 0.9853 | 0.9887 | +0.0034 |
| G_pref | 0.7912 | 0.7173 | -0.0739 |
| G_disp | 0.9992 | 0.9986 | -0.0005 |
| G_merge | 0.0040 | 0.0000 | -0.0040 |
| G_spread | 0.6029 | 0.5907 | -0.0122 |
| G_divide | 0.0000 | 0.0000 | +0.0000 |
| G_context | 0.1902 | 0.1338 | -0.0563 |
| Total cost | $0.8914 | $0.2082 | -0.6832 |
| Total wall time | 24939.4s | 37491.6s | +12552.1947 |

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
| `default_path` | -0.0036 ± 0.0013 | -0.0095 ± 0.0051 | -0.0005 ± 0.0001 | -0.0039 ± 0.0012 | +0.0000 ± 0.0000 | +0.0009 ± 0.0005 | -0.0155 ± 0.0065 | n/a | -0.0005 ± 0.0011 |
| `fit_examples` | -0.0041 ± 0.0013 | -0.0173 ± 0.0051 | -0.0001 ± 0.0001 | +0.0006 ± 0.0012 | -0.0000 ± 0.0000 | -0.0008 ± 0.0005 | -0.0151 ± 0.0065 | n/a | +0.0021 ± 0.0011 |
| `rule_of_thumb_1` | +0.0022 ± 0.0013 | +0.0092 ± 0.0051 | +0.0001 ± 0.0001 | +0.0036 ± 0.0012 | -0.0000 ± 0.0000 | -0.0020 ± 0.0005 | +0.0040 ± 0.0065 | n/a | +0.0013 ± 0.0011 |
| `rule_of_thumb_2` | -0.0001 ± 0.0013 | +0.0001 ± 0.0051 | +0.0001 ± 0.0001 | +0.0029 ± 0.0012 | -0.0000 ± 0.0000 | -0.0027 ± 0.0005 | -0.0009 ± 0.0065 | n/a | -0.0009 ± 0.0011 |
| `rule_of_thumb_3` | -0.0007 ± 0.0013 | -0.0007 ± 0.0051 | +0.0000 ± 0.0001 | -0.0022 ± 0.0012 | -0.0000 ± 0.0000 | -0.0008 ± 0.0005 | -0.0022 ± 0.0065 | n/a | -0.0003 ± 0.0011 |
| `worked_example_standalone` | -0.0031 ± 0.0013 | -0.0092 ± 0.0051 | +0.0001 ± 0.0001 | -0.0032 ± 0.0012 | +0.0000 ± 0.0000 | +0.0002 ± 0.0005 | -0.0127 ± 0.0065 | n/a | -0.0005 ± 0.0011 |
| `worked_example_concurrent` | +0.0007 ± 0.0013 | +0.0031 ± 0.0051 | +0.0000 ± 0.0001 | +0.0011 ± 0.0012 | -0.0000 ± 0.0000 | +0.0003 ± 0.0005 | +0.0005 ± 0.0065 | n/a | +0.0002 ± 0.0011 |
| `worked_example_waking` | +0.0017 ± 0.0013 | +0.0078 ± 0.0051 | -0.0001 ± 0.0001 | -0.0014 ± 0.0012 | +0.0000 ± 0.0000 | -0.0014 ± 0.0005 | +0.0095 ± 0.0065 | n/a | -0.0010 ± 0.0011 |
| `hard_constraints` | -0.0004 ± 0.0013 | -0.0006 ± 0.0051 | +0.0001 ± 0.0001 | -0.0043 ± 0.0012 | +0.0000 ± 0.0000 | +0.0005 ± 0.0005 | +0.0020 ± 0.0065 | n/a | -0.0010 ± 0.0011 |
| `do_not_drop` | +0.0038 ± 0.0013 | +0.0112 ± 0.0051 | -0.0001 ± 0.0001 | +0.0056 ± 0.0012 | +0.0000 ± 0.0000 | -0.0005 ± 0.0005 | +0.0141 ± 0.0065 | n/a | -0.0001 ± 0.0011 |
| `self_verify` | -0.0001 ± 0.0013 | +0.0023 ± 0.0051 | +0.0002 ± 0.0001 | +0.0002 ± 0.0012 | -0.0000 ± 0.0000 | -0.0011 ± 0.0005 | -0.0019 ± 0.0065 | n/a | -0.0019 ± 0.0011 |
| `split_dividable_tasks` | +0.0069 ± 0.0013 | +0.0191 ± 0.0051 | +0.0003 ± 0.0001 | +0.0177 ± 0.0012 | +0.0000 ± 0.0000 | +0.0013 ± 0.0005 | +0.0139 ± 0.0065 | n/a | +0.0009 ± 0.0011 |
| `context_block` | -0.0002 ± 0.0013 | -0.0005 ± 0.0051 | +0.0000 ± 0.0001 | +0.0006 ± 0.0012 | -0.0000 ± 0.0000 | -0.0008 ± 0.0005 | -0.0005 ± 0.0065 | n/a | -0.0009 ± 0.0011 |

#### `Context Aware PB-16 ablation`

| Block | beta(total) | beta(G_cov) | beta(G_cal) | beta(G_pref) | beta(G_disp) | beta(G_merge) | beta(G_spread) | beta(G_divide) | beta(G_context) |
|---|---|---|---|---|---|---|---|---|---|
| `default_path` | -0.0041 ± 0.0003 | -0.0084 ± 0.0031 | -0.0003 ± 0.0001 | -0.0050 ± 0.0017 | +0.0000 ± 0.0000 | -0.0001 ± 0.0008 | -0.0216 ± 0.0012 | -0.0001 ± 0.0001 | +0.0010 ± 0.0026 |
| `fit_examples` | -0.0035 ± 0.0003 | -0.0169 ± 0.0031 | -0.0002 ± 0.0001 | -0.0005 ± 0.0017 | -0.0000 ± 0.0000 | -0.0009 ± 0.0008 | -0.0104 ± 0.0012 | +0.0001 ± 0.0001 | +0.0043 ± 0.0026 |
| `rule_of_thumb_1` | +0.0018 ± 0.0003 | +0.0091 ± 0.0031 | +0.0002 ± 0.0001 | -0.0003 ± 0.0017 | +0.0000 ± 0.0000 | -0.0003 ± 0.0008 | +0.0037 ± 0.0012 | -0.0001 ± 0.0001 | +0.0005 ± 0.0026 |
| `rule_of_thumb_2` | +0.0003 ± 0.0003 | +0.0010 ± 0.0031 | +0.0002 ± 0.0001 | +0.0012 ± 0.0017 | -0.0000 ± 0.0000 | -0.0010 ± 0.0008 | +0.0038 ± 0.0012 | -0.0001 ± 0.0001 | -0.0028 ± 0.0026 |
| `rule_of_thumb_3` | -0.0012 ± 0.0003 | -0.0011 ± 0.0031 | +0.0001 ± 0.0001 | +0.0015 ± 0.0017 | -0.0000 ± 0.0000 | +0.0006 ± 0.0008 | -0.0082 ± 0.0012 | +0.0001 ± 0.0001 | -0.0050 ± 0.0026 |
| `worked_example_standalone` | -0.0013 ± 0.0003 | -0.0078 ± 0.0031 | -0.0000 ± 0.0001 | +0.0015 ± 0.0017 | -0.0000 ± 0.0000 | -0.0013 ± 0.0008 | -0.0038 ± 0.0012 | -0.0001 ± 0.0001 | +0.0027 ± 0.0026 |
| `worked_example_concurrent` | +0.0016 ± 0.0003 | +0.0008 ± 0.0031 | +0.0002 ± 0.0001 | +0.0057 ± 0.0017 | -0.0000 ± 0.0000 | +0.0030 ± 0.0008 | +0.0005 ± 0.0012 | +0.0001 ± 0.0001 | +0.0035 ± 0.0026 |
| `worked_example_waking` | +0.0005 ± 0.0003 | +0.0061 ± 0.0031 | -0.0000 ± 0.0001 | -0.0013 ± 0.0017 | +0.0000 ± 0.0000 | +0.0006 ± 0.0008 | +0.0016 ± 0.0012 | -0.0001 ± 0.0001 | -0.0051 ± 0.0026 |
| `hard_constraints` | -0.0008 ± 0.0003 | -0.0022 ± 0.0031 | -0.0002 ± 0.0001 | -0.0050 ± 0.0017 | -0.0000 ± 0.0000 | +0.0014 ± 0.0008 | +0.0059 ± 0.0012 | +0.0001 ± 0.0001 | -0.0056 ± 0.0026 |
| `do_not_drop` | +0.0055 ± 0.0003 | +0.0185 ± 0.0031 | -0.0002 ± 0.0001 | +0.0035 ± 0.0017 | +0.0000 ± 0.0000 | -0.0016 ± 0.0008 | +0.0228 ± 0.0012 | -0.0001 ± 0.0001 | +0.0007 ± 0.0026 |
| `self_verify` | -0.0003 ± 0.0003 | -0.0014 ± 0.0031 | -0.0000 ± 0.0001 | +0.0022 ± 0.0017 | -0.0000 ± 0.0000 | -0.0014 ± 0.0008 | +0.0005 ± 0.0012 | +0.0001 ± 0.0001 | -0.0026 ± 0.0026 |
| `split_dividable_tasks` | +0.0071 ± 0.0003 | +0.0180 ± 0.0031 | +0.0004 ± 0.0001 | +0.0192 ± 0.0017 | -0.0000 ± 0.0000 | -0.0006 ± 0.0008 | +0.0141 ± 0.0012 | +0.0001 ± 0.0001 | +0.0050 ± 0.0026 |
| `context_block` | -0.0028 ± 0.0003 | -0.0176 ± 0.0031 | +0.0010 ± 0.0001 | -0.0107 ± 0.0017 | +0.0000 ± 0.0000 | -0.0016 ± 0.0008 | -0.0083 ± 0.0012 | -0.0001 ± 0.0001 | +0.0216 ± 0.0026 |

### Per-block cost effects

#### `Context Blind PB-16 ablation`

| Block | beta(augment tokens) | beta(augment wall, s) | beta(augment USD) |
|---|---|---|---|
| `default_path` | +37173.3750 | +153.7575 | +0.0072 |
| `fit_examples` | +46712.5000 | -93.2828 | +0.0046 |
| `rule_of_thumb_1` | +11737.2500 | +263.0734 | +0.0029 |
| `rule_of_thumb_2` | +16985.7500 | +131.3305 | +0.0020 |
| `rule_of_thumb_3` | +38469.5000 | +74.1696 | +0.0049 |
| `worked_example_standalone` | +5708.1250 | +269.2267 | -0.0007 |
| `worked_example_concurrent` | +17002.8750 | +129.8800 | +0.0031 |
| `worked_example_waking` | +10276.8750 | +121.5011 | +0.0031 |
| `hard_constraints` | +17899.8750 | +128.5230 | +0.0015 |
| `do_not_drop` | +18391.2500 | +376.2445 | +0.0060 |
| `self_verify` | +22582.2500 | -55.3205 | +0.0019 |
| `split_dividable_tasks` | +96955.0000 | +469.5340 | +0.0225 |
| `context_block` | -3057.5000 | +86.8535 | -0.0004 |

#### `Context Aware PB-16 ablation`

| Block | beta(augment tokens) | beta(augment wall, s) | beta(augment USD) |
|---|---|---|---|
| `default_path` | +38552.1875 | +119.5015 | +0.0080 |
| `fit_examples` | +45647.4375 | +22.3311 | +0.0040 |
| `rule_of_thumb_1` | +11721.3125 | +155.4665 | +0.0029 |
| `rule_of_thumb_2` | +16131.1875 | +69.8959 | +0.0015 |
| `rule_of_thumb_3` | +38579.5625 | -101.7185 | +0.0050 |
| `worked_example_standalone` | +5494.3125 | -89.2569 | -0.0008 |
| `worked_example_concurrent` | +17282.9375 | -132.9750 | +0.0032 |
| `worked_example_waking` | +10685.5625 | +165.6919 | +0.0034 |
| `hard_constraints` | +17297.1875 | +140.8980 | +0.0011 |
| `do_not_drop` | +20366.6875 | +395.7648 | +0.0071 |
| `self_verify` | +22143.5625 | -36.9494 | +0.0017 |
| `split_dividable_tasks` | +97617.6875 | +441.1601 | +0.0229 |
| `context_block` | +331015.0625 | +6.5649 | +0.0493 |

<details><summary>Variant detail: <code>Context Blind PB-16 ablation</code></summary>

| Variant id | Ablated blocks | Avg total gain | Augment tokens | Augment wall (s) | Augment USD |
|---|---|---|---|---|---|
| `0000000000000` | (baseline) | 0.5667 | 3351200 | 7949.2s | $0.6876 |
| `1010101010101` | context_block, default_path, hard_constraints, rule_of_thumb_1, rule_of_thumb_3, self_verify, worked_example_concurrent | 0.5697 | 3067705 | 6342.4s | $0.6455 |
| `0110011001100` | do_not_drop, fit_examples, rule_of_thumb_1, self_verify, worked_example_concurrent, worked_example_standalone | 0.5633 | 3105125 | 5862.4s | $0.6509 |
| `1100110011001` | context_block, default_path, do_not_drop, fit_examples, hard_constraints, rule_of_thumb_3, worked_example_standalone | 0.5801 | 3026679 | 5695.9s | $0.6402 |
| `0001111000011` | context_block, rule_of_thumb_2, rule_of_thumb_3, split_dividable_tasks, worked_example_concurrent, worked_example_standalone | 0.5554 | 3005266 | 5320.0s | $0.6238 |
| `1011010010110` | default_path, hard_constraints, rule_of_thumb_1, rule_of_thumb_2, self_verify, split_dividable_tasks, worked_example_standalone | 0.5599 | 2931190 | 4966.7s | $0.6118 |
| `0111100001111` | context_block, do_not_drop, fit_examples, rule_of_thumb_1, rule_of_thumb_2, rule_of_thumb_3, self_verify, split_dividable_tasks | 0.5515 | 2853648 | 5444.0s | $0.5988 |
| `1101001011010` | default_path, do_not_drop, fit_examples, hard_constraints, rule_of_thumb_2, split_dividable_tasks, worked_example_concurrent | 0.5592 | 2849079 | 5312.3s | $0.5940 |
| `0000000111111` | context_block, do_not_drop, hard_constraints, self_verify, split_dividable_tasks, worked_example_waking | 0.5389 | 3023298 | 5387.4s | $0.6173 |
| `1010101101010` | default_path, do_not_drop, rule_of_thumb_1, rule_of_thumb_3, split_dividable_tasks, worked_example_concurrent, worked_example_waking | 0.5411 | 2889261 | 4510.6s | $0.5871 |
| `0110011110011` | context_block, fit_examples, hard_constraints, rule_of_thumb_1, split_dividable_tasks, worked_example_concurrent, worked_example_standalone, worked_example_waking | 0.5594 | 2944730 | 5198.5s | $0.6143 |
| `1100110100110` | default_path, fit_examples, rule_of_thumb_3, self_verify, split_dividable_tasks, worked_example_standalone, worked_example_waking | 0.5715 | 2835565 | 6025.1s | $0.6004 |
| `0001111111100` | do_not_drop, hard_constraints, rule_of_thumb_2, rule_of_thumb_3, self_verify, worked_example_concurrent, worked_example_standalone, worked_example_waking | 0.5631 | 3056567 | 5598.1s | $0.6439 |
| `1011010101001` | context_block, default_path, do_not_drop, rule_of_thumb_1, rule_of_thumb_2, worked_example_standalone, worked_example_waking | 0.5641 | 3156890 | 5100.3s | $0.6476 |
| `0111100110000` | fit_examples, hard_constraints, rule_of_thumb_1, rule_of_thumb_2, rule_of_thumb_3, worked_example_waking | 0.5650 | 3065230 | 6391.4s | $0.6483 |
| `1101001100101` | context_block, default_path, fit_examples, rule_of_thumb_2, self_verify, worked_example_concurrent, worked_example_waking | 0.5746 | 3053921 | 6737.5s | $0.6434 |

</details>

<details><summary>Variant detail: <code>Context Aware PB-16 ablation</code></summary>

| Variant id | Ablated blocks | Avg total gain | Augment tokens | Augment wall (s) | Augment USD |
|---|---|---|---|---|---|
| `0000000000000` | (baseline) | 0.5689 | 4088251 | 6520.9s | $0.7933 |
| `1010101010101` | context_block, default_path, hard_constraints, rule_of_thumb_1, rule_of_thumb_3, self_verify, worked_example_concurrent | 0.5809 | 3135104 | 6410.3s | $0.6508 |
| `0110011001100` | do_not_drop, fit_examples, rule_of_thumb_1, self_verify, worked_example_concurrent, worked_example_standalone | 0.5612 | 3840002 | 5896.4s | $0.7553 |
| `1100110011001` | context_block, default_path, do_not_drop, fit_examples, hard_constraints, rule_of_thumb_3, worked_example_standalone | 0.5846 | 3091373 | 5346.0s | $0.6439 |
| `0001111000011` | context_block, rule_of_thumb_2, rule_of_thumb_3, split_dividable_tasks, worked_example_concurrent, worked_example_standalone | 0.5614 | 3073073 | 6137.8s | $0.6294 |
| `1011010010110` | default_path, hard_constraints, rule_of_thumb_1, rule_of_thumb_2, self_verify, split_dividable_tasks, worked_example_standalone | 0.5628 | 3667363 | 4732.7s | $0.7170 |
| `0111100001111` | context_block, do_not_drop, fit_examples, rule_of_thumb_1, rule_of_thumb_2, rule_of_thumb_3, self_verify, split_dividable_tasks | 0.5551 | 2921806 | 4615.9s | $0.6046 |
| `1101001011010` | default_path, do_not_drop, fit_examples, hard_constraints, rule_of_thumb_2, split_dividable_tasks, worked_example_concurrent | 0.5571 | 3582497 | 4598.7s | $0.6975 |
| `0000000111111` | context_block, do_not_drop, hard_constraints, self_verify, split_dividable_tasks, worked_example_waking | 0.5503 | 3087063 | 4298.9s | $0.6205 |
| `1010101101010` | default_path, do_not_drop, rule_of_thumb_1, rule_of_thumb_3, split_dividable_tasks, worked_example_concurrent, worked_example_waking | 0.5456 | 3615666 | 4248.4s | $0.6864 |
| `0110011110011` | context_block, fit_examples, hard_constraints, rule_of_thumb_1, split_dividable_tasks, worked_example_concurrent, worked_example_standalone, worked_example_waking | 0.5636 | 3014728 | 5101.1s | $0.6212 |
| `1100110100110` | default_path, fit_examples, rule_of_thumb_3, self_verify, split_dividable_tasks, worked_example_standalone, worked_example_waking | 0.5751 | 3570847 | 5670.4s | $0.7050 |
| `0001111111100` | do_not_drop, hard_constraints, rule_of_thumb_2, rule_of_thumb_3, self_verify, worked_example_concurrent, worked_example_standalone, worked_example_waking | 0.5604 | 3792289 | 5698.2s | $0.7488 |
| `1011010101001` | context_block, default_path, do_not_drop, rule_of_thumb_1, rule_of_thumb_2, worked_example_standalone, worked_example_waking | 0.5695 | 3220355 | 5064.6s | $0.6506 |
| `0111100110000` | fit_examples, hard_constraints, rule_of_thumb_1, rule_of_thumb_2, rule_of_thumb_3, worked_example_waking | 0.5743 | 3805190 | 5620.0s | $0.7558 |
| `1101001100101` | context_block, default_path, fit_examples, rule_of_thumb_2, self_verify, worked_example_concurrent, worked_example_waking | 0.5847 | 3122362 | 5906.0s | $0.6493 |

</details>
