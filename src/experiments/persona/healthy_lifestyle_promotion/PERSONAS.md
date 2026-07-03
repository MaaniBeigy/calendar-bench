# Personas: side-by-side comparison

Three personas defined in [`persona_config.yaml`](persona_config.yaml), each sampled
into `instances: 10` persons. Events come from the catalog in
[`event_config.yaml`](event_config.yaml); temporal relations and rules are defined in
`temporal_relation_rules.yaml`.

| | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` |
| --- | --- | --- | --- |
| Archetype | conscientious, anxious, active adult | overwhelmed at-home family caregiver | free-spirited, social young adult |
| Health orientation | high (action stage) | low (constrained / exhausted) | low (spontaneous / reckless) |
| Motivation | intrinsic + planned | external / obligation | low structure, external cues |
| Health events (gym/run/etc.) | yes | none | none |

Legend for the tables below: `–` means the persona does not have that event or context
member. A condition that applies to only some instances is shown in parentheses, for
example `(part-time job, ~20%)`.

---

## 1. Demographics & characteristics

| Feature | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` |
| --- | --- | --- | --- |
| occupation_status | fulltime 100% | unemployed 90%, parttime 10% | student 100% |
| age | norm(40, 9), clip 22-65 | norm(46, 8), clip 30-62 | norm(21, 3), clip 18-29 |
| gender (F/M/NB) | 50 / 40 / 10 | 48 / 44 / 8 | 48 / 44 / 8 |
| socioeconomic (low/mid/high) | 30 / 50 / 20 | 60 / 35 / 5 | 50 / 40 / 10 |
| has_kids | 30% | 60% | 3% |
| has_part_time_job | 0% | 0% | 20% |
| annual income (EUR) | lognorm, median €38,000 | lognorm, median €22,000 | lognorm, median €12,000 |
| weekly_workouts | Poisson(3) | Poisson(1) | Poisson(1) |

### Big Five (loc ± sd)

| Trait | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` |
| --- | --- | --- | --- |
| openness | 0.50 ± 0.18 | 0.40 ± 0.15 | **0.68 ± 0.15** |
| conscientiousness | **0.70 ± 0.05** | 0.45 ± 0.15 | 0.32 ± 0.14 |
| extraversion | 0.50 ± 0.18 | 0.45 ± 0.15 | **0.70 ± 0.14** |
| agreeableness | 0.50 ± 0.18 | **0.68 ± 0.15** | 0.55 ± 0.16 |
| neuroticism | **0.65 ± 0.18** | 0.62 ± 0.16 | 0.45 ± 0.16 |

### Behaviour-change conditions (share of instances meeting each trait threshold)

Shares are the fraction of each persona's instances above the trait threshold.

| Trait threshold → conditional contexts | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` |
| --- | --- | --- | --- |
| conscientiousness ≥0.70 → action_planning, self-efficacy | ~50% | ~5% | ~0% |
| neuroticism ≥0.60 → anxious | ~61% | ~55% | ~17% |
| neuroticism ≥0.55 → avoidance, low self-efficacy | ~71% | ~67% | ~27% |
| agreeableness ≥0.70 → coping planning | ~13% | ~45% | ~17% |
| openness ≥0.70 → need for autonomy | ~13% | ~2% | ~45% |

---

## 2. Daily anchors (sleep and meals)

| Event | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` |
| --- | --- | --- | --- |
| sleep | from 22:30 | from 23:00 | from 23:30 (5-7h) |
| breakfast | 07:00, 15m | 06:30, 15m | 08:00, 15m (0-1×/day) |
| lunch | 13:00, 30m | 12:30, 20m | 13:30, 30m |
| dinner | 19:30, 45m | 18:00, 40m | 20:30, 45m |

---

## 3. Events / stages (`time [days]`)

All three have `drink_water`, `snacking`, `napping` (flexible, daily) and `groceries`.

| Event | Category | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` |
| --- | --- | --- | --- | --- |
| office_work | work | 08:30 [Mon-Fri] | 09:00 [Mon-Fri] (part-time only) | morning [Mon/Wed/Fri] (part-time job, ~20%) |
| meeting | work | morning 120m [Mon-Fri] | – | – |
| studying | education | – | – | 10:00 [Mon-Fri] |
| childcare | caregiving | – | 07:00 [daily] (with children) | – |
| eldercare | caregiving | – | afternoon [daily] (without children) | – |
| cooking | chores | 17:30 30m [Mon/Wed/Sun] | 17:00 [daily] | 19:30 20m [Mon-Fri] |
| cleaning | chores | 11:00 90m [Sat] | 10:00 [daily] | 14:00 60m [Sat] |
| laundry | chores | 09:00 120m [Sun] | 09:00 120m [Thu/Sun] | 12:00 90m [Sun] |
| groceries | chores | flex [Wed/Sat] | flex [Mon/Wed/Fri/Sat] | flex [Sat] |
| study_reading | education | – | – | afternoon [daily] |
| shower | wellbeing | 21:30 15m [daily] | 21:30 15m [daily] | 21:30 15m [daily] |
| journaling | wellbeing | 21:00 30m [daily] | – | – |
| cycling | sports | 07:45 & 17:15, 15m [Mon-Fri] | – | – |
| walking | sports | m/a/e 20-60m [daily] | morning 20m [Mon/Wed/Sat] | evening 20m [Mon/Wed/Fri] |
| running | sports | morning 30m [Tue/Thu] | – | – |
| gym | sports | evening 90m [Tue/Thu] | – | – |
| swimming | sports | afternoon 60m [Sat] | – | – |
| socialising | social | afternoon [Sat/Sun] | – | evening [daily] |
| going_out | social | 21:00 [Fri] | – | 22:00 [Sat] |
| visit_friends_or_family | social | 18:30 120m [Sat] | – | 16:00 120m [Sun] |
| gaming | leisure | – | – | night [daily] |
| reading | leisure | 22:30 30m [daily] | 22:30 30m [daily] | – |

---

## 4. Event overrides (episodes, duration, patterns)

| Event | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` |
| --- | --- | --- | --- |
| office_work | 1-3×/day, 6-9h | 1-2×/day, 1-6h | 1×/day, 90-150m; morning (part-time job, ~20%) |
| meeting | 2-4×/day, 150-360m; morning/afternoon | – | – |
| studying | – | – | 2-3×/day, 240-420m; morning/afternoon |
| study_reading | – | – | 1-2×/day, 120-240m; morning/afternoon |
| childcare | – | 3-5×/day, 330-540m; all daytime windows (with children) | – |
| eldercare | – | 2-4×/day, 300-480m; all daytime windows (without children) | – |
| cooking | – | 2-3×/day, 120-240m | 0-1×/day, 10-20m; weekday |
| cleaning | – | 1-2×/day, 90-180m | – |
| walking | 1×/day, 20-60m; morning/afternoon/evening | 0-1×/day, 30-60m | 0-2×/day, 5-30m; evening |
| gym | 60-240m; +30m Sat, +30m/week | – | – |
| running | 0-1×/day, 0-45m | – | – |
| swimming | 1×/day, 90-150m | – | – |
| socialising | 1-2×/day, 90-180m; afternoon | – | 2-3×/day, 150-300m; afternoon/evening |
| going_out | 0-1×/week, 45-120m; evening/night | – | 1×/week, 60-120m; evening/night |
| gaming | – | – | 1-2×/day, 90-180m; evening/night |
| breakfast | – | – | 0-1×/day, 0-30m |
| snacking | 1-3×/day, 5-60m | 1-4×/day, 15-90m | 0-6×/day, 5-120m |
| journaling | 0-1×/day, 5-90m | – | – |

---

## 5. Contexts

Each cell reads as `episodes per day · total daily duration`; for example
`0-3× · 30-180m` is 0-3 episodes per day totalling 30-180 minutes (`h` denotes hours).
Bold marks the persona for whom a member is most pronounced. The Condition column states
the trait threshold or flag under which a persona has that member.

### Mood and energy

| Member | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` | Condition |
| --- | --- | --- | --- | --- |
| happy | 0-3× · 30-180m | 0-2× · 15-90m | **2-4× · 60-240m** | – |
| calm | 2-3× · 30-90m | 1-2× · 20-60m | 1-2× · 30-90m | – |
| anxious | 0-2× · 0-90m | – | – | neuroticism ≥0.60 |
| energetic | 3-4× · 270-720m | 1-2× · 60-240m | 3-4× · 240-720m | – |
| tired | 2× · 5-60m | **3-4× · 60-240m** | 1-2× · 30-120m | – |

### Physiology, stress

| Member | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` | Condition |
| --- | --- | --- | --- | --- |
| hungry | 1-10× · 30-180m | 1-8× · 30-150m | 1-8× · 30-180m | – |
| stress_episode | 0-2× · 0-90m | **1-3× · 30-180m** | 0-2× · 0-90m | – |

### Location, social, weather

| Member | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` | Condition |
| --- | --- | --- | --- | --- |
| home | 3-6× · 8-14h | **3-6× · 10-16h** | 3-6× · 6-12h | – |
| workplace | 0-1× · 0-8h | 0-1× · 0-5h | – | `unemployed_family_caregiver`: part-time |
| public_park | 0-1× · 0-90m | 0-1× · 0-60m | 1-2× · 0-180m | – |
| alone | 1-3× · 2-8h | 1-2× · 1-5h | 1-3× · 1-6h | – |
| with_family | 0-3× · 0-6h | **1-4× · 2-10h** | 1-4× · 1-8h | – |
| suitable_for_outdoor | 0-3× · 0-6h | 0-3× · 0-6h | 0-3× · 0-6h | – |
| raining_or_snowing | 0-3× · 0-4h | 0-3× · 0-4h | 0-3× · 0-4h | – |

### Behaviour state (COM-B / HAPA)

| Member | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` | Condition |
| --- | --- | --- | --- | --- |
| action_planning | 0-2× · 10-60m | – | – | conscientiousness ≥0.70 |
| intention_to_enact_a_behaviour | 2-3× · 20-120m | – | – | – |
| self_efficacy_belief_for_a_behaviour | 3× · 90-240m | – | – | conscientiousness ≥0.70 |
| habitual_behaviour | 2-3× · 60-240m | 2-4× · 60-300m | 1-2× · 30-120m | – |
| intention_not_to_enact_a_behaviour | 0-2× · 0-60m | 0-3× · 0-120m | 0-3× · 0-120m | neuroticism ≥0.55 |
| coping_planning | 0-2× · 0-40m | 0-2× · 0-60m | – | agreeableness ≥0.70 |

### Capability / opportunity

| Member | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` | Condition |
| --- | --- | --- | --- | --- |
| physical_behavioural_opportunity | 2-4× · 240-540m | 2-3× · 120-360m | 3-4× · 240-600m | – |
| social_behavioural_opportunity | 2-3× · 120-360m | 1-2× · 60-240m | **3-4× · 240-540m** | – |
| psychological_behavioural_capability | 2-3× · 120-300m | 1-2× · 60-240m | 1-2× · 60-240m | – |
| low_physical_opportunity_to_enact_a_behaviour | 0-2× · 0-4h | **2-4× · 2-4h** | – | – |
| low_self_efficacy | 0-2× · 0-90m | 0-3× · 0-120m | – | neuroticism ≥0.55 |
| need_for_autonomy | 0-2× · 0-90m | – | 0-3× · 0-120m | openness ≥0.70 |

### Goals and motivation (SDT)

| Member | `diligent_anxious_active` | `unemployed_family_caregiver` | `free_spirited_social_student` | Condition |
| --- | --- | --- | --- | --- |
| intend_to_walk | 0-2× · 0-60m | – | – | – |
| disposition_to_attend_to_one_s_goals | 0-2× · 0-60m | – | – | – |
| intrinsic_motivation | 0-3× · 0-4h | 0-1× · 0-1h | 0-2× · 0-2h | – |
| external_regulation | 0-2× · 0-3h | **1-3× · 0-4h** | 1-2× · 0-3h | – |

---

## 6. Distinct characteristics of personas

- `diligent_anxious_active` is the only health-oriented persona: it includes the full sports set
  (gym, running, cycling, swimming, walking) and the deliberate planning contexts
  (`action_planning`, `intention_to_enact_a_behaviour`,
  `self_efficacy_belief_for_a_behaviour`, `journaling`). `action_planning` and
  `self_efficacy_belief_for_a_behaviour` require conscientiousness ≥0.70, which about half of
  its instances have. Their motivation is intrinsic and goal-directed.
- `unemployed_family_caregiver` is defined by load and constraint: highest `tired`, `stress_episode`,
  `home`, `with_family`, and `low_physical_opportunity_to_enact_a_behaviour`, with
  `external_regulation` dominant. Care work is mutually exclusive by `has_kids`:
  the 60% with children do `childcare`, the 40% without do `eldercare`. No gym or
  running, and no planning contexts.
- `free_spirited_social_student` is defined by freedom and sociability: highest `happy` and
  `social_behavioural_opportunity`, high `energetic`, plus `need_for_autonomy` (openness ≥0.70, ~45%). Late
  sleep, skippable breakfast, daily `socialising`, weekend `going_out`, and `gaming`.
  All are students; the ~20% who also hold a part-time job add a morning `office_work`
  shift on Mon/Wed/Fri, which lowers their free time. Low conscientiousness means the
  planning contexts effectively never occur.

`unemployed_family_caregiver` and `free_spirited_social_student` omit gym/running/journaling
and are not full-time, so the health rules (for example `F(gym AND habitual_behaviour)`
and `weekly_gym_target_fulltime`) do not apply to them and generation stays feasible.
