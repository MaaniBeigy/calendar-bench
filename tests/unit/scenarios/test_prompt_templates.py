"""Unit tests for src.scripts.scenarios.task_generation.prompt_templates.

Coverage targets:
  _render_expr:
    - simple variable substitution
    - join filter with various separators
    - unknown variable to empty string

  _render:
    - all {{ }} placeholders substituted
    - no placeholders to text returned as-is

  render:
    - each named template renders without error
    - unknown template name to raises ValueError
    - substituted values appear in output

  list_templates:
    - returns sorted list of all registered names
"""

from __future__ import annotations

import pytest

from src.scripts.scenarios.task_generation.prompt_templates import (
    _render,
    _render_expr,
    list_templates,
    render,
)

# ---------------------------------------------------------------------------
# _render_expr
# ---------------------------------------------------------------------------


class TestRenderExpr:
    def test_simple_variable(self):
        assert _render_expr("name", {"name": "Alice"}) == "Alice"

    def test_unknown_variable_gives_empty_string(self):
        assert _render_expr("missing", {}) == ""

    def test_join_filter_comma_sep(self):
        result = _render_expr('items | join(", ")', {"items": ["a", "b", "c"]})
        assert result == "a, b, c"

    def test_join_filter_and_sep(self):
        result = _render_expr('domains | join(" and ")', {"domains": ["X", "Y"]})
        assert result == "X and Y"

    def test_join_filter_or_sep(self):
        result = _render_expr('difficulty | join(" or ")', {"difficulty": ["L1", "L2"]})
        assert result == "L1 or L2"

    def test_join_filter_single_item(self):
        result = _render_expr('items | join(", ")', {"items": ["only"]})
        assert result == "only"

    def test_join_filter_empty_list(self):
        result = _render_expr('items | join(", ")', {"items": []})
        assert result == ""

    def test_non_list_value_in_join(self):
        """A scalar value in a join expression should be converted to string."""
        result = _render_expr('val | join(", ")', {"val": "scalar"})
        assert result == "scalar"

    def test_integer_variable_stringified(self):
        assert _render_expr("count", {"count": 5}) == "5"


# ---------------------------------------------------------------------------
# _render (internal renderer)
# ---------------------------------------------------------------------------


class TestRenderInternal:
    def test_no_placeholders_returned_unchanged(self):
        template = "Hello, world."
        assert _render(template, {}) == template

    def test_single_variable_substituted(self):
        result = _render("Hello, {{ name }}!", {"name": "Bob"})
        assert result == "Hello, Bob!"

    def test_multiple_variables_substituted(self):
        result = _render("{{ a }} and {{ b }}", {"a": "foo", "b": "bar"})
        assert result == "foo and bar"

    def test_join_filter_in_template(self):
        result = _render(
            "Tasks: {{ items | join(', ') }}",
            {"items": ["yoga", "running"]},
        )
        assert result == "Tasks: yoga, running"

    def test_whitespace_around_expr_ignored(self):
        result = _render("{{  name  }}", {"name": "test"})
        assert result == "test"


# ---------------------------------------------------------------------------
# render (public API)
# ---------------------------------------------------------------------------


class TestRender:
    def test_unknown_template_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown template"):
            render("nonexistent_template")

    def test_error_message_lists_available_templates(self):
        with pytest.raises(ValueError) as exc_info:
            render("bad_name")
        assert "available" in str(exc_info.value).lower()

    # --- health_improvement ---

    def test_health_improvement_renders(self):
        result = render(
            "health_improvement",
            num_tasks=5,
            domains=["PhysicalActivityTask", "NutritionTask"],
            difficulty=["Level2", "Level3"],
            profile_block="  - occupation_status: fulltime\n  - age: 43",
            existing_event_labels=["sleep", "office_work"],
            scenario_description="Increase physical activity.",
        )
        assert "5" in result
        assert "PhysicalActivityTask" in result
        assert "NutritionTask" in result
        assert "Level2" in result
        assert "occupation_status: fulltime" in result
        assert "age: 43" in result
        assert "sleep" in result
        assert "Increase physical activity." in result

    def test_generation_templates_render_full_profile_block(self):
        """Every generation template carries the multi-axis profile block."""
        block = (
            "  - occupation_status: fulltime\n"
            "  - gender: female\n"
            "  - has_kids: true\n"
            "  - neuroticism: 0.39"
        )
        for name in _GENERATION_TEMPLATE_NAMES:
            result = render(
                name,
                num_tasks=3,
                domains=["PhysicalActivityTask"],
                difficulty=["Level2"],
                profile_block=block,
                existing_event_labels=["sleep"],
                scenario_description="Goal.",
            )
            assert "occupation_status: fulltime" in result, name
            assert "gender: female" in result, name
            assert "has_kids: true" in result, name
            assert "neuroticism: 0.39" in result, name

    def test_health_improvement_requests_display_name(self):
        result = render(
            "health_improvement",
            num_tasks=3,
            domains=["PhysicalActivityTask"],
            difficulty=["Level1"],
            profile_block="  - occupation_status: fulltime",
            existing_event_labels=[],
            scenario_description="Goal.",
        )
        assert "display_name" in result

    def test_health_improvement_requests_description_with_citation(self):
        result = render(
            "health_improvement",
            num_tasks=3,
            domains=["PhysicalActivityTask"],
            difficulty=["Level1"],
            profile_block="  - occupation_status: fulltime",
            existing_event_labels=[],
            scenario_description="Goal.",
        )
        assert "description" in result
        assert "citation" in result.lower() or "[" in result

    def test_health_improvement_domains_joined(self):
        result = render(
            "health_improvement",
            num_tasks=3,
            domains=["PhysicalActivityTask", "NutritionTask"],
            difficulty=["Level2"],
            profile_block="  - occupation_status: student",
            existing_event_labels=[],
            scenario_description="Goal.",
        )
        assert "PhysicalActivityTask and NutritionTask" in result

    def test_health_improvement_difficulty_joined_with_or(self):
        result = render(
            "health_improvement",
            num_tasks=3,
            domains=["PhysicalActivityTask"],
            difficulty=["Level2", "Level3"],
            profile_block="  - occupation_status: student",
            existing_event_labels=[],
            scenario_description="Goal.",
        )
        assert "Level2 or Level3" in result

    # --- senior_lifestyle ---

    def test_senior_lifestyle_requests_display_name_and_description(self):
        result = render(
            "senior_lifestyle",
            num_tasks=3,
            domains=["NutritionTask"],
            difficulty=["Level1"],
            profile_block="  - occupation_status: retired",
            existing_event_labels=[],
            scenario_description="Goal.",
        )
        assert "display_name" in result
        assert "description" in result

    def test_senior_lifestyle_renders(self):
        result = render(
            "senior_lifestyle",
            num_tasks=3,
            domains=["NutritionTask"],
            difficulty=["Level1"],
            profile_block="  - occupation_status: retired",
            existing_event_labels=["lunch", "dinner"],
            scenario_description="Improve nutrition.",
        )
        assert "3" in result
        assert "NutritionTask" in result
        assert "retired" in result

    # --- productivity ---

    def test_productivity_requests_display_name_and_description(self):
        result = render(
            "productivity",
            num_tasks=3,
            domains=["MentalWellbeingTask"],
            difficulty=["Level2"],
            profile_block="  - occupation_status: fulltime",
            existing_event_labels=[],
            scenario_description="Goal.",
        )
        assert "display_name" in result
        assert "description" in result

    def test_productivity_renders(self):
        result = render(
            "productivity",
            num_tasks=4,
            domains=["MentalWellbeingTask"],
            difficulty=["Level2"],
            profile_block="  - occupation_status: fulltime",
            existing_event_labels=["office_work"],
            scenario_description="Better focus.",
        )
        assert "4" in result
        assert "MentalWellbeingTask" in result

    # --- augment_agent ---

    def test_augment_agent_renders(self):
        result = render(
            "augment_agent",
            wake_start="07:00",
            sleep_start="23:00",
        )
        assert "07:00" in result
        assert "23:00" in result
        assert "task_label" in result
        assert "concurrent_flag" in result
        assert "concurrent_with" in result

    def test_augment_agent_contains_skip_instruction(self):
        result = render(
            "augment_agent",
            wake_start="06:00",
            sleep_start="22:00",
        )
        assert "skip" in result.lower()

    # --- semantic_merge_score ---

    def test_semantic_merge_score_renders(self):
        result = render(
            "semantic_merge_score",
            label_a="mindful_eating",
            label_b="lunch",
        )
        assert "mindful_eating" in result
        assert "lunch" in result
        assert "0.0" in result
        assert "1.0" in result
        assert "score" in result

    def test_semantic_merge_score_json_format_mentioned(self):
        result = render(
            "semantic_merge_score",
            label_a="walking",
            label_b="podcast",
        )
        assert "reason" in result


# ---------------------------------------------------------------------------
# list_templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    def test_returns_list(self):
        assert isinstance(list_templates(), list)

    def test_contains_all_expected_names(self):
        names = set(list_templates())
        assert "health_improvement" in names
        assert "senior_lifestyle" in names
        assert "productivity" in names
        assert "augment_agent" in names
        assert "semantic_merge_score" in names

    def test_returns_sorted_order(self):
        names = list_templates()
        assert names == sorted(names)

    def test_no_duplicates(self):
        names = list_templates()
        assert len(names) == len(set(names))

    def test_render_works_for_every_listed_template(self):
        """Each template in the registry must render without KeyError."""
        base_kwargs = dict(
            num_tasks=3,
            domains=["PhysicalActivityTask"],
            difficulty=["Level2"],
            profile_block="  - occupation_status: fulltime",
            existing_event_labels=["sleep"],
            scenario_description="Test.",
            wake_start="07:00",
            sleep_start="23:00",
            admissible_epochs=["morning"],
            label_a="a",
            label_b="b",
            week_start="Monday 04 May 2026",
            week_end="Sunday 10 May 2026",
            calendar_summary="<<calendar>>",
            tasks_list="<<tasks>>",
        )
        for name in list_templates():
            result = render(name, **base_kwargs)
            assert isinstance(result, str)
            assert len(result) > 10  # non-trivial output


# ---------------------------------------------------------------------------
# Generation templates; verbatim GraphRAG content
# ---------------------------------------------------------------------------


_GENERATION_TEMPLATE_NAMES = ("health_improvement", "senior_lifestyle", "productivity")


def _render_generation_template(name: str) -> str:
    """Render one of the three generation templates with placeholder kwargs."""
    return render(
        name,
        num_tasks=3,
        domains=["PhysicalActivityTask"],
        difficulty=["Level1"],
        profile_block="  - occupation_status: fulltime",
        existing_event_labels=[],
        scenario_description="Goal.",
    )


class TestGenerationTemplatesNoEmojiInvention:
    """Generation templates must not invite the LLM to invent emojis."""

    def test_no_bodyweight_circuit_example(self):
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            assert "Bodyweight Circuit" not in out, name
            assert "💪" not in out, name


class TestGenerationTemplatesVerbatimContent:
    """`display_name` and `description` must be copied verbatim from
    the GraphRAG retrieval context, emojis included."""

    def test_template_contains_verbatim_instruction(self):
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            assert "Copy it VERBATIM from the GraphRAG retrieval context" in out, name

    def test_template_requires_emoji_preservation(self):
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            assert "PRESERVE any emojis exactly as they appear" in out, name

    def test_template_forbids_emoji_invention(self):
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            normalised = " ".join(out.split())
            assert "do NOT add emojis the context does not contain" in normalised, name

    def test_template_forbids_emoji_stripping(self):
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            normalised = " ".join(out.split())
            assert "do NOT remove emojis the context does contain" in normalised, name

    def test_template_includes_description_verbatim_rule(self):
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            assert "rdfs:comment" in out, name


class TestGenerationTemplatesInstanceUriSteering:
    """Generation templates require a specific instance URI from the
    GraphRAG retrieval context, never a generic class URI.  Only
    instance nodes carry `dcterms:title` and `dcterms:description`."""

    def test_template_requires_uri_from_retrieval_context(self):
        """v3 wording: instead of asking for a "SPECIFIC INSTANCE URI"
        (which the LLM picked from prompt examples), the template now
        anchors on the `uri:` line of every retrieved context block
        and asks the LLM to copy from there."""
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            normalised = " ".join(out.split())
            assert "MUST cite a URI you SEE in those context blocks" in normalised, name

    def test_template_does_not_carry_concrete_uri_examples(self):
        """v3 cleanup: concrete URI examples were removed because the
        v2 prompt's three example URIs caused 22 / 30 personas to
        return EXACTLY those three.  The new wording
        relies on the retrieval context blocks instead."""
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            assert "/task/blend-a-smoothie" not in out, name
            assert "/task/tidy-up-for-5-minutes" not in out, name
            assert "/task/drink-a-glass-of-water" not in out, name

    def test_template_anchors_on_retrieval_context_uri_lines(self):
        """The new instruction tells the LLM to look at the `uri:`
        line at the bottom of every retrieved context block and copy
        any URI it sees there byte-for-byte."""
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            normalised = " ".join(out.split())
            assert "uri:" in normalised, name
            assert "Copy the URI byte-for-byte" in normalised, name
            assert "Do NOT extrapolate" in normalised, name

    def test_template_demands_diversity_across_picks(self):
        """v3 also pushes the LLM to spread picks across different
        retrieval blocks; fixes the diversity collapse where 30
        personas all picked the same 3 URIs."""
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            normalised = " ".join(out.split())
            assert (
                "Spread your picks across DIFFERENT context blocks" in normalised
            ), name

    def test_template_forbids_class_uri_suffixes(self):
        """Class-name suffixes are still called out so the LLM knows
        not to pick `health/HealthTask` / `…Level1` / `…Activity` -
        same forbidden categories as v2, no concrete examples."""
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            normalised = " ".join(out.split())
            for suffix in ("Task", "LevelN", "Activity"):
                assert suffix in normalised, (name, suffix)

    def test_template_does_not_request_uri_in_description(self):
        """v3 removes the `[<uri>]` citation requirement from the
        description; the URI lives in its own field, and appending it
        to the description was creating noise in the augmented JSON
        and ICS outputs."""
        for name in _GENERATION_TEMPLATE_NAMES:
            out = _render_generation_template(name)
            normalised = " ".join(out.split())
            assert "Do NOT append the URI in square brackets" in normalised, name

    def test_ontology_uri_field_points_to_content_rules(self):
        """The per-task `ontology_uri` field cross-references the
        CONTENT RULES block instead of describing the URI itself."""
        out = _render_generation_template("health_improvement")
        assert "full URI from HealthTasks or HumanActivities" not in out
        assert "specific INSTANCE URI" in out


# ---------------------------------------------------------------------------
# augment_oneshot; worked-example clarifications
# ---------------------------------------------------------------------------


def _render_oneshot(wake_start: str = "06:00", sleep_start: str = "22:00") -> str:
    """Render augment_oneshot with placeholder-friendly stubs for assertion."""
    return render(
        "augment_oneshot",
        num_tasks=2,
        week_start="Monday 04 May 2026",
        week_end="Sunday 10 May 2026",
        wake_start=wake_start,
        sleep_start=sleep_start,
        calendar_summary="<<calendar>>",
        tasks_list="<<tasks>>",
    )


class TestAugmentOneshotContainmentExample:
    def test_containment_inequalities_listed(self):
        out = _render_oneshot()
        assert "busy.start ≤ task.start" in out
        assert "task.end" in out and "busy.end" in out
        assert "task.start < task.end" in out
        assert "duration_min ≤ (task.end − task.start) ≤ duration_max" in out

    def test_reading_worked_example_present(self):
        """The 30-minute worked example must appear so the LLM sees a
        concrete reference for the task-end vs user-event-end constraint."""
        out = _render_oneshot()
        assert '"Reading" planned 20:00–20:30' in out
        assert "20:00–20:20" in out  # VALID
        assert "20:10–20:30" in out  # VALID; task ends exactly at user event end
        assert "20:30–20:50" in out  # INVALID; starts at user event end
        assert "20:25–20:45" in out  # INVALID; ends past user event end
        assert "19:55–20:15" in out  # INVALID; starts before user event

    def test_containment_invalid_examples_explained(self):
        out = _render_oneshot()
        assert "starts at user's event end" in out
        assert "past user's event end" in out
        assert "before user's event" in out


class TestAugmentOneshotWindowExample:
    def test_window_inequalities_split(self):
        """The window rule lists the two inequalities separately so the LLM
        cannot conflate `task.start` with `task.end`."""
        out = _render_oneshot()
        assert "06:00 ≤ task.start" in out
        assert "task.end" in out and "≤ 22:00" in out

    def test_window_worked_example_uses_actual_times(self):
        """The worked example must reflect the supplied wake/sleep values, not
        a hardcoded 06:00–22:00 literal."""
        out = _render_oneshot(wake_start="07:00", sleep_start="23:00")
        assert "waking_window 07:00–23:00" in out
        # The default 06:00 literal must not leak into the rendered output.
        assert "06:00 ≤ task.start" not in out

    def test_window_invalid_pre_wake_example(self):
        out = _render_oneshot()
        assert "04:22–04:32" in out
        assert "starts before wake" in out

    def test_window_invalid_post_sleep_example(self):
        out = _render_oneshot()
        assert "21:55–22:05" in out
        assert "5 min past sleep" in out

    def test_window_invalid_at_sleep_boundary(self):
        out = _render_oneshot()
        assert "22:00–22:10" in out
        assert "starts at sleep boundary" in out

    def test_window_valid_examples_at_window_edges(self):
        out = _render_oneshot()
        # Default-rendered VALID examples include placements right at each edge.
        assert "06:00–06:30" in out
        assert "21:40–22:00" in out

    def test_strictly_keyword_used(self):
        """Hint that all containment conditions must hold STRICTLY; paired
        with the new inequality list."""
        out = _render_oneshot()
        assert "STRICTLY" in out


class TestAugmentOneshotSubstitution:
    def test_wake_substitutes_into_constraint_block(self):
        out = _render_oneshot(wake_start="08:00", sleep_start="20:00")
        assert "08:00 ≤ task.start" in out
        assert "≤ 20:00" in out

    def test_no_unrendered_placeholders_remain(self):
        """Sanity: every `{{ … }}` placeholder must have been substituted."""
        out = _render_oneshot()
        assert "{{" not in out and "}}" not in out


# ---------------------------------------------------------------------------
# augment_oneshot; content boundaries
# ---------------------------------------------------------------------------


class TestAugmentOneshotNoAcceptsConcurrent:
    """The LLM never sees a host's `is_concurrent` flag.  Concurrency
    is decided semantically from titles and descriptions; the validator
    enforces `host.is_concurrent` post-hoc."""

    def test_template_does_not_mention_accepts_concurrent(self):
        out = _render_oneshot()
        assert "accepts_concurrent" not in out

    def test_busy_legend_carries_only_label_start_end(self):
        """The STEP 1 legend describes busy intervals using only
        `label`, `start`, `end`; no compatibility flag."""
        out = _render_oneshot()
        normalised = " ".join(out.lower().split())
        assert "with `label`, `start`, `end`" in normalised

    def test_concurrent_placement_requires_proper_fit(self):
        """CONCURRENT placement requires the proper-fit test (see DEFAULT PATH)."""
        out = _render_oneshot()
        normalised = " ".join(out.lower().split())
        assert "proper concurrent fit" in normalised


class TestAugmentOneshotPlacementFraming:
    """The default path picks the better of standalone or a proper concurrent fit; CONCURRENT stays gated by the proper-fit test."""

    def test_default_path_block_present(self):
        out = _render_oneshot()
        assert "DEFAULT PATH" in out

    def test_states_pick_better_of_standalone_or_concurrent(self):
        out = _render_oneshot()
        normalised = " ".join(out.lower().split())
        assert "actively try a concurrent placement" in normalised
        assert "not standalone by default" in normalised

    def test_states_concurrent_is_a_conditional_choice(self):
        """CONCURRENT placement is allowed only when the proper-fit test passes."""
        out = _render_oneshot()
        normalised = " ".join(out.lower().split())
        assert "concurrent placement is encouraged" in normalised
        # Hard gate: must be a proper concurrent fit with an event or context.
        assert "proper concurrent fit" in normalised

    def test_lists_proper_concurrent_fits(self):
        """Good-fit examples cover every concurrency-friendly host
        in the catalog (2026-05-14): Lunch, Dinner, Office Work,
        Reading.  Each example pairs a habit / micro-action with
        one of those hosts."""
        out = _render_oneshot()
        assert "Lunch" in out and "Dinner" in out
        assert "Office Work" in out
        assert "Reading" in out
        normalised = " ".join(out.lower().split())
        # Canonical meal-friendly habit.
        assert "mindful eating" in normalised
        # Canonical seated-activity micro-actions.
        assert "breathing exercise" in normalised or "meditation" in normalised
        assert "posture check" in normalised or "hydration sip" in normalised
        # Canonical office-friendly micro-actions.
        assert (
            "squats" in normalised
            or "hydration" in normalised
            or "looking outside" in normalised
        )
        # Moderate-fit micro-actions on seated hosts.
        assert (
            "drink a bottle of water" in normalised or "short rest break" in normalised
        )
        assert "calm stretch" in normalised

    # The good-fit list ends at the "Examples that are NOT proper fits"
    # heading; the prompt no longer carries a standalone "exclusive
    # hosts" list (it was removed 2026-05-14 because it was reverse-
    # engineering the evaluator's catalog into the prompt rather than
    # describing genuine scheduling intent).  Tests below use the
    # bad-fit heading as the split boundary.
    _BAD_FIT_HEADING = "Examples that are NOT proper fits"

    def test_good_fit_list_excludes_first_eat(self):
        """First Eat must not appear in the good-fit list."""
        out = _render_oneshot()
        block = out.split("Examples of PROPER concurrent fits", 1)[1]
        block = block.split(self._BAD_FIT_HEADING, 1)[0]
        assert "First Eat" not in block

    def test_good_fit_list_excludes_study(self):
        """Study must not appear in the good-fit list.  (Office Work
        was historically also excluded here, but the 2026-05-14
        catalog edit promoted it to a concurrent-friendly host
        alongside Reading.)
        """
        out = _render_oneshot()
        block = out.split("Examples of PROPER concurrent fits", 1)[1]
        block = block.split(self._BAD_FIT_HEADING, 1)[0]
        assert "Study" not in block

    def test_good_fit_list_excludes_morning_routine(self):
        """Morning Routine is not a configured event and must not
        appear in the good-fit list."""
        out = _render_oneshot()
        block = out.split("Examples of PROPER concurrent fits", 1)[1]
        block = block.split(self._BAD_FIT_HEADING, 1)[0]
        assert "Morning Routine" not in block

    def test_no_explicit_exclusive_hosts_block(self):
        """Sanity guard: the 2026-05-14 cleanup removed the explicit
        "exclusive hosts" enumeration from the prompt.  Listing every
        exclusive event by name was reverse-engineering the catalog's
        `is_concurrent: false` flag into the prompt; a separation-
        of-concerns violation.  The bad-fit examples and Rule of Thumb
        3 already steer the LLM; the validator catches the rest.
        """
        out = _render_oneshot()
        normalised = " ".join(out.split())
        assert "The following hosts seem exclusive" not in normalised
        assert "configured as EXCLUSIVE activities" not in normalised

    def test_lists_bad_hosts_for_concurrent(self):
        """Bad-fit examples list every host that physically occupies the
        person and steers the LLM to STANDALONE placement."""
        out = _render_oneshot()
        # Bad-fit examples list hosts that genuinely cannot host a
        # layered habit OR a refinement.  Walking / Cycling / Yoga
        # were removed from this list 2026-05-14 alongside the
        # catalog flip to `is_concurrent: true`; refinement tasks
        # like `Walk 5,000 Steps` are now legitimate OPTION B
        # placements against those hosts.
        for host in (
            "Grocery Shopping",
            "Visit Family",
            "Gaming",
            "Gardening",
        ):
            assert host in out, f"missing bad-fit host {host!r} in template"

    def test_distinguishes_during_hint_from_proper_fit(self):
        """A task title containing `during X` / `while X` / `with X`
        is described as a hint, not a rule; the proper-fit test is the
        gate."""
        out = _render_oneshot()
        normalised = " ".join(out.lower().split())
        assert "is a hint, not a rule" in normalised


class TestAugmentOneshotPreparationRule:
    """Tasks meaning Preparation must be placed STANDALONE in the free
    gap immediately preceding the user's event; never concurrent."""

    def test_preparation_rule_present(self):
        out = _render_oneshot()
        normalised = " ".join(out.split())
        assert "Preparation" in normalised and "BEFORE the user's event" in normalised

    def test_preparation_rule_lists_prepare_trigger(self):
        """The rule catches the `prepare X` / `before X` / `ready for X` triggers."""
        out = _render_oneshot()
        normalised = " ".join(out.split())
        assert "prepare X" in normalised
        assert "before X" in normalised

    def test_preparation_examples_listed(self):
        out = _render_oneshot()
        assert "Prepare a Salad" in out
        assert "Light Dinner Preparation" in out

    def test_preparation_sequential_to_eating_event(self):
        """Preparation belongs in the free gap BEFORE eating events."""
        out = _render_oneshot()
        normalised = " ".join(out.split())
        assert "preparation is sequential" in normalised.lower()


class TestAugmentOneshotHostNameDiscriminator:
    """Rule of Thumb 3 routes a task to OPTION B (CONCURRENT) when the
    task title either contains a calendar event's name OR describes the
    same activity at minor changes in intensity, count, style, or pace.
    """

    def test_host_name_discriminator_present(self):
        out = _render_oneshot()
        normalised = " ".join(out.split())
        # Same-activity examples on live concurrent-friendly hosts.
        assert "Eat Mindfully" in normalised
        assert "Read Something Calm" in normalised

    def test_host_name_refinement_is_concurrent(self):
        """Headline conclusion: refinement tasks are CONCURRENT with the user's event, not standalone."""
        out = _render_oneshot()
        normalised = " ".join(out.split())
        # The rule recommends CONCURRENT placement and frames the task as
        # the same activity as the user's event during that slot.
        assert "placed CONCURRENT with the user's event" in normalised
        assert "happen simultaneously by being the same activity" in normalised

    def test_host_name_counter_cases_present(self):
        """The rule also lists counter-cases where the host name
        appears in the task but the activities are NOT the same -
        these fall back to Rule of Thumb 2 (preparation) or to
        standalone."""
        out = _render_oneshot()
        normalised = " ".join(out.split())
        # Planning-style counter-case.
        assert "Plan Tomorrow's Lunch" in normalised
        # Different-activity counter-case.
        assert "Walk With Family" in normalised


class TestAugmentOneshotStandaloneWorkedExample:
    """OPTION A carries a worked-success example for the common
    standalone path."""

    def test_worked_standalone_example_present(self):
        out = _render_oneshot()
        assert "Do Light Stretching" in out
        assert '"start":"07:07","end":"08:40"' in out
        assert 'start="07:30", end="07:45"' in out

    def test_standalone_is_first_option(self):
        """STEP 3 lists STANDALONE placement as the first numbered option."""
        out = _render_oneshot()
        assert "(1) STANDALONE placement" in out

    def test_concurrent_placement_is_second_option(self):
        """STEP 3 lists CONCURRENT placement as the second numbered option."""
        out = _render_oneshot()
        assert "(2) CONCURRENT placement" in out


class TestAugmentOneshotNoSkipRule:
    """The no-skip rule names the 50%-free-time fact and the consequence
    of dropping a task."""

    def test_schedule_every_task_phrase(self):
        out = _render_oneshot()
        assert "SCHEDULE EVERY TASK. Leave NO task unscheduled." in out

    def test_mentions_fifty_percent_free_time(self):
        out = _render_oneshot()
        assert "50% of the day free" in out

    def test_warns_about_dropping_a_task(self):
        out = _render_oneshot()
        normalised = " ".join(out.split())
        assert "drops the task" in normalised


class TestAugmentOneshotDropsAllenLegend:
    """The Allen-relations legend is omitted from the LLM-facing prompt;
    the validator enforces those rules behind the scenes."""

    def test_no_meets_keyword(self):
        out = _render_oneshot()
        assert "MEETS" not in out
        assert "MET_BY" not in out

    def test_no_finishes_or_starts_keyword(self):
        out = _render_oneshot()
        assert "FINISHES" not in out
        assert "STARTS free" not in out

    def test_no_precedes_or_preceded_by(self):
        out = _render_oneshot()
        assert "PRECEDES" not in out
        assert "PRECEDED_BY" not in out

    def test_template_does_not_expose_allen_legend(self):
        """The Allen legend is not surfaced in the LLM prompt; the validator
        applies admissible-relation checks server-side at scoring time."""
        out = _render_oneshot()
        # Allen-keyword check (above) already guards the legend itself; this
        # test just pins that the prompt does not advertise the legend's
        # absence in a way future edits could reverse.
        assert "Allen relations" not in out


class TestAugmentOneshotSelfVerify:
    """The in-prompt self-verify block re-checks waking-window
    inequalities, duration band, and host containment in the same
    response."""

    def test_self_verify_block_present(self):
        out = _render_oneshot()
        assert "BEFORE finalizing" in out

    def test_self_verify_mentions_window(self):
        out = _render_oneshot(wake_start="06:00", sleep_start="22:00")
        assert "06:00 ≤ task.start" in out
        assert "task.end ≤ 22:00" in out

    def test_self_verify_mentions_duration_band(self):
        out = _render_oneshot()
        assert "duration_min" in out and "duration_max" in out

    def test_self_verify_mentions_host_containment(self):
        out = _render_oneshot()
        # Containment inequalities may wrap across lines; normalise
        # whitespace so the line break in `task.end\n      ≤ event.end`
        # does not break the assertion.
        normalised = " ".join(out.split())
        assert "event.start ≤ task.start" in normalised
        assert "task.end ≤ event.end" in normalised

    def test_self_verify_silent_in_output(self):
        """The block instructs the LLM not to mention the check in the
        final JSON output."""
        out = _render_oneshot()
        assert "do NOT mention" in out
