"""Tests for the Jinja2-backed `augment_oneshot` renderer and its
length-matched placebo catalog."""

from __future__ import annotations

import pytest

from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
    ABLATABLE_BLOCKS,
    default_placebo_for,
    get_template_source,
    render,
)
from src.scripts.scenarios.augmentation.prompts.augment_oneshot_placebos import (
    DEFAULT_PLACEBOS,
)
from src.scripts.scenarios.task_generation import prompt_templates

# ---------------------------------------------------------------------------
# Reference fixtures
# ---------------------------------------------------------------------------

_CTX = {
    "week_start": "2026-05-18",
    "week_end": "2026-05-24",
    "wake_start": "06:00",
    "sleep_start": "22:00",
    "calendar_summary": "<<CAL>>",
    "num_tasks": 3,
    "tasks_list": "<<TASKS>>",
}

# Markers that, when present in the rendered prompt, uniquely identify
# a given block. Use these to assert presence / absence rather than
# the entire block body, so prose tweaks do not break the test suite.
_BLOCK_MARKERS: dict[str, str] = {
    "default_path": "Pick the BETTER of standalone vs a proper concurrent fit",
    "fit_examples": 'task::"mindful eating"',
    "rule_of_thumb_1": "RULE OF THUMB 1",
    "rule_of_thumb_2": "RULE OF THUMB 2",
    "rule_of_thumb_3": "RULE OF THUMB 3",
    "worked_example_standalone": '"07:07"',
    "worked_example_concurrent": '"Reading" planned 20:00',
    "worked_example_waking": "INVALID: 04:22",
    "hard_constraints": "HARD CONSTRAINTS",
    "do_not_drop": "SCHEDULE EVERY TASK",
    "self_verify": "SELF-VERIFY",
    "split_dividable_tasks": "DIVIDE-FRIENDLY TASKS",
}


# ---------------------------------------------------------------------------
# Baseline render: shape + identity
# ---------------------------------------------------------------------------


class TestBaselineRender:
    def test_renders_non_empty(self):
        out = render(_CTX)
        assert isinstance(out, str)
        assert len(out) > 8000

    def test_substitutes_every_context_variable(self):
        out = render(_CTX)
        assert "2026-05-18" in out
        assert "2026-05-24" in out
        assert "06:00" in out
        assert "22:00" in out
        assert "<<CAL>>" in out
        assert "<<TASKS>>" in out
        assert "3 total" in out

    def test_every_block_present_in_baseline(self):
        out = render(_CTX)
        for name, marker in _BLOCK_MARKERS.items():
            assert marker in out, f"block {name!r} missing from baseline render"

    def test_no_jinja_artifacts_in_baseline(self):
        out = render(_CTX)
        for needle in ("{% ", "{%-", "{{", "}}", "{% endblock"):
            assert needle not in out, f"leaked Jinja artifact: {needle!r}"

    def test_no_trailing_extra_blank_line(self):
        """Render must end at the JSON example's closing `]`."""
        out = render(_CTX)
        assert out.endswith("]")

    def test_render_is_pure(self):
        """Two calls with the same context must produce identical output."""
        assert render(_CTX) == render(_CTX)


# ---------------------------------------------------------------------------
# Ablation: deletion path
# ---------------------------------------------------------------------------


class TestDeletionAblation:
    def test_single_block_removed(self):
        out = render(_CTX, ablate=["rule_of_thumb_1"])
        assert _BLOCK_MARKERS["rule_of_thumb_1"] not in out
        # All other blocks must still be present.
        for name, marker in _BLOCK_MARKERS.items():
            if name == "rule_of_thumb_1":
                continue
            assert marker in out, f"block {name!r} should not have been touched"

    def test_multiple_blocks_removed(self):
        out = render(_CTX, ablate=["rule_of_thumb_1", "self_verify"])
        assert _BLOCK_MARKERS["rule_of_thumb_1"] not in out
        assert _BLOCK_MARKERS["self_verify"] not in out
        assert _BLOCK_MARKERS["rule_of_thumb_2"] in out

    def test_every_block_individually_removable(self):
        baseline = render(_CTX)
        for name, marker in _BLOCK_MARKERS.items():
            out = render(_CTX, ablate=[name])
            assert marker not in out, f"failed to remove block {name!r}"
            assert len(out) < len(baseline), f"block {name!r} did not shrink prompt"

    def test_full_ablation_strips_every_block(self):
        out = render(_CTX, ablate=list(ABLATABLE_BLOCKS))
        for name, marker in _BLOCK_MARKERS.items():
            assert marker not in out, f"block {name!r} leaked under full ablation"
        # Structural sections (input window, output schema) must remain.
        assert "INPUT WINDOW" in out
        assert "STEP 4: OUTPUT (STRICT)" in out

    def test_no_jinja_artifacts_after_ablation(self):
        out = render(_CTX, ablate=list(ABLATABLE_BLOCKS))
        for needle in ("{% ", "{%-", "{{", "}}", "{% endblock", "{% raw", "{% endraw"):
            assert needle not in out, f"leaked Jinja artifact: {needle!r}"

    def test_ablation_argument_can_be_iterator(self):
        """`ablate` accepts any iterable, not just a list."""
        out = render(_CTX, ablate=iter(["rule_of_thumb_1"]))
        assert _BLOCK_MARKERS["rule_of_thumb_1"] not in out


# ---------------------------------------------------------------------------
# Ablation: placebo path
# ---------------------------------------------------------------------------


class TestPlaceboAblation:
    def test_user_placebo_appears_verbatim(self):
        placebo = "Stand-in placebo text for rule_of_thumb_1."
        out = render(
            _CTX,
            ablate=["rule_of_thumb_1"],
            placebos={"rule_of_thumb_1": placebo},
        )
        assert placebo in out
        assert _BLOCK_MARKERS["rule_of_thumb_1"] not in out

    def test_placebo_does_not_leak_jinja_syntax(self):
        """Placebos with `{{` or `{%` are treated as literal text."""
        placebo = "Literal {{ var }} and {% if %} markers stay verbatim."
        out = render(
            _CTX,
            ablate=["rule_of_thumb_1"],
            placebos={"rule_of_thumb_1": placebo},
        )
        assert placebo in out

    def test_placebos_for_blocks_not_in_ablate_are_ignored(self):
        """Placebos fire only for blocks that are being ablated."""
        placebo = "This should never appear."
        out = render(
            _CTX,
            ablate=["self_verify"],
            placebos={"rule_of_thumb_1": placebo},
        )
        assert placebo not in out
        assert _BLOCK_MARKERS["rule_of_thumb_1"] in out

    def test_placebos_default_to_none(self):
        """`placebos=None` matches omitting the argument."""
        deleted = render(_CTX, ablate=["self_verify"])
        deleted_explicit_none = render(_CTX, ablate=["self_verify"], placebos=None)
        assert deleted == deleted_explicit_none


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_block_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown ablation target"):
            render(_CTX, ablate=["does_not_exist"])

    def test_unknown_block_lists_valid_blocks(self):
        with pytest.raises(ValueError) as excinfo:
            render(_CTX, ablate=["bogus"])
        for block in ABLATABLE_BLOCKS:
            assert block in str(excinfo.value)


# ---------------------------------------------------------------------------
# Source / placebo helpers
# ---------------------------------------------------------------------------


class TestSourceAndPlacebos:
    def test_get_template_source_is_the_j2_file(self):
        src = get_template_source()
        # The `.j2` source still carries the block markers.
        assert "{% block default_path %}" in src
        assert "{% block self_verify %}" in src

    def test_default_placebo_for_known_block(self):
        for block in ABLATABLE_BLOCKS:
            assert default_placebo_for(block) == DEFAULT_PLACEBOS.get(block, "")

    def test_default_placebo_for_unknown_block_is_empty(self):
        assert default_placebo_for("not_a_real_block") == ""

    def test_default_placebos_cover_every_block(self):
        """Every ablatable block ships with a non-empty default placebo."""
        for block in ABLATABLE_BLOCKS:
            assert DEFAULT_PLACEBOS[
                block
            ].strip(), f"block {block!r} has no default placebo"


# ---------------------------------------------------------------------------
# prompt_templates registry routing
# ---------------------------------------------------------------------------


class TestRegistryRouting:
    def test_registry_lists_augment_oneshot(self):
        assert "augment_oneshot" in prompt_templates.list_templates()

    def test_registry_routes_to_jinja_renderer(self):
        via_registry = prompt_templates.render("augment_oneshot", **_CTX)
        direct = render(_CTX)
        assert via_registry == direct

    def test_registry_supports_ablate_and_placebos_kwargs(self):
        out = prompt_templates.render(
            "augment_oneshot",
            _ablate=["rule_of_thumb_1"],
            _placebos={"rule_of_thumb_1": "stand-in"},
            **_CTX,
        )
        assert "stand-in" in out
        assert _BLOCK_MARKERS["rule_of_thumb_1"] not in out

    def test_registry_unknown_template_still_errors(self):
        with pytest.raises(ValueError, match="unknown template"):
            prompt_templates.render("definitely_not_a_template")

    def test_registry_is_sorted_and_unique(self):
        names = prompt_templates.list_templates()
        assert names == sorted(names)
        assert len(names) == len(set(names))
