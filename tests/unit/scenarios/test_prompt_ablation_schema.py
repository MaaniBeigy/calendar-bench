"""Tests for the `PromptAblationConfig` sub-schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.scripts.scenarios.config.schema import (
    AblationVariantSpec,
    LLMAgentConfig,
    PromptAblationConfig,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_llm_agent_config_default_has_no_ablation(self):
        cfg = LLMAgentConfig()
        assert cfg.prompt_ablation is None

    def test_ablation_default_is_single_design(self):
        cfg = PromptAblationConfig()
        assert cfg.design == "single"
        assert cfg.fold is True
        assert cfg.use_default_placebos is True
        assert cfg.placebos == {}
        assert cfg.variants == []


# ---------------------------------------------------------------------------
# Design enumeration
# ---------------------------------------------------------------------------


class TestDesignNames:
    @pytest.mark.parametrize(
        "name",
        [
            "single",
            "leave_one_out",
            "plackett_burman_12",
            "plackett_burman_24",
            "full_factorial",
        ],
    )
    def test_built_in_designs_accept_no_variants(self, name):
        cfg = PromptAblationConfig(design=name)
        assert cfg.design == name

    def test_rejects_unknown_design(self):
        with pytest.raises(ValidationError):
            PromptAblationConfig(design="not_a_design")


# ---------------------------------------------------------------------------
# Custom design
# ---------------------------------------------------------------------------


class TestCustomDesign:
    def test_custom_requires_variants(self):
        with pytest.raises(ValidationError, match="at least one entry"):
            PromptAblationConfig(design="custom")

    def test_custom_accepts_variants(self):
        cfg = PromptAblationConfig(
            design="custom",
            variants=[
                AblationVariantSpec(id="baseline", ablate=[]),
                AblationVariantSpec(id="no_rot1", ablate=["rule_of_thumb_1"]),
            ],
        )
        assert [v.id for v in cfg.variants] == ["baseline", "no_rot1"]

    def test_custom_duplicate_ids_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            PromptAblationConfig(
                design="custom",
                variants=[
                    AblationVariantSpec(id="dup", ablate=[]),
                    AblationVariantSpec(id="dup", ablate=["rule_of_thumb_1"]),
                ],
            )

    def test_custom_unknown_block_in_variant_rejected(self):
        with pytest.raises(ValidationError, match="unknown block"):
            PromptAblationConfig(
                design="custom",
                variants=[
                    AblationVariantSpec(id="bad", ablate=["does_not_exist"]),
                ],
            )

    def test_non_custom_with_variants_rejected(self):
        with pytest.raises(ValidationError, match="must not declare"):
            PromptAblationConfig(
                design="leave_one_out",
                variants=[AblationVariantSpec(id="extra", ablate=[])],
            )


# ---------------------------------------------------------------------------
# Variant id validation
# ---------------------------------------------------------------------------


class TestVariantIdPattern:
    def test_blank_id_rejected(self):
        with pytest.raises(ValidationError):
            AblationVariantSpec(id="", ablate=[])

    def test_id_with_space_rejected(self):
        with pytest.raises(ValidationError):
            AblationVariantSpec(id="has space", ablate=[])

    def test_id_with_slash_rejected(self):
        with pytest.raises(ValidationError):
            AblationVariantSpec(id="has/slash", ablate=[])

    def test_alnum_underscore_hyphen_allowed(self):
        v = AblationVariantSpec(id="v1_baseline-A", ablate=[])
        assert v.id == "v1_baseline-A"


# ---------------------------------------------------------------------------
# Placebos
# ---------------------------------------------------------------------------


class TestPlacebos:
    def test_known_block_placebo_accepted(self):
        cfg = PromptAblationConfig(
            design="plackett_burman_24",
            placebos={"rule_of_thumb_1": "Apply the proper-fit test."},
        )
        assert "rule_of_thumb_1" in cfg.placebos

    def test_unknown_block_placebo_rejected(self):
        with pytest.raises(ValidationError, match="unknown block"):
            PromptAblationConfig(
                design="plackett_burman_24",
                placebos={"does_not_exist": "placebo"},
            )

    def test_default_placebos_flag_toggleable(self):
        cfg = PromptAblationConfig(
            design="plackett_burman_24",
            use_default_placebos=False,
        )
        assert cfg.use_default_placebos is False


# ---------------------------------------------------------------------------
# Frozen behaviour
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_config_is_immutable(self):
        cfg = PromptAblationConfig()
        with pytest.raises(ValidationError):
            cfg.design = "leave_one_out"  # type: ignore[misc]

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            PromptAblationConfig(design="single", unknown_field=True)


# ---------------------------------------------------------------------------
# LLMAgentConfig integration
# ---------------------------------------------------------------------------


class TestLLMAgentIntegration:
    def test_llm_agent_accepts_ablation_block(self):
        cfg = LLMAgentConfig(
            provider="openai",
            model="gpt-4o-mini",
            prompt_template="augment_oneshot",
            prompt_ablation=PromptAblationConfig(
                design="leave_one_out",
            ),
        )
        assert cfg.prompt_ablation is not None
        assert cfg.prompt_ablation.design == "leave_one_out"

    def test_llm_agent_round_trips_through_model_dump(self):
        cfg = LLMAgentConfig(
            prompt_ablation=PromptAblationConfig(
                design="custom",
                variants=[AblationVariantSpec(id="b", ablate=[])],
            )
        )
        dumped = cfg.model_dump()
        rebuilt = LLMAgentConfig.model_validate(dumped)
        assert rebuilt == cfg
