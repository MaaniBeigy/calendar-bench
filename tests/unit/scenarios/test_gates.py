"""Unit tests for src.scripts.scenarios.task_generation.gates."""

from __future__ import annotations

import logging
import math

import pytest

from src.scripts.scenarios.task_generation.gates import (
    GateDecision,
    _cosine,
    _extract_emojis,
    gate_paraphrase,
)

# ---------------------------------------------------------------------------
# _extract_emojis
# ---------------------------------------------------------------------------


class TestExtractEmojis:
    def test_empty_text_returns_empty_list(self):
        assert _extract_emojis("") == []
        assert _extract_emojis(None) == []  # type: ignore[arg-type]

    def test_no_emoji_returns_empty(self):
        assert _extract_emojis("Plain text only.") == []

    def test_single_emoji_extracted(self):
        assert _extract_emojis("Drink water 🥤") == ["🥤"]

    def test_multiple_distinct_emojis_in_order(self):
        out = _extract_emojis("🥤 then 🤸 finally 🧹")
        assert out == ["🥤", "🤸", "🧹"]

    def test_repeated_emoji_preserved(self):
        out = _extract_emojis("🥤🥤")
        assert out == ["🥤", "🥤"]

    def test_dingbat_range_covered(self):
        # ✅ is U+2705, in the 2600-27BF range.
        assert _extract_emojis("done ✅") == ["✅"]


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors_score_one(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert _cosine([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_empty_vectors_return_zero(self):
        assert _cosine([], []) == 0.0

    def test_length_mismatch_returns_zero(self):
        assert _cosine([1.0, 2.0], [1.0]) == 0.0

    def test_three_dimensional_partial_similarity(self):
        # cos(a,b) for these two = 0.5/sqrt(0.5)/sqrt(0.5) = ?
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        # dot=1, |a|=sqrt(2), |b|=1 to 1/sqrt(2)
        assert _cosine(a, b) == pytest.approx(1 / math.sqrt(2))


# ---------------------------------------------------------------------------
# gate_paraphrase; fakes
# ---------------------------------------------------------------------------


class _IdentityEmbedder:
    """Cheap embedder: returns a length-2 one-hot keyed on the first 8
    chars of the input.  Identical strings to identical vectors to cosine
    1.0; substantially different strings to cosine 0.0."""

    def embed_query(self, text: str) -> list[float]:
        return [float(sum(map(ord, text[:8]))), float(len(text))]


class _ConstantEmbedder:
    """Returns a fixed unit vector for every input to cosine always 1.0."""

    def __init__(self, vec: list[float] | None = None) -> None:
        self._vec = vec or [1.0, 0.0]

    def embed_query(self, _text: str) -> list[float]:
        return list(self._vec)


class _OrthoEmbedder:
    """Returns alternating axes so two distinct strings end up
    orthogonal; cosine 0.0."""

    def __init__(self) -> None:
        self._toggle = False

    def embed_query(self, _text: str) -> list[float]:
        self._toggle = not self._toggle
        return [1.0, 0.0] if self._toggle else [0.0, 1.0]


class _RaisingEmbedder:
    def embed_query(self, _text: str) -> list[float]:
        raise RuntimeError("network down")


# ---------------------------------------------------------------------------
# gate_paraphrase
# ---------------------------------------------------------------------------


def _decide(
    *,
    canonical_description: str,
    paraphrase: str,
    embedder=None,
    canonical_text: str | None = None,
    length_delta_pct: float = 0.10,
    similarity_threshold: float = 0.80,
) -> GateDecision:
    return gate_paraphrase(
        canonical_description=canonical_description,
        canonical_text=canonical_text or canonical_description,
        paraphrase=paraphrase,
        embedder=embedder or _ConstantEmbedder(),
        length_delta_pct=length_delta_pct,
        similarity_threshold=similarity_threshold,
    )


class TestLengthGate:
    def test_within_budget_passes_when_other_gates_pass(self):
        canonical = "Walk daily for clear thinking."
        paraphrase = "Walk daily for clearer thinking."  # +1 char ≈ 3 %
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.used == "paraphrase"
        assert d.failed_gate is None
        assert d.length_delta < 0.10

    def test_over_budget_falls_back_to_canonical(self):
        canonical = "Drink water."
        paraphrase = "Drink water more often, every hour, for hydration."
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.used == "canonical"
        assert d.failed_gate == "length"
        assert d.text == canonical


class TestEmojiGate:
    def test_identical_emojis_pass(self):
        canonical = "Drink 🥤 water."
        paraphrase = "Drink 🥤 fluids."
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.used == "paraphrase"
        assert d.failed_gate is None

    def test_paraphrase_adds_new_emoji_fails(self):
        canonical = "Drink 🥤 water."
        paraphrase = "Drink 🥤 water 😀"
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.failed_gate == "emoji"
        assert d.extra_emojis == {"😀": 1}
        assert d.missing_emojis == {}

    def test_paraphrase_drops_emoji_fails(self):
        # Use a longer canonical so the length-budget headroom is
        # generous and the emoji-drop is the FIRST failure.
        canonical = "Drink a glass of water 🥤 daily for hydration."
        paraphrase = "Drink a glass of water daily for hydration."  # emoji dropped
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.failed_gate == "emoji"
        assert d.missing_emojis == {"🥤": 1}
        assert d.extra_emojis == {}

    def test_paraphrase_swaps_emoji_fails(self):
        canonical = "Drink 🥤 water."
        paraphrase = "Drink 💧 water."
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.failed_gate == "emoji"
        assert d.missing_emojis == {"🥤": 1}
        assert d.extra_emojis == {"💧": 1}

    def test_paraphrase_changes_count_fails(self):
        canonical = "Drink 🥤 water."
        paraphrase = "Drink 🥤🥤 water."  # twice
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.failed_gate == "emoji"
        assert d.extra_emojis == {"🥤": 1}

    def test_canonical_emojiless_paraphrase_must_match(self):
        canonical = "Walk daily."
        paraphrase = "Walk daily 🚶"
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.failed_gate == "emoji"
        assert d.extra_emojis == {"🚶": 1}

    def test_both_emojiless_passes(self):
        canonical = "Walk every day."
        paraphrase = "Walk every day."
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.failed_gate is None
        assert d.used == "paraphrase"

    def test_display_name_emoji_does_not_leak_into_emoji_gate(self):
        """Display_name's emoji is only in canonical_text (gate C),
        not canonical_description (gate B).  Even when display_name
        carries an emoji, gate B requires the paraphrase to also have
        no emojis when the description has none."""
        canonical_desc = "Stretch your shoulders."  # no emoji, 23 chars
        canonical_text = "Stretch for 5 Minutes 🤸. Stretch your shoulders."
        # Paraphrase: same length, no emoji; passes gates A and B.
        d = _decide(
            canonical_description=canonical_desc,
            canonical_text=canonical_text,
            paraphrase="Roll your shoulders out.",
            embedder=_ConstantEmbedder(),
        )
        assert d.failed_gate is None


class TestSimilarityGate:
    def test_low_similarity_falls_back_to_canonical(self):
        canonical = "Drink water for hydration."
        paraphrase = "Drink water for hydration!"  # short, length-ok, emoji-ok
        d = _decide(
            canonical_description=canonical,
            paraphrase=paraphrase,
            embedder=_OrthoEmbedder(),  # cosine ≈ 0
            similarity_threshold=0.80,
        )
        assert d.failed_gate == "similarity"
        assert d.text == canonical
        assert d.similarity == pytest.approx(0.0)

    def test_threshold_at_zero_always_passes_sim_gate(self):
        canonical = "Drink water for hydration."
        paraphrase = "Drink water for hydration!"
        d = _decide(
            canonical_description=canonical,
            paraphrase=paraphrase,
            embedder=_OrthoEmbedder(),
            similarity_threshold=0.0,
        )
        assert d.failed_gate is None

    def test_embedder_failure_treated_as_gate_c_fail(self, caplog):
        canonical = "Drink water for hydration."
        paraphrase = "Drink water for hydration!"
        with caplog.at_level(logging.WARNING):
            d = _decide(
                canonical_description=canonical,
                paraphrase=paraphrase,
                embedder=_RaisingEmbedder(),
            )
        assert d.failed_gate == "similarity"
        assert d.similarity == 0.0
        assert any("embedder failed" in r.getMessage() for r in caplog.records)


class TestPriorityOrder:
    def test_length_failure_reported_before_emoji_or_similarity(self):
        canonical = "Walk."
        # 30+ chars to length fails; also adds an emoji to emoji would
        # fail too; but length is reported first.
        paraphrase = "Walk a lot for the rest of today 😀😀."
        d = _decide(canonical_description=canonical, paraphrase=paraphrase)
        assert d.failed_gate == "length"
        # Emoji counters are still populated for telemetry.
        assert d.extra_emojis == {"😀": 2}

    def test_emoji_failure_reported_before_similarity(self):
        canonical = "Drink water."
        paraphrase = "Drink water 😀"  # length-ok (under 10 % delta), emoji extra
        d = _decide(
            canonical_description=canonical,
            paraphrase=paraphrase,
            embedder=_OrthoEmbedder(),  # similarity would fail too
            length_delta_pct=0.50,  # generous so length passes
        )
        assert d.failed_gate == "emoji"


class TestEmptyParaphrase:
    def test_empty_paraphrase_fails_length_gate(self):
        d = _decide(canonical_description="Walk daily.", paraphrase="")
        # |0 - 11| / 11 = 1.0 to over budget at default 0.10.
        assert d.failed_gate == "length"
        assert d.length_delta == pytest.approx(1.0)
