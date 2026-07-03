"""Tests for `metrics/preference_cache.py`."""

from __future__ import annotations

import os
from unittest.mock import patch

import fakeredis
import pytest

from src.scripts.scenarios.metrics.preference_cache import (
    CachedMapping,
    PreferenceCache,
    _content_hash,
    _decode_payload,
    _field_for,
    _hash_key,
)


@pytest.fixture()
def cache() -> PreferenceCache:
    """Fresh in-memory fakeredis-backed cache per test."""
    return PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))


class TestFieldEncoding:
    def test_field_for_concatenates_with_pipe(self):
        assert _field_for("walk", "walking") == "walk|walking"

    def test_field_for_rejects_pipe_in_inputs(self):
        with pytest.raises(ValueError):
            _field_for("walk|x", "walking")
        with pytest.raises(ValueError):
            _field_for("walk", "walking|y")

    def test_hash_key_uses_pref_mapping_prefix(self):
        assert _hash_key("exp_b", "nutrition_l1") == "pref_mapping:exp_b:nutrition_l1"


class TestContentHash:
    def test_same_inputs_same_hash(self):
        a = _content_hash("walk", "https://x.org/uri", 2, 0.6)
        b = _content_hash("walk", "https://x.org/uri", 2, 0.6)
        assert a == b
        assert len(a) == 32

    def test_different_inputs_different_hash(self):
        a = _content_hash("walk", "https://x.org/uri", 2, 0.6)
        b = _content_hash("walk", "https://x.org/uri", 2, 0.7)
        assert a != b

    def test_none_input_changes_hash(self):
        a = _content_hash("walk", None)
        b = _content_hash("walk", "https://x.org/uri")
        assert a != b


class TestDecodePayload:
    def test_decode_bytes(self):
        assert _decode_payload(b'{"weight": 0.5}') == {"weight": 0.5}

    def test_decode_str(self):
        assert _decode_payload('{"weight": 0.5}') == {"weight": 0.5}

    def test_decode_bad_json_returns_none(self):
        assert _decode_payload("not-json") is None

    def test_decode_non_dict_returns_none(self):
        assert _decode_payload("[1, 2, 3]") is None

    def test_decode_non_str_returns_none(self):
        assert _decode_payload(42) is None


class TestSetAndGet:
    def test_set_then_get_round_trips(self, cache: PreferenceCache):
        mapping = CachedMapping(
            weight=0.6,
            source_tier="ancestor",
            evidence={"hops": 2},
            content_hash="abc",
        )
        cache.set_field("exp", "scen", "walk", "walking", mapping)
        result = cache.get_field("exp", "scen", "walk", "walking")
        assert result is not None
        assert result.weight == 0.6
        assert result.source_tier == "ancestor"
        assert result.evidence == {"hops": 2}
        assert result.content_hash == "abc"

    def test_get_returns_none_on_miss(self, cache: PreferenceCache):
        assert cache.get_field("exp", "scen", "absent", "event") is None

    def test_set_overwrites_existing_entry(self, cache: PreferenceCache):
        m1 = CachedMapping(
            weight=0.1, source_tier="semantic", evidence={}, content_hash="h1"
        )
        m2 = CachedMapping(
            weight=1.0, source_tier="literal", evidence={}, content_hash="h2"
        )
        cache.set_field("exp", "scen", "task", "evt", m1)
        cache.set_field("exp", "scen", "task", "evt", m2)
        out = cache.get_field("exp", "scen", "task", "evt")
        assert out is not None
        assert out.weight == 1.0
        assert out.source_tier == "literal"

    def test_get_stale_entry_with_content_hash_check_returns_none(
        self, cache: PreferenceCache
    ):
        mapping = CachedMapping(
            weight=0.5, source_tier="ancestor", evidence={}, content_hash="OLD"
        )
        cache.set_field("exp", "scen", "task", "evt", mapping)
        # Request with a different expected hash to entry treated as stale.
        out = cache.get_field("exp", "scen", "task", "evt", expected_content_hash="NEW")
        assert out is None
        # Stale entry was HDEL'd.
        assert cache.get_field("exp", "scen", "task", "evt") is None

    def test_get_malformed_payload_returns_none(self, cache: PreferenceCache):
        # Inject bad JSON directly.
        cache._client.hset("pref_mapping:exp:scen", "task|evt", "not-json")
        assert cache.get_field("exp", "scen", "task", "evt") is None


class TestHmgetBatch:
    def test_hmget_round_trips_multiple(self, cache: PreferenceCache):
        cache.set_field(
            "exp",
            "scen",
            "task1",
            "event1",
            CachedMapping(
                weight=1.0, source_tier="literal", evidence={}, content_hash="h"
            ),
        )
        cache.set_field(
            "exp",
            "scen",
            "task2",
            "event2",
            CachedMapping(
                weight=0.5, source_tier="ancestor", evidence={}, content_hash="h"
            ),
        )
        out = cache.hmget_fields(
            "exp", "scen", [("task1", "event1"), ("task2", "event2"), ("missing", "x")]
        )
        assert out[("task1", "event1")] is not None
        assert out[("task1", "event1")].weight == 1.0
        assert out[("task2", "event2")].source_tier == "ancestor"
        assert out[("missing", "x")] is None

    def test_hmget_empty_pairs_returns_empty(self, cache: PreferenceCache):
        assert cache.hmget_fields("exp", "scen", []) == {}

    def test_hmget_stale_entries_dropped(self, cache: PreferenceCache):
        cache.set_field(
            "exp",
            "scen",
            "task1",
            "event1",
            CachedMapping(
                weight=1.0, source_tier="literal", evidence={}, content_hash="OLD"
            ),
        )
        out = cache.hmget_fields(
            "exp",
            "scen",
            [("task1", "event1")],
            expected_content_hash="NEW",
        )
        assert out[("task1", "event1")] is None
        # Verify HDEL ran.
        all_after = cache.all_fields("exp", "scen")
        assert all_after == {}

    def test_hmget_malformed_payload_returns_none(self, cache: PreferenceCache):
        cache._client.hset("pref_mapping:exp:scen", "t|e", "not-json")
        out = cache.hmget_fields("exp", "scen", [("t", "e")])
        assert out == {("t", "e"): None}


class TestPipelineWrite:
    def test_hset_pipeline_writes_all_in_one_round_trip(self, cache: PreferenceCache):
        mappings = {
            ("task1", "event1"): CachedMapping(
                weight=1.0, source_tier="literal", evidence={}, content_hash="h"
            ),
            ("task1", "event2"): CachedMapping(
                weight=0.3,
                source_tier="ancestor",
                evidence={"hops": 2},
                content_hash="h",
            ),
            ("task2", "event1"): CachedMapping(
                weight=0.8,
                source_tier="semantic",
                evidence={"sigma": 0.8},
                content_hash="h",
            ),
        }
        cache.hset_pipeline("exp", "scen", mappings)
        all_ = cache.all_fields("exp", "scen")
        assert len(all_) == 3
        assert all_[("task1", "event1")].weight == 1.0
        assert all_[("task1", "event2")].evidence["hops"] == 2
        assert all_[("task2", "event1")].source_tier == "semantic"

    def test_hset_pipeline_empty_mapping_is_noop(self, cache: PreferenceCache):
        cache.hset_pipeline("exp", "scen", {})
        assert cache.all_fields("exp", "scen") == {}


class TestDiagnostics:
    def test_all_fields_returns_decoded_entries(self, cache: PreferenceCache):
        cache.set_field(
            "exp",
            "scen",
            "task",
            "evt",
            CachedMapping(
                weight=0.5, source_tier="ancestor", evidence={}, content_hash="h"
            ),
        )
        all_ = cache.all_fields("exp", "scen")
        assert ("task", "evt") in all_

    def test_all_fields_skips_malformed_keys(self, cache: PreferenceCache):
        # Inject a malformed field (no pipe).
        cache._client.hset(
            "pref_mapping:exp:scen", "malformed_no_pipe", '{"weight": 0.1}'
        )
        assert cache.all_fields("exp", "scen") == {}

    def test_all_fields_skips_malformed_values(self, cache: PreferenceCache):
        cache._client.hset("pref_mapping:exp:scen", "task|evt", "not-json")
        assert cache.all_fields("exp", "scen") == {}

    def test_delete_scenario_drops_hash(self, cache: PreferenceCache):
        cache.set_field(
            "exp",
            "scen",
            "task",
            "evt",
            CachedMapping(
                weight=1.0, source_tier="literal", evidence={}, content_hash="h"
            ),
        )
        cache.delete_scenario("exp", "scen")
        assert cache.all_fields("exp", "scen") == {}


class TestFromEnv:
    def test_from_env_unset_uses_fakeredis(self):
        with patch.dict(os.environ, {}, clear=True):
            cache = PreferenceCache.from_env()
        # Smoke-test round-trip.
        cache.set_field(
            "e",
            "s",
            "t",
            "ev",
            CachedMapping(
                weight=1.0, source_tier="literal", evidence={}, content_hash="h"
            ),
        )
        assert cache.get_field("e", "s", "t", "ev").weight == 1.0

    def test_from_env_set_invokes_redis_from_url(self):
        """When `REDIS_URL` is set, `from_env` must call
        :meth:`redis.Redis.from_url`.  We patch that classmethod so the
        production dispatch path is exercised without a live Redis
        sidecar."""
        import redis as real_redis

        called = {}

        def fake_from_url(url, decode_responses=True):
            called["url"] = url
            called["decode_responses"] = decode_responses
            return fakeredis.FakeRedis(decode_responses=decode_responses)

        with patch.object(real_redis.Redis, "from_url", staticmethod(fake_from_url)):
            with patch.dict(os.environ, {"REDIS_URL": "redis://nowhere:6379/0"}):
                cache = PreferenceCache.from_env()
        assert called == {
            "url": "redis://nowhere:6379/0",
            "decode_responses": True,
        }
        cache.set_field(
            "e",
            "s",
            "t",
            "ev",
            CachedMapping(
                weight=0.5, source_tier="ancestor", evidence={}, content_hash="h"
            ),
        )
        assert cache.get_field("e", "s", "t", "ev").weight == 0.5
