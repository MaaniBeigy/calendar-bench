"""Redis-backed cache for `PreferenceMapper` mapping resolutions.

Schema:

* HASH key:   `pref_mapping:<experiment>:<scenario>`
* HASH field: `<task_label>|<event_name>` (pipe is forbidden in both
              vocabularies).
* HASH value: JSON `{weight, source_tier, evidence, content_hash, ts}`.

A content hash detects stale fields when matcher knobs (task set,
event catalog, ancestor depth, σ threshold) change between runs.
Stale entries are dropped on read so callers re-compute lazily.

Pure facade; no Redis client is created at import. Use
`PreferenceCache.from_client` or `PreferenceCache.from_env`, which
falls back to `fakeredis` when `REDIS_URL` is unset.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


def _content_hash(*parts: Any) -> str:
    """16-byte hex prefix of SHA-256 over a deterministic JSON-ish string.

    Any change to the inputs that produced a mapping value invalidates
    the cached entry.  Values are
    stringified via :func:`json.dumps` with `sort_keys=True` so dict /
    list ordering does not matter.
    """
    rendered = json.dumps(list(parts), sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:32]


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class CachedMapping:
    """One `(task_label, event_name)` resolution as stored in Redis."""

    weight: float
    source_tier: str  # "literal" | "ancestor" | "semantic" | "none"
    evidence: dict[str, Any]
    content_hash: str
    ts: str = ""


def _field_for(task_label: str, event_name: str) -> str:
    """Build the HASH field name for one `(task, event)` pair."""
    if "|" in task_label or "|" in event_name:
        raise ValueError(
            "task_label / event_name must not contain '|' (the field separator); "
            f"got task={task_label!r}, event={event_name!r}"
        )
    return f"{task_label}|{event_name}"


def _hash_key(experiment: str, scenario: str) -> str:
    return f"pref_mapping:{experiment}:{scenario}"


class PreferenceCache:
    """Redis HASH-backed mapping cache.

    The cache is keyed by `(experiment, scenario)` (one HASH per pair).
    Reads transparently detect stale entries via content-hash comparison
    and return `None` for those fields, so the caller treats them as
    misses and re-computes.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_client(cls, client: Any) -> "PreferenceCache":
        return cls(client)

    @classmethod
    def from_env(cls, env_var: str = "REDIS_URL") -> "PreferenceCache":
        """Build a cache from `$REDIS_URL`; fall back to `fakeredis`.

        Production runs inside docker-compose set `REDIS_URL` so the
        sidecar is used.  In unit-test / offline mode the env var is
        usually unset; we transparently swap in :class:`fakeredis.FakeRedis`
        so behavior stays identical without spinning up Redis.
        """
        url = os.environ.get(env_var)
        if url:
            import redis as _redis  # local import; redis is optional in CI

            client = _redis.Redis.from_url(url, decode_responses=True)
        else:
            import fakeredis as _fr

            client = _fr.FakeRedis(decode_responses=True)
        return cls(client)

    # ------------------------------------------------------------------
    # Single-field helpers
    # ------------------------------------------------------------------

    def get_field(
        self,
        experiment: str,
        scenario: str,
        task_label: str,
        event_name: str,
        *,
        expected_content_hash: str | None = None,
    ) -> CachedMapping | None:
        """Return the cached mapping or `None` (miss / stale).

        When `expected_content_hash` is set, mismatching entries are
        silently dropped; both from the in-Redis HASH (HDEL) and from
        the returned value.
        """
        raw = self._client.hget(
            _hash_key(experiment, scenario), _field_for(task_label, event_name)
        )
        if raw is None:
            return None
        payload = _decode_payload(raw)
        if payload is None:
            return None
        if (
            expected_content_hash is not None
            and payload.get("content_hash") != expected_content_hash
        ):
            # Drop stale entry; the inputs that produced it have moved on.
            self._client.hdel(
                _hash_key(experiment, scenario),
                _field_for(task_label, event_name),
            )
            return None
        return CachedMapping(
            weight=float(payload.get("weight", 0.0)),
            source_tier=str(payload.get("source_tier", "none")),
            evidence=dict(payload.get("evidence", {})),
            content_hash=str(payload.get("content_hash", "")),
            ts=str(payload.get("ts", "")),
        )

    def set_field(
        self,
        experiment: str,
        scenario: str,
        task_label: str,
        event_name: str,
        mapping: CachedMapping,
    ) -> None:
        """Persist one mapping; overwrites any existing entry."""
        payload = {
            "weight": float(mapping.weight),
            "source_tier": mapping.source_tier,
            "evidence": mapping.evidence,
            "content_hash": mapping.content_hash,
            "ts": mapping.ts or _now_iso(),
        }
        self._client.hset(
            _hash_key(experiment, scenario),
            _field_for(task_label, event_name),
            json.dumps(payload),
        )

    # ------------------------------------------------------------------
    # Batched helpers
    # ------------------------------------------------------------------

    def hmget_fields(
        self,
        experiment: str,
        scenario: str,
        pairs: Iterable[tuple[str, str]],
        *,
        expected_content_hash: str | None = None,
    ) -> dict[tuple[str, str], CachedMapping | None]:
        """Batch-fetch many mappings in one HMGET round-trip.

        Returns a dict keyed by `(task_label, event_name)` for each
        pair in `pairs`; values are `None` on miss / stale.
        """
        pair_list = list(pairs)
        if not pair_list:
            return {}
        fields = [_field_for(t, e) for t, e in pair_list]
        raws = self._client.hmget(_hash_key(experiment, scenario), fields)
        out: dict[tuple[str, str], CachedMapping | None] = {}
        stale_fields: list[str] = []
        for (task, event), raw in zip(pair_list, raws):
            if raw is None:
                out[(task, event)] = None
                continue
            payload = _decode_payload(raw)
            if payload is None:
                out[(task, event)] = None
                continue
            if (
                expected_content_hash is not None
                and payload.get("content_hash") != expected_content_hash
            ):
                stale_fields.append(_field_for(task, event))
                out[(task, event)] = None
                continue
            out[(task, event)] = CachedMapping(
                weight=float(payload.get("weight", 0.0)),
                source_tier=str(payload.get("source_tier", "none")),
                evidence=dict(payload.get("evidence", {})),
                content_hash=str(payload.get("content_hash", "")),
                ts=str(payload.get("ts", "")),
            )
        if stale_fields:
            self._client.hdel(_hash_key(experiment, scenario), *stale_fields)
        return out

    def hset_pipeline(
        self,
        experiment: str,
        scenario: str,
        mappings: dict[tuple[str, str], CachedMapping],
    ) -> None:
        """Batch-write many mappings in a single Redis pipeline."""
        if not mappings:
            return
        key = _hash_key(experiment, scenario)
        encoded: dict[str, str] = {}
        for (task, event), mapping in mappings.items():
            payload = {
                "weight": float(mapping.weight),
                "source_tier": mapping.source_tier,
                "evidence": mapping.evidence,
                "content_hash": mapping.content_hash,
                "ts": mapping.ts or _now_iso(),
            }
            encoded[_field_for(task, event)] = json.dumps(payload)
        pipe = self._client.pipeline()
        pipe.hset(key, mapping=encoded)
        pipe.execute()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def all_fields(
        self, experiment: str, scenario: str
    ) -> dict[tuple[str, str], CachedMapping]:
        """Return every cached entry for a scenario (HGETALL).

        Useful for audit / report rendering; not on the hot path.
        Stale entries are filtered out only when an explicit
        `content_hash` was attached to each value at write time.
        """
        raw = self._client.hgetall(_hash_key(experiment, scenario))
        out: dict[tuple[str, str], CachedMapping] = {}
        for field, value in raw.items():
            if "|" not in field:
                continue
            task, event = field.split("|", 1)
            payload = _decode_payload(value)
            if payload is None:
                continue
            out[(task, event)] = CachedMapping(
                weight=float(payload.get("weight", 0.0)),
                source_tier=str(payload.get("source_tier", "none")),
                evidence=dict(payload.get("evidence", {})),
                content_hash=str(payload.get("content_hash", "")),
                ts=str(payload.get("ts", "")),
            )
        return out

    def delete_scenario(self, experiment: str, scenario: str) -> None:
        """Drop every cached entry for a scenario (DEL on the HASH)."""
        self._client.delete(_hash_key(experiment, scenario))


def _decode_payload(raw: Any) -> dict[str, Any] | None:
    """Decode a Redis HASH value to a dict; `None` for malformed entries."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
