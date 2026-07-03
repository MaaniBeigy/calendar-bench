"""Deterministic, order-independent seed mixing.

Each per-person task receives mix_seed(root, persona_id, idx), computed in the parent
before submit. This guarantees the produced output is bit-identical regardless of
executor type or completion order.
"""

from __future__ import annotations

import hashlib

_DIGEST_BYTES = 8  # 64-bit seed
_SEPARATOR = b"\x00"


def mix_seed(root_seed: int, persona_id: str, person_idx: int) -> int:
    """Mix three inputs into a 64-bit seed via BLAKE2b.

    Properties:
    - Same inputs results in same output (pure).
    - Independent of any global state.
    - Sensitive to every input: changing any one of them gives an unrelated seed.
    """
    if root_seed < 0:
        raise ValueError(f"root_seed must be non-negative, got {root_seed}")
    if person_idx < 0:
        raise ValueError(f"person_idx must be non-negative, got {person_idx}")
    if not persona_id:
        raise ValueError("persona_id must be non-empty")

    h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    h.update(root_seed.to_bytes(8, "big", signed=False))
    h.update(_SEPARATOR)
    h.update(persona_id.encode("utf-8"))
    h.update(_SEPARATOR)
    h.update(person_idx.to_bytes(8, "big", signed=False))
    return int.from_bytes(h.digest(), "big", signed=False)


def mix_axis_seed(root_seed: int, persona_id: str, axis: str) -> int:
    """Seed a per-persona, per-axis characteristic draw."""
    if root_seed < 0:
        raise ValueError(f"root_seed must be non-negative, got {root_seed}")
    if not persona_id:
        raise ValueError("persona_id must be non-empty")
    if not axis:
        raise ValueError("axis must be non-empty")

    h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    h.update(root_seed.to_bytes(8, "big", signed=False))
    h.update(_SEPARATOR)
    h.update(b"char")
    h.update(_SEPARATOR)
    h.update(persona_id.encode("utf-8"))
    h.update(_SEPARATOR)
    h.update(axis.encode("utf-8"))
    return int.from_bytes(h.digest(), "big", signed=False)
