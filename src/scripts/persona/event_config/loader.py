"""YAML load + Catalog indexing in one call."""

from __future__ import annotations

from pathlib import Path

from src.scripts.persona.config.loader import load_event
from src.scripts.persona.domain.event import Catalog


def load_catalog(path: Path | str) -> Catalog:
    """Read event_config.yaml and return an indexed Catalog."""
    return Catalog.from_event_config(load_event(Path(path)))
