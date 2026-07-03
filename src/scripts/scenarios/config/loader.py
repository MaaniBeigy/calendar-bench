"""Load and validate scenario YAML config files.

Two formats are supported:

* **Legacy**; top-level `scenario:` key to returns :class:`ScenarioConfig`.
* **Multi-scenario**; top-level `scenarios:` list to returns
  :class:`ExperimentScenariosConfig`.

Use :func:`load_config` to auto-detect which format is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from src.scripts.scenarios.config.schema import (
    ExperimentScenariosConfig,
    ScenarioConfig,
)


class ScenarioConfigError(ValueError):
    """Raised when a scenario YAML cannot be read or fails validation."""


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ScenarioConfigError(f"scenario config not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioConfigError(f"could not read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioConfigError(f"YAML parse error in {path}: {exc}") from exc
    if data is None:
        raise ScenarioConfigError(f"empty scenario config: {path}")
    return data


def _validate(model: type[BaseModel], data: Any, path: Path) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ScenarioConfigError(f"validation error in {path}:\n{exc}") from exc


def load_scenario(path: Path) -> ScenarioConfig:
    """Read a legacy scenario YAML and return a validated `ScenarioConfig`.

    The YAML must have a top-level `scenario:` key.
    """
    data = _read_yaml(Path(path))
    if not isinstance(data, dict) or "scenario" not in data:
        raise ScenarioConfigError(
            f"scenario config must have a top-level 'scenario:' key: {path}"
        )
    return _validate(ScenarioConfig, data["scenario"], path)


def load_experiment_scenarios(path: Path) -> ExperimentScenariosConfig:
    """Read a multi-scenario YAML and return a validated `ExperimentScenariosConfig`.

    The YAML must have a top-level `scenarios:` list and an `experiment_id:` field.
    """
    data = _read_yaml(Path(path))
    if not isinstance(data, dict) or "scenarios" not in data:
        raise ScenarioConfigError(
            f"multi-scenario config must have a top-level 'scenarios:' list: {path}"
        )
    return _validate(ExperimentScenariosConfig, data, path)


def load_config(
    path: Path,
) -> ScenarioConfig | ExperimentScenariosConfig:
    """Auto-detect and load either a legacy or multi-scenario YAML.

    * `scenario:` key  to :class:`ScenarioConfig` (legacy).
    * `scenarios:` key to :class:`ExperimentScenariosConfig` (multi-scenario).

    Raises:
        ScenarioConfigError: file not found, parse error, or validation failure.
    """
    data = _read_yaml(Path(path))
    if not isinstance(data, dict):
        raise ScenarioConfigError(f"YAML root must be a mapping: {path}")

    if "scenario" in data:
        return _validate(ScenarioConfig, data["scenario"], path)

    if "scenarios" in data:
        return _validate(ExperimentScenariosConfig, data, path)

    raise ScenarioConfigError(
        f"config must have a top-level 'scenario:' or 'scenarios:' key: {path}"
    )
