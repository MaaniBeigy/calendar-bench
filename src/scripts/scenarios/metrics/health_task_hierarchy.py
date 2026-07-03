"""HealthTask SUBCLASSOF closure derived from the canonical JSON."""

from __future__ import annotations

import json
from pathlib import Path

HB_TASK_PREFIX = "https://w3id.org/calendar-bench/health/task/"


def load_class_closure(json_path: Path) -> dict[str, frozenset[str]]:
    """Return `{task_uri: frozenset(class_names_leaf_to_root)}` from HealthTasks JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: dict[str, frozenset[str]] = {}
    for domain, branches in data.items():
        for branch, levels in branches.items():
            for level, tasks in levels.items():
                for task_id, _payload in tasks.items():
                    classes = frozenset(
                        {
                            task_id,
                            f"{domain}{branch}{level}",
                            f"{domain}{branch}Task",
                            f"{domain}Task",
                            "HealthTask",
                        }
                    )
                    out[f"{HB_TASK_PREFIX}{task_id}"] = classes
    return out


__all__ = ["HB_TASK_PREFIX", "load_class_closure"]
