"""Per-person task entrypoint and the executor pool wrapper."""

from src.scripts.persona.concurrency.pool import run_pool
from src.scripts.persona.concurrency.tasks import run_person_task

__all__ = ["run_person_task", "run_pool"]
