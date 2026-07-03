"""Abstract base class for calendar augmenters."""

from __future__ import annotations

import abc
import datetime

from src.scripts.scenarios.config.schema import AugmentationConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask

HorizonHint = tuple[datetime.date, int]


class Augmenter(abc.ABC):
    """Plug-in interface for calendar augmentation strategies.

    All augmenters must implement :meth:`augment`, which takes the person's
    current `CalendarTrace`, the list of `RecommendedTask` objects to place,
    and the active `AugmentationConfig`, and returns a `SchedulingSolution`
    containing the placement decisions and the resulting `AugmentedCalendar`.

    Constraint checking is *post-hoc*: the augmenter is not required to produce
    a violation-free schedule.  The scheduling-loss evaluation measures the
    quality of the output after the fact.
    """

    @abc.abstractmethod
    def augment(
        self,
        calendar: CalendarTrace,
        tasks: list[RecommendedTask],
        config: AugmentationConfig,
        horizon: HorizonHint | None = None,
    ) -> SchedulingSolution:
        """Place recommended tasks into the calendar and return a solution.

        Args:
            calendar: the person's existing `CalendarTrace`.
            tasks: recommended-behavior tasks to be scheduled.
            config: augmentation configuration (method, merge settings, etc.).
            horizon: optional `(start_date, days)` override; when set, the
                augmenter places tasks over exactly that range instead of
                deriving the range from the calendar's events. Required
                when a scenario's `timeframe` may leave a person with zero
                base events in the window.

        Returns:
            A `SchedulingSolution` with `scheduled`, `unscheduled`, and
            the resulting `AugmentedCalendar`.
        """
