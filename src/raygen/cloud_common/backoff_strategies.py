#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Back-off strategies for retrying."""

import abc
import logging
import random
import sys
import time
from typing import Any, Callable, Optional

BackOffFactoryType = Callable[[], "BackoffStrategy"]


class BackoffStrategy(metaclass=abc.ABCMeta):
    """Abstract base class for a back-off strategy.

    A backoff strategy that inherits this class must implement the method _backoff_time_s to
    clearly define backoff behaviour.
    """

    def __init__(
        self, max_attempts: int, logger: Optional[logging.Logger] = None
    ) -> None:
        self._max_attempts = max_attempts
        self._attempts = 0
        self._start_seconds = None
        self.logger = logger if logger else logging.getLogger(self.__class__.__name__)

    def __call__(self, *args: Any, **kwargs: Any) -> bool:
        """Perform backoff if it has not expired."""
        if not self.has_expired():
            self.logger.debug("%s starting try #%s", self, self._attempts + 1)
            # Perform backoff
            self._backoff()
            self.logger.debug("%s finished try #%s", self, self._attempts + 1)
            self._attempts += 1
            return True
        else:
            self.logger.debug(
                "%s expired on try #%s -> no back-off applied", self, self._attempts + 1
            )
            return False

    def has_expired(self) -> bool:
        return self._attempts >= self._max_attempts

    def __repr__(self) -> str:
        return f"{self}(id={id(self)})"

    def _backoff(self) -> None:
        sleep_seconds = self._backoff_time_s()
        self.logger.debug("%s sleeping for %s seconds.", self, sleep_seconds)
        time.sleep(sleep_seconds)

    @abc.abstractmethod
    def _backoff_time_s(self) -> float:
        """Calculate wait time before next backoff but do not execute backoff."""
        raise NotImplementedError("Need to implement this method when subclassing!")


class FixedTimeBackoffStrategy(BackoffStrategy):
    def __init__(
        self,
        max_attempts: int,
        wait_time_s: float,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super(FixedTimeBackoffStrategy, self).__init__(
            max_attempts=max_attempts, logger=logger
        )
        self._wait_time_s = wait_time_s

    def _backoff_time_s(self) -> float:
        return self._wait_time_s

    def __repr__(self) -> str:
        return (
            f"FixedTimeBackoff(max_attempts={self._max_attempts}, "
            f"wait_time_s={self._wait_time_s}, id={id(self)})"
        )


class ExponentialPerturbatedBackoffStrategy(BackoffStrategy):
    def __init__(
        self,
        max_attempts: int,
        initial_wait_time_s: float,
        max_random_time_s: float,
        exponent_base: 2.0,
        max_wait_time: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize a back-off strategy with exponential wait times and wait time jitter.

        Args:
            max_attempts: The maximum number of attempts until this back-off expires
            initial_wait_time_s: The number of seconds slept on the first invocation
            max_random_time_s: The maximum random seconds added as jitter.
            exponent_base: base for exponential time calculation
            max_wait_time_s: The maximum sleep time in seconds. Upon reaching this limit,
                                     the back-off will remain max_sleep_seconds and will not
                                     continue to increase exponentially on each invocation.
            logger: optional logger to write back-off log-messages to
        """
        super(ExponentialPerturbatedBackoffStrategy, self).__init__(
            max_attempts=max_attempts, logger=logger
        )
        self._initial_wait_time_s = initial_wait_time_s
        self._exponent_base = exponent_base
        self._max_wait_time_s = max_wait_time
        self._max_random_time_s = max_random_time_s

    def __repr__(self) -> str:
        return (
            f"ExponentialBackoff(max_attempts={self._max_attempts}, "
            f"start_sleep_seconds={self._initial_wait_time_s}, "
            f"max_sleep_seconds={self._max_wait_time_s}, "
            f"factor={self._exponent_base}, id={id(self)})"
        )

    @staticmethod
    def calculate_wait_time_s(
        start_time_s: float,
        exponent_base: float,
        max_random_time_s: float,
        max_time_s: float,
        attempt: int,
    ) -> float:
        """Calculate wait time based on the exponent with random jitter and return.

        Args:
            start_time_s: multiplier for the exponential growth
            exponent_base: base for the exponent
            max_random_time_s: maximum time in seconds to be used for random jitter
            max_time_s: maximum_time_to_wait as cutoff (without jitter)
            attempt: attempt count

        Returns:
            wait time as a float
        """
        base_backoff_s = min(start_time_s * (exponent_base**attempt), max_time_s)
        random_delay_s = random.random() * max_random_time_s
        total_delay_s = base_backoff_s + random_delay_s
        return total_delay_s

    def _backoff_time_s(self) -> float:
        sleep_seconds = ExponentialPerturbatedBackoffStrategy.calculate_wait_time_s(
            start_time_s=self._initial_wait_time_s,
            exponent_base=self._exponent_base,
            attempt=self._attempts,
            max_time_s=self._max_wait_time_s,
            max_random_time_s=self._max_random_time_s,
        )
        return sleep_seconds
