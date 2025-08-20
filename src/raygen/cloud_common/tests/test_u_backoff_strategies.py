#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Unit tests for backoff_strategies module."""

import logging
from typing import List

import pytest

from ..backoff_strategies import (
    ExponentialPerturbatedBackoffStrategy,
    FixedTimeBackoffStrategy,
)


@pytest.mark.parametrize("sleep_time, max_attempts", [(0.01, 4)])
def test_fixed_sleep(sleep_time: float, max_attempts: int) -> None:
    backoff = FixedTimeBackoffStrategy(
        max_attempts=max_attempts, wait_time_s=sleep_time
    )
    # check the sleep time without actually sleeping
    assert backoff._backoff_time_s() == sleep_time


@pytest.mark.parametrize(
    "attempts, exponent, start_time, max_time, random_time, expected_times",
    [
        (5, 2.0, 1.0, 100.0, 0.0, [1.0, 2.0, 4.0, 8.0, 16.0]),
        (5, 2.0, 1.0, 7.0, 0.0, [1.0, 2.0, 4.0, 7.0, 7.0]),
        (5, 2.0, 2.0, 100.0, 0.0, [2.0, 4.0, 8.0, 16.0, 32.0]),
    ],
)
def test_exponential_backoff(
    attempts: int,
    exponent: float,
    start_time: float,
    max_time: float,
    random_time: float,
    expected_times: List[float],
) -> None:
    times: List[float] = []

    for i in range(attempts):
        time_i = ExponentialPerturbatedBackoffStrategy.calculate_wait_time_s(
            start_time_s=start_time,
            exponent_base=exponent,
            max_random_time_s=random_time,
            max_time_s=max_time,
            attempt=i,
        )
        times.append(time_i)

    assert times == expected_times
