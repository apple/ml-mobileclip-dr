#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Unit test for actor smoke tests and api compatibility."""
import pytest
import ray


@ray.remote(num_cpus=1, max_restarts=4, max_task_retries=-1)
class MyTestActor:
    def __init__(self):
        self.counter = 0

    def ready(self):
        return

    def increment(self):
        self.counter += 1
        return self.counter

    def get_count(self):
        return self.counter


@pytest.fixture
def setup():
    from ..driver_utils import initialize_ray
    initialize_ray(local=True)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_actor(count: int):
    """Smoke test for actors."""
    a = MyTestActor.remote()
    a.ready.remote()

    for i in range(count):
        a.increment.remote()

    current_count = ray.get(a.get_count.remote())
    assert current_count == count

    ray.kill(a)
