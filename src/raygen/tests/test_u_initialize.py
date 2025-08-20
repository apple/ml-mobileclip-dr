#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Unit tests for initializing ray."""
import pytest
import ray

@pytest.fixture
def start_clean():
    if ray.is_initialized():
        ray.shutdown()


def test_initialize_ray_no_address():
    ray.init()
    assert ray.is_initialized() == True

    ray.shutdown()


def test_initialize_ray_auto_address():
    ray.init(address="auto")
    assert ray.is_initialized() == True

    ray.shutdown()


def test_initialize_ray_reinit():
    ray.init()
    assert ray.is_initialized() == True
    # Init ray again, with ignore reinit error.
    ray.init(ignore_reinit_error=True)
    assert ray.is_initialized() == True

    ray.shutdown()


@pytest.mark.parametrize("local", [(True,), (False,)])
def test_raygen_initialize_ray(local: bool):
    from ..driver_utils import initialize_ray

    initialize_ray(local=local)
    assert ray.is_initialized()

    ray.shutdown()
