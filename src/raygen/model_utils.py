#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Module for modeling helper functions and model actor related functionalities."""
import logging
import os
from typing import Tuple

import filelock
import open_clip

from raygen.driver_utils import cached_download

_logger = logging.getLogger(__name__)


def open_clip_create_model_from_pretrained(
    s3_client, model_name: str, pretrained: str
) -> Tuple:
    """Create an open_clip model downloading if needed.

    Args:
        s3_client: S3 client to use
        model_name: Name of the model to create.
        pretrained: Path to pretrained model.

    Returns:
        A tuple of model object and transformations.

    """
    # First actor process that tries to grab the lock
    # will also download the model to the `model_cache_dir`
    # and will release the lock. Rest of the actors will
    # load the model from `model_cache_dir` without having
    # to download it.
    model_cache_dir = "/tmp/raygen/model"

    # Create the raygen model cache directory to download the model assets to.
    os.makedirs(model_cache_dir, exist_ok=True)

    with filelock.FileLock("/tmp/raygen/model.lock"):
        if model_name.startswith("hf-hub"):
            model, transforms = open_clip.create_model_from_pretrained(
                model_name,
                cache_dir=model_cache_dir,
            )
        else:
            if pretrained.startswith("s3"):
                _logger.info("Downloading s3 file: %s", pretrained)
                pretrained = str(cached_download(s3_client, pretrained))
            model, _, transforms = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                cache_dir=model_cache_dir,
            )
    return model, transforms
