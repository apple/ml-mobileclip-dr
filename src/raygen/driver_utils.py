#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import hashlib
import logging
import os
import pathlib
from io import BytesIO
from itertools import islice
from typing import Any, Dict, List, Optional

import jsonlines
import numpy as np
import ray
from ray.data import Dataset

from raygen.cloud_common.storage_utils import (
    download_file,
    glob_files,
    parse_s3_url,
    upload_fileobj,
)

_DEFAULT_DASHBOARD_HOST = "0.0.0.0"
_DEFAULT_DASHBOARD_PORT = 8265
_DEFAULT_RAY_LOGGING_LEVEL = logging.INFO

_logger = logging.getLogger(__name__)


def initialize_ray(local: bool = False, **ray_kwargs: Any) -> None:
    """Initialize the ray client and set logging."""
    _logger.info("Initializing ray with local = %s mode.", local)

    if local:
        if ray.is_initialized():
            _logger.warning(
                "ray is already initialized. shutting it down before running `ray.init`."
            )
            ray.shutdown()

        _logger.info("initializing ray client.")
        # Create a local Ray client and runtime and then attach to it.
        ray.init(
            dashboard_host=_DEFAULT_DASHBOARD_HOST,
            dashboard_port=_DEFAULT_DASHBOARD_PORT,
            log_to_driver=True,
            logging_level=_DEFAULT_RAY_LOGGING_LEVEL,
            **ray_kwargs,
        )
    else:
        # Create a Ray client and connect it to a Ray Cluster
        # already running in the cloud job.
        ray.init(
            address="auto",
            log_to_driver=True,
            logging_level=_DEFAULT_RAY_LOGGING_LEVEL,
            **ray_kwargs,
        )


def create_ray_dataset_from_paths(
    uris: List[str],
    override_num_blocks: Optional[int] = None,
    maximize_num_blocks: Optional[bool] = None,
) -> Dataset:
    """Create a Dataset from info.json reading from local or S3.

    Args:
        override_num_blocks: Number of blocks to specify for the underlying Ray
            Dataset to change the default parallelism. If not specified Ray will
            use the `target_min_block_size` and `target_max_block_size` defaults.
        maximize_num_blocks: Boolean to specify whether to maximize number of output
            blocks irrespective of what the user has specified for `override_num_blocks`.
            This will set the number of output blocks to equal to the number of rows.
    Returns:
        A Ray Dataset.
    """
    if maximize_num_blocks:
        override_num_blocks = len(uris)

    return ray.data.from_items(
        [{"path": path} for path in uris], override_num_blocks=override_num_blocks
    )


def write_manifest(
    s3_client: Any,
    jsonl_lines: List[Dict[str, Any]],
    path: str,
    filename: str = "manifest.jsonl",
) -> None:
    full_path = os.path.join(path, filename)
    fileobj = BytesIO()
    try:
        with jsonlines.Writer(fileobj) as writer:
            writer.write_all(jsonl_lines)

        if path.startswith("s3://"):
            s3_bucket, s3_key = parse_s3_url(full_path)
            assert fileobj.readable()
            upload_fileobj(
                s3_client=s3_client,
                fileobj=fileobj,
                s3_bucket=s3_bucket,
                s3_key=s3_key,
            )
        else:
            os.makedirs(path, exist_ok=True)
            with open(full_path, "wb") as fout:
                fout.write(fileobj.getvalue())
                fout.write("\n".encode())
    except Exception as e:
        raise e
    finally:
        fileobj.close()


def create_grouped_paths(
    paths: List[str], glob: bool, glob_suffixes: List[str], grouping_folder_size: int
):
    input_base_path = paths[0].rsplit("/", 1)[0]

    # Uncomment next line for faster debugging
    # input_folders = input_folders[:1]

    all_paths = []
    for path in paths:
        # Assuming glob_files has been adjusted to return a dict even when depth is -1
        if glob:
            # Treat -1 as 0 here for no subfolder grouping
            globed_path = glob_files(path, suffixes=glob_suffixes, depth=0)
        # Assuming all paths are grouped under a single key
        all_paths.extend(globed_path.get("", []))

    grouped_input_paths = {}
    if grouping_folder_size == -1:
        grouped_input_paths[""] = all_paths
    else:
        digits = int(max(0, np.log10(len(all_paths) / grouping_folder_size)) + 1)
        for path_index in range(0, len(all_paths), grouping_folder_size):
            folder_index = path_index // grouping_folder_size
            folder_index_str = f"{folder_index:0{digits}}"
            path_last = min(len(all_paths), path_index + grouping_folder_size)
            grouped_input_paths[folder_index_str] = all_paths[path_index:path_last]
    return grouped_input_paths, input_base_path


def cache_file(url) -> pathlib.Path:
    """Compute environment dependent cache_file for a URL."""
    cache_dir = pathlib.Path("/tmp/")
    cache_dir.mkdir(exist_ok=True, parents=True)
    fname = url.replace("://", "_").replace("/", "_")
    if os.pathconf(cache_dir, "PC_NAME_MAX") <= len(fname):
        fname = fname[:10] + hashlib.sha256(str(fname).encode()).hexdigest() + fname[-10:]
    return cache_dir / fname


def cached_download(s3_client, path) -> pathlib.Path:
    local_path = cache_file(path)
    if not os.path.exists(local_path):
        download_file(s3_client=s3_client, s3_url=path, local_path=local_path)
    return local_path


def batched(iterable, *, step):
    """Create an itertools.batched like iterator over an iterable."""
    # batched('ABCDEFG', 3) → ABC DEF G
    if step < 1:
        raise ValueError("step size must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, step)):
        yield batch
