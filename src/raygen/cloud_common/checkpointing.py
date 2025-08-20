#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Module for checkpointing."""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import ray

from raygen.cloud_common import storage_utils

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = "checkpoint.json"

_logger = logging.getLogger(__name__)


def get_or_create_checkpoint_dir(
    base_dir: str = "/data", clean_dir: bool = False
) -> str:
    """Creates or returns the path to the checkpoint dir in the local fs.

    Args:
        base_dir: Base directory in the local fs to use. Defaults to `/data` dir.
        clean_dir: Whether to clean up the checkpoints dir if exists.

    Returns: Path to the checkpoint directory where checkpoint.json is going to be created.

    """
    ckpt_dir = Path(base_dir, CHECKPOINT_DIR)
    if ckpt_dir.exists() and clean_dir:
        if clean_dir:
            try:
                shutil.rmtree(ckpt_dir)
            except Exception as e:
                _logger.warning(
                    "Could not clean up dir: %s due to: %s. Checkpoint might be stale.",
                    ckpt_dir, str(e)
                )

    ckpt_dir.mkdir(exist_ok=True, parents=True)
    return str(ckpt_dir)


def recover_from_checkpoint(output_path):
    """Check if we need to recover from a checkpoint."""
    # Initialize S3 Client
    s3_client = storage_utils.get_s3_client()

    checkpointed_paths = []
    local_checkpoint_file = os.path.join(
        get_or_create_checkpoint_dir(clean_dir=True),
        CHECKPOINT_FILE,
    )
    remote_checkpoint_path = os.path.join(output_path, CHECKPOINT_FILE)
    if storage_utils.file_exists(remote_checkpoint_path):
        logging.info("Found checkpoint at: %s and will recover from it.", output_path)
        # Download the checkpoint to local FS.
        storage_utils.download_file(
            s3_client=s3_client,
            s3_url=remote_checkpoint_path,
            local_path=local_checkpoint_file,
        )
        # load the checkpoint
        with open(local_checkpoint_file, "r") as fin:
            checkpoint_json = json.load(fin)
        checkpointed_paths = checkpoint_json["input_paths"]
    return remote_checkpoint_path, checkpointed_paths


@ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=5)
class GlobalCounter:
    """Global Ray actor for checkpointing the generation to allow safe resume."""

    def __init__(
        self, remote_checkpoint_path: str, checkpoint_upload_freq: int
    ) -> None:
        """Initialize GlobalCounter from a previous checkpoint if exists.

        Arg:
            remote_checkpoint_path: A string for the S3 path to upload the checkpoint.
            checkpoint_upload_frequency: An integer as the number of shards to process
                between each checkpoint upload.
        """
        level = logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s %(levelname)s : %(message)s"
        )
        root_logger = logging.getLogger(__name__)
        self.logger = root_logger
        self.s3_client = storage_utils.get_s3_client()
        self.local_checkpoint_file = os.path.join(
            get_or_create_checkpoint_dir(), CHECKPOINT_FILE
        )
        self.remote_checkpoint_path = remote_checkpoint_path
        self.checkpoint_upload_freq = checkpoint_upload_freq

        # Recover state from remote checkpoint if exists.
        if storage_utils.file_exists(remote_checkpoint_path):
            # checkpoint.json might already exist in local fs if
            # ray decides to place the GlobalCounter in head node
            # where the entrypoint script runs. So we check that.
            if not os.path.exists(self.local_checkpoint_file):
                storage_utils.download_file(
                    s3_client=self.s3_client,
                    s3_url=remote_checkpoint_path,
                    local_path=self.local_checkpoint_file,
                )
            with open(self.local_checkpoint_file, "r") as fin:
                checkpoint_json = json.load(fin)

            self.checkpointed_input_paths = checkpoint_json["input_paths"]
            self.checkpointed_output_paths = checkpoint_json["output_paths"]
            self.checkpointed_num_uids = checkpoint_json["num_uids"]
            self.time_seconds = checkpoint_json["time_seconds"]
            self.cpu_seconds = checkpoint_json["cpu_seconds"]
            self.gpu_seconds = checkpoint_json["gpu_seconds"]
            self.logger.info(
                "Recovered from checkpoint: %s with total UIDs and shards:\n%s",
                remote_checkpoint_path,
                sum(self.checkpointed_num_uids),
                len(self.checkpointed_input_paths),
            )
        else:
            # checkpoint does not exist so start new.
            self.checkpointed_input_paths = []
            self.checkpointed_output_paths = []
            self.checkpointed_num_uids = []
            self.time_seconds = []
            self.cpu_seconds = []
            self.gpu_seconds = []
        self.index = len(self.checkpointed_input_paths) - 1
        self.last_uploaded_index = self.index

        # Time stats
        self.last_checkpoint_time = datetime.now(timezone.utc)

        self.logger.info("Starting index: %s", self.index + 1)

    def ready(self) -> None:
        return

    def increment(self) -> int:
        self.index += 1
        self.logger.info("Returning index value: %s", self.index)
        return self.index

    def update_checkpoint(
        self, input_path: str, output_path: str, num_uid: int
    ) -> None:
        self.logger.info(
            "Updating checkpoint with payload: %s -> %s : %d",
            input_path,
            output_path,
            num_uid,
        )

        self.checkpointed_input_paths.append(input_path)
        self.checkpointed_output_paths.append(output_path)
        self.checkpointed_num_uids.append(num_uid)
        # TODO: add time left
        current_time = datetime.now(timezone.utc)
        resources = ray.cluster_resources()
        seconds_passed = (current_time - self.last_checkpoint_time).seconds
        self.time_seconds += [seconds_passed]
        self.cpu_seconds += [seconds_passed * resources["CPU"]]
        self.gpu_seconds += [seconds_passed * resources.get("GPU", 0)]
        self.last_checkpoint_time = current_time
        checkpoint_obj = {
            "input_paths": self.checkpointed_input_paths,
            "output_paths": self.checkpointed_output_paths,
            "num_uids": self.checkpointed_num_uids,
            "total_uids": sum(self.checkpointed_num_uids),
            "date": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "time_seconds": self.time_seconds,
            "cpu_seconds": self.cpu_seconds,
            "gpu_seconds": self.gpu_seconds,
            "total_time_h": sum(self.time_seconds) / 60.0,
            "total_cpu_h": sum(self.cpu_seconds) / 60.0,
            "total_gpu_h": sum(self.gpu_seconds) / 60.0,
        }
        if (self.index - self.last_uploaded_index) >= self.checkpoint_upload_freq:
            self.last_uploaded_index = self.index

            # Update local checkpoint file
            with open(self.local_checkpoint_file, "w") as fout:
                json.dump(checkpoint_obj, fout, indent=4)

            # Upload to S3
            bucket, key = storage_utils.parse_s3_url(self.remote_checkpoint_path)
            with open(self.local_checkpoint_file, "rb") as fobj:
                storage_utils.upload_fileobj(
                    s3_client=self.s3_client,
                    fileobj=fobj,
                    s3_bucket=bucket,
                    s3_key=key,
                )
            self.logger.info("Updated checkpoint.")
