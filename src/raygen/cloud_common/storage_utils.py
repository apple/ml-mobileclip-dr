#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Module for cloud storage related utility functions."""

import glob
import hashlib
import logging
import os
from collections import defaultdict
from io import BytesIO
from typing import IO, Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import boto3
import boto3.session
import botocore
import botocore.exceptions
import notary_client.conductor as conductor  # TODO: OSS remove
import urllib3
from boto3.s3.transfer import TransferConfig
from google.cloud import storage as gstorage

from .backoff_strategies import ExponentialPerturbatedBackoffStrategy
from .retrying import retry

_MiB: int = int(1024**2)
_GiB: int = int(1024**3)

_DEFAULT_AWS_REGION = "us-west-2"  # OSS remove
_DEFAULT_AWS_PROFILE = "p8"  # OSS remove

AWS_DEFAULT_REGION_VARNAME: str = "AWS_DEFAULT_REGION"
AWS_ACCESS_KEY_ID_VARNAME: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_VARNAME: str = "AWS_SECRET_ACCESS_KEY"
AWS_SESSION_TOKEN_VARNAME: str = "AWS_SESSION_TOKEN"
AWS_PROFILE_VARNAME: str = "AWS_PROFILE"

MAX_ATTEMPTS: int = 10
INITIAL_WAIT_TIME_S: float = 1.0

_logger = logging.getLogger(__name__)

_s3_retry_exceptions = [
    boto3.exceptions.S3UploadFailedError,
    botocore.exceptions.ClientError,
    botocore.exceptions.CredentialRetrievalError,
    botocore.exceptions.EndpointConnectionError,
    botocore.exceptions.ReadTimeoutError,
    botocore.exceptions.WaiterError,
    urllib3.exceptions.ProtocolError,
    ConnectionResetError,
    ValueError,
]

s3_retry = retry(
    on_exceptions=tuple(_s3_retry_exceptions),
    backoff_factory=lambda: ExponentialPerturbatedBackoffStrategy(
        max_attempts=MAX_ATTEMPTS,
        initial_wait_time_s=INITIAL_WAIT_TIME_S,
        max_random_time_s=4.0,
        exponent_base=2.0,
        max_wait_time=60.0,
        logger=_logger,
    ),
)


def aws_session_from_env_vars() -> boto3.Session:
    """Create and return an AWS session from env variables."""

    env_region = os.environ.get(AWS_DEFAULT_REGION_VARNAME)
    env_key = os.environ.get(AWS_ACCESS_KEY_ID_VARNAME)
    env_secret = os.environ.get(AWS_SECRET_ACCESS_KEY_VARNAME)
    env_session_token = os.environ.get(AWS_SESSION_TOKEN_VARNAME)
    env_profile = os.environ.get(AWS_PROFILE_VARNAME)

    if not env_region:
        _logger.debug("Missing env region. Using default of: %s", _DEFAULT_AWS_REGION)
        env_region = _DEFAULT_AWS_REGION

    if not env_profile:
        _logger.debug(
            "Missing env profile `AWS_PROFILE` using default of: %s",
            _DEFAULT_AWS_PROFILE,
        )
        env_profile = _DEFAULT_AWS_PROFILE

    _logger.debug(
        "Attempting to creating an AWS session from profile: %s and region_name: %s",
        env_profile,
        env_region,
    )

    try:
        session = boto3.session.Session(
            profile_name=env_profile, region_name=env_region
        )
        return session

    except Exception:
        _logger.warning(
            "Could not create boto session from profile. Trying with env variables.",
            exc_info=1,
        )

        if not env_key:
            raise Exception("Missing env key `AWS_ACCESS_KEY_ID_VARNAME`")

        if not env_secret:
            raise Exception("Missing env secret `AWS_SECRET_ACCESS_KEY_VARNAME`")

        if not env_session_token:
            raise Exception("Missing env session token `AWS_SESSION_TOKEN_VARNAME`")

        return boto3.session.Session(
            aws_access_key_id=env_key,
            aws_secret_access_key=env_secret,
            aws_session_token=env_session_token,
            region_name=env_region,
        )


def aws_session_from_path(path) -> boto3.Session:
    """Create and return an AWS session from path prefix <profile>://."""

    profile = path.split(":")[0]
    session = boto3.session.Session(profile_name=profile)
    return session


def get_s3_client(path: str = None):
    if path is None or path.startswith("s3"):
        s3_session = aws_session_from_env_vars()
        s3_client = s3_session.client("s3")
    elif path.startswith("conductor"):
        # TODO: OSS remove
        s3_client = conductor.get_s3_client()
    else:
        s3_session = aws_session_from_path(path)
        s3_client = s3_session.client("s3")
    return s3_client


def get_client(path: str = None):
    if path is None or path.startswith("s3"):
        s3_session = aws_session_from_env_vars()
        client = s3_session.client("s3")
    elif path.startswith("conductor"):
        # TODO: OSS remove
        client = conductor.get_s3_client()
    elif path.startswith("gs"):
        project_name = parse_gs_project_name(path)
        client = gstorage.Client(project_name)
    else:
        s3_session = aws_session_from_path(path)
        client = s3_session.client("s3")
    return client


def is_s3_path(path: str) -> bool:
    return not path.startswith("gs")


def parse_gs_project_name(path: str) -> str:
    """Returns the project name from gs paths of the form gs://bucket_name@project_name/."""
    remote, path = path.split("://")
    parts = path.split("/", 1)
    bucket = parts[0]
    return bucket.split("@")[1] if "@" in bucket else None


def parse_remote_path(path: str) -> Tuple[str, str, str]:
    """
    Extract the bucket and key from an S3 or GS path.

    Args:
    - remote_path (str): The remote path in the format
        "<s3|gs|conductor|profile>://bucket/key"  # OSS remove

    Returns:
    - tuple: A tuple containing the bucket and key

    Raises ValueError when path is invalid.
    """
    if "://" not in path:
        raise ValueError("Invalid S3/GS path format. Must include '://'")

    remote, path = path.split("://")
    parts = path.split("/", 1)
    bucket = parts[0]
    bucket = bucket.split("@")[0]  # Drop project-name from a gs path

    if len(parts) > 1:
        key = parts[1]
    else:
        key = ""
    return remote, bucket, key


def get_upload_transfer_config() -> TransferConfig:
    """Create a transfer config for data uploads.

    This is the default transfer config. It is optimized for very large files (multiple GB)
    and quite conservative in throughput. It is likely useful to provide a TransferConfig for
    your use cases if you have know your data properties and bucket policies well.
    We plan to provide pre-configured S3 transfer policies for common data patterns in future.

    Returns:
        upload transfer config

    """
    transfer_config = TransferConfig(
        multipart_threshold=5 * _GiB,
        multipart_chunksize=5 * _GiB,
        max_concurrency=1,
        use_threads=True,
    )
    return transfer_config


def get_download_transfer_config() -> TransferConfig:
    """Create a transfer config for data downloads.

    This is the default transfer config. It is optimized for very large files (multiple GB)
    and quite conservative in throughput. It is likely useful to provide a TransferConfig for
    your use cases if you have know your data properties and bucket policies well.
    We plan to provide pre-configured S3 transfer policies for common data patterns in future.

    Returns:
        download transfer config

    """
    transfer_config = TransferConfig(
        multipart_threshold=2 * _GiB,
        multipart_chunksize=2 * _GiB,
        max_concurrency=2,
        use_threads=True,
    )
    return transfer_config


def parse_s3_url(s3url: str) -> Tuple[str, str]:
    """Parse S3 URL to into bucket and key components.

    Args:
        s3url: URL to parse

    Returns:
        string tuple of bucket and object key

    """
    parsed_url = urlparse(s3url, allow_fragments=False)
    if not parsed_url.netloc:
        raise Exception('Please provide a bucket_name instead of "%s"' % s3url)
    else:
        bucket_name = parsed_url.netloc
        # Remove both leading and trailing slashes from key.  We want only the path and query
        # portion of the url.
        key = urlunparse(parsed_url._replace(scheme="", netloc="")).strip("/")
        return bucket_name, key


@s3_retry
def list_keys(bucket_name: str, prefix: str, delimiter: str = "") -> List[str]:
    """List keys given bucket name and prefix.

    Args:
        bucket_name: Bucket name.
        prefix: Prefix.
        delimiter: Delimiter, default=''.

    Returns:
        List of keys found with the bucket name and prefix.

    """
    s3_session = aws_session_from_env_vars()
    paginator = s3_session.resource("s3").meta.client.get_paginator("list_objects_v2")
    response = paginator.paginate(
        Bucket=bucket_name, Prefix=prefix, Delimiter=delimiter
    )
    keys = []
    for page in response:
        if "Contents" in page:
            for k in page["Contents"]:
                keys.append(k["Key"])
    return keys


def get_file_list(
    bucket_name: str, prefix: str, file_ext: Optional[str] = None
) -> List[str]:
    """Get all s3 keys with the given bucket and prefix.

    Args:
        bucket_name: Bucket name.
        prefix: Prefix
        file_ext: Optional file extension to filter file list.

    Returns:
        List of keys.

    """
    if not prefix.endswith("/"):
        prefix += "/"
    file_urls = [url for url in list_keys(bucket_name, prefix) if prefix in url]
    if file_ext:
        file_urls = [file_url for file_url in file_urls if file_url.endswith(file_ext)]
    return file_urls


def file_exists(s3_url: str) -> bool:
    """Check whether a remote file exists at the given url.

    Args:
        s3_url: S3 url of the file

    Returns:
        True if file exists, False otherwise.

    Raises:
        ValueError if s3_url is not a file.

    """
    if s3_url.endswith("/"):
        raise ValueError(f"Expected a file but got a directory instead: {s3_url}")

    bucket_name, key = parse_s3_url(s3_url)
    keys = list_keys(bucket_name=bucket_name, prefix=key)
    if keys:
        return True
    return False


@s3_retry
def download_fileobj(
    s3_client: Any, s3_url: str, s3_transfer_config: Optional[TransferConfig] = None
) -> BytesIO:
    """Download S3 object to a file-like object.

    Example:
        # Retrieving bytes content
        with s3_utils.download_fileobj(client, s3_url) as fileobj:
            bytes_content = fileobj.read()

        # Retrieving and converting to json
        with s3_utils.download_fileobj(client, json_s3_url) as fileobj:
            json_object = json.load(fileobj)

        with s3_utils.download_fileobj(client, json_s3_url) as fileobj:
            bytes_content = fileobj.read()
            json_object = json.loads(bytes_content.decode())

    Args:
        s3_client: S3 client to use for download
        s3_url: S3 URL to download from
        s3_transfer_config: optional transfer policy to use case specific throughput optimization

    Returns:
        Ready to read BytesIO object containing the bytes content of the S3 object

    """
    download_bucket, download_key = parse_s3_url(s3_url)
    logging.debug("Downloading %s/%s to BytesIO object", download_bucket, download_key)
    effective_transfer_config = (
        s3_transfer_config if s3_transfer_config else get_download_transfer_config()
    )
    fileobj = BytesIO()
    s3_client.download_fileobj(
        Bucket=download_bucket,
        Key=download_key,
        Fileobj=fileobj,
        Config=effective_transfer_config,
    )
    # boto3 download_fileobj does not reset position of the file-like
    # object after writing of content so seek to the start of the buffer
    # to allow immediate usage of contents
    fileobj.seek(0)
    return fileobj


@s3_retry
def download_file(
    s3_client: Any,
    s3_url: str,
    local_path: str,
    s3_transfer_config: Optional[TransferConfig] = None,
) -> None:
    """Download a single file from S3 to a local path with rate-limiting handling retries.

    Args:
        s3_client: S3 client to use for download
        s3_url: S3 URL to download from
        local_path: local path to store the file in
        s3_transfer_config: optional transfer policy to use case specific throughput optimization

    """
    download_bucket, download_key = parse_s3_url(s3_url)
    effective_transfer_config = (
        s3_transfer_config if s3_transfer_config else get_download_transfer_config()
    )
    try:
        s3_client.download_file(
            download_bucket, download_key, local_path, Config=effective_transfer_config
        )
    except botocore.exceptions.ClientError:
        logging.error("Error downloading key: %s", s3_url)
        if os.path.exists(local_path):
            logging.debug("Clean-up downloaded file: %s", local_path)
            os.remove(local_path)
        raise


@s3_retry
def upload_fileobj(
    s3_client: Any,
    fileobj: IO[bytes],
    s3_bucket: str,
    s3_key: str,
    s3_transfer_config: Optional[TransferConfig] = None,
    validate_upload: bool = False,
) -> None:
    """Upload a file-like object to an S3 object.

    Args:
        s3_client: S3 client to use for upload
        fileobj: A file-like object to upload to S3
        s3_bucket: S3 bucket to upload to
        s3_key: S3 key (including prefix) to upload to
        s3_transfer_config: optional transfer policy to use case specific throughput optimization
        validate_upload: optionally validate the upload

    """
    logging.debug("Uploading file-like object to %s/%s", s3_bucket, s3_key)
    fileobj.seek(0)
    data = fileobj.read()
    # Compute checksum hash using SHA256 prior to upload.
    sha256_hash = hashlib.sha256()
    sha256_hash.update(data)
    local_sha256 = sha256_hash.hexdigest()

    effective_transfer_config = (
        s3_transfer_config if s3_transfer_config else get_upload_transfer_config()
    )

    try:
        # Reset the fileobj pointer to the beginning before uploading.
        fileobj.seek(0)
        s3_client.upload_fileobj(
            Bucket=s3_bucket,
            Key=s3_key,
            Fileobj=BytesIO(data),
            Config=effective_transfer_config,
            ExtraArgs={'ChecksumAlgorithm': 'SHA256'},
        )
    except Exception:
        raise
    else:
        if validate_upload:
            logging.debug(
                "Validating the upload to the bucket %s, object %s", s3_bucket, s3_key
            )
            object_waiter = s3_client.get_waiter("object_exists")
            object_waiter.wait(Bucket=s3_bucket, Key=s3_key)
            obj_head = s3_client.head_object(Bucket=s3_bucket, Key=s3_key)

            # Get the sha256 checksum hash from S3 API response.
            returned_sha256 = obj_head.get("ChecksumSHA256")
            if returned_sha256 and returned_sha256 != local_sha256:
                raise ValueError("Corrupted upload to S3.")


def glob_files(path: str, suffixes: List[str], depth: int = 0) -> Dict[str, List[str]]:
    """
    Glob files based on a given path and a list of suffixes, grouping them by subfolder depth.
    Supports both local and S3 paths. Ensures that files matching multiple suffixes are only added once,
    prioritizing longer suffixes to avoid duplicates. Always returns a dictionary where each key
    represents a subfolder at the specified depth, and each value is a list of unique file
    paths with suffixes matching those provided in the list. A depth of 0 implies no
    subfolder grouping, but files are still returned within a single key.

    :param path: path to glob. Can be local or S3 (e.g., s3://bucket-name/path/)
    :param suffixes: list of suffixes of files to match
    :param depth: depth of subfolders for grouping. Defaults to 0 (group all at root)
    :return: dict of grouped file paths, with subfolder paths as keys
    """
    grouped_files = defaultdict(list)
    # Sort suffixes by length (longest first) to prioritize specific matches
    sorted_suffixes = sorted(suffixes, key=len, reverse=True)

    matched_files = set()  # To keep track of files already added

    if is_s3_path(path):
        s3 = get_s3_client(path)
        remote, bucket_name, prefix = parse_remote_path(path)
        if not prefix.endswith("/"):
            prefix += "/"
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                if (
                    any(suffix in obj["Key"] for suffix in sorted_suffixes)
                    and obj["Key"] not in matched_files
                ):
                    parts = obj["Key"][len(prefix) :].split("/")
                    group_key = "/".join(parts[:depth]) if depth > 0 else ""
                    grouped_files[group_key].append(
                        f"{remote}://{bucket_name}/{obj['Key']}"
                    )
                    matched_files.add(obj["Key"])

    elif path.startswith("gs://"):
        # Initialise a client
        project_name = parse_gs_project_name(path)
        storage_client = gstorage.Client(project_name)
        remote, bucket_name, prefix = parse_remote_path(path)
        # Create a bucket object for our bucket
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
        for blob in blobs:
            file_name = blob.name
            if (
                any(suffix in file_name for suffix in sorted_suffixes)
                and file_name not in matched_files
            ):
                parts = file_name[len(prefix) :].split("/")
                group_key = "/".join(parts[:depth]) if depth > 0 else ""
                grouped_files[group_key].append(f"gs://{bucket_name}/{file_name}")
                matched_files.add(file_name)

    else:
        for suffix in sorted_suffixes:
            search_pattern = f"{path.rstrip('/')}/**/*{suffix}"
            for file_path in glob.glob(search_pattern, recursive=True):
                if file_path not in matched_files:
                    relative_path = os.path.relpath(file_path, start=path)
                    parts = relative_path.split(os.sep)
                    group_key = os.sep.join(parts[:depth]) if depth > 0 else ""
                    grouped_files[group_key].append(file_path)
                    matched_files.add(file_path)

    return dict(grouped_files)
