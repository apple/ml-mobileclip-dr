#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Tools for reading and writing sharded datasets.

Parts of this code have been taken from https://github.com/mlfoundations/open_lm/blob/main/open_lm/datapreprocess/ray/tokenize_shuffle.py
"""
import abc
import enum
import gzip
import io
import json
import logging
from datetime import datetime, timezone
import tempfile
import os
import pathlib
import pickle
import re
import tarfile
import time
from io import BytesIO
from typing import IO, Any, BinaryIO, Dict, List

import jsonlines
import numpy as np
import pybase64
import ray
import torch
import webdataset as wds
import zstandard as zstd
from PIL import Image
from pympler import asizeof
from ray.actor import ActorHandle

from raygen.cloud_common.storage_utils import (
    _MiB,
    download_fileobj,
    download_file,
    get_client,
    is_s3_path,
    parse_remote_path,
    parse_s3_url,
    upload_fileobj,
)

_logger = logging.getLogger(__name__)
DIR = pathlib.Path(__file__).parent.absolute()
INFO_JSON_SUFFIX = "info.json"


class RawFileType(enum.Enum):
    """Supported raw shard file types."""

    JSONL = 1
    ZSTD_JSONL_COMPRESSED = 2
    GZIP_JSONL_COMPRESSED = 3
    TAR = 4
    TFRECORD = 5
    UNKNOWN = -1


class ShardReader(metaclass=abc.ABCMeta):
    """Base abstract class for shard readers."""

    def __init__(self, content_key: str, rekey: Dict[str, str]) -> None:
        """Initialiaztion with default filtering/conversion arguments.

        Args:
            content_key: A regular expression for filterring for a selection of keys.
            rekey: A dictionary for mapping input keys to output keys.
            enable_convert_value: A boolean that controls whether to apply key mapping.
        """
        self.content_key = content_key
        self.rekey = rekey
        self.enable_convert_value = False

    def shard_iterator(self, fh: BinaryIO):
        """Return a shard iterator that returns items given a file handle."""
        raise NotImplementedError("`shard_iterator' not implemented.")

    def convert_key(self, k: str) -> str:
        """Convert keys given a mapping of input to output keys in self.rekey."""
        return self.rekey.get(k, k) if self.rekey is not None else k

    def convert_value(self, k: str, v: str) -> str:
        """Potentially convert the value with some logic."""
        return v

    def convert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Filter and convert a single (key, value) item in the dataset."""
        if (
            (self.content_key is None or self.content_key == ".*")
            and self.rekey is None
            and not self.enable_convert_value
        ):
            return item
        return {
            self.convert_key(k, self.rekey): self.convert_value(v)
            for k, v in item.items()
            if re.match(self.content_key, k)
        }


class JsonlReader(ShardReader):
    """The reader for jsonl shard format where a line contains a single item as dict."""

    def shard_iterator(self, fh: BinaryIO):
        """Return a shard iterator that returns items given a file handle."""
        with io.TextIOWrapper(fh, encoding="utf-8") as text_reader:
            with jsonlines.Reader(text_reader) as jsonl_reader:
                for item in jsonl_reader.iter(type=dict, skip_invalid=True):
                    yield self.convert_item(item)


class ZstdCompressedReader(JsonlReader):
    """The reader for jsonl.zstd format where a line contains a single item as dict."""

    def shard_iterator(self, fh: BinaryIO):
        """Return a shard iterator that returns items given a file handle."""
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            for item in super().shard_iterator(reader):
                yield item


class GzipCompressedReader(ShardReader):
    """The reader for jsonl.gz format where a line contains a single item as dict."""

    def shard_iterator(self, fh: BinaryIO):
        """Return a shard iterator that returns items given a file handle."""
        with gzip.open(fh, "rb") as f_in:
            with jsonlines.Reader(f_in) as jsonl_reader:
                for item in jsonl_reader.iter(type=dict, skip_invalid=True):
                    yield self.convert_item(item)


class TarReader(ShardReader):
    """The reader for .tar webdataset shards.

    Webdataset format represents keys as individual files in the shard where file name
    is the item key.
        UID1.txt
        UID1.jpg
        UID1.json
        ...
        UIDn.txt
        UIDn.jpg
        UIDn.json
    """

    def shard_iterator(self, fh: BinaryIO):
        """Return a shard iterator that returns items given a file handle.

        The webdataset shard iteror currently handles a selected content types.
        """
        # TODO: use webdataset loader
        buffer = io.BytesIO(fh.read())
        sample_key, sample = None, {}
        with tarfile.open(fileobj=buffer, mode="r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Assume an item is saved sequentially in the tar
                    with tar.extractfile(member) as fileobj:
                        if fileobj:  # Ensure fileobj is not None
                            content_key, content_ext = member.name.split(".", 1)
                            if sample_key is not None and sample_key != content_key:
                                yield self.convert_item(sample)
                                sample_key, sample = None, {}
                            sample_key = member.name
                            sample["uid"] = sample_key = content_key
                            if content_ext == "url.txt":
                                sample["url"] = fileobj.read().decode("utf-8")
                            elif content_ext == "txt":
                                sample["text"] = fileobj.read().decode("utf-8")
                            elif content_ext == "jpg":
                                sample["image"] = fileobj.read()
                            elif content_ext.endswith(
                                "json"
                            ):  # json, syn.json, paug.json
                                sample.update(json.load(fileobj))
                            elif content_ext == "npy" or content_ext == "npz":
                                emb_dict = np.load(
                                    io.BytesIO(fileobj.read()), allow_pickle=True
                                )
                                sample.update(emb_dict)
                            elif content_ext == "pth.gz":
                                emb_dict = torch.load(
                                    gzip.GzipFile(fileobj=fileobj, mode="rb")
                                )
                                sample.update(emb_dict)
                            else:
                                raise ValueError(
                                    f"Unsupported content key extension: {content_key}"
                                )
        if sample_key is not None:
            yield self.convert_item(sample)


class TFRecordReader(ShardReader):
    """The reader for .tfrecord shard formats."""

    def __init__(self, *args, shuffle=False, **kwargs):
        """Initialize TFRecord reader and the default disable value conversion."""
        super().__init__(*args, **kwargs)
        self.enable_convert_value = True
        self.shuffle = shuffle

    def convert_value(k, v):
        """Convert the key and values from the encoding of tfrecord."""
        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None
        import warnings

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
        try:
            if k == "image":
                image = Image.open(BytesIO(v[0]))
                buffer = BytesIO()
                image.save(buffer, format=image.format)
                return pybase64.b64encode(buffer.getvalue()).decode("utf-8")
            elif isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return v.item()
                return v
            else:
                return v[0].decode("utf-8")
        except Exception as e:
            _logger.warning(f"Skipping tfrecord conversion: {e}")
            return None

    def shard_reader(self, fh: BinaryIO):
        """Return a shard iterator that returns items given a file handle."""
        from torchdata.datapipes.iter import TFRecordLoader

        buffer = io.BytesIO(fh.read())
        if self.shuffle:
            tfrecord_reader = TFRecordLoader([("", buffer)]).shuffle()
        else:
            tfrecord_reader = TFRecordLoader([("", buffer)])
        for item in tfrecord_reader:
            yield self.convert_item(item)


def create_paths_from_info_json(
    s3_client: Any,
    path: str,
) -> List[str]:
    """Create a path list from info.json reading from local or S3.

    Args:
        s3_client: S3 client to use
        path: Path to info.json to read from
    Returns:
        A list of path strings.

    Raises:
        ValueError if path is malformed or does not end with info.json.
    """
    if not path.endswith(INFO_JSON_SUFFIX):
        raise ValueError("Provided path is not info.json: ", path)

    info_json = None
    if os.path.isfile(path):
        with open(path, "r") as fin:
            info_json = json.load(fin)

    elif path.startswith("s3://"):
        _logger.info("Path is: %s", path)
        with tempfile.TemporaryDirectory() as temp_dir:
            _logger.info("Downloading s3 file: %s to local dir: %s", path, temp_dir)
            download_file(
                s3_client=s3_client,
                s3_url=path,
                local_path=os.path.join(temp_dir, INFO_JSON_SUFFIX),
            )
            with open(os.path.join(temp_dir, INFO_JSON_SUFFIX), "r") as fin:
                info_json = json.load(fin)

    else:
        raise ValueError("Input path must be a local file or on s3.")
    paths = info_json["lengths"].keys()
    return paths


def create_info_json_data(
    jsonl_lines: List[Dict[str, Any]], path: str
) -> Dict[str, Any]:
    """Create info.json file with information about all shards and number of samples."""
    info_dict = {
        "name": path.strip("/"),
        "lengths": {
            os.path.join(path.strip("/"), "{}.jsonl".format(item["shard"])): str(
                item["num_uids"]
            )
            for item in jsonl_lines
        },
        "totalLength": str(sum([item["num_uids"] for item in jsonl_lines])),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return info_dict


def create_paths(
    s3_client: Any,
    path: str,
) -> List[str]:
    """Create a path list reading from local or S3.

    Args:
        s3_client: S3 client to use
        path: Path to info.json to read from
    Returns:
        A list of path strings.

    Raises:
        ValueError if path is malformed or does not end with info.json.
    """
    if path.endswith("info.json"):
        paths = create_paths_from_info_json(s3_client, path)
    else:
        paths = wds.shardlists.expand_urls(path)
    return paths


def get_image_bytes(item: Dict[str, Any], *, image_key: str) -> bytes:
    """Get the image inside the record and return as bytes.

    Args:
        item: A dictionary representing a record
        image_key: Key to use to read the image in the record

    Returns:
        Read image in bytes if exists if not empty bytes.
    """
    image = item.get(image_key)
    if hasattr(image, "tobytes"):
        img_bytes = image.tobytes()
    elif isinstance(image, str):
        img_bytes = pybase64.b64decode(image, validate=True)
    elif isinstance(image, bytes):
        img_bytes = image
    else:
        img_bytes = b""
        _logger.warning("Failed to decode image bytes. Returning empty.")
    return img_bytes


def get_reader(
    path: str,
    content_key: str = None,
    shuffle: bool = False,
    rekey: Dict[str, str] = None,
) -> ShardReader:
    """Initilize and return a shard reader."""
    file_type = get_raw_filetype(path)
    if file_type == RawFileType.JSONL:
        return JsonlReader(content_key=content_key, rekey=rekey)
    if file_type == RawFileType.ZSTD_JSONL_COMPRESSED:
        return ZstdCompressedReader(content_key=content_key, rekey=rekey)
    if file_type == RawFileType.GZIP_JSONL_COMPRESSED:
        return GzipCompressedReader(content_key=content_key, rekey=rekey)
    if file_type == RawFileType.TAR:
        return TarReader(content_key=content_key, rekey=rekey)
    if file_type == RawFileType.TFRECORD:
        return TFRecordReader(content_key=content_key, rekey=rekey, shuffle=shuffle)
    else:
        raise Exception("Unsupported filetype")


def get_raw_filetype(key: str) -> RawFileType:
    """Return shard raw file type format."""
    if any(key.endswith(e) for e in [".jsonl", ".json"]):
        return RawFileType.JSONL
    elif any(
        key.endswith(e) for e in [".jsonl.zst", "json.zst", "jsonl.zstd", "json.zstd"]
    ):
        return RawFileType.ZSTD_JSONL_COMPRESSED
    elif any(key.endswith(e) for e in [".jsonl.gz", ".json.gz"]):
        return RawFileType.GZIP_JSONL_COMPRESSED
    elif key.endswith(".tar"):
        return RawFileType.TAR
    elif ".tfrecord" in key:
        return RawFileType.TFRECORD
    else:
        _logger.warning(f"Unknown filetype: {key}")
        return RawFileType.UNKNOWN


def get_reader_handle(
    path: str,
    content_key: str = None,
    rekey: Dict[str, str] = None,
    shuffle: bool = False,
):
    _logger.info("Downloading: %s to BytesIO object", path)
    download_tik = time.perf_counter()
    if is_s3_path(path):
        s3_client = get_client(path)
        remote, bucket, key = parse_remote_path(path)
        # response = s3_client.get_object(Bucket=bucket, Key=key)
        # fh = response["Body"]
        # Download is retry-safe compared with object-handle
        fileobj = download_fileobj(s3_client, path)
    elif path.startswith("gs"):
        remote, bucket_name, key = parse_remote_path(path)
        # Initialise a client
        gs_client = get_client(path)
        # Create a bucket object for our bucket
        bucket = gs_client.get_bucket(bucket_name)
        # Create a blob object from the filepath
        blob = bucket.blob(key)
        # fh = blob.download_as_bytes()
        fileobj = blob.open("rb")
    else:
        key = path
        fileobj = open(path, "rb")
    download_tok = time.perf_counter()
    _logger.info(
        "Downloaded: %s to BytesIO object. Duration: %.4f seconds",
        path,
        (download_tok - download_tik),
    )
    fileobj.seek(0)

    file_reader_f = get_reader(path, content_key, rekey, shuffle)
    file_reader_fh = file_reader_f.shard_iterator(fileobj)
    return file_reader_fh


def read_shard(s3_client, path) -> List[Dict[str, Any]]:
    """Read all the data from a single shard path."""
    data: List[Dict[str, Any]] = []

    _logger.info("Downloading: %s to BytesIO object", path)
    download_tik = time.perf_counter()
    fileobj = download_fileobj(s3_client=s3_client, s3_url=path)
    download_tok = time.perf_counter()
    _logger.info(
        "Downloaded: %s to BytesIO object. Duration: %.4f seconds",
        path,
        (download_tok - download_tik),
    )
    fileobj.seek(0)

    file_reader_f = get_reader(path)  # TODO(fartash): add support for key/rekey args
    file_reader_fh = file_reader_f.shard_iterator(fileobj)
    for obj in file_reader_fh:
        # TODO(Cem): Add a counter to expose how many we skip.
        data.append(obj)
    fileobj.close()
    _logger.info(
        "Materialized dataset: %s with items: %s. Total size: %.4f MiB",
        path,
        len(data),
        asizeof.asizeof(data) / _MiB,
    )

    return data


def lossless_compress(v: np.array) -> str:
    """Return a compressed string encoding of a numpy array."""
    v = np.array(v, dtype="float32")
    v = pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
    v = gzip.compress(v)
    v = pybase64.b64encode_as_string(v)
    return v


def lossless_decompress(v: str) -> np.array:
    """Return a decompressed numpy array from the string encoding."""
    if isinstance(v, list) or isinstance(v, torch.Tensor):
        # Not compressed
        return v
    v = pybase64.b64decode(v)
    v = gzip.decompress(v)
    v = pickle.loads(v)
    return v


class ShardWriter(metaclass=abc.ABCMeta):
    """Abstract class for dataset shard writers."""

    def __init__(
        self,
        s3_client,
        output_path: str,
        output_shard_name_format: str,
        bfloat16: bool,
        counter_actor: ActorHandle,
    ) -> None:
        """Initialize shard writer.

        Args:
            s3_client: A S3 client.
            output_path: A string for the base output path.
            output_shard_name_format: A string that specifies output shard format
                (count|orig).
            bfloat16: A boolean that if true stores embeddings in BFloat16.
            counter_actor: A Ray ActorHandle.
        """
        self.s3_client = s3_client
        self.output_path = output_path
        self.output_shard_name_format = output_shard_name_format
        self.bfloat16 = bfloat16
        self.counter_actor = counter_actor

    def write_shard(self, data, input_path: str):
        """Create a shard and write to output location."""
        shard_name_format = self.output_shard_name_format
        shard_index = ray.get(self.counter_actor.increment.remote())
        shard_name: str = ""
        if shard_name_format == "count":
            digits = 8
            shard_name = f"{shard_index:0{digits}}.{self.shard_extension}"
        elif shard_name_format == "orig":
            shard_name = os.path.basename(input_path)
        else:
            raise ValueError(f"Unsupported shard_names: {shard_name_format}")

        with BytesIO() as fileobj:
            uid_counter = self._write_fileobj(data, fileobj)
            fileobj.seek(0)
            _logger.info(
                "Writing shard: %s with size: %.4f MiB to location: %s",
                shard_name,
                asizeof.asizeof(fileobj) / _MiB,
                self.output_path,
            )
            self._write_to_location(
                output_path=self.output_path, shard_name=shard_name, fileobj=fileobj
            )
        # Ask the global counter actor to checkpoint (non-blocking)
        self.counter_actor.update_checkpoint.remote(
            input_path=input_path,
            output_path=os.path.join(self.output_path, shard_name),
            num_uid=uid_counter,
        )
        return shard_name, uid_counter

    def _write_to_location(
        self, output_path: str, shard_name: str, fileobj: IO[bytes]
    ) -> None:
        """Write a file object to output shard path."""
        full_path = os.path.join(output_path, shard_name)
        if full_path.startswith("s3://"):
            s3_bucket, s3_key = parse_s3_url(full_path)
            upload_fileobj(
                s3_client=self.s3_client,
                fileobj=fileobj,
                s3_bucket=s3_bucket,
                s3_key=s3_key,
            )
        else:
            os.makedirs(output_path, exist_ok=True)
            with open(full_path, "wb") as fout:
                fout.write(fileobj.getvalue())

        _logger.info("Finished writing shard: %s to: %s", shard_name, output_path)


class TarWriter(ShardWriter):
    """A Webdataset shard writer that writes data to a tarball."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the shard writer."""
        super().__init__(*args, **kwargs)
        self.shard_extension = "tar"

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an item to a dictionary of filetype to content for tarball writer."""
        sample = {"__key__": item.pop("uid", None) or item.pop("id", None)}
        # url
        if "url" in item:
            sample["url.txt"] = item.pop("url")

        # image
        image_bytes = get_image_bytes(item, image_key="image")
        if len(image_bytes) > 0:
            sample["jpg"] = image_bytes
            # pop image key from the item as we dont need it
            del item["image"]

        # caption
        if "caption" in item:
            sample["txt"] = item.pop("caption")
        if "text" in item:
            sample["txt"] = item.pop("text")

        # synthetic text
        json_data = {
            k: item.pop(k)
            for k in list(item.keys())
            if (k.startswith("syn_text") or "_captions" in k)
            and not (k.endswith("embedding") or k.endswith("emb"))
        }

        if len(json_data) > 0:
            json_string = json.dumps(json_data)
            sample["syn.json"] = json_string

        # augmentation parameters
        if "param_aug" in item:
            json_data = {"param_aug": item.pop("param_aug")}
            json_string = json.dumps(json_data)
            sample["paug.json"] = json_string

        # embedding data
        def convert_emb(v):
            if self.bfloat16:
                return torch.tensor(v, dtype=torch.bfloat16)
            return v

        emb_data = {
            k.replace("embedding", "emb"): convert_emb(item.pop(k))
            for k in list(item.keys())
            if "emb" in k
        }
        if len(emb_data) > 0:
            has_tensor = any([isinstance(x, torch.Tensor) for x in emb_data.values()])
            if self.bfloat16 or has_tensor:
                sample["pth.gz"] = emb_data
            else:
                sample["npz"] = emb_data

        # all else is saved in json
        json_string = json.dumps(item)
        sample["json"] = json_string
        return sample

    def _write_fileobj(self, data: List[Dict[str, Any]], fileobj: IO[bytes]) -> int:
        """Write a dictionary of filetype->data to a tarball file object."""
        uid_counter = 0
        with wds.TarWriter(fileobj) as sink:
            for item in data:
                uid_counter += 1
                sink.write(self._process_item(item))
        return uid_counter


class JsonlWriter(ShardWriter):
    """Jsonl shard format writer."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the shard writer."""
        super().__init__(*args, **kwargs)
        self.shard_extension = "jsonl"

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an item for writing in .jsonl format."""
        if "image" in item:
            if isinstance(item["image"], bytes):
                item["image"] = pybase64.b64encode_as_string(item["image"])

        # embedding data
        def convert_emb(v) -> str:
            if self.bfloat16:
                return lossless_compress(torch.tensor(v, dtype=torch.bfloat16))
            return lossless_compress(v)

        for k in list(item.keys()):
            if "emb" in k:
                item[k.replace("embedding", "emb")] = convert_emb(item.pop(k))
        return item

    def _write_fileobj(self, data: List[Dict[str, Any]], fileobj: IO[bytes]) -> int:
        """Write a converted data dictionary to an open file object."""
        uid_counter = 0
        for item in data:
            uid_counter += 1
            item = self._process_item(item)
            fileobj.write((json.dumps(item) + "\n").encode("utf-8"))
        return uid_counter


def get_shard_writer(s3_client, output_format: str, *args, **kwargs) -> ShardWriter:
    """Initialize and return a shard writer."""
    if output_format == "wds":
        writer = TarWriter(s3_client, *args, **kwargs)
    elif output_format == "jsonl":
        writer = JsonlWriter(s3_client, *args, **kwargs)
    return writer
