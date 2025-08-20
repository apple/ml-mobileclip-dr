#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Generate synthetic captions or CLIP embeddings for an image-text datasets.

Synthetic Caption Generation
----------------------------
python3 scripts/gen.py \
    --datagen-type caption \
    --input s3://some_bucket/some_path/info.json \
    --output s3://some_bucket/some_path/ \
    --batch-size 256 \
    --min-actors 1 \
    --verbose \
    --local

Embedding Generation
--------------------
python3 scripts/gen.py \
    --datagen-type embedding \
    --input s3://some_bucket/some_path/info.json \
    --output s3://some_bucket/some_path/ \
    --batch-size 256 \
    --min-actors 1 \
    --model-name 'hf-hub:apple/DFN2B-CLIP-ViT-L-14,hf-hub:apple/DFN2B-CLIP-ViT-L-14-39B' \
    --pretrained 'N/A,N/A' \
    --num-samples 2 \
    --syn-text-key-regex "^syn_text$" \
    --aug-config '{"normalize": {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}, "rand_augment": {"enable": true, "p": 1.0}, "random_resized_crop": {"interpolation": "bicubic", "size": 224}, "to_rgb": {"enable": true}, "to_tensor": {"enable": true}}' \
    --verbose \
    --local
"""

import abc
import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import open_clip
import ray
import torch
from PIL import Image
from ray.data import DataContext as ctx
from ray.util.placement_group import placement_group
from torch.nn import functional as F

from raygen.cloud_common import checkpointing, storage_utils
from raygen.datasetio import get_shard_writer, read_shard
from raygen.dr.transforms import compose_from_config
from raygen.datasetio import get_image_bytes, create_paths, create_info_json_data
from raygen.model_utils import open_clip_create_model_from_pretrained
from raygen.driver_utils import (
    batched,
    create_ray_dataset_from_paths,
    initialize_ray,
    write_manifest,
)

EOT = "<end_of_text>"
SOT = "<start_of_text>"


def _make_output_path(output_path: str, is_local_run: bool) -> str:
    """Generate an output path.

    This is intended for local runs only. This will generate a UTC
    based suffix to be added to the output path user supplied
    which makes it easier for local testing.
    """
    if is_local_run:
        utc_timestamp = datetime.now(timezone.utc)
        utc_suffix = utc_timestamp.strftime("%Y%m%d/%H_%M_%S")
        return os.path.join(output_path, utc_suffix)

    return output_path


class Model(metaclass=abc.ABCMeta):
    """Abstarct class for computing the output of a model on a data shard."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        batch_size: int,
        shard_writer_kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the model and shard writer."""
        self.model_name = model_name
        self.pretrained = pretrained
        self.s3_client = storage_utils.get_s3_client()
        self.batch_size = batch_size
        self.total = 0

        # Setup logger
        level = logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s %(levelname)s : %(message)s"
        )
        root_logger = logging.getLogger(__name__)
        self.logger = root_logger

        # Setup shard writer
        self.shard_writer = get_shard_writer(self.s3_client, **shard_writer_kwargs)

        # Setup GPU
        devices = ray.get_gpu_ids()
        self.logger.info("Actor visible devices: %s", devices)
        self.logger.info(
            "Ray reported visible gpus: %s",
            ray.get_runtime_context().get_accelerator_ids().get("GPU", 0),
        )

        if len(devices) > 1:
            self.logger.warning(
                "Multi GPU model sharding is not supported. Picking one."
            )

        device = devices[0]
        self.logger.info("Using device id: %s", device)

        self.device_id = "cuda"
        self.device = torch.device(self.device_id)

        self.logger.info("Set to use device: %s", self.device)


class CaptionModel(Model):
    """A class for generating captions from a CoCa model given an image dataset."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        batch_size: int,
        num_samples: int,
        tag_suffix: str,
        shard_writer_kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the caption generator and shard writer.

        Args:
            model_name: The name of a captioning model architecture defined in OpenCILP.
            pretrained: A string with a path or pretrained model name in OpenCLIP.
            batch_size: An integer as the size of the batch size for one capgen step.
            num_samples: An integer as the number of samples to generate from captioner.
            tag_suffix: A string as the suffix for the new key in output dataset where
                the final key will be `syn_text{tag_suffix}'.
            shard_writer_kwargs: A dictionary of arguments passed to shard writer.
        """
        super(CaptionModel, self).__init__(
            model_name=model_name,
            pretrained=pretrained,
            batch_size=batch_size,
            shard_writer_kwargs=shard_writer_kwargs,
        )
        self.num_samples = num_samples
        self.tag_suffix = tag_suffix

        self.model, self.transforms = open_clip_create_model_from_pretrained(
            self.s3_client, model_name, pretrained
        )

        self.model.to(self.device)
        self.model.eval()

        self.logger.info("Successfully loaded model on device: %s", self.device)

    def _get_image_tensor_from_items(
        self, items, start
    ) -> Tuple[torch.Tensor, List[int]]:
        """Read a batch of data into Tensors and return a list of valid indices.

        Args:
            items: A list of images or dictionaries with image keys.
            start: An integer as the start index of the indices in the current batch
                used only for setting the image indices.

        Returns:
            A Torch Tensor of images and a list of indices.
        """
        valid_indices: List[int] = []
        image_tensors = []
        for img_idx, item in enumerate(items, start=start):
            try:
                img_bytes = get_image_bytes(item, image_key="image")
                if len(img_bytes) == 0:
                    self.logger.warning(
                        "Ignored record at index: %s due to invalid image.", img_idx
                    )
                    continue
                image_tensors.append(
                    self.transforms(Image.open(BytesIO(img_bytes))).unsqueeze(0)
                )
                valid_indices.append(img_idx)
            except Exception as e:
                self.logger.warning(
                    "Ignored record at index: %s because of exception: %s",
                    img_idx,
                    e,
                )
        return torch.cat(image_tensors, dim=0), valid_indices

    def __call__(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the data of a new shard with added captions given an input shard path.

        Args:
            row: A dictionary with a single key `path' containing the input shard path.

        Returns:
            A list of dictionaries containing all the input data in the input shard
            together with added generated captions on images.
        """
        path = row["path"]
        data: List[Dict[str, Any]] = read_shard(self.s3_client, path)

        valid_indices: List[int] = []
        generated_captions: Dict[int, List[str]] = defaultdict(list)
        start_idx = 0
        download_tik = time.perf_counter()
        for batch in batched(data, step=self.batch_size):
            download_tok = time.perf_counter()
            time_str = (
                ", Elapsed time: {:.4f}s, Remaining time: {:.4f}s".format(
                    (download_tok - download_tik),
                    (len(data) - start_idx) * (download_tok - download_tik) / start_idx,
                )
                if start_idx > 0
                else ""
            )
            self.logger.info(
                "Processing shard: [%s:%s]/%s%s",
                start_idx,
                start_idx + len(batch),
                len(data),
                time_str,
            )

            with torch.inference_mode():
                # Read batch of images into tensors
                (
                    batch_image_tensors,
                    batch_valid_indices,
                ) = self._get_image_tensor_from_items(batch, start=start_idx)
                valid_indices.extend(batch_valid_indices)
                # Move batch of images to device
                batch_image_tensors = batch_image_tensors.to(self.device)
                for cap_idx in range(self.num_samples):
                    # Do inference here
                    out_tensor = self.model.generate(
                        batch_image_tensors,
                        generation_type="top_p",
                        top_p=0.9,
                        fixed_output_length=True,
                    ).to(
                        "cpu"
                    )  # shape: [B, num_tokens] B is the `batch_size`

                    generated_captions[cap_idx].extend(
                        [
                            open_clip.decode(tensor)
                            .split(EOT)[0]
                            .split(SOT)[-1]
                            .split(".")[0]
                            .rstrip()
                            for tensor in out_tensor
                        ]
                    )
                    del out_tensor
                del batch_image_tensors
            start_idx += len(batch)

        # Add generated synthetic captions to all valid records
        for tensor_index, item_idx in enumerate(valid_indices):
            item_captions = []
            for cap_idx in range(self.num_samples):
                item_captions.append(generated_captions[cap_idx][tensor_index])

            item = data[item_idx]
            item[f"syn_text{self.tag_suffix}"] = item_captions

        tar_name, uid_counter = self.shard_writer.write_shard(data, input_path=path)
        self.total += uid_counter

        return [{"shard": tar_name.split(".")[0], "num_uids": uid_counter}]


class EmbeddingModel(Model):
    """Store image/text embeddings from OpenCLIP pretrained models."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        batch_size: int,
        num_samples: int,
        tag_suffix: str,
        syn_text_key_regex: str,
        aug_config: Dict,
        shard_writer_kwargs: Dict[str, Any],
    ) -> None:
        """Initialize a CLIP embedding generation and shard writer.

        Args:
            model_name: The name of a CLIP model architecture defined in OpenCILP.
            pretrained: A string with a path or pretrained model name in OpenCLIP.
            batch_size: An integer as the size of the batch size for one one forward.
            num_samples: An integer as the number of image augmentations to generate.
            tag_suffix: A string as the suffix for the new key in output dataset where
                the final key will be `.*_emb{tag_suffix}'.
            syn_text_key_regex: A regular expression string to select a subset of
                synthetic captions in the dataset for embedding computation.
            aug_config: A dictionary of augmentation configurations supported by DR.
            shard_writer_kwargs: A dictionary of arguments passed to shard writer.
        """
        super(EmbeddingModel, self).__init__(
            model_name=model_name.split(","),
            pretrained=pretrained.split(","),
            batch_size=batch_size,
            shard_writer_kwargs=shard_writer_kwargs,
        )

        self.num_samples = num_samples
        self.tag_suffix = tag_suffix
        self.syn_text_key_regex = syn_text_key_regex
        self.dr_transforms = compose_from_config(aug_config)
        self.crop_size = []

        self.model = []
        self.tokenizer = []
        for m, p in zip(self.model_name, self.pretrained):
            model, transforms = open_clip_create_model_from_pretrained(
                self.s3_client, m, p
            )
            model.eval()
            self.model += [model.to(self.device)]
            self.tokenizer += [open_clip.get_tokenizer(m)]
            self.crop_size += [transforms.transforms[0].size]
            if isinstance(self.crop_size[-1], int):
                self.crop_size[-1] = (self.crop_size[-1], self.crop_size[-1])
            cs = [
                t[1].size
                for t in self.dr_transforms.transforms
                if t[0] == "random_resized_crop"
            ][0]
            assert self.crop_size[-1][0] <= cs[0], "Set crop size to max of all models."
            mean = [
                t[1].mean for t in self.dr_transforms.transforms if t[0] == "normalize"
            ][0]
            np.testing.assert_allclose(
                transforms.transforms[-1].mean,
                mean,
                err_msg="Normalization mean does not match teacher's.",
            )
        self.logger.info(f"Models: {self.model}")
        self.logger.info(f"Crop sizes: {self.crop_size}")
        self.logger.info(f"Tokenizers: {self.tokenizer}")
        self.logger.info("Successfully loaded model on device: %s", self.device)

    def _get_tensors_from_items(self, items, start) -> Dict[str, Any]:
        """Read a batch of data into Tensors and return a list of valid indices.

        Args:
            items: A list of dictionaries with image and caption keys.
            start: An integer as the start index of the indices in the current batch
                used only for setting the image indices.

        Returns:
            A dictionary of (image tensors, combined text tensors, augmentation
            parameters, dictionary of keys to lengths of texts) and a list of valid
            indices.
        """
        valid_indices: List[int] = []
        num_texts: List[int] = []

        image_tensors = []
        texts = []
        param_augs_all = []
        for img_idx, item in enumerate(items, start=start):
            try:
                # Image.
                img_bytes = get_image_bytes(item, image_key="image")
                if len(img_bytes) == 0:
                    self.logger.warning(
                        "Ignored record at index: %s due to invalid image.", img_idx
                    )
                    continue
                image = Image.open(BytesIO(img_bytes)).convert("RGBA").convert("RGB")
                image_augs = []
                param_augs = []
                for _ in range(self.num_samples):
                    image_aug, param_aug = self.dr_transforms(image)
                    image_augs += [image_aug.unsqueeze(0)]
                    param_augs += [self.dr_transforms.compress(param_aug)]
                image_augs = torch.cat(image_augs, dim=0)
                image_tensors.append(image_augs.unsqueeze(0))
                param_augs_all += [param_augs]
                valid_indices.append(img_idx)

                # Keep all texts as a list
                text = []
                if "caption" in item:
                    # DFN dataset
                    assert (
                        "text" not in item or item["text"] is None
                    ), "Both caption and text entries exist."
                    assert (
                        "texts" not in item
                        or item["texts"] is None
                        or item["caption"] == item["texts"]
                    ), "Texts and caption are assumed to be equal if both exist."
                    item["text"] = item["caption"]
                    if "texts" in item:
                        del item["texts"]
                # Ground-truth text comes first
                for k, v in item.items():
                    if k == "text" or k == "texts":
                        vv = v if isinstance(v, list) else [v]
                        text += vv
                        num_text = [[k, len(text)]]
                # Synthetic text comes next
                for k, v in item.items():
                    if re.match(self.syn_text_key_regex, k):
                        text += v
                        num_text += [[k, len(v)]]
                # tokenize per model
                text_teach = []
                for tok in self.tokenizer:
                    text_teach += [tok(text).unsqueeze(1)]
                texts += [torch.cat(text_teach, dim=1).unsqueeze(0)]
                num_texts.extend([num_text])
            except Exception as e:
                self.logger.warning(
                    "Ignored record at index: %s because of exception: %s",
                    img_idx,
                    e,
                )
                import traceback

                print(traceback.format_exc())

        # Pad texts by the number of captions
        total_num_texts = [t.shape[1] for t in texts]
        max_num_texts = max(total_num_texts)
        texts = [
            torch.cat(
                (
                    t,
                    torch.zeros(
                        [1, max_num_texts - t.shape[1]] + list(t.shape[2:]),
                        dtype=t.dtype,
                        device=t.device,
                    ),
                ),
                dim=1,
            )
            for t in texts
        ]
        output = dict(
            image=torch.cat(image_tensors, dim=0),
            text=torch.cat(texts, dim=0),
            param_aug=param_augs_all,
            total_num_texts=total_num_texts,
        )
        return output, valid_indices, num_texts

    def __call__(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the data of a new shard with added embeddings given an input path.

        Args:
            row: A dictionary with a single key `path' containing the input shard path.

        Returns:
            A list of dictionaries containing all the input data in the input shard
            together with added embeddings for all images and selected captions.
        """
        path = row["path"]
        data: List[Dict[str, Any]] = read_shard(self.s3_client, path)

        valid_indices: List[int] = []
        num_texts: List[int] = []
        image_emb_all = []
        text_emb_all = []
        start_idx = 0
        num_img_samples = None
        download_tik = time.perf_counter()
        for batch in batched(data, step=self.batch_size):
            download_tok = time.perf_counter()
            time_str = (
                ", Elapsed time: {:.4f}s, Remaining time: {:.4f}s".format(
                    (download_tok - download_tik),
                    (len(data) - start_idx) * (download_tok - download_tik) / start_idx,
                )
                if start_idx > 0
                else ""
            )
            self.logger.info(
                "Processing shard: [%s:%s]/%s%s",
                start_idx,
                start_idx + len(batch),
                len(data),
                time_str,
            )

            with torch.inference_mode():
                # Read batch of images into tensors
                (
                    batch_tensors,
                    batch_valid_indices,
                    batch_num_texts,
                ) = self._get_tensors_from_items(batch, start=start_idx)
                valid_indices.extend(batch_valid_indices)
                num_texts.extend(batch_num_texts)
                batch_image_tensors = batch_tensors["image"]
                _, num_img_samples = batch_image_tensors.shape[:2]
                # Move batch of images to device
                batch_image_tensors = batch_image_tensors.to(self.device)
                # Compute embeddings for multiple random samples
                for img_idx in range(num_img_samples):
                    image_embedding = []
                    image = batch_image_tensors[:, img_idx].to(self.device)
                    for model, cs in zip(self.model, self.crop_size):
                        if image.shape[-1] != cs[1]:
                            image = F.interpolate(image, size=cs, mode="bicubic")
                        img_emb, _, _ = model.forward(image=image)
                        image_embedding += [img_emb]
                    image_embedding = torch.cat(image_embedding, dim=1)
                    image_emb_all += [image_embedding.cpu()]

                del batch_image_tensors

                # Compute text embeddings
                max_text_samples = max(batch_tensors["total_num_texts"])
                for cap_idx in range(max_text_samples):
                    text_embedding = []
                    for j, model in enumerate(self.model):
                        text = batch_tensors["text"][:, cap_idx, j].to(self.device)
                        _, t_emb, _ = model.forward(text=text)
                        text_embedding += [t_emb]
                    text_embedding = torch.cat(text_embedding, dim=1)
                    text_emb_all += [text_embedding.cpu()]

                del batch_tensors
            start_idx += len(batch)

        # Add generated synthetic captions to all valid records
        # TODO(Cem): Investigate writing the tarball here as an optimization.
        for tensor_index, item_idx in enumerate(valid_indices):
            # Concatenate embeddings of image samples
            image_embedding = torch.cat(
                [
                    image_emb_all[j][tensor_index : tensor_index + 1]
                    for j in range(num_img_samples)
                ],
                dim=0,
            )
            item = data[item_idx]
            item["image_emb{self.tag_suffix}"] = image_embedding.cpu().numpy()
            num_text = num_texts[tensor_index]
            num_text_cur = 0
            for k, v in num_text:
                text_embedding = torch.cat(
                    [
                        text_emb_all[j][tensor_index : tensor_index + 1]
                        for j in range(num_text_cur, num_text_cur + v)
                    ],
                    dim=0,
                )
                item[f"{k}_emb{self.tag_suffix}"] = text_embedding.cpu().numpy()
                num_text_cur += v

        tar_name, uid_counter = self.shard_writer.write_shard(data, input_path=path)
        self.total += uid_counter

        return [{"shard": tar_name.split(".")[0], "num_uids": uid_counter}]


def run(args):
    """Initialize Ray and run a processing pipeline."""
    logging.info("ray version: %s", ray.__version__)

    min_actors = args.min_actors
    if args.max_actors:
        max_actors = args.max_actors
    else:
        max_actors = min_actors

    logging.info("Creating an actor pool of size: (%s, %s)", min_actors, max_actors)

    # Ray will place actors on the head pod if it sees resources
    # If you want to prevent this from happening then uncomment
    # the following lines.
    # if is_cloud_job():
    #     actor_options['resources'] = {"gpu_worker": 0.01}

    # Cloud job specific configurations and program flow.
    if not args.local:
        # Disable progress bars for less clutter in logging.
        ctx.get_current().enable_progress_bars = False
        wait_for_min_actors_s = args.ray_data_wait_for_min_actor_s
        if (
            hasattr(ctx.get_current(), "wait_for_min_actors_s")
            and wait_for_min_actors_s > 0
        ):
            # https://github.com/ray-project/ray/pull/45508
            ctx.get_current().wait_for_min_actors_s = wait_for_min_actors_s
        else:
            logging.warning(
                "Unable to set `DataContext.wait_for_min_actors_s` because it is not yet supported in ray v.%s",
                ray.__version__,
            )
            # We will create a PlacementGroup for one of the GPU worker nodes
            # to make sure that at least one GPU worker node has started before
            # creating a Dataset.
            bundle = {"gpu_worker": 1}
            pg = placement_group([bundle])
            if wait_for_min_actors_s > 0:
                logging.info(
                    "Placement group %s is submitted; will block for %s seconds until ready.",
                    bundle,
                    wait_for_min_actors_s,
                )
                try:
                    ray.get(pg.ready(), timeout=wait_for_min_actors_s)
                except Exception as e:
                    logging.warning(
                        "Placement group %s timed out due to: %s. Exiting", bundle, e
                    )
                    sys.exit(1)
            else:
                logging.info(
                    "Placement group %s is submitted; will block indefinitely until it is ready.",
                    bundle,
                )
                ray.get(pg.ready())

            logging.info("Placement group %s is ready!", bundle)

    output_path = _make_output_path(output_path=args.output, is_local_run=args.local)
    logging.info("Output path: %s", output_path)

    # Check if we need to recover from a checkpoint.
    start_time = time.perf_counter()
    remote_checkpoint_path, checkpointed_paths = checkpointing.recover_from_checkpoint(
        output_path
    )
    # Materialize dataset
    s3_client = storage_utils.get_s3_client()
    paths = create_paths(s3_client=s3_client, path=args.input)
    if checkpointed_paths and len(checkpointed_paths) > 0:
        paths = set(paths) - set(checkpointed_paths)
    if len(paths) == 0:
        logging.info("Checkpoint contains all paths. Exiting.")
        return

    counter_actor = checkpointing.GlobalCounter.remote(
        remote_checkpoint_path=remote_checkpoint_path,
        checkpoint_upload_freq=args.checkpoint_upload_freq,
    )
    # Schedule a dummy task on counter actor to make sure it is scheduled eagerly
    counter_actor.ready.remote()
    dataset = create_ray_dataset_from_paths(
        paths,
        maximize_num_blocks=True,  # maximize output blocks for max parallelism
    )

    logging.info(
        "Created dataset from input: %s. Num Blocks: %s Num Rows: %s",
        args.input,
        dataset.num_blocks(),
        dataset.count(),
    )

    actor_options = {
        "max_restarts": -1,
        "max_task_retries": 5,
    }

    shard_writer_kwargs = {
        "output_format": args.output_format,
        "output_shard_name_format": args.output_shard_name_format,
        "output_path": output_path,
        "bfloat16": args.bfloat16,
        "counter_actor": counter_actor,
    }
    if args.datagen_type == "caption":
        model = CaptionModel
        model_params = {
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "tag_suffix": args.tag_suffix,
            "shard_writer_kwargs": shard_writer_kwargs,
        }
    elif args.datagen_type == "embedding":
        model = EmbeddingModel
        if isinstance(args.aug_config, str):
            aug_config = json.loads(args.aug_config)
        model_params = {
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "tag_suffix": args.tag_suffix,
            "syn_text_key_regex": args.syn_text_key_regex,
            "aug_config": aug_config,
            "shard_writer_kwargs": shard_writer_kwargs,
        }
        # TODO: support writing config.json if previous job was Emb. gen.
        # Here we write config.json which is only important for reproducing embeddings.
        # Capgen does not produce a config.json and we usually run embgen after capgen
        write_manifest(
            s3_client=s3_client,
            jsonl_lines=[aug_config],
            path=output_path,
            filename="config.json",
        )
    else:
        msg = "Datagen type {} is not supported"
        logging.error(msg.format(args.datagen_type))
        raise Exception(msg.format(args.datagen_type))

    transformed_ds = dataset.flat_map(
        model,
        fn_constructor_kwargs=model_params,
        concurrency=(min_actors, max_actors),
        num_gpus=1,
        **actor_options,
    )

    transformed_ds = transformed_ds.repartition(1).sort(key="shard")
    manifest_data = transformed_ds.take_all()

    if args.output_format == "wds":
        write_manifest(
            s3_client=s3_client,
            jsonl_lines=manifest_data,
            path=output_path,
            filename="manifest.jsonl",
        )
    elif args.output_format == "jsonl":
        info_json_data = create_info_json_data(manifest_data, output_path)
        write_manifest(
            s3_client=s3_client,
            jsonl_lines=info_json_data,
            path=output_path,
            filename="info.json",
        )

    end_time = time.perf_counter()

    logging.info(
        "Finished. Duration: %.4f seconds. Manifest path: %s",
        (end_time - start_time),
        output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datagen-type",
        help="Data generation type",
        type=str,
        choices=["caption", "embedding"],
        required=True,
    )
    parser.add_argument(
        "--input",
        help="path to info.json or webdataset shard list",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output", help="output path to write, can be local or s3", type=str
    )
    parser.add_argument(
        "--num-samples",
        help="Number of captions/augmentations to generate",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--model-name",
        help="Model name from OpenCLIP",
        type=str,
        default="coca_ViT-L-14",
    )
    parser.add_argument(
        "--pretrained",
        help="Pretrained checkpoint name from OpenCLIP",
        type=str,
        default="mscoco_finetuned_laion2B-s13B-b90k",
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for each model actor to process at a time.",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--tag-suffix", help="Synthetic caption column suffix.", type=str, default=""
    )
    parser.add_argument(
        "--syn-text-key-regex",
        help="Synthetic caption key regular expression to match.",
        type=str,
        default="syn_text",
    )
    parser.add_argument(
        "--aug-config", help="Image augmentation configuration.", type=str, default=""
    )
    parser.add_argument(
        "--bfloat16",
        help="Store embeddings as BFloat16 tensors",
        action="store_true",
    )
    parser.add_argument(
        "--min-actors", help="Min number of model actors.", type=int, default=1
    )
    parser.add_argument("--max-actors", help="Max number of model actors.", type=int)
    parser.add_argument(
        "--verbose", help="Turns on DEBUG level logging.", action="store_true"
    )
    parser.add_argument("--local", help="Runs in local mode.", action="store_true")
    parser.add_argument(
        "--ray-data-wait-for-min-actor-s",
        help="Wait seconds for actors to be brought up before failing Dataset job.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="wds",
        choices=["wds", "jsonl"],
        help="Format of the output shards.",
    )
    parser.add_argument(
        "--output-shard-name-format",
        type=str,
        default="count",
        choices=["count", "orig"],
        help="Format of the output shard name. `count' stores as XXXXXXXX.EXT while "
        "`orig' saves the output with the same input name.",
    )
    parser.add_argument(
        "--checkpoint-upload-freq",
        help="Frequency of uploading checkpoint to S3. Default: 1000 shards.",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=level)

    if args.verbose:
        logging.info("DEBUG level turned on.")

    # Set ray runtime env variables that will be reflected on each worker
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("AWS")}
    ray_runtime_env = {"env_vars": env_vars}

    logging.info("ray runtime env: %s", ray_runtime_env)

    initialize_ray(local=args.local, runtime_env=ray_runtime_env)
    run(args)

    logging.info("Done.")
