#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
source /miniconda/etc/profile.d/conda.sh
conda activate

input_path="s3://some_bucket/some_path"  # info.json path for JSONL format or webdataset regexp
output_path="s3://some_bucket/some_path"  # base path without info.json or .tar regexp

echo  "Generate CLIP embeddings from an ensemble of CLIP models."
python3 ./scripts/gen.py \
    --datagen-type embedding \
    --input $input_path \
    --output $output_path \
    --batch-size 256 \
    --min-actors 8 \
    --max-actors 512 \
    --model-name 'hf-hub:apple/DFN2B-CLIP-ViT-L-14,hf-hub:apple/DFN2B-CLIP-ViT-L-14-39B' \
    --pretrained 'N/A,N/A' \
    --num-samples 2 \
    --syn-text-key-regex "^syn_text$" \
    --aug-config '{"normalize": {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}, "rand_augment": {"enable": true, "p": 1.0}, "random_resized_crop": {"interpolation": "bicubic", "size": 224}, "to_rgb": {"enable": true}, "to_tensor": {"enable": true}}' \
    --output-format webdataset \
    --bfloat16
# --verbose --local: for debugging locally
# --output-format webdataset: store in webdataset format, a collection of tars
# --batch-size: per-GPU batch size, modify depending on the teacher model
# --min-actors: minimum actors to start the job
# --max-actors: maximum actors to grow to
# --tag-suffix: captions are stored as "syn_text_{tag_suffix}", defaults to ""
# --ray-data-wait-for-min-actor-s 3600: to timeout when waiting on gpu workers
# --bfloat16: store embedding features with bfloat16 compression as pth.tar
# --checkpoint-upload-freq 1000: checkpoint every 1000 shards. Increase if too 
# many small shards, decrease for more fault-tolerance.
