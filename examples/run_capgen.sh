#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
source /miniconda/etc/profile.d/conda.sh
conda activate

input_path="s3://some_bucket/some_path"  # info.json path for JSONL format or webdataset regexp
output_path="s3://some_bucket/some_path"  # base path without info.json or .tar regexp

echo  "Generate synthetic captions from CoCa for a dataset stored in JSONL format."
python3 ./scripts/gen.py \
    --datagen-type caption \
    --input $input_path \
    --output $output_path \
    --model-name coca_ViT-L-14 \
    --tag-suffix "_coca" \
    --batch-size 1024 \
    --min-actors 8 \
    --max-actors 512 \
    --output-format jsonl
# --verbose --local: for debugging locally
# --output-format webdataset: store in webdataset format, a collection of tars
# --batch-size: per-GPU batch size, modify depending on the teacher model
# --min-actors: minimum actors to start the job
# --max-actors: maximum actors to grow to
# --tag-suffix: captions are stored as "syn_text_{tag_suffix}", defaults to ""
# --ray-data-wait-for-min-actor-s 3600: to timeout when waiting on gpu workers
# --checkpoint-upload-freq 1000: checkpoint every 1000 shards. Increase if too 
# many small shards, decrease for more fault-tolerance.
