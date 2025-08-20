#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

nvidia-smi

echo "PATH is: ${PATH} and CUDA_HOME is: ${CUDA_HOME}"

python3 -m pip install loguru
python3 -m pip install -U "ray[data,train,tune,serve]==2.22.0"
python3 -m pip install cloudpathlib jsonlines webdataset nltk sentencepiece retrie pybase64
python3 -m pip uninstall -y transformer-engine  # conflict with transformers
python3 -m pip install -U open_clip_torch transformers
python3 -m pip install Pympler
python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchdata==0.9.0

echo "setup is done."
