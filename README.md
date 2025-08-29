# RayGen: Multi-Modal Dataset Reinforcement for MobileCLIP

**For MobileCLIP and MobileCLIP2 models and demos see 
[ml-mobileclip](https://github.com/apple/ml-mobileclip) repository.**

This repository contains data generation code used in the following papers to 
improve and augment multi-modal and single-modal datasets. This code can be 
used to run distributed inference at scale to compute the output of an ML model 
on a large-scale dataset and store efficiently in a new dataset.  The code 
supports processing petabytes of data and billions of samples on 10,000 GPUs 
and more.  The code also supports elasticity where processing can start with 
a minimum number of workers and gradually scaled up.

- **[MobileCLIP2: Improving Multi-Modal Reinforced Training](http://arxiv.org/abs/2508.20691). (TMLR August 2025 <mark>Featured</mark>)**
Fartash Faghri, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, 
Alexander Toshev, Oncel Tuzel, Hadi Pouransari.
- **[MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced 
Training](https://arxiv.org/pdf/2311.17049.pdf). (CVPR 2024)**
*Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel.*
- **[Reinforce Data, Multiply Impact: Improved Model Accuracy and Robustness 
with Dataset Reinforcement](https://arxiv.org/abs/2303.08983). (ICCV 2023)**
Fartash Faghri, Hadi Pouransari, Sachin Mehta, Mehrdad Farajtabar, Ali Farhadi, 
Mohammad Rastegari, Oncel Tuzel.


The repository contains the RayGen package, a package that simplifies data 
generation using the distributed processing framework 
[Ray](https://github.com/ray-project/ray).

The training code and models are accessible in the following repositories
- [Dataset Reinforcement (ml-dr)](https://github.com/apple/ml-dr): Single 
 modality training on reinforced image classification datasets.
- [MobileCLIP (ml-mobileclip)](https://github.com/apple/ml-mobileclip): 
 Multi-modal reinforced training on image-text datasets.

## Installation

For local development we recommend installing a Miniforge based Conda 
environment with Python 3.10 as the base environment. This code is developed 
and tested on Ubuntu 20.04-x86_64 compute environment.

Create a Conda environment and install the environment dependencies.
```
conda create -n raygen python=3.10
conda activate raygen
```

Run the setup script:
```
bash scripts/setup_gpu_worker.sh  # or setup_cpu_worker.sh for cpu only
```

Install `raygen` python package to the local environment using:
```
pip install -e .
```


## Local Development

To run RayGen scripts locally on a CPU or GPU machine first follow the 
installation instructions above.  At least one GPU is needed for data 
generation scripts but not for dataset conversion.

Additionally you will want to install the "dev" dependencies which installs Ray.
```shell
pip install -e ".[dev]"
```

Once everything is installed, you can then start the local Ray cluster via:
```shell
ray start --head --port=6379
```

Next, run the generation code on a toy dataset:

```shell
INPUT=s3://some_bucket/some_path/
OUTPUT=s3://some_bucket/some_path/
GPU_COUNT=$(nvidia-smi -L | wc -l)
BATCH_SIZE=256

python3 scripts/gen.py \
    --input $INPUT \
    --output $OUTPUT \
    --batch-size $BATCH_SIZE \
    --min-actors $GPU_COUNT \
    --verbose \
    --local
```

If the batch size of 256 is too large for the local GPU then you can reduce it. I ran this with 16 cores on my host.

## Example Commands

The example script [run_capgen.py](./examples/run_capgen.sh) generates 
synthetic captions from a CoCa model for a dataset in JSONL format and saves it 
as a new dataset in JSONL format.

Next example script [run_embgen.py](./examples/run_embgen.sh) generates  CLIP 
embeddings. We generate CLIP image-text embeddings from an ensemble of two CLIP 
models. The models are loaded from HuggingFace. The number of image 
augmentations is specified as 2. We also specify a regular expression to select 
a subset of the synthetic captions for compute synthetic text embeddings. The 
image augmentation parameters are also specified

## Supported sharded dataset formats

The code supports datasets as input and output in JSONL format (a collection of 
`.jsonl` files with a single `.jsonl` file as the manifest) as well as 
Webdataset format (a collection of `.tar` files passed with a regular 
expression). Additional file formats such as `jsonl.zstd`, `jsonl.gz`, 
`.tfrecord` are also supported as inputs but not as output.

To pass an input dataset, one can pass a path like 
`{bucket}/path/{00000000..00001023}.tar`.  Alternatively, one may pass 
a `info.json` file that contains a list of all the shards in the dataset.  The 
minimal content of such a file is `{'lengths': {'shard1.jsonl': 0, 
'shard2.jsonl': 0, …}}`. As the generation code does not utilize the value of 
lengths, they can be set to zero. If the output dataset format is JSONL, a new 
`info.json` will be written that contains additional information such as 
`totalLength` of the dataset.

## Checkpointing

The code saves a checkpoint of the generation process and by default resumes 
from a checkpoint if available in the path. The checkpoint is 
a `checkpoint.json` file containing information about processed input and 
output paths so far together with the number of processed samples.

## License

This software and accompanying data and models have been released under the 
following licenses:
- Code: [Apple Sample Code License (ASCL)](./LICENSE)
- ML models: [Apple ML Research Model TOU](./LICENSE_MODELS)
- Data: [CC-BY-NC-ND](./LICENSE_DATA) [Deed](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 

## Citation

If you found this code useful, please cite the following papers:

    @article{faghri2025mobileclip2,
      title={Mobile{CLIP}2: Improving Multi-Modal Reinforced Training},
      author={Fartash Faghri and Pavan Kumar Anasosalu Vasu and Cem Koc and Vaishaal Shankar and Alexander T Toshev and Oncel Tuzel and Hadi Pouransari},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2025},
      url={https://openreview.net/forum?id=WeF9zolng8},
      note={Featured Certification}
    }


    @InProceedings{vasu2024mobileclip,
      author = {Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi and Faghri, 
      Fartash and Vemulapalli, Raviteja and Tuzel, Oncel},
      title = {MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2024},
    }


    @InProceedings{faghri2023reinforce,
        author    = {Faghri, Fartash and Pouransari, Hadi and Mehta, Sachin and Farajtabar, Mehrdad and Farhadi, Ali and Rastegari, Mohammad and Tuzel, Oncel},
        title     = {Reinforce Data, Multiply Impact: Improved Model Accuracy and Robustness with Dataset Reinforcement},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2023},
    }
