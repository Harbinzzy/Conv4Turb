# A High-Dimensional Spatial-Physical Coupling Operator for Generalizable Super-Resolution of Three-Dimensional Turbulent Flows

This repository provides the PyTorch Lightning implementation of the paper
"A High-Dimensional Spatial-Physical Coupling Operator for Generalizable
Super-Resolution of Three-Dimensional Turbulent Flows."

## Requirements

- Python 3.9 or newer
- A CUDA-capable GPU is recommended for training and evaluation

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Datasets

The dataset is available on
[Hugging Face](https://huggingface.co/datasets/hitzzy/BLASTNet/tree/main).

The data are arranged as follows:

```text
DATA_ROOT/
├── train_data_summary.csv
├── test_data_summary.csv
├── paramvar_data_summary.csv
├── forcedhit_data_summary.csv
├── HR/
│   ├── train/
│   ├── test/
│   ├── paramvar/
│   └── forcedhit/
├── LR_8x/
├── LR_16x/
└── LR_32x/
```

High-resolution fields are read as `128 x 128 x 128` volumes. Low-resolution
fields are read as `(128 / scale)^3` volumes.

## Repository structure

```text
.
├── common/
│   ├── data.py       # Dataset loading, normalization, and augmentation
│   ├── metrics.py    # 3D SSIM and gradient loss
│   └── physics.py    # Spatial derivatives and SGS diagnostics
├── models/
│   ├── edsr.py       # 3D EDSR
│   ├── edsr4d.py     # EDSR with 4D convolution
│   ├── rcan.py       # 3D RCAN
│   ├── rcan4d.py     # RCAN with 4D convolution
│   ├── registry.py   # Model registry and supported sizes
│   └── tasks.py      # Lightning training and evaluation tasks
├── train.py          # Training entry point
├── test.py           # Evaluation entry point
├── train.sh          # Configurable training wrapper
└── test.sh           # Configurable evaluation wrapper
```

## Training

The shell wrapper exposes the most common settings through environment
variables:

```bash
DATA_PATH=/path/to/data \
MODEL_TYPE=edsr4d \
APPROX_PARAM=0.5M \
UPSCALE=8 \
DEVICES=0 \
./train.sh
```

Additional options can be passed directly to `train.py`. For example:

```bash
python train.py \
  --model-type edsr \
  --approx-param 5M \
  --upscale 16 \
  --data-path /path/to/data \
  --train-meta /path/to/data/train_data_summary.csv \
  --val-meta /path/to/data/test_data_summary.csv \
  --checkpoint-dir checkpoints/edsr_5M_16x \
  --devices 0,1 \
  --strategy ddp_find_unused_parameters_false
```

Resume training with `--resume /path/to/last.ckpt`. Run
`python train.py --help` for the complete option list.

## Evaluation

Evaluate a checkpoint on all three evaluation splits:

```bash
DATA_PATH=/path/to/data \
MODEL_TYPE=edsr \
APPROX_PARAM=0.5M \
UPSCALE=8 \
DEVICES=0 \
./test.sh /path/to/checkpoint.ckpt
```

To evaluate a single split or disable spectrum plots, call the Python entry
point directly:

```bash
python test.py \
  --checkpoint /path/to/checkpoint.ckpt \
  --model-type edsr \
  --approx-param 0.5M \
  --upscale 8 \
  --test-data test \
  --data-path /path/to/data \
  --test-meta /path/to/data/test_data_summary.csv \
  --output-dir eval/edsr_0.5M_8x \
  --no-spectrum
```

## Acknowledgments

Our code is based on the following awesome repository:
[BLASTNet 2.0 Super-Resolution Benchmark](https://github.com/blastnet/blastnet2_sr_benchmark).

We thank the authors for releasing their code!
