# DeepFreqMark

The official pytorch implementation of the paper [DeepFreqMark: End-to-end Learnable Frequency-domain Watermarking with Spherical Attack Simulation for Latent Diffusion Models]().

## Setup and Installation

This project uses [uv](https://astral.sh/uv/) for fast, reliable Python package and environment management. 

### 1. Prerequisites

First, install `uv` on your system if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Installation & Environment Sync

Clone this repository and run `uv sync`. This will automatically download the correct Python version, create a local virtual environment (`.venv`), and install all dependencies (including PyTorch, Hugging Face ecosystem, and utilities) with exact locked versions.

```bash
git clone git@github.com:chenhsiu48/DeepFreqMark.git
cd DeepFreqMark
uv sync
```

## Execution

### Training

```bash
uv run trans.py --train --type FFT
```

### Testing

```bash
uv run trans.py --embed --resume logger/THE_TRAINED_FOLDER --type FFT
```

## Acknowledgement

The codebase in the `main` sub-folder is adapted from the following project:

* [ZoDiac](https://github.com/zhanglijun95/ZoDiac)

We sincerely appreciate the authors for sharing their valuable work with the community.

## Citation

If you find this work useful for your research, please cite:

```
TBD
```
