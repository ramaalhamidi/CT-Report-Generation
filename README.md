# CT Report Generation

A pipeline for preprocessing 3D chest CT scans, extracting visual embeddings with CT-ViT/CT-CLIP, and preparing them for radiology report generation with an R2Gen-based decoder.

## Installation

Clone this repository:

```bash
git clone https://github.com/ramaalhamidi/CT-Report-Generation.git
cd CT-Report-Generation
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate ctvit-val
```

Alternatively, use the provided setup script.

Windows:

```bash
.\setup.ps1
```

Linux:

```bash
chmod +x setup.sh
./setup.sh
```

## Required Repositories
Clone CT2Rep and R2Gen:

```bash
git clone https://github.com/ibrahimethemhamamci/CT2Rep.git CT2Rep
git clone https://github.com/zhjohnchan/R2Gen.git R2Gen
```

Install the CT-ViT package included with CT2Rep:

```bash
cd CT2Rep/ctvit
pip install -e .
cd ../..
```

## CT-CLIP Weights

Download the pretrained CT-CLIP weights from [HuggingFace](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT-CLIP_v2.pt). 

Update the checkpoint path in config.py if needed.

## Usage

### Preprocess a CT scan:

```bash
python scripts/preprocess.py
```

### Validate the preprocessing pipeline:

```bash
python scripts/validate_pipeline.py
```

### Generate and cache CT-ViT embeddings:

```bash
python scripts/cache_embeddings.py
```

See SHAPE_CONTRACT.md for the expected tensor shapes.

