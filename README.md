# Desert Tree-Cover Evaluation

This repository implements a parallelized pipeline for evaluating tree cover in desert regions using Sentinel-2 imagery and deep learning.

## Repository Structure
```
├── data/
│   ├── raw/              # Sentinel-2 JP2 files (organized by scene)
│   ├── tiles/            # Preprocessed tile .npz files
│   └── results/          # Predicted mask PNGs
├── docs/
│   └── IMPLEMENTATION.md # Profiling results and challenges
├── notebooks/            # Exploratory Jupyter notebooks
├── src/                  # Source code modules
│   ├── preprocess.py     # Tile generation and multiprocessing
│   ├── dataset.py        # PyTorch Dataset definition
│   ├── model.py          # U-Net model
│   ├── train.py          # Training and validation
│   └── infer.py          # Model inference and output
├── checkpoints/          # Saved model weights
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions workflow
├── requirements.txt      # Python dependencies
└── README.md             # Project overview and instructions
```

## Setup
1. **Clone the repository**
   ```bash
    git clone https://github.com/username/desert-tree-cover.git
    cd desert-tree-cover
    ```

2. **Create a virtual environment**
   ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
   ```bash
    pip install -r requirements.txt
    ```

4. **Prepare data**
   - Download Sentinel-2 JP2 files and place them under `data/raw/<scene_id>/` with filenames `<scene_id>_B02.jp2`, etc.

## Usage

### Preprocessing
```bash
python src/preprocess.py
```
Generates tile `.npz` files in `data/tiles` and a manifest `tiles_manifest.json`.

### Training
```bash
python src/train.py
```
Trains the U-Net model, saving checkpoints in `checkpoints/`.

### Inference
```bash
# Update checkpoint_path in infer.py if needed
ython src/infer.py
```
Outputs PNG masks to `data/results/`.
