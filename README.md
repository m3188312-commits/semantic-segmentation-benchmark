# semantic-segmentation-benchmark

> Benchmark and compare DeepLabV3, YOLO‑Seg, U‑Net and Random Forest models for semantic segmentation on satellite imagery.

---

## 🚀 Overview

**semantic-segmentation-benchmark** is an end-to-end suite designed to:

* **Train**: Fit multiple segmentation models (DeepLabV3, YOLO‑Seg, U‑Net, Random Forest) on curated datasets.
* **Evaluate**: Quantify per-class and aggregate metrics (Precision, Recall, F1, pixel accuracy) on validation splits or custom image subsets.
* **Infer**: Generate segmentation masks for new images via a unified CLI interface.
* **Visualize**: Produce professional visual artifacts—heatmaps, grouped bar charts, radar plots—for transparent comparative analysis.


---

## 📁 Repository Structure

```
semantic-segmentation-benchmark/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excluded files
├── data/                         # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── models/                       # Model training and checkpoints
│   ├── deeplab/
│   ├── yolo/
│   ├── unet/
│   └── random_forest/
├── scripts/                      # Orchestrator scripts
│   ├── train_all.py              # Train all models sequentially
│   ├── eval_all.py               # Batch evaluation script
│   └── infer.py                  # Single-image inference wrapper
├── notebooks/                    # Exploratory analysis and dashboards
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_comparison.ipynb
│   └── 03_visualization_dashboard.ipynb
└── output/                       # Generated artifacts
    ├── checkpoints/              # Best/final model weights
    ├── metrics/                  # CSVs of computed metrics
    └── figures/                  # Visualization outputs
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/aggelosntou/semantic-segmentation-benchmark.git
   cd semantic-segmentation-benchmark
   ```

2. **Create & activate** a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate         # macOS/Linux
   .\.venv\Scripts\activate.ps1    # Windows PowerShell
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🎯 Usage

### 1. Train & Evaluate

Run a full training cycle followed by validation:

```bash
python trainingDL.py
```

### 2. Inference Only

Generate a mask for a single image:

```bash
python trainingDL.py \
  --infer \
  --image /path/to/input.jpg \
  --out   ./output/mask.png
```

### 3. Full Validation Only

Compute metrics on the entire validation set:

```bash
python trainingDL.py --eval
```

### 4. Subset Evaluation

Evaluate on a specific list of images:

```bash
python trainingDL.py --eval_images img1.jpg img2.jpg img3.jpg
```

---

## 📊 Visualization

Leverage the notebooks or scripts to generate:

* **Heatmaps** of per-class F1.
* **Grouped bar charts** comparing models across classes.
* **Radar charts** illustrating holistic model performance profiles.


---

## 🛠️ Extensibility

* Add new architectures by placing training logic in `models/<model_name>/` and updating `scripts/train_all.py`.
* Integrate additional metrics (e.g. boundary F1, ROC curves) by extending the evaluation functions in `scripts/eval_all.py`.

---
