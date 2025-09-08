# Semantic Segmentation Benchmark

Semantic segmentation project comparing 4 models (U-Net, DeepLab, YOLO, Random Forest) on satellite imagery for land use classification.

## Models
- **U-Net** - ResNet34 encoder
- **DeepLab** - v3+ architecture  
- **YOLO** - v8 segmentation
- **Random Forest** - Traditional ML baseline

## Setup

```bash
git clone https://github.com/aggelosntou/semantic-segmentation-benchmark.git
cd semantic-segmentation-benchmark
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For GPU training:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Training

Train individual models:
```bash
cd models/unet_no_patches && python train.py
cd models/deeplab && python train.py
cd models/yolo && python train.py
cd models/random_forest && python train.py
```

## Semi-Supervised Learning (Pseudo-Labeling)

This project includes a complete semi-supervised learning pipeline using model agreement for pseudo-labeling.

### Pipeline Overview
1. **Generate predictions** from 4 trained models on 100 unlabeled images
2. **Build consensus masks** where pixels are labeled if N≥2,3,4 models agree
3. **Create two image variants**: original and gray-painted (unknown regions grayed out)
4. **Train DeepLab** with different pseudo-labeling strategies

### Run Experiments
```bash
python run_experiments.py
```

This runs 18 pseudo-labeling experiments + baseline with combinations of:
- **Pseudo images**: 10, 50, 100 (selected by highest coverage)
- **Agreement levels**: 2, 3, 4 models must agree
- **Image variants**: original vs gray-painted unknown regions

Results saved to `experiment_results.csv` and models to `checkpoints/`.

### Manual Pipeline Steps
If you want to run the pipeline manually:

1. **Generate model predictions**:
```bash
python scripts/infer_deeplab.py
python scripts/infer_unet.py  
python scripts/infer_yolo.py
python scripts/infer_rf.py
```

2. **Build consensus masks**:
```bash
python scripts/build_consensus.py
```

3. **Run experiments**:
```bash
python pseudo_labeling_experiments.py
```

## Evaluation

Evaluate models:
```bash
python models/unet_no_patches/eval.py
python models/deeplab/eval.py
python models/yolo/eval.py
python models/random_forest/eval.py
```

## Dataset Structure

```
dataset/
├── train/          # 90 labeled training images + masks
├── test/           # 23 test images + masks
└── lowres/         # Low resolution version

data/
├── unlabeled/      # 100 unlabeled images
└── pseudo_images/  # Gray-painted versions by agreement level

masks_agree{2,3,4}/ # Consensus masks (agreement ≥ N models)
```

## Classes (8 total)

1. Unknown (155,155,155)
2. Artificial Land (226,169,41)  
3. Woodland (60,16,152)
4. Arable Land (132,41,246)
5. Frygana (0,255,0)
6. Bareland (255,255,255)
7. Water (0,0,255)
8. Permanent Cultivation (255,255,0)

## Key Features

- All images standardized to 512×512
- Early stopping to prevent overfitting
- CUDA support for GPU training
- Comprehensive pseudo-labeling pipeline
- Automated evaluation metrics (precision, recall, F1)

## Requirements

- Python 3.8+
- CUDA 12.1+ (for GPU)
- 8GB+ GPU memory recommended
