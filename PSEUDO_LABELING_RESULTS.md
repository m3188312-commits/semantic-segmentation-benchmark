# Pseudo-Labeling Pipeline Results

## Overview
Successfully completed pseudo-labeling pipeline for 100 unlabeled images using 4 trained models with consensus-based agreement thresholds.

## Generated Outputs

### 1. Individual Model Predictions (`outputs/unlabeled_preds/`)
- **DeepLab**: 200 files (100 ID masks + 100 visualizations)
- **U-Net**: 200 files (100 ID masks + 100 visualizations)  
- **YOLO**: 200 files (100 ID masks + 100 visualizations)
- **Random Forest**: 200 files (100 ID masks + 100 visualizations)
- **Total**: 800 files (all predictions at 512×512 resolution)

### 2. Consensus Pseudo-Masks (`data/pseudo_labels/`)
- **agreement_2/**: 100 masks (≥2 models agree)
- **agreement_3/**: 100 masks (≥3 models agree)  
- **agreement_4/**: 100 masks (all 4 models agree)

### 3. Gray-Painted Images (`data/pseudo_images/`)
- **agreement_2/**: 100 images with unknown regions painted gray
- **agreement_3/**: 100 images with unknown regions painted gray
- **agreement_4/**: 100 images with unknown regions painted gray

### 4. Combined Training Datasets
Ready-to-use datasets combining original labeled data + pseudo-labeled data:

#### `masks_agree2/` (Agreement ≥2)
- **90 original masks** (from `dataset/train/mask/`)
- **100 pseudo-masks** (from unlabeled images)  
- **100 colorized visualizations** (`*_mask_agree2.png`)
- **Total**: 290 files

#### `masks_agree3/` (Agreement ≥3)  
- **90 original masks** (from `dataset/train/mask/`)
- **100 pseudo-masks** (from unlabeled images)
- **100 colorized visualizations** (`*_mask_agree3.png`)
- **Total**: 290 files

#### `masks_agree4/` (Agreement ≥4)
- **90 original masks** (from `dataset/train/mask/`)  
- **100 pseudo-masks** (from unlabeled images)
- **100 colorized visualizations** (`*_mask_agree4.png`)
- **Total**: 290 files

## Class Mapping
All predictions use consistent 8-class semantic segmentation:

```python
CLASS_RGB = {
    (155, 155, 155): 0,  # Unknown
    (226, 169, 41):   1,  # Artificial Land
    (60, 16, 152):    2,  # Woodland
    (132, 41, 246):   3,  # Arable Land
    (0, 255, 0):      4,  # Frygana
    (255, 255, 255):  5,  # Bareland
    (0, 0, 255):      6,  # Water
    (255, 255, 0):    7,  # Permanent Cultivation
}
```

## Key Features
✅ **Consistent Resolution**: All predictions standardized to 512×512  
✅ **Model Agreement**: Consensus-based pseudo-labeling with N∈{2,3,4} thresholds  
✅ **Unknown Handling**: Pixels with insufficient agreement marked as "Unknown" (class 0)  
✅ **Visualization**: Colorized masks for easy inspection  
✅ **Ready for Training**: Combined datasets with proper file naming conventions

## Next Steps
You can now train your models on the enhanced datasets:
- Use `masks_agree2/` for maximum data (looser agreement)
- Use `masks_agree3/` for balanced quality/quantity  
- Use `masks_agree4/` for highest confidence (strictest agreement)

## Usage
To reproduce this pipeline:
```bash
python run_full_pipeline.py
```

Generated on: $(date)
