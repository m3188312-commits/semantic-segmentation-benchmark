# 🚀 CUDA Laptop Setup Guide for Pseudo-Label Evaluation

## 📋 Overview
This guide will help you set up and run the full pseudo-label evaluation on your CUDA-enabled laptop. The pipeline has been tested on Mac and is ready for GPU acceleration.

## 🖥️ Prerequisites
- CUDA-enabled GPU (NVIDIA)
- Python 3.8+ with pip
- Git

## 📥 Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url> semantic-segmentation-benchmark
cd semantic-segmentation-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For latest CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 🔍 Step 2: Verify CUDA Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8 (or your version)
GPU count: 1 (or more)
```

## 🧪 Step 3: Test Single Case (Optional)

Before running the full evaluation, test with a single case:

```bash
# Test with 5 pseudo-label images, K=2 agreement, keep unknown pixels
python scripts/test_single_case.py --img_count 5 --agreement 2 --handling keep
```

This will:
- Create training dataset with 90 original + 5 pseudo-label images
- Train DeepLab model (real training on GPU)
- Save model to `evaluations/test_case/deeplab/`

## 🚀 Step 4: Run Full Evaluation

The full evaluation will test all 18 cases:

```bash
# Run evaluation for all models
python scripts/evaluate_pseudo_labels.py --models deeplab,unet,yolo,rf

# Or run for specific models only
python scripts/evaluate_pseudo_labels.py --models deeplab,unet

# Or test a specific case first
python scripts/evaluate_pseudo_labels.py --case "10_2_keep"
```

## 📊 Evaluation Cases

The script will evaluate these combinations:

| Images | Agreement (K) | Unknown Handling | Total Cases |
|--------|---------------|------------------|-------------|
| 10     | 2, 3, 4      | keep, remove     | 6 cases     |
| 50     | 2, 3, 4      | keep, remove     | 6 cases     |
| 78     | 2, 3, 4      | keep, remove     | 6 cases     |

**Total: 18 evaluation cases**

## 🏗️ What Each Case Does

1. **Dataset Creation**: Combines original training data (90 images) + pseudo-labels
2. **Model Training**: Trains all 4 models on the combined dataset
3. **Evaluation**: Tests trained models on test set (23 images)
4. **Metrics Collection**: Calculates precision, recall, F1-score, IoU

## 📁 Output Structure

```
evaluations/pseudo_label_results/
├── case_10_2_keep/
│   ├── deeplab/
│   │   ├── best_model.pth
│   │   └── results/
│   │       └── metrics.json
│   ├── unet/
│   ├── yolo/
│   ├── rf/
│   └── case_results.json
├── case_10_2_remove/
├── case_10_3_keep/
├── ... (18 total cases)
└── overall_results.json
```

## ⏱️ Expected Runtime

**On CUDA GPU:**
- Single case: 15-30 minutes
- Full evaluation: 8-12 hours (depending on GPU)

**Breakdown per case:**
- DeepLab: 10-15 min
- UNet: 8-12 min  
- YOLO: 5-8 min
- Random Forest: 2-3 min

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in training scripts
--batch_size 2  # Instead of 4
```

### Training Too Slow
```bash
# Reduce epochs for testing
--epochs 20  # Instead of 80
```

### Model Not Found
```bash
# Check if model files exist
ls -la models/*/train.py
```

## 📈 Results Analysis

After completion, analyze results:

```bash
# View overall results
cat evaluations/pseudo_label_results/overall_results.json

# Compare specific cases
python -c "
import json
with open('evaluations/pseudo_label_results/overall_results.json') as f:
    results = json.load(f)
    
# Find best performing case
best_case = max(results.values(), key=lambda x: x['metrics']['deeplab']['f1_score'])
print(f'Best case: {best_case[\"case_name\"]}')
print(f'DeepLab F1: {best_case[\"metrics\"][\"deeplab\"][\"f1_score\"]:.3f}')
"
```

## 🎯 Key Metrics to Compare

1. **F1-Score**: Overall performance
2. **IoU**: Segmentation quality
3. **Precision vs Recall**: Trade-off analysis
4. **Model Agreement Impact**: K=2 vs K=3 vs K=4
5. **Unknown Pixel Handling**: keep vs remove

## 📝 Next Steps After Evaluation

1. **Generate Report**: Document all findings
2. **Visualize Results**: Create comparison charts
3. **Identify Best Configuration**: Optimal pseudo-label strategy
4. **Recommendations**: For future pseudo-label usage

## 🆘 Need Help?

If you encounter issues:

1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check GPU memory: `nvidia-smi -l 1`
4. Monitor training: Check console output for errors

## 🎉 Success Indicators

✅ **Training**: Models save successfully to output directories  
✅ **Evaluation**: Metrics files generated for each model  
✅ **Results**: `overall_results.json` contains all 18 cases  
✅ **Performance**: GPU utilization during training  

---

**Good luck with your evaluation! 🚀**

The pipeline is designed to be robust and will provide comprehensive results for your pseudo-label research.
