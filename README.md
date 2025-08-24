# Semantic Segmentation Benchmark

A comprehensive benchmark for semantic segmentation using multiple deep learning architectures and traditional machine learning approaches.

## 🎯 Project Overview

This project implements and evaluates various semantic segmentation models on satellite imagery data, focusing on land use classification. The goal is to compare different approaches and provide a robust framework for semantic segmentation tasks.

## 🏗️ Architecture

### Deep Learning Models
- **U-Net**: Vanilla U-Net with ResNet34 encoder
- **DeepLab**: DeepLab v3+ architecture
- **YOLO**: YOLOv8 adapted for segmentation
- **Random Forest**: Traditional ML approach for comparison

### Dataset Structure
```
dataset/
├── train/          # Training images and masks
├── test/           # Test images and masks  
├── lowres/         # Low-resolution images and masks
└── unlabeled/      # Unlabeled images for pseudo-labeling
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 12.1+ (for GPU training)
- 8GB+ GPU memory recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aggelosntou/semantic-segmentation-benchmark.git
   cd semantic-segmentation-benchmark
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### CUDA Installation (GPU Users)

For CUDA 12.1 support, install PyTorch with:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📊 Model Training

### U-Net Training
```bash
cd models/unet_no_patches
python train.py
```

**Configuration:**
- Batch size: 8
- Learning rate: 1e-3
- Epochs: 100
- Early stopping patience: 7
- Encoder: ResNet34 (ImageNet pretrained)

### DeepLab Training
```bash
cd models/deeplab
python train.py
```

### YOLO Training
```bash
cd models/yolo
python train.py
```

### Random Forest Training
```bash
cd models/random_forest
python train.py
```

## 🔍 Model Evaluation

### Full Pipeline (Predict + Evaluate)
```bash
# U-Net
python models/unet_no_patches/eval.py

# DeepLab  
python models/deeplab/eval.py

# Random Forest
python models/random_forest/eval.py

# YOLO
python models/yolo/eval.py
```

**Note**: By default, all models evaluate on **train**, **test**, and **lowres** datasets.

### Evaluate Only (Requires existing predictions)
```bash
python models/unet_no_patches/eval.py --evaluate
```

### Predict Only
```bash
python models/unet_no_patches/eval.py --predict
```

### Single Image Evaluation
```bash
python models/unet_no_patches/eval.py --single-image test 0000.jpg
```

### Custom Model Path
```bash
python models/unet_no_patches/eval.py --model_path /path/to/model.pth
```

## 📈 Evaluation Metrics

All models output the following metrics to terminal:
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores  
- **F1-Score**: Per-class F1 scores
- **Summary**: Average metrics across all classes

## 🏷️ Class Labels

The dataset contains 8 land use classes:
1. **Unknown/Background** (155, 155, 155)
2. **Artificial Land** (226, 169, 41)
3. **Woodland** (60, 16, 152)
4. **Arable Land** (132, 41, 246)
5. **Frygana** (0, 255, 0)
6. **Bareland** (255, 255, 255)
7. **Water** (0, 0, 255)
8. **Permanent Cultivation** (255, 255, 0)

## 🔧 Configuration

### Training Parameters
- **Batch Size**: Configurable per model
- **Learning Rate**: Adaptive learning rate with schedulers
- **Early Stopping**: Prevents overfitting
- **Class Balancing**: Automatic class weight calculation

### Data Preprocessing
- **Image Size**: 512x512 pixels
- **Normalization**: ImageNet mean/std values
- **Augmentation**: Basic transforms (resize, normalize, ToTensor)

## 📁 Project Structure

```
semantic-segmentation-benchmark/
├── models/                 # Model implementations
│   ├── unet_no_patches/   # U-Net model
│   ├── deeplab/           # DeepLab model
│   ├── yolo/              # YOLO model
│   └── random_forest/     # Random Forest model
├── dataset/               # Dataset files
├── scripts/               # Utility scripts
├── predictions/           # Model predictions
├── evaluations/           # Evaluation results
└── requirements.txt       # Dependencies
```

## 🚀 Advanced Usage

### Pseudo-Labeling Pipeline
The project includes a complete pipeline for using unlabeled data:
1. Generate predictions from multiple models
2. Build consensus masks with agreement thresholds
3. Evaluate different pseudo-labeling strategies

### Multi-Split Evaluation
Models can be evaluated on multiple dataset splits:
- **Train set**: Training data evaluation (overfitting check)
- **Test set**: Standard evaluation
- **Lowres set**: Lower resolution images
- **Custom splits**: User-defined data splits

## 🐛 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size
2. **Import Errors**: Ensure virtual environment is activated
3. **Model Not Found**: Check model weights path in scripts folder

### GPU Requirements
- **Minimum**: 4GB GPU memory
- **Recommended**: 8GB+ GPU memory
- **CUDA Version**: 12.1+ for optimal performance

## 📚 References

- **U-Net**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **DeepLab**: [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets](https://arxiv.org/abs/1606.00915)
- **YOLO**: [YOLOv8: You Only Look Once](https://github.com/ultralytics/ultralytics)
- **Random Forest**: [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Aggelos Ntou** - Initial work

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Segmentation Models PyTorch for model implementations
- Ultralytics for YOLO implementation

---

**Note**: This project is designed for research and educational purposes. Ensure you have proper licenses for any datasets used.
