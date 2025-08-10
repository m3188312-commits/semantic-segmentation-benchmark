# Semantic Segmentation Benchmark for Satellite Data

This project implements and benchmarks multiple semantic segmentation models for satellite imagery, specifically designed for 8 land cover classes.

## 🎯 Classes

1. **Unknown** - Unclassified areas
2. **Artificial Land** - Urban areas, buildings, roads
3. **Woodland** - Forest and wooded areas
4. **Arable Land** - Agricultural fields
5. **Frygana** - Mediterranean shrubland
6. **Bareland** - Exposed soil/rock
7. **Water** - Lakes, rivers, sea
8. **Permanent Cultivation** - Orchards, vineyards

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install using pip
pip install -r requirements.txt

# Or install using setup.py
pip install -e .
```

### 2. Prepare Dataset

Your dataset should have the following structure:
```
dataset/
├── train/
│   ├── image/     # Training images (.jpg, .png)
│   ├── mask/      # Ground truth masks (.png)
│   └── labels/    # Generated YOLO labels (.txt)
├── test/
│   ├── image/     # Test images
│   ├── mask/      # Test masks
│   └── labels/    # Test labels
└── lowres/
    ├── image/     # Low resolution images
    ├── mask/      # Low resolution masks
    └── labels/    # Low resolution labels
```

### 3. Convert Masks to YOLO Format

```bash
cd models/yolo
python dataset.py
```

This script will:
- Convert your mask images to YOLO polygon format
- Generate label files for each split
- Handle color tolerance for accurate class detection

### 4. Train YOLO Model

```bash
# Train on full resolution data
python train.py

# Or train individual models
python -c "
from train import train_model
train_model(
    data_yaml='train.yaml',
    save_path='yolo_train.pt',
    epochs=100,
    imgsz=640,
    batch=8
)
"
```

## 🔧 Model Configurations

### YOLO Configuration

- **Base Model**: YOLOv8n-seg (nano segmentation model)
- **Image Size**: 640x640 pixels
- **Batch Size**: 8 (adjust based on GPU memory)
- **Epochs**: 100 with early stopping (patience=10)
- **Optimizer**: AdamW with learning rate 0.01
- **Loss Weights**: Optimized for segmentation tasks

### Training Parameters

The training script includes several optimizations:
- **Warmup**: 3 epochs with momentum scheduling
- **Label Smoothing**: Disabled for precise segmentation
- **Mask Overlap**: Enabled for better mask quality
- **Dropout**: Configurable regularization

## 📊 Performance Monitoring

Training progress is automatically logged and includes:
- Loss curves (box, cls, dfl, pose)
- Validation metrics (mAP, precision, recall)
- Learning rate scheduling
- Early stopping based on validation performance

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'ultralytics'"**
   ```bash
   pip install ultralytics>=8.0.0
   ```

2. **CUDA out of memory**
   - Reduce batch size in `train.py`
   - Use smaller image size (e.g., 512 instead of 640)
   - Use CPU training: `device="cpu"`

3. **Dataset not found**
   - Ensure you're running from the project root
   - Check that `dataset/` directory exists
   - Verify mask files are in PNG format

4. **Poor training results**
   - Check class balance in your dataset
   - Verify mask quality and color accuracy
   - Increase training epochs
   - Use data augmentation

### Validation

Before training, the script validates:
- Dataset directory structure
- Presence of label files
- YAML configuration files
- Required dependencies

## 📁 Project Structure

```
├── models/
│   ├── yolo/           # YOLO implementation
│   ├── unet_patches/   # U-Net with patches
│   ├── unet_no_patches/# U-Net without patches
│   ├── deeplab/        # DeepLab implementation
│   └── random_forest/  # Random Forest baseline
├── dataset/            # Training and test data
├── evaluations/        # Model performance metrics
└── scripts/           # Utility scripts
```

## 🔬 Advanced Usage

### Custom Training

```python
from models.yolo.train import train_model

# Custom configuration
train_model(
    data_yaml='custom.yaml',
    save_path='custom_model.pt',
    epochs=200,
    imgsz=1024,
    batch=4,
    device="0",  # GPU 0
    patience=20,
    base_model="yolov8s-seg.pt"  # Small model
)
```

### Multi-GPU Training

```python
# Use multiple GPUs
train_model(
    data_yaml='train.yaml',
    save_path='multi_gpu.pt',
    device="0,1,2,3"  # Use GPUs 0,1,2,3
)
```

## 📈 Expected Results

With proper training, you should expect:
- **mAP@0.5**: 0.7-0.9 for well-defined classes
- **mAP@0.5:0.95**: 0.4-0.7 overall
- **Training Time**: 2-8 hours on modern GPU
- **Model Size**: ~6MB (YOLOv8n-seg)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO implementation based on Ultralytics
- Dataset structure inspired by COCO format
- Satellite imagery processing best practices
