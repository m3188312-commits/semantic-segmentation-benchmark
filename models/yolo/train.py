from pathlib import Path
from ultralytics import YOLO
import shutil
import os
import sys

YOLO_DIR = Path(__file__).parent
PROJECT_ROOT = YOLO_DIR.parent.parent

def validate_dataset_structure():
    """Validate that the dataset structure is correct before training."""
    required_dirs = [
        PROJECT_ROOT / "dataset" / "train" / "image",
        PROJECT_ROOT / "dataset" / "train" / "labels",
        PROJECT_ROOT / "dataset" / "test" / "image",
        PROJECT_ROOT / "dataset" / "test" / "labels",
        PROJECT_ROOT / "dataset" / "lowres" / "image",
        PROJECT_ROOT / "dataset" / "lowres" / "labels"
    ]
    
    missing_dirs = [d for d in required_dirs if not d.exists()]
    if missing_dirs:
        print("❌ Missing required dataset directories:")
        for d in missing_dirs:
            print(f"   - {d}")
        print("\nPlease run the dataset conversion script first:")
        print("   python models/yolo/dataset.py")
        return False
    
    # Check if labels exist
    train_labels = list((PROJECT_ROOT / "dataset" / "train" / "labels").glob("*.txt"))
    test_labels = list((PROJECT_ROOT / "dataset" / "test" / "labels").glob("*.txt"))
    lowres_labels = list((PROJECT_ROOT / "dataset" / "lowres" / "labels").glob("*.txt"))
    
    if len(train_labels) == 0:
        print("❌ No label files found in train/labels/")
        print("Please run the dataset conversion script first:")
        print("   python models/yolo/dataset.py")
        return False
    
    print(f"✅ Dataset validation passed:")
    print(f"   - Train labels: {len(train_labels)}")
    print(f"   - Test labels: {len(test_labels)}")
    print(f"   - Lowres labels: {len(lowres_labels)}")
    return True

def train_model(data_yaml, save_path, epochs=100, imgsz=640, batch=8, device="0", patience=10, base_model="yolov8n-seg.pt"):
    """Train YOLO model with improved error handling and validation."""
    
    # Validate data yaml exists
    if not Path(data_yaml).exists():
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    print(f"🚀 Starting training with {base_model}")
    print(f"   - Data: {data_yaml}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Image size: {imgsz}")
    print(f"   - Batch size: {batch}")
    print(f"   - Device: {device}")
    
    try:
        model = YOLO(base_model)
        
        # Launch training with GPU + early stopping
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=patience,
            project=None,
            name=None,
            save=True,
            save_period=-1,
            exist_ok=True,
            verbose=True,
            # Additional parameters for better training
            optimizer="auto",
            lr0=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,  # box loss gain
            cls=0.5,  # cls loss gain
            dfl=1.5,  # dfl loss gain
            pose=12.0,  # pose loss gain
            kobj=2.0,  # keypoint obj loss gain
            label_smoothing=0.0,  # label smoothing epsilon
            nbs=64,  # nominal batch size
            overlap_mask=True,  # masks should overlap during training
            mask_ratio=4,  # mask downsample ratio
            dropout=0.0,  # use dropout regularization
            val=True,  # run validation during training
        )
        
        # Resolve best checkpoint robustly
        save_dir = Path(results.save_dir)
        best_weights = save_dir / "weights" / "best.pt"
        
        if best_weights.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_weights, save_path)
            print(f"✅ Saved best model to {save_path}")
            
            # Also save last weights
            last_weights = save_dir / "weights" / "last.pt"
            if last_weights.exists():
                last_save_path = save_path.parent / f"{save_path.stem}_last{save_path.suffix}"
                shutil.copy2(last_weights, last_save_path)
                print(f"✅ Saved last model to {last_save_path}")
        else:
            print(f"⚠️ Best weights not found at {best_weights}")
            return False
        
        # Clean up training folder to keep repo lean
        try:
            shutil.rmtree(save_dir)
            print(f"🧹 Cleaned temp training dir: {save_dir}")
        except Exception as e:
            print(f"ℹ️ Skipped cleanup ({e})")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    # Change to project root for relative paths to work
    os.chdir(PROJECT_ROOT)
    
    print("🔍 Validating dataset structure...")
    if not validate_dataset_structure():
        sys.exit(1)
    
    print("\n🎯 Starting YOLO training...")
    
    # Model trained on dataset/train
    print("\n📚 Training on full resolution dataset...")
    success1 = train_model(
        data_yaml=YOLO_DIR / "train.yaml",
        save_path=YOLO_DIR / "yolo_train.pt",
        epochs=100,
        imgsz=640,
        batch=8,
        device="0",
        patience=10,
        base_model="yolov8n-seg.pt"
    )
    
    # Model trained on dataset/lowres
    print("\n📚 Training on low resolution dataset...")
    success2 = train_model(
        data_yaml=YOLO_DIR / "lowres.yaml",
        save_path=YOLO_DIR / "yolo_lowres.pt",
        epochs=100,
        imgsz=640,
        batch=8,
        device="0",
        patience=10,
        base_model="yolov8n-seg.pt"
    )
    
    if success1 and success2:
        print("\n🎉 All training completed successfully!")
    else:
        print("\n⚠️ Some training runs failed. Check the logs above.")
        sys.exit(1)
