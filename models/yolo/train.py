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

    if len(train_labels) == 0:
        print("❌ No label files found in train/labels/")
        print("Please run the dataset conversion script first:")
        print("   python models/yolo/dataset.py")
        return False

    print(f"✅ Dataset validation passed:")
    print(f"   - Train labels: {len(train_labels)}")
    print(f"   - Test labels: {len(test_labels)}")
    return True


def train_model(data_yaml, save_path, epochs=100, imgsz=640, batch=8, device="0", patience=10,
                base_model="yolov8n-seg.pt"):
    """Train YOLO model and save best weights."""
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
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=patience,
            save=True,
            exist_ok=True,
            verbose=True,
        )

        save_dir = Path(results.save_dir)
        best_weights = save_dir / "weights" / "best.pt"

        if best_weights.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_weights, save_path)
            print(f"✅ Saved best model to {save_path}")
        else:
            print(f"⚠️ Best weights not found at {best_weights}")
            return False

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)

    print("🔍 Validating dataset structure...")
    if not validate_dataset_structure():
        sys.exit(1)

    print("\n🎯 Starting YOLO training on train set only...")

    success = train_model(
        data_yaml=YOLO_DIR / "train.yaml",
        save_path=YOLO_DIR / "yolo_train.pt",
        epochs=100,
        imgsz=640,
        batch=8,
        device="0",
        patience=10,
        base_model="yolov8n-seg.pt"
    )

    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n⚠️ Training failed. Check logs above.")
        sys.exit(1)
