from pathlib import Path
from ultralytics import YOLO
import shutil

YOLO_DIR = Path("models/yolo")

def train_model(data_yaml, save_path, epochs=100, imgsz=640, batch=8, device="0", patience=10, base_model="yolov8n-seg.pt"):
    model = YOLO(base_model)

    # Launch training with GPU + early stopping
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,          # "0" = first GPU
        patience=patience,      # <-- early stopping patience
        project=None,           # avoid runs/ default project
        name=None,              # avoid nested subfolders
        save=True,
        save_period=-1,         # no periodic checkpoints
        exist_ok=True,
        verbose=True
    )

    # Resolve best checkpoint robustly
    save_dir = Path(results.save_dir)               # training output dir created by Ultralytics
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, save_path)
        print(f"✅ Saved best model to {save_path}")
    else:
        print(f"⚠ Best weights not found at {best_weights}")

    # Optional: clean up training folder to keep repo lean
    try:
        shutil.rmtree(save_dir)
        print(f"🧹 Cleaned temp training dir: {save_dir}")
    except Exception as e:
        print(f"ℹ Skipped cleanup ({e})")

if __name__ == "__main__":
    # Model trained on dataset/train
    train_model(
        data_yaml=YOLO_DIR / "train.yaml",
        save_path=YOLO_DIR / "yolo_train.pt",
        epochs=100,
        imgsz=640,
        batch=8,
        device="0",         # force GPU
        patience=10,        # early stopping
        base_model="yolov8n-seg.pt"
    )

    # Model trained on dataset/lowres
    train_model(
        data_yaml=YOLO_DIR / "lowres.yaml",
        save_path=YOLO_DIR / "yolo_lowres.pt",
        epochs=100,
        imgsz=640,
        batch=8,
        device="0",         # force GPU
        patience=10,        # early stopping
        base_model="yolov8n-seg.pt"
    )
