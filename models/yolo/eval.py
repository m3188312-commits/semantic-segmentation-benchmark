from pathlib import Path
import argparse
import torch
from ultralytics import YOLO

# ---------- Settings ----------
PROJECT_ROOT   = Path(__file__).resolve().parents[2]    # repo root
DATASET_ROOT   = PROJECT_ROOT / "dataset"               # dataset/train|test|lowres/image
SCRIPTS_ROOT   = PROJECT_ROOT / "scripts"               # holds yolo_train.pt, yolo_lowres.pt
OUT_ROOT       = PROJECT_ROOT / "predictions" / "yolo"  # predictions/yolo/<model>/<split>/

MODELS = {
    "yolo_train":  SCRIPTS_ROOT / "yolo_train.pt",
    "yolo_lowres": SCRIPTS_ROOT / "yolo_lowres.pt",
}

SPLITS = ["train", "test", "lowres"]
# -------------------------------


def resolve_image_dir(split: str) -> Path:
    """
    Resolve the 'image' directory for a split, tolerating either `image` or `images`.
    Prefers 'image' (your layout), falls back to 'images' if present.
    """
    d1 = DATASET_ROOT / split / "image"
    d2 = DATASET_ROOT / split / "images"
    if d1.exists():
        return d1
    if d2.exists():
        return d2
    raise FileNotFoundError(f"No image directory found for split='{split}' "
                            f"(checked: {d1} and {d2}).")


def run_predictions(weights: Path, split: str, out_dir: Path, conf: float, imgsz: int, device: str):
    """
    Execute YOLO segmentation inference for a given model+split and write overlays to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto device if requested
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights))
    source = str(resolve_image_dir(split))

    # We set project=out_dir and name="" to avoid Ultralytics creating nested runs/<name> folders
    model.predict(
        source=source,
        task="segment",
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(out_dir),
        name="",           # write files directly under out_dir
        exist_ok=True,
        verbose=True,
        workers=0          # deterministic on Windows
    )


def main():
    ap = argparse.ArgumentParser(description="Batch-generate YOLO segmentation predictions for all models and splits.")
    ap.add_argument("--conf",  type=float, default=0.01, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int,   default=640,  help="Inference image size")
    ap.add_argument("--device", default="auto",          help="'auto', 'cpu', or CUDA index (e.g., '0')")
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help=f"Subset of models to run (default: {list(MODELS.keys())})")
    ap.add_argument("--splits", nargs="*", default=SPLITS,
                    help=f"Subset of splits to run (default: {SPLITS})")
    args = ap.parse_args()

    # Execute matrix: models × splits
    for model_name in args.models:
        weights = MODELS.get(model_name)
        if not weights or not weights.exists():
            print(f"⚠️  Skipping '{model_name}': weights not found at {weights}")
            continue

        for split in args.splits:
            try:
                out_dir = OUT_ROOT / model_name / split
                print(f"\n▶️  {model_name} on {split} → {out_dir}")
                run_predictions(weights, split, out_dir, conf=args.conf, imgsz=args.imgsz, device=args.device)
                print(f"✅ Done: {model_name} / {split}")
            except Exception as e:
                print(f"❌ Failed: {model_name} / {split} — {e}")


if __name__ == "__main__":
    main()
