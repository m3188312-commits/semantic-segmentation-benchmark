from pathlib import Path
import yaml

ROOT = Path("dataset").resolve()  # absolute path to dataset folder
YOLO_DIR = Path("models/yolo")

with open(YOLO_DIR / "classes.yaml", "r") as f:
    cfg = yaml.safe_load(f)
names = cfg["names"]

def write_yaml(path, train_dir, val_dir, test_dir):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "nc": len(names),
        "names": names,
        "train": str(train_dir.resolve()),
        "val":   str(val_dir.resolve()),
        "test":  str(test_dir.resolve()),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

if __name__ == "__main__":
    write_yaml(
        YOLO_DIR / "train.yaml",
        ROOT / "train" / "image",
        ROOT / "test" / "image",
        ROOT / "test" / "image",
    )
    write_yaml(
        YOLO_DIR / "lowres.yaml",
        ROOT / "lowres" / "image",
        ROOT / "test" / "image",
        ROOT / "test" / "image",
    )
    print("✅ Wrote models/yolo/train.yaml and models/yolo/lowres.yaml with absolute paths")
