# scripts/infer_rf.py
# Run from repo root:
#   python -m scripts.infer_rf

from pathlib import Path
import numpy as np
from PIL import Image

from models.random_forest.model   import load_model
from models.random_forest.dataset import extract_features, COLOR2CLASS

# === Paths ===
WEIGHTS = Path("scripts/rf_model_train.pkl")       # adjust if needed
LIST    = Path("data/unlabeled_100_list.txt")      # one path per line (full or filename)
BASE    = Path("dataset/unlabeled/image")          # used if LIST has bare filenames

OUT_MASK_ID  = Path("outputs/unlabeled_preds/rf/masks")      # uint8 [0..7]
OUT_MASK_VIS = Path("outputs/unlabeled_preds/rf/masks_vis")  # RGB visualization

# Build inverse palette: class_idx -> (r,g,b)
INV_MAP = {cls: rgb for rgb, cls in COLOR2CLASS.items()}
UNKNOWN_ID = 0

def colorize_ids(idmask: np.ndarray) -> np.ndarray:
    """Map class IDs (H,W) -> RGB (H,W,3) using the project palette."""
    h, w = idmask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in INV_MAP.items():
        rgb[idmask == cls] = col
    return rgb

def read_list_bom_safe(path: Path) -> list[str]:
    # utf-8-sig strips BOM if present
    return [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

def main():
    OUT_MASK_ID.mkdir(parents=True, exist_ok=True)
    OUT_MASK_VIS.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"RF weights not found: {WEIGHTS}")
    clf = load_model(str(WEIGHTS))

    # Resolve paths (support either full paths in the list, or bare filenames)
    entries = read_list_bom_safe(LIST)
    paths = []
    for e in entries:
        p = Path(e)
        paths.append(p if p.is_absolute() or (len(p.parts) > 0 and p.parts[0] == "dataset") else (BASE / p))

    for p in paths:
        if not p.exists():
            print(f"[WARN] missing image: {p} — skipping")
            continue

        # Load image and extract per-pixel features
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.uint8)                # (H,W,3)
        feats = extract_features(arr)                      # (H,W,F)
        H, W, F = feats.shape
        X = feats.reshape(-1, F)

        # Predict class id per pixel
        y = clf.predict(X).astype(np.uint8)                # (H*W,)
        pred = y.reshape(H, W)                             # (H,W) in [0..7]

        # Save ID mask
        id_path = OUT_MASK_ID / (p.stem + ".png")
        Image.fromarray(pred, mode="L").save(id_path)

        # Save color visualization
        vis = colorize_ids(pred)
        vis_path = OUT_MASK_VIS / (p.stem + ".png")
        Image.fromarray(vis, mode="RGB").save(vis_path)

        print(f"saved id:  {id_path}")
        print(f"saved vis: {vis_path}")

if __name__ == "__main__":
    main()
