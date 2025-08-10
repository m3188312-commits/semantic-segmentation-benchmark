import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

# === CONFIGURATION ===
ROOT_DIR    = Path("dataset")  # contains train/test/lowres
TOLERANCE   = 8                # color tolerance per channel (tight to catch subtle shades)
MIN_AREA_PX = 10               # keep tiny regions
MAX_PTS     = 1000             # cap polygon vertices to avoid pathological files
IMAGE_EXTS  = (".jpg", ".jpeg", ".png")
MASK_EXTS   = (".jpg", ".jpeg", ".png")

# --- Load classes & colors from YAML ---
with open(Path(__file__).parent / "classes.yaml", "r") as f:
    cfg = yaml.safe_load(f)

names = cfg["names"]
mcm   = cfg["mask_color_map"]  # class name -> "R,G,B"
# ID order follows 'names'
color_map = {i: tuple(map(int, mcm[n].split(","))) for i, n in enumerate(names)}

def build_mask_index(mask_dir: Path):
    """Index masks by stem, accepting allowed extensions."""
    index = {}
    for ext in MASK_EXTS:
        for p in mask_dir.glob(f"*{ext}"):
            index[p.stem] = p
    return index

def simplify_if_needed(cnt, max_pts=MAX_PTS):
    """Optionally reduce vertices very slightly if contour is extremely dense."""
    if len(cnt) <= max_pts:
        return cnt
    peri = cv2.arcLength(cnt, True)
    # tiny epsilon to preserve detail while reducing point count
    eps  = max(0.001 * peri, 1e-6)
    simp = cv2.approxPolyDP(cnt, eps, True)
    # if still too large, increase epsilon progressively
    while len(simp) > max_pts:
        eps *= 1.5
        simp = cv2.approxPolyDP(cnt, eps, True)
    return simp

def process_split(split_name):
    split_dir = ROOT_DIR / split_name
    img_dir   = split_dir / "image"   # your structure (we'll address training separately)
    mask_dir  = split_dir / "mask"
    label_dir = split_dir / "labels"
    label_dir.mkdir(exist_ok=True)

    if not img_dir.exists():
        print(f"[WARN] Missing image dir: {img_dir}")
        return
    if not mask_dir.exists():
        print(f"[WARN] Missing mask dir: {mask_dir}")
        return

    mask_idx = build_mask_index(mask_dir)

    images = []
    for ext in IMAGE_EXTS:
        images.extend(img_dir.glob(f"*{ext}"))
    images = sorted(images)

    total_polygons = 0
    missing_masks = 0

    for img_path in tqdm(images, desc=f"Processing {split_name}"):
        stem = img_path.stem
        mask_path = mask_idx.get(stem, None)
        if mask_path is None:
            missing_masks += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask is None:
            print(f"[WARN] Cannot read mask: {mask_path}")
            continue

        H, W = mask.shape[:2]
        polygons = []

        for cls_id, rgb in color_map.items():
            # OpenCV is BGR; we tolerance-match around target
            bgr_target = (rgb[2], rgb[1], rgb[0])
            lb = np.clip(np.array(bgr_target) - TOLERANCE, 0, 255).astype(np.uint8)
            ub = np.clip(np.array(bgr_target) + TOLERANCE, 0, 255).astype(np.uint8)

            bin_mask = cv2.inRange(mask, lb, ub)

            # Capture maximum detail: keep all points; external contours only (YOLO polygons don't support holes)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_AREA_PX:
                    continue

                cnt = simplify_if_needed(cnt, MAX_PTS)

                # flatten and normalize
                cnt = cnt.squeeze(1)  # (N,2)
                if cnt.ndim != 2 or cnt.shape[0] < 3:
                    continue
                xs = cnt[:, 0] / W
                ys = cnt[:, 1] / H
                pts = np.stack([xs, ys], axis=1).reshape(-1)
                if pts.size < 6:  # need >=3 points
                    continue

                polygons.append((cls_id, pts.tolist()))

        # Write YOLO label file (even if empty -> explicit)
        label_path = label_dir / (stem + ".txt")
        with open(label_path, "w") as fout:
            for cid, poly in polygons:
                coords = " ".join(f"{v:.6f}" for v in poly)
                fout.write(f"{cid} {coords}\n")

        total_polygons += len(polygons)

    print(f"[INFO] {split_name}: {total_polygons} polygons written. Missing masks: {missing_masks}")

if __name__ == "__main__":
    for split in ["train", "test", "lowres"]:
        process_split(split)
    print("🏁 Conversion complete: YOLO label files saved in each split/labels/")
