from pathlib import Path
import argparse, cv2, numpy as np, torch
from ultralytics import YOLO
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PRED_ROOT    = PROJECT_ROOT / "predictions" / "yolo"

# Hard palette (RGB) — includes class 0
CLASS_RGB = {
    0: (155, 155, 155),  # Unknown
    1: (226, 169,  41),  # Artificial Land
    2: ( 60,  16, 152),  # Woodland
    3: (132,  41, 246),  # Arable Land
    4: (  0, 255,   0),  # Frygana
    5: (255, 255, 255),  # Bareland
    6: (  0,   0, 255),  # Water
    7: (255, 255,   0),  # Permanent Cultivation
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def image_dir(split: str) -> Path:
    d1 = DATASET_ROOT / split / "image"
    d2 = DATASET_ROOT / split / "images"
    if d1.exists(): return d1
    if d2.exists(): return d2
    raise FileNotFoundError(f"No image dir for split '{split}'")

def list_images(p: Path):
    return sorted([x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS])

def combine_instance_masks(result, H, W, conf_thr=0.001, mask_thr=0.05) -> np.ndarray:
    """
    Build class-index mask (H,W) from YOLO instance outputs.
    - Very permissive thresholds to keep tiny instances.
    - Per-pixel winner is the highest-confidence instance; ties allowed (>=).
    """
    out_idx  = np.zeros((H, W), dtype=np.uint8)
    out_conf = np.zeros((H, W), dtype=np.float32)

    if result.masks is None or len(result.masks.data) == 0:
        return out_idx

    masks = result.masks.data.cpu().numpy()              # (N,h,w)
    cls   = result.boxes.cls.cpu().numpy().astype(int)   # (N,)
    conf  = result.boxes.conf.cpu().numpy().astype(np.float32)

    # Process in descending confidence so strong objects paint first;
    # pixel tie-breaker still done via out_conf comparison.
    order = np.argsort(-conf)
    for k in order:
        if conf[k] < conf_thr:
            continue
        m = (masks[k] >= mask_thr).astype(np.uint8)
        if m.sum() == 0:
            continue
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        c = conf[k]; cid = int(cls[k])

        update = (m == 1) & (c >= out_conf)  # '>=' keeps tiny ties
        if update.any():
            out_idx[update]  = cid
            out_conf[update] = c
    return out_idx

def make_color_mask(id_mask: np.ndarray) -> np.ndarray:
    """
    Produce a HARD color mask (no transparency) in BGR.
    Every pixel is exactly one class color from CLASS_RGB.
    """
    h, w = id_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in CLASS_RGB.items():
        rgb[id_mask == cid] = col
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def run(weights: Path, split: str, conf: float, imgsz: int, device: str,
        save_overlay: bool, mask_thr: float):
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights))
    src   = image_dir(split)
    out   = PRED_ROOT / weights.stem / split
    (out / "masks").mkdir(parents=True, exist_ok=True)     # class-index masks
    (out / "color").mkdir(parents=True, exist_ok=True)     # HARD color masks
    if save_overlay:
        (out / "overlay").mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list_images(src), desc=f"{weights.stem}:{split}"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        r = model.predict(source=img, task="segment", conf=conf, imgsz=imgsz,
                          device=device, save=False, verbose=False, workers=0)[0]

        id_mask = combine_instance_masks(r, H, W, conf_thr=conf, mask_thr=mask_thr)

        # 1) class-index PNG for metrics
        cv2.imwrite(str(out / "masks" / f"{img_path.stem}.png"), id_mask)

        # 2) HARD color PNG (no transparency)
        hard = make_color_mask(id_mask)
        cv2.imwrite(str(out / "color" / f"{img_path.stem}.png"), hard)

        # 3) Optional overlay for quick visual QA
        if save_overlay:
            ov = cv2.addWeighted(img, 1.0, hard, 1.0, 0.0)  # alpha=1.0 => hard draw
            cv2.imwrite(str(out / "overlay" / f"{img_path.stem}.jpg"), ov)

def main():
    ap = argparse.ArgumentParser("Emit HARD color masks + class-index masks from YOLO models.")
    ap.add_argument("--models", nargs="*", default=["yolo_train.pt", "yolo_lowres.pt"])
    ap.add_argument("--splits", nargs="*", default=["train", "test", "lowres"])
    ap.add_argument("--conf", type=float, default=0.001, help="confidence threshold (very low to keep tiny objs)")
    ap.add_argument("--mask-thr", type=float, default=0.05, help="probability threshold to binarize instance masks")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")  # auto / cpu / "0"
    ap.add_argument("--overlay", action="store_true", help="also write hard overlays")
    args = ap.parse_args()

    for m in args.models:
        w = SCRIPTS_ROOT / m
        if not w.exists():
            print(f"⚠️  Missing weights: {w}")
            continue
        for s in args.splits:
            run(w, s, args.conf, args.imgsz, args.device, args.overlay, args.mask_thr)

if __name__ == "__main__":
    main()
