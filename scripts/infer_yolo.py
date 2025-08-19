# scripts/infer_yolo.py
# Run:
#   python -m scripts.infer_yolo

from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
from models.deeplab.dataset import CLASS_RGB  # reuse your palette

WEIGHTS = Path("scripts/yolo_train.pt")
LIST    = Path("data/unlabeled_100_list.txt")
BASE    = Path("dataset/unlabeled/image")

OUT_MASK_ID  = Path("outputs/unlabeled_preds/yolo/masks")
OUT_MASK_VIS = Path("outputs/unlabeled_preds/yolo/masks_vis")

INV_MAP = {cls: rgb for rgb, cls in CLASS_RGB.items()}

def read_list_bom_safe(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

def colorize_ids(idmask: np.ndarray) -> np.ndarray:
    h, w = idmask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in INV_MAP.items():
        rgb[idmask == cls] = col
    return rgb

def upsample_to(img_h, img_w, m: np.ndarray) -> np.ndarray:
    """Nearest-neighbor upsample of a (h,w) float/bool mask to (img_h,img_w)."""
    # m is float [0,1] or bool; convert to uint8 image for PIL resize
    m_u8 = (m > 0.5).astype(np.uint8) * 255
    m_resized = Image.fromarray(m_u8, mode="L").resize((img_w, img_h), resample=Image.NEAREST)
    return np.array(m_resized, dtype=np.uint8) > 0

def main():
    OUT_MASK_ID.mkdir(parents=True, exist_ok=True)
    OUT_MASK_VIS.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    # Resolve paths
    entries = read_list_bom_safe(LIST)
    paths = []
    for e in entries:
        p = Path(e)
        paths.append(p if p.is_absolute() or (len(p.parts) > 0 and p.parts[0] == "dataset") else (BASE / p))

    for p in paths:
        if not p.exists():
            print(f"[WARN] missing image: {p} — skipping")
            continue

        res = model.predict(source=str(p), imgsz=640, verbose=False)[0]
        H, W = res.orig_shape
        sem = np.zeros((H, W), dtype=np.uint8)  # Unknown=0

        if res.masks is not None and res.boxes is not None and len(res.masks.data) > 0:
            # res.masks.data: (N, h, w) at model scale; must upsample to (H, W)
            masks_np = res.masks.data.cpu().numpy()           # float [0..1], shape (N,h,w)
            cls_ids  = res.boxes.cls.cpu().numpy().astype(np.int64)  # (N,)
            for mi, cls in zip(masks_np, cls_ids):
                mi_up = upsample_to(H, W, mi)                 # (H,W) bool
                sem[mi_up] = int(cls)

        # Persist artifacts
        id_path  = OUT_MASK_ID / (p.stem + ".png")
        vis_path = OUT_MASK_VIS / (p.stem + ".png")
        Image.fromarray(sem, mode="L").save(id_path)
        Image.fromarray(colorize_ids(sem), mode="RGB").save(vis_path)
        print(f"saved id:  {id_path}")
        print(f"saved vis: {vis_path}")

if __name__ == "__main__":
    main()
