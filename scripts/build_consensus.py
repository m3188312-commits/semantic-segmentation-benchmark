# scripts/build_consensus.py
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

UNKNOWN_ID  = 0
UNKNOWN_RGB = (155, 155, 155)

def read_mask(path: Path) -> np.ndarray:
    """Read a class-ID mask as uint8 (H, W)."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def save_mask(path: Path, mask: np.ndarray) -> None:
    """Save a class-ID mask (uint8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)

def save_masked_image(img_path: Path, pseudo: np.ndarray, out_path: Path) -> None:
    """Save an RGB image where Unknown pixels are painted UNKNOWN_RGB."""
    arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    arr[pseudo == UNKNOWN_ID] = UNKNOWN_RGB
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)

def consensus(masks: list[np.ndarray], k: int, n_classes: int = 8) -> np.ndarray:
    """Pixelwise majority with threshold k over a list of class-ID masks."""
    s = np.stack(masks, axis=0)          # (M, H, W) where M=#models=4
    H, W = s.shape[1:]
    flat = s.reshape(s.shape[0], -1).T   # (N, M)
    out  = np.full(flat.shape[0], UNKNOWN_ID, dtype=np.uint8)
    for i in range(flat.shape[0]):
        counts = np.bincount(flat[i], minlength=n_classes)
        cls    = int(np.argmax(counts))
        cnt    = int(counts[cls])
        out[i] = cls if cnt >= k else UNKNOWN_ID
    return out.reshape(H, W)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks",          default="2,3,4", help="comma-separated K values (agreement thresholds)")
    ap.add_argument("--unlabeled",   default="data/unlabeled", help="folder with unlabeled images")
    ap.add_argument("--deeplab",     default="outputs/unlabeled_preds/deeplab/masks")
    ap.add_argument("--unet",        default="outputs/unlabeled_preds/unet/masks")
    ap.add_argument("--yolo",        default="outputs/unlabeled_preds/yolo/masks")
    ap.add_argument("--rf",          default="outputs/unlabeled_preds/rf/masks")
    ap.add_argument("--out_pseudo",  default="data/pseudo_labels")
    ap.add_argument("--out_masked",  default="data/pseudo_images")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]

    unlabeled_dir = Path(args.unlabeled)
    deeplab_dir   = Path(args.deeplab)
    unet_dir      = Path(args.unet)
    yolo_dir      = Path(args.yolo)
    rf_dir        = Path(args.rf)
    out_pseudo    = Path(args.out_pseudo)
    out_masked    = Path(args.out_masked)

    processed = 0
    skipped   = 0

    for img_path in sorted(unlabeled_dir.glob("*.*")):
        stem      = img_path.stem
        mask_name = stem + ".png"  # all model masks are saved as PNG

        mpaths = [
            deeplab_dir / mask_name,
            unet_dir    / mask_name,
            yolo_dir    / mask_name,
            rf_dir      / mask_name,
        ]
        if not all(p.exists() for p in mpaths):
            missing = [str(p) for p in mpaths if not p.exists()]
            print(f"[WARN] missing masks for {img_path.name} → {', '.join(missing)}")
            skipped += 1
            continue

        masks = [read_mask(p) for p in mpaths]

        for k in ks:
            pseudo = consensus(masks, k=k, n_classes=8)

            # Save class-ID pseudo mask as PNG
            save_mask(out_pseudo / f"agreement_{k}" / mask_name, pseudo)

            # Save masked image; keep original image extension
            save_masked_image(img_path, pseudo, out_masked / f"agreement_{k}" / img_path.name)

        processed += 1

    print(f"✅ Consensus complete. Images processed: {processed}, skipped: {skipped}")
    print(f"   → Pseudo masks: {out_pseudo}")
    print(f"   → Masked images: {out_masked}")

if __name__ == "__main__":
    main()
