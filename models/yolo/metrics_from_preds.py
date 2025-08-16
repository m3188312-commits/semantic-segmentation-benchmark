# models/yolo/metrics_from_preds.py
from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
PRED_ROOT = PROJECT_ROOT / "predictions" / "yolo"
EVAL_ROOT = PROJECT_ROOT / "evaluations" / "yolo"

# Class color map (RGB)
CLASS_RGB = {
    (155, 155, 155): 0,  # Unknown
    (226, 169,  41): 1,  # Artificial Land
    ( 60,  16, 152): 2,  # Woodland
    (132,  41, 246): 3,  # Arable Land
    (  0, 255,   0): 4,  # Frygana
    (255, 255, 255): 5,  # Bareland
    (  0,   0, 255): 6,  # Water
    (255, 255,   0): 7,  # Permanent Cultivation
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def gt_mask_dir(split: str) -> Path:
    d = DATASET_ROOT / split / "mask"
    if not d.exists():
        raise FileNotFoundError(f"GT masks not found for split='{split}': {d}")
    return d


def list_images(img_dir: Path):
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def rgb_to_id_lut(class_rgb: dict) -> np.ndarray:
    """Build a LUT from RGB -> class_id"""
    lut = np.full((256, 256, 256), 255, dtype=np.uint8)  # 255 means invalid
    for rgb, cid in class_rgb.items():
        r, g, b = rgb
        lut[r, g, b] = cid
    return lut


def gt_color_to_ids(gt_bgr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    ids = lut[rgb[..., 0], rgb[..., 1], rgb[..., 2]].astype(np.int16)
    ids[ids == 255] = -1  # ignore unknown pixels
    return ids


def per_class_metrics(pred: np.ndarray, gt: np.ndarray, nc: int):
    valid = gt != -1
    eps = 1e-9
    rows = []
    tp_sum = fp_sum = fn_sum = 0
    precs, recs, f1s, ious = [], [], [], []

    for c in range(nc):
        p = (pred == c) & valid
        g = (gt == c) & valid
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        iou = tp / (tp + fp + fn + eps)

        rows.append({
            "class_id": c,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "iou": iou,
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        ious.append(iou)

    macro = {
        "precision_macro": np.mean(precs),
        "recall_macro": np.mean(recs),
        "f1_macro": np.mean(f1s),
        "iou_macro": np.mean(ious)
    }
    prec_micro = tp_sum / (tp_sum + fp_sum + eps)
    rec_micro = tp_sum / (tp_sum + fn_sum + eps)
    f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro + eps)
    iou_micro = tp_sum / (tp_sum + fp_sum + fn_sum + eps)

    micro = {
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "iou_micro": iou_micro
    }
    return rows, macro, micro


def evaluate_model_split(model_name: str, split: str, nc: int, class_rgb: dict):
    gt_dir = gt_mask_dir(split)
    pred_dir = PRED_ROOT / model_name / split / "masks"

    if not pred_dir.exists():
        print(f"⚠️ No predicted masks found: {pred_dir}")
        return []

    lut = rgb_to_id_lut(class_rgb)
    per_image_rows = []

    for gt_path in tqdm(list_images(gt_dir), desc=f"{model_name}:{split}"):
        stem = gt_path.stem
        pred_path = pred_dir / f"{stem}.png"

        if not pred_path.exists():
            continue

        # Load GT
        gt_bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        gt_ids = gt_color_to_ids(gt_bgr, lut)

        # Load prediction (already class-index mask)
        pred_ids = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if pred_ids.ndim == 3:
            pred_ids = cv2.cvtColor(pred_ids, cv2.COLOR_BGR2GRAY)

        if pred_ids.shape != gt_ids.shape:
            pred_ids = cv2.resize(pred_ids, (gt_ids.shape[1], gt_ids.shape[0]), interpolation=cv2.INTER_NEAREST)

        per_class, macro, micro = per_class_metrics(pred_ids, gt_ids, nc)

        row = {
            "model": model_name,
            "split": split,
            "image": stem,
            **micro,
            **macro
        }
        for c in range(nc):
            row[f"iou_c{c}"] = per_class[c]["iou"]
            row[f"f1_c{c}"] = per_class[c]["f1"]
            row[f"pr_c{c}"] = per_class[c]["precision"]
            row[f"rec_c{c}"] = per_class[c]["recall"]
        per_image_rows.append(row)

    return per_image_rows


def main():
    ap = argparse.ArgumentParser(description="Compute pixel metrics from saved YOLO predictions.")
    ap.add_argument("--models", nargs="*", default=["yolo_train", "yolo_lowres"])
    ap.add_argument("--splits", nargs="*", default=["train", "test", "lowres"])
    ap.add_argument("--out_csv", default=str(EVAL_ROOT / "pixel_metrics.csv"))
    args = ap.parse_args()

    nc = len(CLASS_RGB)
    all_rows = []

    for model_name in args.models:
        for split in args.splits:
            rows = evaluate_model_split(model_name, split, nc, CLASS_RGB)
            all_rows.extend(rows)

    if not all_rows:
        print("⚠️ No metrics computed. Check your predictions/masks folder.")
        return

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Metrics saved to {args.out_csv}")


if __name__ == "__main__":
    main()
