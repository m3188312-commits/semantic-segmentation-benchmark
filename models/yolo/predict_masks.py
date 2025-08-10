# models/yolo/predict_masks.py
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def load_classes(yaml_path: Path):
    cfg = yaml.safe_load(yaml_path.read_text())
    names = cfg["names"]
    # class_id is index in names (0..7)
    return names

def predict_split(weights: Path, split_dir: Path, out_dir: Path, conf=0.01, imgsz=640, device="0"):
    img_dir = split_dir / "image"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    # force device
    if device != "cpu" and torch.cuda.is_available():
        device = device
    else:
        device = "cpu"

    # enumerate images
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]

        # run prediction on this image only (no saving of overlays)
        results = model.predict(
            source=img, task="segment", conf=conf, imgsz=imgsz,
            save=False, verbose=False, workers=0, device=device
        )
        r = results[0]

        # initialize prediction maps
        pred_idx = np.zeros((H, W), dtype=np.uint8)     # class id per pixel
        pred_conf = np.zeros((H, W), dtype=np.float32)  # best confidence per pixel (for tie-breaking)

        if r.masks is not None and len(r.masks.data) > 0:
            masks = r.masks.data.cpu().numpy()          # (N, h, w) at model scale
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
            confs = r.boxes.conf.cpu().numpy().astype(np.float32)

            # resize each mask to original image size and write by best-conf policy
            for k in range(masks.shape[0]):
                m = (masks[k] > 0.5).astype(np.uint8)
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                cls_id = int(cls_ids[k])
                conf_k = float(confs[k])

                # update pixels where this instance is present and confidence is higher
                update = (m == 1) & (conf_k > pred_conf)
                pred_idx[update] = cls_id
                pred_conf[update] = conf_k

        # save single-channel class-index mask (uint8)
        out_path = out_dir / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), pred_idx)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    ap.add_argument("--split", required=True, choices=["train","test","lowres"])
    ap.add_argument("--out", required=True, help="Output folder for predicted masks")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    split_dir = Path("dataset") / args.split
    out_dir = Path(args.out)

    predict_split(Path(args.weights), split_dir, out_dir, conf=args.conf, imgsz=args.imgsz, device=args.device)
    print(f"✅ Saved predicted masks to {out_dir}")

if __name__ == "__main__":
    main()
