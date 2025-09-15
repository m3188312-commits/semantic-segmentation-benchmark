#!/usr/bin/env python3
"""
Evaluate or predict DeepLab models on the test set.
Supports full 18-run loop, single-model prediction, or single-model evaluation.
Saves:
  - evaluation_results.csv (macro summary per run)
  - evaluation_per_class.csv (per-class metrics per run)
Predictions saved under pseudo_predictions/.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.metrics import precision_recall_fscore_support

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 2

CHECKPOINT_DIR = Path("checkpoints")
TEST_IMAGES_DIR = Path("dataset/test/image")
TEST_MASKS_DIR = Path("dataset/test/mask")
DEFAULT_PRED_DIR = Path("pseudo_predictions")

CLASS_RGB = {
    (155, 155, 155): 0,  # Unknown
    (226, 169, 41): 1,   # Artificial
    (60, 16, 152): 2,    # Woodland
    (132, 41, 246): 3,   # Arable
    (0, 255, 0): 4,      # Frygana
    (255, 255, 255): 5,  # Bareland
    (0, 0, 255): 6,      # Water
    (255, 255, 0): 7,    # Permanent
}

CLASS_NAMES = [
    'Unknown', 'Artificial', 'Woodland', 'Arable',
    'Frygana', 'Bareland', 'Water', 'Permanent'
]

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("deeplab_eval")

# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------
def build_deeplab(num_classes=NUM_CLASSES):
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def rgb_to_class_id(rgb_mask: np.ndarray) -> np.ndarray:
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb_color, class_id in CLASS_RGB.items():
        matches = np.all(rgb_mask == rgb_color, axis=2)
        class_mask[matches] = class_id
    return class_mask

def class_id_to_rgb(class_mask: np.ndarray) -> np.ndarray:
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for rgb_color, class_id in CLASS_RGB.items():
        rgb_mask[class_mask == class_id] = rgb_color
    return rgb_mask

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform or T.Compose([
            T.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        mask = mask.resize(IMAGE_SIZE, resample=Image.NEAREST)

        img_tensor = self.transform(img)
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(rgb_to_class_id(mask_np).astype(np.int64))

        return img_tensor, mask_tensor, self.image_paths[idx].name

def create_test_loader():
    img_paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    mask_paths = [TEST_MASKS_DIR / f"{p.stem}.png" for p in img_paths]
    valid_pairs = [(i, m) for i, m in zip(img_paths, mask_paths) if m.exists()]

    if not valid_pairs:
        raise RuntimeError("❌ No matching test images/masks found!")

    imgs, masks = zip(*valid_pairs)
    dataset = SegmentationDataset(list(imgs), list(masks))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def predict_and_or_evaluate(model: nn.Module, test_loader: DataLoader, save_dir: Path,
                            run_id: str, do_predict=True, do_eval=True):
    model.eval()
    all_preds, all_targets = [], []
    if do_predict:
        save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, masks, names in test_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            if do_eval:
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())

            if do_predict:
                for pred_mask, fname in zip(preds.cpu().numpy(), names):
                    rgb_mask = class_id_to_rgb(pred_mask)
                    Image.fromarray(rgb_mask).save(save_dir / fname.replace(".jpg", ".png"))

    if do_eval:
        # Per-class metrics
        p, r, f1, _ = precision_recall_fscore_support(
            np.array(all_targets), np.array(all_preds),
            labels=list(range(NUM_CLASSES)),
            zero_division=0
        )

        per_class_rows = []
        for i, cname in enumerate(CLASS_NAMES):
            per_class_rows.append({
                "run_id": run_id,
                "class_id": i,
                "class_name": cname,
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i])
            })

        # Macro metrics
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            np.array(all_targets), np.array(all_preds),
            labels=list(range(NUM_CLASSES)),
            average="macro", zero_division=0
        )
        summary = {"precision": float(precision), "recall": float(recall), "f1": float(f1_macro)}

        return summary, per_class_rows
    else:
        return {}, []

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DeepLab evaluation & prediction runner")
    parser.add_argument("--predict", action="store_true", help="Only run prediction")
    parser.add_argument("--evaluate", action="store_true", help="Only run evaluation")
    parser.add_argument("--model", type=str, help="Path to a single checkpoint to evaluate/predict")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_PRED_DIR), help="Directory for predictions")
    args = parser.parse_args()

    do_predict = args.predict or (not args.evaluate)
    do_eval = args.evaluate or (not args.predict)
    output_dir = Path(args.output_dir)

    test_loader = create_test_loader()
    results_summary = []
    results_per_class = []

    if args.model:
        run_id = Path(args.model).stem
        ckpt_path = Path(args.model)
        pred_dir = output_dir / run_id

        logger.info(f"=== Running single model: {run_id} ===")
        model = build_deeplab(NUM_CLASSES).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)

        metrics, per_class = predict_and_or_evaluate(model, test_loader, pred_dir, run_id, do_predict, do_eval)
        if do_eval:
            logger.info(f"✅ {run_id}: F1={metrics['f1']:.4f}")
            results_summary.append({"run_id": run_id, **metrics})
            results_per_class.extend(per_class)

    else:
        K_values = [10, 50, 100]
        N_values = [2, 3, 4]
        variants = ["no-remove", "remove-unknown"]

        for K in K_values:
            for N in N_values:
                for variant in variants:
                    run_id = f"deeplab_K{K}_N{N}_{variant}"
                    ckpt_path = CHECKPOINT_DIR / f"{run_id}.pth"
                    pred_dir = output_dir / run_id

                    logger.info(f"=== Evaluating {run_id} ===")

                    if not ckpt_path.exists():
                        logger.error(f"❌ Missing checkpoint: {ckpt_path}")
                        continue

                    try:
                        model = build_deeplab(NUM_CLASSES).to(DEVICE)
                        state = torch.load(ckpt_path, map_location=DEVICE)
                        model.load_state_dict(state, strict=False)

                        metrics, per_class = predict_and_or_evaluate(model, test_loader, pred_dir, run_id, do_predict, do_eval)
                        if do_eval:
                            results_summary.append({"run_id": run_id, "K": K, "N": N, "variant": variant, **metrics})
                            results_per_class.extend(per_class)
                        logger.info(f"✅ {run_id} done. Predictions -> {pred_dir}")

                    except Exception as e:
                        logger.error(f"❌ {run_id} failed: {e}")
                        results_summary.append({"run_id": run_id, "error": str(e)})

    # Save results
    if do_eval and results_summary:
        df_summary = pd.DataFrame(results_summary)
        df_summary.to_csv("evaluation_results.csv", index=False)

        df_per_class = pd.DataFrame(results_per_class)
        df_per_class.to_csv("evaluation_per_class.csv", index=False)

        print("\n=== SUMMARY ===")
        print(df_summary.to_string(index=False, float_format="%.4f"))
        best = df_summary.loc[df_summary['f1'].idxmax()]
        print(f"\n🏆 Best F1: {best['f1']:.4f} ({best['run_id']})")

if __name__ == "__main__":
    main()
