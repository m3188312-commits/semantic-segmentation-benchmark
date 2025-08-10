import os
import sys
import argparse
import numpy as np
from glob import glob
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Ensure project root is on PYTHONPATH so imports resolve
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.random_forest.dataset import (
    load_dataset,
    rgb_to_mask,
    COLOR2CLASS,
    extract_features
)
from models.random_forest.model import load_model

CLASS_NAMES = [
    'Unknown', 'Artificial', 'Woodland', 'Arable',
    'Frygana', 'Bareland', 'Water', 'Permanent'
]


def predict_and_save(clf, img_dir: str, out_dir: str, single_image: str = None):
    """Run inference on images and save RGB masks."""
    os.makedirs(out_dir, exist_ok=True)
    inv_map = {v: k for k, v in COLOR2CLASS.items()}

    if single_image:
        img_paths = [os.path.join(img_dir, single_image)]
    else:
        img_paths = sorted(
            glob(os.path.join(img_dir, '*.png')) +
            glob(os.path.join(img_dir, '*.jpg')) +
            glob(os.path.join(img_dir, '*.jpeg'))
        )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    for img_path in img_paths:
        img = np.array(Image.open(img_path))
        feats = extract_features(img)
        h, w, f = feats.shape
        preds = clf.predict(feats.reshape(-1, f)).reshape(h, w)

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx, color in inv_map.items():
            rgb[preds == cls_idx] = color

        base = os.path.basename(img_path).rsplit('.', 1)[0] + '.png'
        Image.fromarray(rgb).save(os.path.join(out_dir, base))


def evaluate_split(pred_dir: str, gt_img_dir: str, gt_mask_dir: str, single_image: str = None):
    """Compute precision, recall, and F1 for predictions vs. ground truth."""
    # Batch mode: load all GT masks
    if single_image is None:
        _, true_masks = load_dataset(gt_img_dir, gt_mask_dir)
        pred_paths = sorted(glob(os.path.join(pred_dir, '*.png')))
        if len(pred_paths) != len(true_masks):
            raise RuntimeError(
                f"Count mismatch: {len(pred_paths)} preds vs {len(true_masks)} GT masks"
            )
        y_true = []
        y_pred = []
        for gt, pf in zip(true_masks, pred_paths):
            y_true.append(gt.flatten())
            pm = rgb_to_mask(np.array(Image.open(pf).convert('RGB')))
            y_pred.append(pm.flatten())
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
    else:
        # Single-image mode: load GT mask and prediction directly
        base = os.path.splitext(single_image)[0]
        # Find GT mask file
        gt_pattern = os.path.join(gt_mask_dir, base + '.*')
        gt_files = glob(gt_pattern)
        if not gt_files:
            raise FileNotFoundError(f"No GT mask found for {single_image} in {gt_mask_dir}")
        gt_path = gt_files[0]
        gt_mask = rgb_to_mask(np.array(Image.open(gt_path).convert('RGB')))
        y_true = gt_mask.flatten()
        # Prediction mask is always .png
        pred_path = os.path.join(pred_dir, base + '.png')
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"No prediction found for {single_image} in {pred_dir}")
        pred_mask = rgb_to_mask(np.array(Image.open(pred_path).convert('RGB')))
        y_pred = pred_mask.flatten()
    # Compute metrics
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0
    )
    return {'p': p, 'r': r, 'f1': f1}


def generate_pdf(results: dict, out_path: str):
    """Generate a PDF report with metric tables for each split."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    for split, metrics in results.items():
        elements.append(Paragraph(f"Metrics for '{split}'", styles['Heading2']))
        data = [['Class', 'Precision', 'Recall', 'F1']]
        for idx, name in enumerate(CLASS_NAMES):
            data.append([
                name,
                f"{metrics['p'][idx]:.3f}",
                f"{metrics['r'][idx]:.3f}",
                f"{metrics['f1'][idx]:.3f}"
            ])
        table = Table(data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT')
        ])
        elements.append(table)
        elements.append(Spacer(1, 12))

    doc.build(elements)


def main():
    parser = argparse.ArgumentParser(
        description="RF evaluation: predict+evaluate on train, lowres, test."
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip prediction; only compute metrics."
    )
    parser.add_argument(
        "--single-image",
        nargs=2,
        metavar=('SPLIT', 'IMAGE'),
        help="Evaluate a single IMAGE from SPLIT (train, lowres, test)."
    )
    args = parser.parse_args()

    data_root = os.path.join(PROJECT_ROOT, 'dataset')
    scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
    preds_root = os.path.join(PROJECT_ROOT, 'predictions', 'random_forest')

    weight_map = {
        'train': 'rf_model_train.pkl',
        'lowres': 'rf_model_lowres.pkl',
        'test': 'rf_model_train.pkl'
    }

    # Single-image mode
    if args.single_image:
        split, img_name = args.single_image
        if split not in weight_map:
            print(f"Invalid split '{split}'. Choose from train, lowres, test.")
            sys.exit(1)
        model_path = os.path.join(scripts_dir, weight_map[split])
        clf = load_model(model_path)
        img_dir = os.path.join(data_root, split, 'image')
        mask_dir = os.path.join(data_root, split, 'mask')
        if not os.path.isdir(mask_dir):
            mask_dir = os.path.join(data_root, split, 'masks')
        pred_dir = os.path.join(preds_root, split)
        if not args.evaluate_only:
            predict_and_save(clf, img_dir, pred_dir, single_image=img_name)
        metrics = evaluate_split(pred_dir, img_dir, mask_dir, single_image=img_name)
        print(f"Metrics for '{img_name}' in split '{split}':")
        print(f"{'Class':<12}{'P':>8}{'R':>8}{'F1':>8}")
        print('-'*36)
        for idx, name in enumerate(CLASS_NAMES):
            print(f"{name:<12}{metrics['p'][idx]:8.3f}{metrics['r'][idx]:8.3f}{metrics['f1'][idx]:8.3f}")
        return

    # Full evaluation mode
    results = {}
    for split in ['train', 'lowres', 'test']:
        model_path = os.path.join(scripts_dir, weight_map[split])
        clf = load_model(model_path)
        img_dir = os.path.join(data_root, split, 'image')
        mask_dir = os.path.join(data_root, split, 'mask')
        if not os.path.isdir(mask_dir):
            mask_dir = os.path.join(data_root, split, 'masks')
        pred_dir = os.path.join(preds_root, split)

        if not args.evaluate_only:
            print(f"Generating predictions for '{split}' → {pred_dir}")
            predict_and_save(clf, img_dir, pred_dir)
        print(f"Computing metrics for '{split}'")
        results[split] = evaluate_split(pred_dir, img_dir, mask_dir)

    pdf_path = os.path.join(PROJECT_ROOT, 'evaluations', 'rf_metrics.pdf')
    generate_pdf(results, pdf_path)
    print(f"PDF report written to {pdf_path}")

if __name__ == '__main__':
    main()
