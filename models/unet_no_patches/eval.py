#!/usr/bin/env python3
"""
U-Net Evaluation Script (mirroring DeepLab’s pipeline)

Usage:
  python models/unet_no_patches/eval.py
  python models/unet_no_patches/eval.py --predict
  python models/unet_no_patches/eval.py --evaluate
  python models/unet_no_patches/eval.py --single-image <split> <image_filename>
"""
import os
import sys
import argparse
from glob import glob

import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import torch
from torchvision import transforms as T
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ─── Ensure project root on PYTHONPATH ─────────────────────────
SCRIPT_DIR   = os.path.dirname(__file__)
# Go up two levels: models/unet_no_patches -> models -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────

from models.unet_no_patches.dataset import rgb_to_mask, CLASS_RGB
from models.unet_no_patches.model import build_pretrained_unet

# Class labels (optional custom names)
CLASS_NAMES = ['Unknown','Artificial','Woodland','Arable','Frygana','Bareland','Water','Permanent']

# Checkpoint filenames per split
WEIGHTS = {
    'train':  'unet_train.pth',
    'lowres': 'unet_lowres.pth',
    'test':   'unet_train.pth'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TF = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def list_files(directory: str):
    """Collect and sort image files in a directory."""
    patterns = ['*.png', '*.jpg', '*.jpeg']
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(directory, p)))
    return sorted(files)


def predict_split(model, img_dir: str, out_dir: str, single_image: str = None):
    """Generate and save color-coded mask predictions."""
    if single_image:
        paths = [os.path.join(img_dir, single_image)]
    else:
        paths = list_files(img_dir)

    os.makedirs(out_dir, exist_ok=True)
    inv_map = {v: rgb for rgb, v in CLASS_RGB.items()}
    model.eval()
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert('RGB')
            inp = TF(img).unsqueeze(0).to(DEVICE)
            out = model(inp)
            logits = out['out'] if isinstance(out, dict) else out
            pred = logits[0].argmax(0).cpu().numpy()
            # Colorize
            h, w = pred.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_idx, rgb in inv_map.items():
                vis[pred == cls_idx] = rgb
            name = os.path.basename(p).split('.')[0] + '.png'
            Image.fromarray(vis).save(os.path.join(out_dir, name))
    print(f"[INFO] Saved {len(paths)} masks to {out_dir}")


def evaluate_split(pred_dir: str, img_dir: str, mask_dir: str, single_image: str = None):
    """Compute precision, recall, and F1 for a split or single image."""
    if single_image:
        base = os.path.splitext(single_image)[0]
        # Ground truth
        gt_files = glob(os.path.join(mask_dir, base + '.*'))
        if not gt_files:
            raise FileNotFoundError(f"No GT mask for {single_image} in {mask_dir}")
        y_true = rgb_to_mask(Image.open(gt_files[0]).convert('RGB')).flatten()
        # Prediction
        pred_path = os.path.join(pred_dir, base + '.png')
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"No pred mask for {single_image} in {pred_dir}")
        y_pred = rgb_to_mask(Image.open(pred_path).convert('RGB')).flatten()
    else:
        # All GT masks
        gt_paths   = list_files(mask_dir)
        true_masks = [rgb_to_mask(Image.open(fp).convert('RGB')).flatten() for fp in gt_paths]
        # All preds
        pred_paths = list_files(pred_dir)
        if len(pred_paths) != len(true_masks):
            raise RuntimeError(f"{len(pred_paths)} preds vs {len(true_masks)} GT")
        preds = [rgb_to_mask(Image.open(pp).convert('RGB')).flatten() for pp in pred_paths]
        y_true = np.hstack(true_masks)
        y_pred = np.hstack(preds)

    labels = list(range(len(CLASS_NAMES)))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {'p': p, 'r': r, 'f1': f1}


def generate_pdf(results: dict, out_path: str):
    """Compile metric tables into a PDF."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=letter)
    elems = []
    styles = getSampleStyleSheet()

    for split, metrics in results.items():
        elems.append(Paragraph(f"Metrics for '{split}'", styles['Heading2']))
        data = [['Class','Precision','Recall','F1']]
        for idx, name in enumerate(CLASS_NAMES):
            data.append([
                name,
                f"{metrics['p'][idx]:.3f}",
                f"{metrics['r'][idx]:.3f}",
                f"{metrics['f1'][idx]:.3f}"
            ])
        table = Table(data)
        table.setStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
            ('GRID',(0,0),(-1,-1),0.5,colors.black),
            ('ALIGN',(1,1),(-1,-1),'RIGHT')
        ])
        elems.append(table)
        elems.append(Spacer(1,12))

    doc.build(elems)
    print(f"[INFO] PDF report generated at {out_path}")


def main():
    parser = argparse.ArgumentParser(description="U-Net eval: predict + evaluate + PDF")
    parser.add_argument('--predict', action='store_true', help='only generate masks')
    parser.add_argument('--evaluate', action='store_true', help='only compute metrics')
    parser.add_argument('--single-image', nargs=2, metavar=('SPLIT','IMAGE'),
                        help='evaluate one image in dataset/<split>/image')
    args = parser.parse_args()

    do_pred = args.predict or not args.evaluate
    do_eval = args.evaluate or not args.predict

    results = {}
    for split in ['train','lowres','test']:
        print(f"\n=== [{split.upper()}] Loading {WEIGHTS[split]} ===")
        model = build_pretrained_unet(in_channels=3, num_classes=len(CLASS_RGB), device=DEVICE)
        ckpt = os.path.join(PROJECT_ROOT, 'scripts', WEIGHTS[split])
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE)

        img_dir  = os.path.join(PROJECT_ROOT, 'dataset', split, 'image')
        mask_dir = os.path.join(PROJECT_ROOT, 'dataset', split, 'mask')
        pred_dir = os.path.join(PROJECT_ROOT, 'predictions', 'unet_no_patches', split)

        if args.single_image:
            if split == args.single_image[0]:
                if do_pred:
                    predict_split(model, img_dir, pred_dir, single_image=args.single_image[1])
                if do_eval:
                    m = evaluate_split(pred_dir, img_dir, mask_dir, single_image=args.single_image[1])
                    # Print per-class metrics
                    for i, name in enumerate(CLASS_NAMES):
                        print(f"{name:<12} P={m['p'][i]:.3f} R={m['r'][i]:.3f} F1={m['f1'][i]:.3f}")
            return

        if do_pred:
            predict_split(model, img_dir, pred_dir)
        if do_eval:
            print(f"Evaluating '{split}'...")
            results[split] = evaluate_split(pred_dir, img_dir, mask_dir)

    if do_eval and not args.single_image:
        pdf_path = os.path.join(PROJECT_ROOT, 'evaluations', 'unet_metrics.pdf')
        generate_pdf(results, pdf_path)

if __name__ == '__main__':
    main()
