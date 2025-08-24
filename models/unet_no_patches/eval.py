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
    'lowres': 'unet_train.pth',
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
            # UNet returns tensor directly, not dict
            pred = out[0].argmax(0).cpu().numpy()
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





def main():
    parser = argparse.ArgumentParser(description="U-Net eval: predict + evaluate with terminal output")
    parser.add_argument('--predict', action='store_true', help='only generate masks')
    parser.add_argument('--evaluate', action='store_true', help='only compute metrics')
    parser.add_argument('--single-image', nargs=2, metavar=('SPLIT','IMAGE'),
                        help='evaluate one image in dataset/<split>/image')
    parser.add_argument('--model_path', type=str, help='path to model weights file')
    args = parser.parse_args()

    do_pred = args.predict or not args.evaluate
    do_eval = args.evaluate or not args.predict

    # Determine which splits to process
    if args.single_image:
        splits = [args.single_image[0]]
    else:
        # Run on both test and lowres sets by default
        splits = ['test', 'lowres']

    results = {}
    for split in splits:
        # Determine model weights path
        if args.model_path:
            weights_path = args.model_path
        else:
            # Look for weights in scripts folder
            weights_path = os.path.join(PROJECT_ROOT, 'scripts', WEIGHTS.get(split, 'unet_train.pth'))
        
        if not os.path.exists(weights_path):
            print(f"❌ Model weights not found at: {weights_path}")
            print("Available files in scripts folder:")
            scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
            if os.path.exists(scripts_dir):
                for f in os.listdir(scripts_dir):
                    if f.endswith('.pth'):
                        print(f"  - {f}")
            continue
            
        print(f"\n=== [{split.upper()}] Loading {os.path.basename(weights_path)} ===")
        model = build_pretrained_unet(in_channels=3, num_classes=len(CLASS_RGB), device=DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
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
                    print(f"\n📊 Metrics for '{args.single_image[1]}' in split '{split}':")
                    print(f"{'Class':<12}{'P':>6}{'R':>6}{'F1':>6}")
                    print('-'*30)
                    for i, name in enumerate(CLASS_NAMES):
                        print(f"{name:<12}{m['p'][i]:6.3f}{m['r'][i]:6.3f}{m['f1'][i]:6.3f}")
                return

        if do_pred:
            predict_split(model, img_dir, pred_dir)
        if do_eval:
            print(f"Evaluating '{split}'...")
            res = evaluate_split(pred_dir, img_dir, mask_dir)
            results[split] = res
            
            # Print metrics to terminal
            print(f"\n📊 Metrics for '{split}':")
            print(f"{'Class':<12}{'P':>6}{'R':>6}{'F1':>6}")
            print('-'*30)
            for i, name in enumerate(CLASS_NAMES):
                print(f"{name:<12}{res['p'][i]:6.3f}{res['r'][i]:6.3f}{res['f1'][i]:6.3f}")
    
    # Print summary if multiple splits
    if len(results) > 1:
        print(f"\n📋 SUMMARY:")
        for split, res in results.items():
            print(f"{split}: P={res['p'].mean():.3f}, R={res['r'].mean():.3f}, F1={res['f1'].mean():.3f}")

if __name__ == '__main__':
    main()
