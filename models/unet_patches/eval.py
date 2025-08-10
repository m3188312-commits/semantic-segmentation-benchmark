#!/usr/bin/env python3
# Evaluate patch‑based U‑Net on test set: stitch patches, overlay, compute metrics

import os, sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
from torch.utils.data import DataLoader

# Auto‑inject project root
ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, ROOT)

from models.unet_patches.dataset import PatchesSegmentationDataset, COMMON_SIZE, rgb_to_class
from models.unet_patches.model import build_unet_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate patch‑based U‑Net: stitch outputs, save overlays, compute metrics"
    )
    parser.add_argument("--data-dir",    type=str, required=True,
                        help="Root folder with 'image/' and 'mask/' subdirs for test images")
    parser.add_argument("--ckpt-path",   type=str, required=True,
                        help="Path to trained U‑Net checkpoint (.pth)")
    parser.add_argument("--output-dir",  type=str, required=True,
                        help="Where to save reconstructed overlays and metrics")
    parser.add_argument("--batch-size",  type=int, default=16,
                        help="Batch size for patch inference")
    parser.add_argument("--patch-size",  type=int, default=128,
                        help="Patch size used during training")
    parser.add_argument("--stride",      type=int, default=128,
                        help="Stride between patches used during training")
    parser.add_argument("--selected-classes", type=int, nargs='+', default=[1,2,3],
                        help="Classes to overlay and evaluate")
    return parser.parse_args()


def apply_color_map_on_image(image_np, pred_mask, selected_classes):
    color_map = {1:[226,169,41],2:[60,16,152],3:[132,41,246]}
    overlay = image_np.copy()
    mask_draw = np.isin(pred_mask, selected_classes)
    for cls in selected_classes:
        overlay[np.logical_and(pred_mask==cls, mask_draw)] = color_map[cls]
    return overlay


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    test_img_dir  = os.path.join(args.data_dir, 'image')
    test_mask_dir = os.path.join(args.data_dir, 'mask')
    out_overlays  = Path(args.output_dir)/'overlays'
    out_overlays.mkdir(parents=True, exist_ok=True)

    # Dataset & Loader
    ds = PatchesSegmentationDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        patch_size=args.patch_size,
        stride=args.stride
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = build_unet_model(in_channels=3, num_classes=4, device=device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Collect all patch predictions and truths
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()  # [B, ph, pw]
            all_preds.append(preds)
            all_trues.append(masks.numpy())            # [B, ph, pw]
    preds_patches = np.concatenate(all_preds, axis=0)
    true_patches = np.concatenate(all_trues, axis=0)

    # Stitch back per image
    n_images = len(ds.images)
    img_W, img_H = COMMON_SIZE
    ps = args.patch_size; st = args.stride
    patches_per_row = (img_W - ps) // st + 1
    patches_per_col = (img_H - ps) // st + 1
    patches_per_image = patches_per_row * patches_per_col

    # compute metrics lists
    y_true_all, y_pred_all = [], []

    for img_idx in range(n_images):
        # initialize full masks
        full_pred = np.zeros((img_H, img_W), dtype=np.int64)
        full_true = np.zeros((img_H, img_W), dtype=np.int64)
        # get image-level original and mask
        orig = Image.open(os.path.join(test_img_dir, ds.images[img_idx])).convert('RGB').resize(COMMON_SIZE)
        orig_np = np.array(orig)
        # mask true from dataset? reload and rgb_to_class
        true_mask = Image.open(os.path.join(test_mask_dir, ds.masks[img_idx])).convert('RGB').resize(COMMON_SIZE)
        full_true = rgb_to_class(true_mask)

        # slice patches
        start = img_idx * patches_per_image
        end   = start + patches_per_image
        img_patches = preds_patches[start:end]
        # place each
        pi = 0
        for row in range(patches_per_col):
            for col in range(patches_per_row):
                top = row * st; left = col * st
                full_pred[top:top+ps, left:left+ps] = img_patches[pi]
                pi += 1
        # overlay
        overlay = apply_color_map_on_image(orig_np, full_pred, args.selected_classes)
        Image.fromarray(overlay).save(out_overlays / f'overlay_{img_idx:04d}.png')
        # accumulate metrics, ignore background
        mask = full_true.flatten() != 0
        y_true_all.extend(full_true.flatten()[mask])
        y_pred_all.extend(full_pred.flatten()[mask])

    # metrics
    cls = args.selected_classes
    f1 = f1_score(y_true_all, y_pred_all, average=None, labels=cls)
    rec = recall_score(y_true_all, y_pred_all, average=None, labels=cls)
    prec = precision_score(y_true_all, y_pred_all, average=None, labels=cls)

    # save summary
    summary = Path(args.output_dir)/'metrics_summary.csv'
    with open(summary,'w') as f:
        f.write('class,f1,recall,precision\n')
        for c,a,b,d in zip(cls,f1,rec,prec):
            f.write(f'{c},{a:.4f},{b:.4f},{d:.4f}\n')

    print(f"✅ Overlays in: {out_overlays}")
    print(f"✅ Metrics saved: {summary}")

if __name__=='__main__':
    main()
