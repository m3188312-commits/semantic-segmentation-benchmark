"""
YOLO Evaluation Script

Usage:
  # Full pipeline: predict + evaluate
  python models/yolo/eval.py

  # Predict-only (generate segmentations)
  python models/yolo/eval.py --predict

  # Evaluate-only (use existing segmentations)
  python models/yolo/eval.py --evaluate

  # Single-image evaluation with new prediction
  python models/yolo/eval.py --single-image <split> <image_filename>

  # Single-image metrics-only
  python models/yolo/eval.py --evaluate --single-image <split> <image_filename>

Options:
  <split>           one of: train, lowres, test
  <image_filename>  exact filename (e.g. 0000.jpg) in dataset/<split>/image/
"""
import os
import sys
import argparse
import numpy as np
from glob import glob
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import torch
from pathlib import Path
import cv2

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

# Class labels
CLASS_NAMES = [
    'Unknown', 'Artificial', 'Woodland', 'Arable',
    'Frygana', 'Bareland', 'Water', 'Permanent'
]

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

WEIGHTS = {
    'train':  'yolo_train.pt',
    'lowres': 'yolo_train.pt',
    'test':   'yolo_train.pt'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rgb_to_mask(rgb_image):
    """Convert RGB image to class indices using CLASS_RGB mapping."""
    if isinstance(rgb_image, Image.Image):
        rgb_array = np.array(rgb_image)
    else:
        rgb_array = rgb_image
    
    h, w = rgb_array.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for rgb, class_id in CLASS_RGB.items():
        mask[np.all(rgb_array == rgb, axis=2)] = class_id
    
    return mask


def predict_split(model, img_dir, out_dir, single_image=None, conf=0.01, imgsz=640):
    """Run YOLO on images and save RGB masks."""
    if single_image:
        paths = [os.path.join(img_dir, single_image)]
    else:
        paths = sorted(glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg')))
    
    os.makedirs(out_dir, exist_ok=True)
    inv_map = {v: rgb for rgb, v in CLASS_RGB.items()}
    
    model.eval()
    
    for p in paths:
        
        img = np.array(Image.open(p).convert('RGB'))
        H, W = img.shape[:2]
        
        
        results = model.predict(
            source=img, 
            task="segment", 
            conf=conf, 
            imgsz=imgsz,
            save=False, 
            verbose=False, 
            workers=0, 
            device=DEVICE
        )
        r = results[0]
        
        # Initialize prediction maps
        pred_idx = np.zeros((H, W), dtype=np.uint8)
        pred_conf = np.zeros((H, W), dtype=np.float32)
        
        if r.masks is not None and len(r.masks.data) > 0:
            masks = r.masks.data.cpu().numpy()          # (N, h, w) at model scale
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
            confs = r.boxes.conf.cpu().numpy().astype(np.float32)
            
            # Resize each mask to original image size and write by best-conf policy
            for k in range(masks.shape[0]):
                m = (masks[k] > 0.5).astype(np.uint8)
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                cls_id = int(cls_ids[k])
                conf_k = float(confs[k])
                
                # Update pixels where this instance is present and confidence is higher
                update = (m == 1) & (conf_k > pred_conf)
                pred_idx[update] = cls_id
                pred_conf[update] = conf_k
        
        # Colorize prediction
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        for cls_idx, rgb in inv_map.items():
            vis[pred_idx == cls_idx] = rgb
        
        name = os.path.basename(p).rsplit('.', 1)[0] + '.png'
        Image.fromarray(vis).save(os.path.join(out_dir, name))
    
    print(f"[INFO] Saved {len(paths)} masks to {out_dir}")


def evaluate_split(pred_dir, img_dir, mask_dir, single_image=None):
    """Compute and return P/R/F1 metrics for one split."""
    if single_image:
        base = os.path.splitext(single_image)[0]
        # GT
        gt_pattern = os.path.join(mask_dir, base + '.*')
        gt_files = glob(gt_pattern)
        if not gt_files:
            raise FileNotFoundError(f"No GT mask for {single_image} in {mask_dir}")
        gt_mask = rgb_to_mask(Image.open(gt_files[0]).convert('RGB'))
        y_true = gt_mask.flatten()
        # Pred
        pred_path = os.path.join(pred_dir, base + '.png')
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"No prediction for {single_image} in {pred_dir}")
        pred_mask = rgb_to_mask(Image.open(pred_path).convert('RGB'))
        y_pred = pred_mask.flatten()
    else:
        # Load all GT masks
        gt_paths = sorted(glob(os.path.join(mask_dir, '*.png')))
        y_true = []
        for gt_path in gt_paths:
            gt_mask = rgb_to_mask(Image.open(gt_path).convert('RGB'))
            y_true.append(gt_mask.flatten())
        
        # Load all predictions
        pred_paths = sorted(glob(os.path.join(pred_dir, '*.png')))
        if len(pred_paths) != len(gt_paths):
            raise RuntimeError(f"{len(pred_paths)} preds vs {len(gt_paths)} GT")
        
        y_pred = []
        for pred_path in pred_paths:
            pred_mask = rgb_to_mask(Image.open(pred_path).convert('RGB'))
            y_pred.append(pred_mask.flatten())
        
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
    
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    return {'p': p, 'r': r, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description="YOLO pipeline: predict/evaluate with terminal output.")
    parser.add_argument('--predict', action='store_true', help='only generate segmentations')
    parser.add_argument('--evaluate', action='store_true', help='only compute metrics')
    parser.add_argument('--single-image', nargs=2, metavar=('SPLIT','IMAGE'), help='evaluate one image')
    parser.add_argument('--model_path', type=str, help='path to model weights file')
    parser.add_argument('--conf', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='inference image size')
    args = parser.parse_args()
    
    do_pred = args.predict or not args.evaluate
    do_eval = args.evaluate or not args.predict

    # Determine which splits to process
    if args.single_image:
        splits = [args.single_image[0]]
    else:
        # Run on train, test and lowres sets by default
        splits = ['train', 'test', 'lowres']

    results = {}
    for split in splits:
        # Determine model weights path
        if args.model_path:
            weights_path = args.model_path
        else:
            # Look for weights in scripts folder
            weights_path = os.path.join(PROJECT_ROOT, 'scripts', WEIGHTS.get(split, 'yolo_train.pt'))
        
        if not os.path.exists(weights_path):
            print(f"❌ Model weights not found at: {weights_path}")
            print("Available files in scripts folder:")
            scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
            if os.path.exists(scripts_dir):
                for f in os.listdir(scripts_dir):
                    if f.endswith('.pt'):
                        print(f"  - {f}")
            continue
            
        # Load model
        print(f"\n=== [{split.upper()}] Loading {os.path.basename(weights_path)} ===")
        model = YOLO(weights_path)
        
        img_dir = os.path.join(PROJECT_ROOT, 'dataset', split, 'image')
        mask_dir = os.path.join(PROJECT_ROOT, 'dataset', split, 'mask')
        if not os.path.isdir(mask_dir): 
            mask_dir = os.path.join(PROJECT_ROOT, 'dataset', split, 'masks')
        pred_dir = os.path.join(PROJECT_ROOT, 'predictions', 'yolo', split)
        
        if args.single_image:
            # Only run for that split
            if split == args.single_image[0]:
                if do_pred:
                    predict_split(model, img_dir, pred_dir, single_image=args.single_image[1], 
                                conf=args.conf, imgsz=args.imgsz)
                if do_eval:
                    m = evaluate_split(pred_dir, img_dir, mask_dir, single_image=args.single_image[1])
                    print(f"\n📊 Metrics for '{args.single_image[1]}' in {split}:")
                    print(f"{'Class':<12}{'P':>6}{'R':>6}{'F1':>6}")
                    print('-'*30)
                    for i, n in enumerate(CLASS_NAMES): 
                        print(f"{n:<12}{m['p'][i]:6.3f}{m['r'][i]:6.3f}{m['f1'][i]:6.3f}")
                return
        else:
            if do_pred:
                predict_split(model, img_dir, pred_dir, conf=args.conf, imgsz=args.imgsz)
            if do_eval:
                print(f"Evaluating '{split}'")
                res = evaluate_split(pred_dir, img_dir, mask_dir)
                results[split] = res
                
                
                print(f"\n📊 Metrics for '{split}':")
                print(f"{'Class':<12}{'P':>6}{'R':>6}{'F1':>6}")
                print('-'*30)
                for i, n in enumerate(CLASS_NAMES): 
                    print(f"{n:<12}{res['p'][i]:6.3f}{res['r'][i]:6.3f}{res['f1'][i]:6.3f}")
    
    
    if len(results) > 1:
        print(f"\n📋 SUMMARY:")
        for split, res in results.items():
            print(f"{split}: P={res['p'].mean():.3f}, R={res['r'].mean():.3f}, F1={res['f1'].mean():.3f}")


if __name__ == '__main__':
    main()
