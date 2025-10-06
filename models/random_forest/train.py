# models/random_forest/train.py
# Run from repo root:
#   python -m models.random_forest.train
#   python -m models.random_forest.train --sample_fraction 0.05
#   python -m models.random_forest.eval --model scripts/rf_model_train_5pct.pkl 
#   python -m models.random_forest.train --gridsearch

import os
import argparse
from .dataset import load_dataset, prepare_training_data
from .model import get_model, save_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of pixels to use for training"
    )
    parser.add_argument(
        "--gridsearch",
        action="store_true",
        help="Use GridSearchCV instead of baseline RandomForest"
    )
    args = parser.parse_args()  

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    base = os.path.join(root, 'dataset')
    scripts_dir = os.path.join(root, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)

    train_img_dir  = os.path.join(base, 'train', 'image')
    train_mask_dir = os.path.join(base, 'train', 'mask')
    print(f"[RF] Loading train split:")
    print(f"      images: {train_img_dir}")
    print(f"      masks : {train_mask_dir}")

    images, masks = load_dataset(train_img_dir, train_mask_dir)
    X, y = prepare_training_data(images, masks)

    #apply sample fraction
    if args.sample_fraction < 1.0:
        n = int(len(y) * args.sample_fraction)
        X, y = X[:n], y[:n]
        print(f"[INFO] Using {args.sample_fraction*100:.1f}% of pixels "
              f"({n:,} / {len(y):,}) for training.")
        
    print(f"[RF] Prepared features: X={X.shape}, y={y.shape}")

    if args.gridsearch:
        clf = get_model(X, y, use_gridsearch=True)
        print("[RF] Training with Grid SearchCV")
    else:
        clf = get_model()
        print("[RF] Training with baseline Random Forest")
        clf.fit(X, y)

    print("[RF] Final model parameter: ")
    print(clf.get_params())

    #file name encodes fraction
    suffix = f"{int(args.sample_fraction*100)}pct" if args.sample_fraction < 1.0 else "full"
    out_path = os.path.join(scripts_dir, f"rf_model_train_{suffix}.pkl")
    save_model(clf, out_path)
    print(f"[RF] Saved model → {out_path}")

if __name__ == "__main__":
    main()
