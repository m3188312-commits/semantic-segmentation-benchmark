# models/random_forest/train.py
# Run from repo root:
#   python -m models.random_forest.train

import os
from .dataset import load_dataset, prepare_training_data
from .model   import get_model, save_model

def main():
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
    print(f"[RF] Prepared features: X={X.shape}, y={y.shape}")

    clf = get_model()  
    print("[RF] Fitting RandomForestClassifier on train split...")
    clf.fit(X, y)

    out_path = os.path.join(scripts_dir, 'rf_model_train.pkl')
    save_model(clf, out_path)
    print(f"[RF] Saved model → {out_path}")

if __name__ == "__main__":
    main()
