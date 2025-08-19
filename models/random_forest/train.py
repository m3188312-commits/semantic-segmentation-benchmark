# models/random_forest/train.py
# Run from repo root:
#   python -m models.random_forest.train

import os
from .dataset import load_dataset, prepare_training_data
from .model   import get_model, save_model
    # clf_lo.fit(X_lo, y_lo)
    # save_model(clf_lo, os.path.join(scripts_dir, 'rf_model_lowres.pkl'))
    # print("Saved low-res model.")_dataset, prepare_training_data
from .model   import get_model, save_model


def main():
    # Repo root = two levels up from this file
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    base = os.path.join(root, 'dataset')
    scripts_dir = os.path.join(root, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)

<<<<<<< HEAD
    # --- Train split only ---
    train_img_dir  = os.path.join(base, 'train', 'image')
    train_mask_dir = os.path.join(base, 'train', 'mask')
    print(f"[RF] Loading train split:\n      images: {train_img_dir}\n      masks : {train_mask_dir}")

    images, masks = load_dataset(train_img_dir, train_mask_dir)
    X, y = prepare_training_data(images, masks)
    print(f"[RF] Prepared features: X={X.shape}, y={y.shape}")

    clf = get_model()  # defaults defined in models/random_forest/model.py
    print("[RF] Fitting RandomForestClassifier on train split...")
    clf.fit(X, y)

    out_path = os.path.join(scripts_dir, 'rf_model_train.pkl')
    save_model(clf, out_path)
    print(f"[RF] Saved model → {out_path}")

=======
        # — Train only on "train" split —
    train_img  = os.path.join(base, 'train',  'image')
    train_mask = os.path.join(base, 'train',  'mask')
    print(f"Loading train set: {train_img} + {train_mask}")
    imgs_tr, masks_tr = load_dataset(train_img, train_mask)
    X_tr, y_tr = prepare_training_data(imgs_tr, masks_tr)
    clf_tr = get_model()
    print("Fitting random forest model...")
    clf_tr.fit(X_tr, y_tr)
    save_model(clf_tr, os.path.join(scripts_dir, 'rf_model_train.pkl'))
    print("Saved train model.")
>>>>>>> 7049c64f4f20a1277151ee25b10ac3948734a6ad

if __name__ == "__main__":
    main()
