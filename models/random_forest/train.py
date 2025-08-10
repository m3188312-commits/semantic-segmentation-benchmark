import os
from .dataset import load_dataset, prepare_training_data
from .model   import get_model, save_model

def main():
    # Project root = two levels up
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    base = os.path.join(root, 'dataset')
    scripts_dir = os.path.join(root, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    print("Scripts directory:", scripts_dir)

    # — Train on high-res “train” split —
    train_img  = os.path.join(base, 'train',  'image')
    train_mask = os.path.join(base, 'train',  'mask')
    print(f"Loading high-res: {train_img} + {train_mask}")
    imgs_tr, masks_tr = load_dataset(train_img, train_mask)
    X_tr, y_tr = prepare_training_data(imgs_tr, masks_tr)
    clf_tr = get_model()
    print("Fitting high-res RF…")
    clf_tr.fit(X_tr, y_tr)
    save_model(clf_tr, os.path.join(scripts_dir, 'rf_model_train.pkl'))
    print("Saved high-res model.")

    # — Train on low-res “lowres” split —
    low_img  = os.path.join(base, 'lowres', 'image')
    low_mask = os.path.join(base, 'lowres', 'mask')
    print(f"Loading low-res: {low_img} + {low_mask}")
    imgs_lo, masks_lo = load_dataset(low_img, low_mask)
    X_lo, y_lo = prepare_training_data(imgs_lo, masks_lo)
    clf_lo = get_model()
    print("Fitting low-res RF…")
    clf_lo.fit(X_lo, y_lo)
    save_model(clf_lo, os.path.join(scripts_dir, 'rf_model_lowres.pkl'))
    print("Saved low-res model.")

if __name__ == "__main__":
    main()
