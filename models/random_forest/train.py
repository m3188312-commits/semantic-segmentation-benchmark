import os
from .dataset import load_dataset, prepare_training_data
from .model   import get_model, save_model
    # clf_lo.fit(X_lo, y_lo)
    # save_model(clf_lo, os.path.join(scripts_dir, 'rf_model_lowres.pkl'))
    # print("Saved low-res model.")_dataset, prepare_training_data
from .model   import get_model, save_model

def main():
    # Project root = two levels up
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    base = os.path.join(root, 'dataset')
    scripts_dir = os.path.join(root, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    print("Scripts directory:", scripts_dir)

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

if __name__ == "__main__":
    main()
