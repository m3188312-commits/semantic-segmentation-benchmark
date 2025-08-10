# dataset.py
import os
import numpy as np
from skimage import io, color, filters, feature

# --- Class mapping ---
COLOR2CLASS = {
    (155, 155, 155): 0,  # Unknown
    (226, 169, 41):   1,  # Artificial Land
    (60, 16, 152):    2,  # Woodland
    (132, 41, 246):   3,  # Arable Land
    (0, 255, 0):      4,  # Frygana
    (255, 255, 255):  5,  # Bareland
    (0, 0, 255):      6,  # Water
    (255, 255, 0):    7,  # Permanent Cultivation
}

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def list_image_files(directory: str):
    """
    List and sort all image files in a directory by common extensions.
    """
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(IMG_EXTENSIONS):
            files.append(os.path.join(directory, f))
    return sorted(files)


def rgb_to_mask(label_img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB mask image to a 2D array of class indices.
    """
    h, w, _ = label_img.shape
    mask = np.zeros((h, w), dtype=np.int64)
    for col, cls in COLOR2CLASS.items():
        mask[np.all(label_img == col, axis=-1)] = cls
    return mask


def load_dataset(img_dir: str, mask_dir: str):
    """
    Read images and corresponding RGB masks; return lists of numpy arrays.
    Supports multiple image formats.
    """
    img_paths = list_image_files(img_dir)
    mask_paths = list_image_files(mask_dir)

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    images, masks = [], []
    for ip, mp in zip(img_paths, mask_paths):
        img = io.imread(ip)
        if img.ndim == 2:
            img = color.gray2rgb(img)
        mask_rgb = io.imread(mp)
        mask = rgb_to_mask(mask_rgb)
        images.append(img)
        masks.append(mask)
    return images, masks


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel features: intensity, edges, LBP, and RGB channels.
    Returns array of shape (H, W, F).
    """
    gray = color.rgb2gray(img)
    feats = [
        gray,
        filters.sobel(gray),
        feature.local_binary_pattern(gray, P=8, R=1),
        img[:, :, 0], img[:, :, 1], img[:, :, 2]
    ]
    return np.stack(feats, axis=-1)


def prepare_training_data(images, masks):
    """
    Flatten per-pixel features and labels across all images.
    Returns X (N x F) and y (N, ) including Unknown as class 0.
    """
    X_list, y_list = [], []
    for img, mask in zip(images, masks):
        feats = extract_features(img)
        h, w, f = feats.shape
        X_list.append(feats.reshape(-1, f))
        y_list.append(mask.flatten())
    if not X_list:
        raise ValueError("No data found for training. Check your image/mask directories.")
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y
