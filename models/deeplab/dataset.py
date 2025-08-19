"""
Semantic Segmentation Dataset for DeepLab training.
"""
import torch
import os
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# --- Configuration ---
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
CLASS_RGB = {
    (155, 155, 155): 0,  # Unknown
    (226, 169, 41):   1,  # Artificial Land
    (60, 16, 152):    2,  # Woodland
    (132, 41, 246):   3,  # Arable Land
    (0, 255, 0):      4,  # Frygana
    (255, 255, 255):  5,  # Bareland
    (0, 0, 255):      6,  # Water
    (255, 255, 0):    7,  # Permanent Cultivation
}
NUM_CLASSES = len(CLASS_RGB)


def list_files(directory: str) -> List[str]:
    """
    Enumerate all image files in `directory` with supported extensions,
    returned in sorted order.
    """
    files: List[str] = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob(os.path.join(directory, f'*{ext}')))
    return sorted(files)


def rgb_to_mask(mask_img: Image.Image) -> np.ndarray:
    """
    Convert an RGB PIL image (color-coded labels) to a H×W int64 mask
    where each pixel’s value is the class index.
    """
    arr = np.array(mask_img)
    h, w, _ = arr.shape
    label = np.zeros((h, w), dtype=np.int64)
    for rgb, cls_idx in CLASS_RGB.items():
        mask = np.all(arr == rgb, axis=-1)
        label[mask] = cls_idx
    return label


def load_dataset(img_dir: str, mask_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load all images and their corresponding masks from parallel directories.

    Returns:
        imgs  – list of H×W×3 uint8 RGB arrays
        masks – list of H×W int64 class-index arrays
    """
    img_paths = list_files(img_dir)
    mask_paths = list_files(mask_dir)
    if len(img_paths) != len(mask_paths):
        raise ValueError(
            f"Image/mask count mismatch: {len(img_paths)} images vs {len(mask_paths)} masks"
        )

    imgs, masks = [], []
    for img_p, mask_p in zip(img_paths, mask_paths):
        imgs.append(np.array(Image.open(img_p).convert('RGB')))
        masks.append(rgb_to_mask(Image.open(mask_p).convert('RGB')))
    return imgs, masks


class SemanticSegmentationDataset(Dataset):
    """
    Torch Dataset for DeepLab training/inference.
    """
    def __init__(self, base_dir: str, split: str, transforms: T.Compose = None):
        img_dir = os.path.join(base_dir, split, 'image')
        mask_dir = os.path.join(base_dir, split, 'mask')

        self.img_paths = list_files(img_dir)
        self.mask_paths = list_files(mask_dir)
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(f"Image/mask count mismatch for split '{split}'")

        self.transforms = transforms or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img_tensor = self.transforms(img)

        mask = Image.open(self.mask_paths[idx]).convert('RGB')
        mask_tensor = torch.from_numpy(rgb_to_mask(mask)).long()

        return img_tensor, mask_tensor
