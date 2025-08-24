import os
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Config ---
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')
COMMON_SIZE   = (512, 512)  # (width, height)
CLASS_RGB     = {
    (155,155,155): 0,    # Unknown/Background
    (226,169,41):  1,    # Artificial Land
    (60,16,152):   2,    # Woodland
    (132,41,246):  3,    # Arable Land
    (0,255,0):     4,    # Frygana
    (255,255,255): 5,    # Bareland
    (0,0,255):     6,    # Water
    (255,255,0):   7     # Permanent Cultivation
}
NUM_CLASSES = len(CLASS_RGB)

def list_files(directory: str) -> List[str]:
    """Return sorted list of all image files under `directory`."""
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob(os.path.join(directory, f'*{ext}')))
    return sorted(files)

def rgb_to_mask(mask: Image.Image) -> np.ndarray:
    """
    Convert a color-coded PIL mask → H×W numpy array of class-indices.
    """
    arr = np.array(mask)
    h, w = arr.shape[:2]
    mask_idx = np.zeros((h, w), dtype=np.int64)
    for rgb, cls in CLASS_RGB.items():
        mask_idx[np.all(arr == rgb, axis=-1)] = cls
    return mask_idx

class UNetSegmentationDataset(Dataset):
    """
    BaseDir/ ├─ train/ ├─ image/ ├─ mask/
              ├─ lowres/ ├─ image/ ├─ mask/
              └─ test/  ├─ image/ ├─ mask/
    """
    def __init__(
        self,
        base_dir: str,
        split: str,
        augment: bool = True
    ):
        img_dir  = os.path.join(base_dir, split, 'image')
        mask_dir = os.path.join(base_dir, split, 'mask')

        self.img_paths  = list_files(img_dir)
        self.mask_paths = list_files(mask_dir)
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(
                f"[UNetDataset] {split}: "
                f"{len(self.img_paths)} images vs {len(self.mask_paths)} masks"
            )

        if augment:
            self.transforms = A.Compose([
                A.Resize(COMMON_SIZE[1], COMMON_SIZE[0]),     # (H, W)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                        rotate=(-15, 15), p=0.5),  # Correct Affine parameters
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
            # Separate mask transforms (no normalization, no ToTensor)
            self.mask_transforms = A.Compose([
                A.Resize(COMMON_SIZE[1], COMMON_SIZE[0]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                        rotate=(-15, 15), p=0.5),
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(COMMON_SIZE[1], COMMON_SIZE[0]),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
            self.mask_transforms = A.Compose([
                A.Resize(COMMON_SIZE[1], COMMON_SIZE[0]),
            ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # load & preprocess image and mask
        img = np.array(Image.open(self.img_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('RGB'))

        # Apply transforms separately
        img_transformed = self.transforms(image=img)["image"]
        mask_transformed = self.mask_transforms(image=mask)["image"]
        
        # Convert RGB mask to class indices
        mask_idx = rgb_to_mask(Image.fromarray(mask_transformed))
        
        return img_transformed, torch.from_numpy(mask_idx).long()
