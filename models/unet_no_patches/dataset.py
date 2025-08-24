import os
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

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
        transforms: T.Compose = None
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

        # use default normalization if none provided
        self.transforms = transforms or T.Compose([
            T.Resize(COMMON_SIZE, interpolation=Image.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),  # Data augmentation
            T.RandomRotation(degrees=10),   # Small rotations
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Color variation
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # load & preprocess image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transforms(img)

        # load & preprocess mask
        mask = Image.open(self.mask_paths[idx]).convert('RGB')
        mask = mask.resize(COMMON_SIZE, resample=Image.NEAREST)
        mask_idx = rgb_to_mask(mask)
        mask_tensor = torch.from_numpy(mask_idx).long()

        return img, mask_tensor
