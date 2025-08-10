import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Color-to-class mapping for RGB masks
CLASS_RGB = {
    (155, 155, 155): 0,  # background
    (226, 169, 41):  1,  # houses
    (60, 16, 152):   2,  # trees
    (132, 41, 246):  3,  # agriculture
}

# Common spatial size constants
COMMON_SIZE = (512, 512)  # full image size (W, H)


def rgb_to_class(mask: Image.Image) -> np.ndarray:
    """
    Convert a PIL RGB mask into a 2D array of class indices according
    to the predefined CLASS_RGB mapping.
    """
    arr = np.array(mask)
    h, w = arr.shape[:2]
    cmap = np.zeros((h, w), dtype=np.int64)
    for rgb, cls in CLASS_RGB.items():
        match = np.all(arr == rgb, axis=-1)
        cmap[match] = cls
    return cmap


class PatchesSegmentationDataset(Dataset):
    """
    PyTorch Dataset that splits full-size images and RGB masks into patches.

    Each mask is mapped via CLASS_RGB to class indices.

    Parameters:
      image_dir (str): path to folder containing input images
      mask_dir  (str): path to folder containing corresponding RGB masks
      patch_size (int): size of the square patch (e.g. 128)
      stride (int): stride between patch top-left corners (e.g. 128)
      transforms (callable): image transforms applied to each patch
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patch_size: int = 128,
        stride: int = 128,
        transforms: T.Compose = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transforms = transforms or T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # list image and mask filenames
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ])
        self.masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ])
        assert len(self.images) == len(self.masks), \
            f"Image/mask count mismatch {len(self.images)} vs {len(self.masks)}"

        # precompute patch coordinates for all images
        self.index_map = []  # list of tuples (img_idx, left, top)
        W, H = COMMON_SIZE
        pw = self.patch_size
        for img_idx in range(len(self.images)):
            for top in range(0, H - pw + 1, self.stride):
                for left in range(0, W - pw + 1, self.stride):
                    self.index_map.append((img_idx, left, top))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        img_idx, left, top = self.index_map[idx]
        img_path = os.path.join(self.image_dir, self.images[img_idx])
        mask_path = os.path.join(self.mask_dir, self.masks[img_idx])

        # Load full image and mask, resize to COMMON_SIZE
        img = Image.open(img_path).convert('RGB').resize(
            COMMON_SIZE, resample=Image.BILINEAR
        )
        mask = Image.open(mask_path).convert('RGB').resize(
            COMMON_SIZE, resample=Image.NEAREST
        )

        # Crop patches
        patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
        mask_patch = mask.crop((left, top, left + self.patch_size, top + self.patch_size))

        # Convert mask RGB to class indices
        mask_idx = rgb_to_class(mask_patch)  # shape: (ph, pw)

        # Apply transforms to patch
        patch_tensor = self.transforms(patch)  # shape: (3, ph, pw)
        mask_tensor = torch.from_numpy(mask_idx).long()  # (ph, pw)

        return patch_tensor, mask_tensor
