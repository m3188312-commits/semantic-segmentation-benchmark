#!/usr/bin/env python3
"""
Pseudo-Labeling Experiments for Semantic Segmentation
Runs 18 training/evaluation cases: 3 K values × 3 N values × 2 variants
"""

import argparse
import copy
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import precision_recall_fscore_support
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths (adjust as needed)
LABELED_IMAGES_DIR = Path("dataset/train/image")
LABELED_MASKS_DIR = Path("dataset/train/mask")
TEST_IMAGES_DIR = Path("dataset/test/image")
TEST_MASKS_DIR = Path("dataset/test/mask")

UNLABELED_IMAGES_DIR = Path("data/unlabeled")
PSEUDO_MASKS_DIRS = {
    2: Path("masks_agree2"),
    3: Path("masks_agree3"),
    4: Path("masks_agree4")
}
PSEUDO_IMAGES_DIRS = {
    2: Path("data/pseudo_images/agreement_2"),
    3: Path("data/pseudo_images/agreement_3"),
    4: Path("data/pseudo_images/agreement_4")
}

# Training parameters (match deeplab/train.py)
DEVICE = torch.device('cuda', index=0)  # enforce GPU
NUM_CLASSES = 8
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 80
EARLY_STOP_PATIENCE = 10

# Class mapping
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

# ============================================================================
# DATA UTILITIES
# ============================================================================

def rgb_to_class_id(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert RGB mask to class ID mask."""
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)

    for rgb_color, class_id in CLASS_RGB.items():
        if len(rgb_mask.shape) == 3:
            matches = np.all(rgb_mask == rgb_color, axis=2)
        else:
            return rgb_mask.astype(np.uint8)
        class_mask[matches] = class_id

    return class_mask

class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation."""

    def __init__(self, image_paths: List[Path], mask_paths: List[Path], transform=None):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform or self._default_transform()

        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks"

    def _default_transform(self):
        return T.Compose([
            T.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image_tensor = self.transform(image)

        mask = Image.open(self.mask_paths[idx])
        if mask.mode != 'RGB':
            mask = mask.convert('RGB')
        mask = mask.resize(IMAGE_SIZE, resample=Image.NEAREST)
        mask_array = np.array(mask)

        class_mask = rgb_to_class_id(mask_array)
        mask_tensor = torch.from_numpy(class_mask.astype(np.int64))

        return image_tensor, mask_tensor

def build_deeplab_model(num_classes: int = NUM_CLASSES):
    """Build DeepLab model for semantic segmentation."""
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model

def get_pseudo_mask_coverage(mask_path: Path) -> float:
    """Calculate percentage of non-unknown pixels in pseudo-mask."""
    try:
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
        class_mask = rgb_to_class_id(mask_rgb)

        total_pixels = class_mask.size
        unknown_pixels = np.sum(class_mask == 0)
        coverage = (total_pixels - unknown_pixels) / total_pixels
        return coverage
    except Exception as e:
        logger.warning(f"Error calculating coverage for {mask_path}: {e}")
        return 0.0

def select_top_k_pseudo_images(K: int, N: int) -> List[str]:
    """Select top K pseudo-images by mask coverage."""
    pseudo_masks_dir = PSEUDO_MASKS_DIRS[N]
    mask_files = list(pseudo_masks_dir.glob(f"*_mask_agree{N}.png"))

    coverages = []
    for mask_file in mask_files:
        image_id = mask_file.name.split('_')[0]
        coverage = get_pseudo_mask_coverage(mask_file)
        coverages.append((image_id, coverage))

    coverages.sort(key=lambda x: (-x[1], x[0]))
    selected = [image_id for image_id, _ in coverages[:K]]

    return selected

def build_subset(K: int, N: int, variant: str) -> Tuple[List[Path], List[Path]]:
    """Build training subset with K pseudo-images."""
    image_paths = []
    mask_paths = []

    labeled_images = list(LABELED_IMAGES_DIR.glob("*.jpg"))
    labeled_pairs = []
    for img_path in labeled_images:
        mask_path = LABELED_MASKS_DIR / f"{img_path.stem}.png"
        if mask_path.exists():
            labeled_pairs.append((img_path, mask_path))

    for img_path, mask_path in labeled_pairs:
        image_paths.append(img_path)
        mask_paths.append(mask_path)

    if K > 0:
        selected_stems = select_top_k_pseudo_images(K, N)
        for stem in selected_stems:
            if variant == "no-remove":
                img_path = UNLABELED_IMAGES_DIR / f"{stem}.jpg"
            elif variant == "remove-unknown":
                img_path = PSEUDO_IMAGES_DIRS[N] / f"{stem}.jpg"
            else:
                raise ValueError(f"Unknown variant: {variant}")

            mask_path = PSEUDO_MASKS_DIRS[N] / f"{stem}_mask_agree{N}.png"

            if img_path.exists() and mask_path.exists():
                image_paths.append(img_path)
                mask_paths.append(mask_path)

    return image_paths, mask_paths

# ============================================================================
# TRAINING / EVALUATION
# ============================================================================

def train_model(image_paths: List[Path], mask_paths: List[Path], run_id: str,
                num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE,
                learning_rate: float = LEARNING_RATE, early_stop_patience: int = EARLY_STOP_PATIENCE) -> nn.Module:
    """Train DeepLab model on given dataset with early stopping + scheduler."""
    logger.info(f"Training model for run: {run_id}")

    dataset = SegmentationDataset(image_paths, mask_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = build_deeplab_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        logger.info(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    checkpoint_path = Path("checkpoints") / f"{run_id}.pth"
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")

    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            preds_np = preds.cpu().numpy().flatten()
            targets_np = masks.cpu().numpy().flatten()
            valid_mask = targets_np != 255
            all_preds.extend(preds_np[valid_mask])
            all_targets.extend(targets_np[valid_mask])

    precision, recall, f1, _ = precision_recall_fscore_support(
        np.array(all_targets), np.array(all_preds), average='macro', zero_division=0
    )

    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}

def create_test_loader(batch_size: int = BATCH_SIZE) -> DataLoader:
    """Create test data loader."""
    test_images = list(TEST_IMAGES_DIR.glob("*.jpg"))
    test_masks = list(TEST_MASKS_DIR.glob("*.png"))
    test_pairs = [(img, TEST_MASKS_DIR / f"{img.stem}.png") for img in test_images if (TEST_MASKS_DIR / f"{img.stem}.png").exists()]
    image_paths, mask_paths = zip(*test_pairs)
    test_dataset = SegmentationDataset(list(image_paths), list(mask_paths))
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run pseudo-labeling experiments")
    parser.add_argument("--output", type=str, default="experiment_results.csv", help="Output CSV file")
    args = parser.parse_args()

    logger.info(f"Using device: {DEVICE}")

    test_loader = create_test_loader(BATCH_SIZE)
    results = []

    K_values = [10, 50, 100]
    N_values = [2, 3, 4]
    variants = ["no-remove", "remove-unknown"]

    experiment_count = 0
    total_experiments = len(K_values) * len(N_values) * len(variants)

    for K in K_values:
        for N in N_values:
            for variant in variants:
                experiment_count += 1
                run_id = f"deeplab_K{K}_N{N}_{variant}"
                logger.info("="*80)
                logger.info(f"EXPERIMENT {experiment_count}/{total_experiments}: {run_id}")
                logger.info("="*80)

                try:
                    image_paths, mask_paths = build_subset(K, N, variant)
                    model = train_model(image_paths, mask_paths, run_id)
                    metrics = evaluate_model(model, test_loader)

                    results.append({
                        'run_id': run_id,
                        'K': K,
                        'N': N,
                        'variant': variant,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1']
                    })

                    logger.info(f"Experiment {run_id} completed: {metrics}")

                except Exception as e:
                    logger.error(f"Experiment {run_id} failed: {e}")
                    results.append({
                        'run_id': run_id,
                        'K': K,
                        'N': N,
                        'variant': variant,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'error': str(e)
                    })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(df.to_string(index=False, float_format='%.4f'))

    if len(df) > 0:
        best_f1 = df.loc[df['f1'].idxmax()]
        print(f"\nBest F1 Score: {best_f1['f1']:.4f} (Run: {best_f1['run_id']})")

if __name__ == "__main__":
    main()
