#!/usr/bin/env python3
"""
Pseudo-Labeling Experiments for Semantic Segmentation
Runs 18 training/evaluation cases: 3 K values × 3 N values × 2 variants + baseline
"""

import argparse
import json
import csv
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Any
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

# Training parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 8
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10  # Stop if no improvement for 10 epochs

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
# UTILITY FUNCTIONS
# ============================================================================

def rgb_to_class_id(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert RGB mask to class ID mask."""
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    for rgb_color, class_id in CLASS_RGB.items():
        # Find pixels matching this RGB color
        if len(rgb_mask.shape) == 3:
            matches = np.all(rgb_mask == rgb_color, axis=2)
        else:
            # Grayscale mask - assume it's already class IDs
            return rgb_mask.astype(np.uint8)
        class_mask[matches] = class_id
    
    return class_mask

# ============================================================================
# DATASET CLASS
# ============================================================================

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
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image_tensor = self.transform(image)
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        
        # Convert to RGB if not already, then resize
        if mask.mode != 'RGB':
            mask = mask.convert('RGB')
        mask = mask.resize(IMAGE_SIZE, resample=Image.NEAREST)
        mask_array = np.array(mask)
        
        # Convert RGB mask to class IDs
        class_mask = rgb_to_class_id(mask_array)
        mask_tensor = torch.from_numpy(class_mask.astype(np.int64))
        
        return image_tensor, mask_tensor

# ============================================================================
# DEEPLAB MODEL
# ============================================================================

def build_deeplab_model(num_classes: int = NUM_CLASSES):
    """Build DeepLab model for semantic segmentation."""
    from torchvision.models.segmentation import deeplabv3_resnet50
    
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    return model

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_pseudo_mask_coverage(mask_path: Path) -> float:
    """Calculate percentage of non-unknown pixels in pseudo-mask."""
    try:
        # Load RGB mask and convert to class IDs
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
        class_mask = rgb_to_class_id(mask_rgb)
        
        total_pixels = class_mask.size
        unknown_pixels = np.sum(class_mask == 0)  # Unknown class = 0
        coverage = (total_pixels - unknown_pixels) / total_pixels
        return coverage
    except Exception as e:
        logger.warning(f"Error calculating coverage for {mask_path}: {e}")
        return 0.0

def select_top_k_pseudo_images(K: int, N: int) -> List[str]:
    """Select top K pseudo-images by mask coverage."""
    pseudo_masks_dir = PSEUDO_MASKS_DIRS[N]
    
    # Get all visualization mask files (format: XXX_mask_agreeN.png)
    mask_files = list(pseudo_masks_dir.glob(f"*_mask_agree{N}.png"))
    
    # Calculate coverage for each mask
    coverages = []
    for mask_file in mask_files:
        # Extract the image ID from filename (e.g., "001_mask_agree2.png" -> "001")
        image_id = mask_file.name.split('_')[0]
        coverage = get_pseudo_mask_coverage(mask_file)
        coverages.append((image_id, coverage))
    
    # Sort by coverage (desc) then by filename (asc)
    coverages.sort(key=lambda x: (-x[1], x[0]))
    
    # Select top K
    selected = [image_id for image_id, _ in coverages[:K]]
    
    if len(coverages) > 0:
        coverage_range = f"{coverages[min(K-1, len(coverages)-1)][1]:.3f} - {coverages[0][1]:.3f}"
        logger.info(f"Selected top {K} pseudo-images for N={N}: coverage range {coverage_range}")
    else:
        logger.warning(f"No pseudo-masks found in {pseudo_masks_dir}")
    
    return selected

def build_subset(K: int, N: int, variant: str) -> Tuple[List[Path], List[Path]]:
    """
    Build training subset with K pseudo-images.
    
    Args:
        K: Number of pseudo-images to include
        N: Agreement level (2, 3, or 4)
        variant: "no-remove" or "remove-unknown"
    
    Returns:
        Tuple of (image_paths, mask_paths)
    """
    image_paths = []
    mask_paths = []
    
    # 1. Add all original labeled images and masks
    labeled_images = list(LABELED_IMAGES_DIR.glob("*.jpg"))
    labeled_masks = list(LABELED_MASKS_DIR.glob("*.png"))
    
    # Match images to masks by stem
    labeled_pairs = []
    for img_path in labeled_images:
        mask_path = LABELED_MASKS_DIR / f"{img_path.stem}.png"
        if mask_path.exists():
            labeled_pairs.append((img_path, mask_path))
    
    logger.info(f"Found {len(labeled_pairs)} original labeled image-mask pairs")
    
    for img_path, mask_path in labeled_pairs:
        image_paths.append(img_path)
        mask_paths.append(mask_path)
    
    # 2. Select and add top K pseudo-images
    if K > 0:
        selected_stems = select_top_k_pseudo_images(K, N)
        
        for stem in selected_stems:
            # Choose image source based on variant
            if variant == "no-remove":
                img_path = UNLABELED_IMAGES_DIR / f"{stem}.jpg"
            elif variant == "remove-unknown":
                img_path = PSEUDO_IMAGES_DIRS[N] / f"{stem}.jpg"
            else:
                raise ValueError(f"Unknown variant: {variant}")
            
            # Mask comes from visualization masks (format: XXX_mask_agreeN.png)
            mask_path = PSEUDO_MASKS_DIRS[N] / f"{stem}_mask_agree{N}.png"
            
            if img_path.exists() and mask_path.exists():
                image_paths.append(img_path)
                mask_paths.append(mask_path)
            else:
                logger.warning(f"Missing files for {stem}: img={img_path.exists()}, mask={mask_path.exists()}")
    
    logger.info(f"Built subset: {len(image_paths)} total images ({len(labeled_pairs)} labeled + {len(image_paths)-len(labeled_pairs)} pseudo)")
    
    return image_paths, mask_paths

def train_model(image_paths: List[Path], mask_paths: List[Path], run_id: str, 
                num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE, 
                learning_rate: float = LEARNING_RATE, early_stop_patience: int = 10) -> nn.Module:
    """Train DeepLab model on given dataset with early stopping."""
    logger.info(f"Training model for run: {run_id}")
    
    # Create dataset and dataloader
    dataset = SegmentationDataset(image_paths, mask_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = build_deeplab_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(images)['out']  # DeepLab returns dict
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"  Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
        
        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"  New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs (patience: {early_stop_patience})")
            
            if patience_counter >= early_stop_patience:
                logger.info(f"  Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with loss: {best_loss:.4f}")
    
    # Save model checkpoint
    checkpoint_path = Path("checkpoints") / f"{run_id}.pth"
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")
    
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """Evaluate model on test set and return metrics."""
    logger.info("Evaluating model...")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            
            # Flatten and convert to numpy
            preds_np = preds.cpu().numpy().flatten()
            targets_np = masks.cpu().numpy().flatten()
            
            # Filter out ignore_index values if any
            valid_mask = targets_np != 255
            preds_np = preds_np[valid_mask]
            targets_np = targets_np[valid_mask]
            
            all_preds.extend(preds_np)
            all_targets.extend(targets_np)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    logger.info(f"Evaluation metrics: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    
    return metrics

def create_test_loader(batch_size: int = BATCH_SIZE) -> DataLoader:
    """Create test data loader."""
    test_images = list(TEST_IMAGES_DIR.glob("*.jpg"))
    test_masks = list(TEST_MASKS_DIR.glob("*.png"))
    
    # Match images to masks
    test_pairs = []
    for img_path in test_images:
        mask_path = TEST_MASKS_DIR / f"{img_path.stem}.png"
        if mask_path.exists():
            test_pairs.append((img_path, mask_path))
    
    if not test_pairs:
        raise RuntimeError(f"No test image-mask pairs found in {TEST_IMAGES_DIR} and {TEST_MASKS_DIR}")
    
    logger.info(f"Found {len(test_pairs)} test image-mask pairs")
    
    image_paths, mask_paths = zip(*test_pairs)
    test_dataset = SegmentationDataset(list(image_paths), list(mask_paths))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run pseudo-labeling experiments")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--output", type=str, default="experiment_results.csv", help="Output CSV file")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline training")
    
    args = parser.parse_args()
    
    # Use local variables instead of modifying globals
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    
    logger.info(f"Starting experiments with {num_epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"Using device: {DEVICE}")
    
    # Create test loader (same for all experiments)
    test_loader = create_test_loader(batch_size)
    
    # Results storage
    results = []
    
    # Experiment parameters
    K_values = [10, 50, 100]
    N_values = [2, 3, 4]
    variants = ["no-remove", "remove-unknown"]
    
    # Run baseline (original labeled data only)
    if not args.skip_baseline:
        logger.info("="*80)
        logger.info("RUNNING BASELINE EXPERIMENT")
        logger.info("="*80)
        
        baseline_images, baseline_masks = build_subset(K=0, N=2, variant="no-remove")  # K=0 means no pseudo-images
        baseline_model = train_model(baseline_images, baseline_masks, "baseline", num_epochs, batch_size, learning_rate, EARLY_STOP_PATIENCE)
        baseline_metrics = evaluate_model(baseline_model, test_loader)
        
        results.append({
            'run_id': 'baseline',
            'K': 0,
            'N': 'N/A',
            'variant': 'labeled_only',
            'precision': baseline_metrics['precision'],
            'recall': baseline_metrics['recall'],
            'f1': baseline_metrics['f1']
        })
        
        logger.info(f"Baseline results: {baseline_metrics}")
    
    # Run all 18 experiments
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
                    # Build dataset
                    image_paths, mask_paths = build_subset(K, N, variant)
                    
                    # Train model
                    model = train_model(image_paths, mask_paths, run_id, num_epochs, batch_size, learning_rate, EARLY_STOP_PATIENCE)
                    
                    # Evaluate model
                    metrics = evaluate_model(model, test_loader)
                    
                    # Store results
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
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Print summary table
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Print best results
    if len(df) > 1:
        best_f1 = df.loc[df['f1'].idxmax()]
        print(f"\nBest F1 Score: {best_f1['f1']:.4f} (Run: {best_f1['run_id']})")
        
        if not args.skip_baseline:
            baseline_f1 = df[df['run_id'] == 'baseline']['f1'].iloc[0]
            improvement = best_f1['f1'] - baseline_f1
            print(f"Improvement over baseline: {improvement:+.4f} ({improvement/baseline_f1*100:+.1f}%)")

if __name__ == "__main__":
    main()
