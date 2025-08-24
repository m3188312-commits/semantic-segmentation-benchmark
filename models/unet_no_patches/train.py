# models/unet_no_patches/train.py
# Run from repo root:
#   python -m models.unet_no_patches.train

import multiprocessing
import os
import copy
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Ensure project root on PYTHONPATH for relative imports
import sys
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.unet_no_patches.dataset import UNetSegmentationDataset as SegmentationDataset
from models.unet_no_patches.model   import build_pretrained_unet as build_model

# ─── Configuration ──────────────────────────────────
device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size          = 2  # Reduced for better stability
num_epochs          = 100  # More epochs for complex task
early_stop_patience = 7  # More patience
lr                  = 5e-5  # Lower learning rate for stability
train_val_split     = 0.8
num_workers         = 0     # Windows-friendly; no prefetch_factor
# ────────────────────────────────────────────────────

def train_split(base_dir: str, split_name: str, scripts_dir: str):
    """Train U-Net on a specific split (here we'll call only 'train')."""
    # Dataset and DataLoaders
    full_ds = SegmentationDataset(base_dir, split_name, augment=True)  # Enable augmentations for training
    n_train = int(len(full_ds) * train_val_split)
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Build model
    model     = build_model(device=device)
    
    # Better loss function with class balancing
    # Calculate class weights from training data
    print("Calculating class weights...")
    all_masks = []
    for i in range(len(full_ds)):
        _, mask = full_ds[i]
        all_masks.append(mask.flatten())
    
    all_masks = torch.cat(all_masks)
    class_counts = torch.bincount(all_masks, minlength=8)
    class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_wts  = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    # Checkpoint pathing
    os.makedirs(scripts_dir, exist_ok=True)
    ckpt_path  = os.path.join(scripts_dir, 'unet_train.pth')
    final_path = os.path.join(scripts_dir, 'unet_train.pth')

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # — Training —
        model.train()
        running_train = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)                 # (B, C, H, W)
            loss    = criterion(outputs, masks)   # masks are (B, H, W) with class indices
            loss.backward()
            optimizer.step()
            running_train += loss.item() * imgs.size(0)
        train_loss = running_train / len(train_loader.dataset)

        # — Validation —
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                running_val += criterion(model(imgs), masks).item() * imgs.size(0)
        val_loss = running_val / len(val_loader.dataset)

        print(f"[train] Epoch {epoch:02d}/{num_epochs}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        # — Early stopping & checkpointing —
        if val_loss < best_loss:
            best_loss = val_loss
            best_wts  = copy.deepcopy(model.state_dict())
            torch.save(best_wts, ckpt_path)
            print(f"  → New best, saved to {ckpt_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"  → No improvement for {early_stop_patience} epochs; stopping.")
                break

        scheduler.step(val_loss)

    # Finalize
    torch.save(best_wts, final_path)
    print(f"[train] Training complete. Final weights saved to {final_path}")

def main():
    base_dir   = os.path.join(PROJECT_ROOT, 'dataset')
    scripts_dir= os.path.join(PROJECT_ROOT, 'scripts')

    # Train only on the original training set
    train_split(base_dir, 'train', scripts_dir)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
