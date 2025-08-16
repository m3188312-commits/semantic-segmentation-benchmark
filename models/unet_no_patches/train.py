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
from models.unet_no_patches.model import build_pretrained_unet as build_model

# ─── Configuration ──────────────────────────────────
device               = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size           = 8
num_epochs           = 80
early_stop_patience  = 5
lr                   = 1e-4
train_val_split      = 0.8
num_workers          = 8
# ────────────────────────────────────────────────────

def train_split(base_dir, split_name, scripts_dir):
    """Train U-Net on a specific split: 'train' or 'lowres'."""
    # Dataset and DataLoaders
    full_ds = SegmentationDataset(base_dir, split_name)
    n_train = int(len(full_ds) * train_val_split)
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )

    # Build model
    model = build_model(device=device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    # Prepare checkpoint path
    os.makedirs(scripts_dir, exist_ok=True)
    ckpt_path = os.path.join(scripts_dir, f'unet_{split_name}.pth')

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # — Training —
        model.train()
        running_train = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
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

        # Logging
        print(f"[{split_name}] Epoch {epoch:02d}/{num_epochs}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        # — Early stopping —
        if val_loss < best_loss:
            best_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, ckpt_path)
            print(f"  → New best, saved to {ckpt_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"  → No improvement for {early_stop_patience} epochs; stopping.")
                break

        scheduler.step()


def main():
    # Define base dataset path and scripts dir
    base_dir = os.path.join(PROJECT_ROOT, 'dataset')
    scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')

    # Train only on 'train' set
    train_split(base_dir, 'train', scripts_dir)
    # Removed lowres training as requested
    # train_split(base_dir, 'lowres', scripts_dir)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()