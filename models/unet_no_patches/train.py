# To run: python models/unet_no_patches/train.py 

import multiprocessing
import os
import copy
import random
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import sys

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.unet_no_patches.dataset import UNetSegmentationDataset as SegmentationDataset
from models.unet_no_patches.model import build_pretrained_unet as build_model


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8    
num_epochs = 100  
early_stop_patience = 7  
lr = 1e-4         
train_val_split = 0.8
num_workers = 0   
NUM_CLASSES = 8         

def train_split(base_dir: str, split_name: str, scripts_dir: str):
    full_ds = SegmentationDataset(base_dir, split_name)
    n_train = int(len(full_ds) * train_val_split)
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    
    model = build_model(device=device)

    print("Calculating class weights from training set...")
    all_masks = []
    for i in range(len(train_ds)):
        _, mask = train_ds[i]
        all_masks.append(mask.flatten())

    all_masks = torch.cat(all_masks)
    class_counts = torch.bincount(all_masks, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # Normalize

    print(f"Class counts (train only): {class_counts}")
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(scripts_dir, exist_ok=True)
    ckpt_path = os.path.join(scripts_dir, 'unet_train.pth')
    final_path = os.path.join(scripts_dir, 'unet_train.pth')

    for epoch in range(1, num_epochs + 1):
        
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

        
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                running_val += criterion(model(imgs), masks).item() * imgs.size(0)
        val_loss = running_val / len(val_loader.dataset)

        print(f"[train] Epoch {epoch:02d}/{num_epochs}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        
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

    torch.save(best_wts, final_path)
    print(f"[train] Training complete. Final weights saved to {final_path}")


def main():
    base_dir = os.path.join(PROJECT_ROOT, 'dataset')
    scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')

    train_split(base_dir, 'train', scripts_dir)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
