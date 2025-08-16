# train.py
import multiprocessing
import os
import copy
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Absolute imports from the deeplab package
from models.deeplab.dataset import SemanticSegmentationDataset
from models.deeplab.model   import build_model


# ─── Configuration ──────────────────────────────────
device          = torch.device('cuda', index=0)  # enforce GPU
batch_size      = 8      # see guidance below
num_epochs      = 80
early_stop_patience = 10
lr              = 1e-4
train_val_split = 0.8
num_workers     = 8
# ────────────────────────────────────────────────────

def train_split(base_dir, split_name, scripts_dir):
    # Prepare datasets and loaders
    full_ds = SemanticSegmentationDataset(base_dir, split_name)
    n_train = int(len(full_ds) * train_val_split)
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    # Build model on GPU
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)      # AdamW often yields better generalization
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)   # slower LR decay

    best_wts    = copy.deepcopy(model.state_dict())
    best_loss   = float('inf')
    epochs_no_improve = 0

    os.makedirs(scripts_dir, exist_ok=True)
    ckpt_path = os.path.join(scripts_dir, f'deeplab_{split_name}.pth')

    for epoch in range(1, num_epochs+1):
        # — Training —
        model.train()
        running_train = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)['out']
            loss    = criterion(outputs, masks)
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
                running_val += criterion(model(imgs)['out'], masks).item() * imgs.size(0)
        val_loss = running_val / len(val_loader.dataset)

        print(f"[{split_name}] Epoch {epoch:02d}/{num_epochs}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        # — Early stopping check —
        if val_loss < best_loss:
            best_loss = val_loss
            best_wts  = copy.deepcopy(model.state_dict())
            torch.save(best_wts, ckpt_path)
            print(f"  → New best, saving to {ckpt_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"  → No improvement for {early_stop_patience} epochs. Early stopping.")
                break

        scheduler.step()

    # — Final checkpoint —
    final_path = os.path.join(scripts_dir, f'deeplab_{split_name}_final.pth')
    torch.save(best_wts, final_path)
    print(f"[{split_name}] Training complete. Final weights saved to {final_path}")

def main():
    root        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    base        = os.path.join(root, 'dataset')
    scripts_dir = os.path.join(root, 'scripts')

    train_split(base, 'train',  scripts_dir)
    # Removed lowres training as requested
    # train_split(base, 'lowres', scripts_dir)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
