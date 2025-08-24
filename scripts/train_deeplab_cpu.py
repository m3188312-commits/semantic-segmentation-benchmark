#!/usr/bin/env python3
"""
Modified DeepLab training script for CPU usage and command line arguments.
Based on the original models/deeplab/train.py but adapted for evaluation pipeline.
"""

import multiprocessing
import os
import copy
import argparse
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deeplab.dataset import SemanticSegmentationDataset
from models.deeplab.model   import build_model

def train_split(base_dir, split_name, scripts_dir, device, batch_size=4, num_epochs=10):
    """Train the model with the given parameters."""
    print(f"🚀 Starting training for {split_name}")
    print(f"   Data directory: {base_dir}")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    
    # Prepare datasets and loaders
    try:
        full_ds = SemanticSegmentationDataset(base_dir, split_name)
        print(f"   Dataset size: {len(full_ds)}")
        
        n_train = int(len(full_ds) * 0.8)
        train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])
        print(f"   Train/Val split: {len(train_ds)}/{len(val_ds)}")
        
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False  # Disable pin_memory for CPU
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )
        
    except Exception as e:
        print(f"   ❌ Dataset creation failed: {e}")
        return False

    # Build model
    try:
        model = build_model(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        print(f"   ✅ Model built successfully")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 5

    os.makedirs(scripts_dir, exist_ok=True)
    ckpt_path = os.path.join(scripts_dir, 'best_model.pth')

    print(f"   🔄 Starting training loop...")
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        running_train = 0.0
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * imgs.size(0)
            
            if batch_idx % 10 == 0:
                print(f"     Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss = running_train / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                running_val += criterion(model(imgs)['out'], masks).item() * imgs.size(0)
        val_loss = running_val / len(val_loader.dataset)

        print(f"   📊 Epoch {epoch:02d}/{num_epochs}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, ckpt_path)
            print(f"     → New best, saving to {ckpt_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"     → No improvement for {early_stop_patience} epochs. Early stopping.")
                break

        scheduler.step()

    print(f"   ✅ Training complete. Best weights saved to {ckpt_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Train DeepLab model for evaluation")
    parser.add_argument("--data_dir", required=True, help="Training data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()
    
    # Determine device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("🖥️  Using CPU")
    else:
        device = torch.device('cuda')
        print("🚀 Using CUDA")
    
    print("🧠 DeepLab Training Script")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    success = train_split(
        base_dir=args.data_dir,
        split_name='train',  # We'll use the full directory as training data
        scripts_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {args.output_dir}")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
