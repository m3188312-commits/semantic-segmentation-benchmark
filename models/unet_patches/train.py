import os
import argparse
from pathlib import Path

import numpy as np
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split

from models.unet_patches.dataset import PatchesSegmentationDataset
from models.unet_patches.model import build_unet_model

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on patch segmentation dataset")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root folder containing 'image/' and 'mask/' subdirs for training")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save checkpoints, logs, and figures")
    parser.add_argument("--patch-size", type=int, default=128,
                        help="Patch size (square) for dataset")
    parser.add_argument("--stride", type=int, default=128,
                        help="Stride between patches in dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create output directories
    ckpt_dir = Path(args.output_dir)/"checkpoints"
    fig_dir  = Path(args.output_dir)/"figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    img_dir = os.path.join(args.data_dir, "image")
    mask_dir = os.path.join(args.data_dir, "mask")
    full_ds = PatchesSegmentationDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        patch_size=args.patch_size,
        stride=args.stride
    )
    total = len(full_ds)
    val_size = int(total * args.val_split)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model, loss, optimizer, scheduler
    model = build_unet_model(in_channels=3, num_classes=4, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs+1):
        # training
        model.train()
        running = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                running_val += criterion(outputs, masks).item() * imgs.size(0)
        val_loss = running_val / len(val_loader.dataset)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir/"best.pth")
            print("  Saved new best model")
        scheduler.step()

    torch.save(model.state_dict(), ckpt_dir/"final.pth")
    print("Training complete.")

    # save CSV log
    csv_path = Path(args.output_dir)/"training_log.csv"
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['epoch','train_loss','val_loss'])
        for i,(tl,vl) in enumerate(zip(history['train_loss'], history['val_loss']),1):
            writer.writerow([i, tl, vl])

    # plot
    epochs = range(1, args.epochs+1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(fig_dir/"loss_curve.png")

if __name__ == '__main__':
    args = parse_args()
    train(args)
