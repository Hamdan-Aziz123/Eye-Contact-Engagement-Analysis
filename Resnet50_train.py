#!/usr/bin/env python3
"""
Resnet50_train.py (windows-safe, updated)
Fine-tune a pretrained ResNet-50 on your Deliverable-1 processed dataset (3-class classification).

Expected CSVs (created by your preprocessing script):
  <processed_dir>/train_labels.csv
  <processed_dir>/val_labels.csv
  <processed_dir>/test_labels.csv

Each CSV row should contain columns: image_path, pitch, vert, horiz (others allowed).
This script maps pitch/vert/horiz -> 3 classes using simple thresholds (thr_att, thr_dist).
"""

import os
import argparse
from pathlib import Path
import random
import time
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)

from tqdm import tqdm

# ---------------------------
# Utilities / Seeds
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # make cudnn deterministic for reproducibility (may slow training)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------
# Label mapping (3 classes)
# ---------------------------
# 0 -> Attentive
# 1 -> Distracted
# 2 -> Disengaged
def map_angles_to_label(pitch, vert, horiz, thr_att=10.0, thr_dist=25.0):
    """
    Simple rule:
      magnitude = max(|pitch|, |vert|, |horiz|)
      if magnitude <= thr_att       -> Attentive (0)
      elif magnitude <= thr_dist    -> Distracted (1)
      else                          -> Disengaged (2)
    thr_att and thr_dist are in degrees (or same units as csv)
    """
    mag = max(abs(float(pitch)), abs(float(vert)), abs(float(horiz)))
    if mag <= thr_att:
        return 0
    elif mag <= thr_dist:
        return 1
    else:
        return 2


# ---------------------------
# Dataset
# ---------------------------
class GazeDataset(Dataset):
    def __init__(self, csv_file, transform=None, thr_att=10.0, thr_dist=25.0):
        """
        csv_file: path to processed CSV created by Deliverable-1
        transform: torchvision transforms applied to PIL image
        thr_att, thr_dist: thresholds used to compute labels (so we avoid passing local functions)
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.thr_att = thr_att
        self.thr_dist = thr_dist

        # verify required columns
        for col in ['image_path', 'pitch', 'vert', 'horiz']:
            if col not in self.df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        # drop rows where image not found (defensive)
        def exists(p): return os.path.exists(str(p))
        exists_mask = self.df['image_path'].apply(exists)
        if not exists_mask.all():
            missing = (~exists_mask).sum()
            print(f"Warning: {missing} images listed in {csv_file} not found on disk. These will be skipped.")
            self.df = self.df[exists_mask].reset_index(drop=True)

        # precompute labels for speed (avoid passing functions to workers)
        self.labels = []
        for _, row in self.df.iterrows():
            lab = map_angles_to_label(row['pitch'], row['vert'], row['horiz'],
                                      thr_att=self.thr_att, thr_dist=self.thr_dist)
            self.labels.append(int(lab))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        # defensive image loading
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # if image fails to load, return a zero image (shouldn't happen often)
            print(f"Warning: failed to open image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        label = int(self.labels[idx])
        return image, label


# ---------------------------
# Training & Validation loops
# ---------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds = []
    gts = []
    for imgs, labels in tqdm(dataloader, desc="Train batches", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs.detach(), 1)
        preds.extend(predicted.cpu().numpy().tolist())
        gts.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    acc = accuracy_score(gts, preds) if len(gts) > 0 else 0.0
    return epoch_loss, acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    gts = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Val batches", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy().tolist())
            gts.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    acc = accuracy_score(gts, preds) if len(gts) > 0 else 0.0
    return epoch_loss, acc, gts, preds


# ---------------------------
# Build model (ResNet-50) - compatible with old/new torchvision API
# ---------------------------
def build_resnet50(num_classes=3, pretrained=True):
    # Try new weights API if available; otherwise fall back to deprecated 'pretrained' arg
    try:
        if pretrained:
            # newer torchvision: use weights enum
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.resnet50(weights=weights)
    except Exception:
        # older torchvision
        model = models.resnet50(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------
# Helper: compute class weights from label list (robust)
# ---------------------------
def compute_class_weights(labels, num_classes=3):
    """
    labels: list-like of integer labels
    returns: tensor of length num_classes with weights (inverse freq)
    """
    labels = np.array(labels, dtype=np.int64)
    total = max(1, labels.shape[0])
    counts = np.bincount(labels, minlength=num_classes)
    # avoid zero division
    weights = []
    for c in range(num_classes):
        cnt = counts[c]
        if cnt > 0:
            w = float(total) / (num_classes * float(cnt))
        else:
            w = 1.0  # default for missing class
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------
# Main trainer
# ---------------------------
def main(args):
    set_seed(args.seed)

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise ValueError(f"Processed dir not found: {processed_dir}")

    train_csv = processed_dir / 'train_labels.csv'
    val_csv = processed_dir / 'val_labels.csv'
    test_csv = processed_dir / 'test_labels.csv'
    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise ValueError(f"Missing CSV: {p}")

    # Build transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        # small augmentations (avoid horizontal flip because labels depend on gaze)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomRotation(degrees=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Datasets (pass thresholds, avoid passing local functions)
    train_ds = GazeDataset(train_csv, transform=train_transforms,
                           thr_att=args.thr_att, thr_dist=args.thr_dist)
    val_ds = GazeDataset(val_csv, transform=val_transforms,
                         thr_att=args.thr_att, thr_dist=args.thr_dist)
    test_ds = GazeDataset(test_csv, transform=val_transforms,
                          thr_att=args.thr_att, thr_dist=args.thr_dist)

    # Determine num_workers safely on Windows
    if os.name == 'nt':
        num_workers = 0
    else:
        num_workers = max(0, args.num_workers)

    pin_memory_flag = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory_flag)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory_flag)

    print(f"Dataset sizes - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    # Compute class weights
    class_weights = compute_class_weights(train_ds.labels, num_classes=args.num_classes)
    print("Class weights (train):", class_weights.numpy())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Build model
    model = build_resnet50(num_classes=args.num_classes, pretrained=not args.no_pretrained)
    model = model.to(device)

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # If freezing, set requires_grad appropriately and build optimizer on trainable params
    if args.freeze_epochs > 0:
        for name, param in model.named_parameters():
            if name.startswith('fc'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"Backbone frozen. Training head for {args.freeze_epochs} epochs.")
    else:
        print("No freezing phase. Training entire network from start.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        # safety: ensure at least fc is trainable
        for param in model.fc.parameters():
            param.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        # If entering fine-tune phase (unfreeze)
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.fine_tune_lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}")

        # scheduler step (ReduceLROnPlateau expects metric)
        try:
            scheduler.step(val_acc)
        except Exception:
            pass

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = out_dir / 'best_resnet50.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_mapping': {0: 'Attentive', 1: 'Distracted', 2: 'Disengaged'},
                'args': vars(args)
            }, best_path)
            print(f"Saved best model -> {best_path} (val_acc={val_acc:.4f})")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.2f} minutes. Best val acc: {best_val_acc:.4f}")

    # Load best model for testing (if exists)
    best_path = out_dir / 'best_resnet50.pth'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    else:
        print("Warning: best model not found, using last model state for testing.")

    # Evaluate on test set and print classification metrics
    _, _, y_true, y_pred = validate(model, test_loader, criterion, device)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, target_names=['Attentive', 'Distracted', 'Disengaged'], zero_division=0)

    print("\n=== Test Results ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall:    {rec:.4f}")
    print(f"Weighted F1:        {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:\n", cls_report)

    # Save report
    report = {
        'test_accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report
    }
    with open(out_dir / 'test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(cls_report)

    print(f"\nSaved test report to {out_dir}")

    # optionally save full history as csv
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / 'train_history.csv', index=False)
    print(f"Saved training history to {out_dir / 'train_history.csv'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 for 3-class engagement classification")
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Path to processed CSVs (train_labels.csv, val_labels.csv, test_labels.csv)')
    parser.add_argument('--output_dir', type=str, default='outputs/resnet50', help='Where to save models and reports')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--freeze_epochs', type=int, default=5, help='Number of epochs to train only head (backbone frozen)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for initial head training')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5, help='Learning rate for fine-tuning phase')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--thr_att', type=float, default=10.0, help='Threshold for attentive class (max angle <= thr_att)')
    parser.add_argument('--thr_dist', type=float, default=25.0, help='Threshold for distracted class (max angle <= thr_dist)')
    parser.add_argument('--no_pretrained', action='store_true', help='If set, do not use ImageNet pretrained weights')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of target classes (default 3)')

    args = parser.parse_args()
    main(args)
