# File: ViT_train.py
"""
ViT_train.py
------------- (same header as before)
"""
import argparse
import os
import sys
import random
import json
import math
import time
import warnings
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Try to import timm; fallback to torchvision
USE_TIMM = True
try:
    import timm
except Exception:
    USE_TIMM = False
    warnings.warn("timm not found. Falling back to torchvision ViT (if available). Install timm for more models.")

# -------------------------
# Utility functions
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False

def map_angles_to_label(pitch: float, vert: float, horiz: float, thr_att: float = 10.0, thr_dist: float = 25.0) -> int:
    mag = float(max(abs(float(pitch)), abs(float(vert)), abs(float(horiz))))
    if mag <= thr_att:
        return 0
    elif mag <= thr_dist:
        return 1
    else:
        return 2

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
class GazeCSVImageDataset(Dataset):
    def __init__(self, csv_path: str, processed_dir: Optional[str] = None,
                 transform=None, thr_att: float = 10.0, thr_dist: float = 25.0,
                 verbose: bool = True):
        self.csv_path = csv_path
        self.processed_dir = processed_dir
        self.transform = transform
        self.thr_att = thr_att
        self.thr_dist = thr_dist
        self.verbose = verbose

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        required_cols = {'image_path', 'pitch', 'vert', 'horiz'}
        if not required_cols.issubset(set(self.df.columns)):
            raise ValueError(f"CSV {csv_path} missing required columns. Required: {required_cols}. Found: {set(self.df.columns)}")

        records = []
        missing = 0
        for _, row in self.df.iterrows():
            img_path = str(row['image_path'])
            if self.processed_dir and not os.path.isabs(img_path):
                img_path = os.path.join(self.processed_dir, img_path)
            if not os.path.isabs(img_path):
                img_path = os.path.abspath(img_path)
            if not os.path.isfile(img_path):
                missing += 1
                if self.verbose and missing <= 10:
                    warnings.warn(f"Missing image file, skipping: {img_path}")
                continue
            pitch = safe_float(row['pitch'])
            vert = safe_float(row['vert'])
            horiz = safe_float(row['horiz'])
            label = map_angles_to_label(pitch, vert, horiz, thr_att=self.thr_att, thr_dist=self.thr_dist)
            records.append({
                'image_path': img_path,
                'label': int(label),
                'pitch': pitch,
                'vert': vert,
                'horiz': horiz,
                **{c: row[c] for c in row.index if c not in ['image_path', 'pitch', 'vert', 'horiz']}
            })
        if missing and self.verbose:
            print(f"[Dataset] {missing} missing files skipped from {csv_path}. Kept {len(records)} samples.")
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = rec['label']
        return img, int(label)

# -------------------------
# Model builder
# -------------------------
def build_vit_model(model_name: str, num_classes: int = 3, pretrained: bool = True, image_size: int = 224):
    if USE_TIMM:
        try:
            backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            feat_dim = getattr(backbone, 'num_features', None)
            if feat_dim is None:
                raise RuntimeError("Couldn't determine feature dimension from timm model.")
            head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, num_classes)
            )
            backbone.head = head
            return backbone, feat_dim
        except Exception as e:
            warnings.warn(f"timm model build failed: {e}. Trying torchvision fallback.")
    try:
        import torchvision
        if hasattr(torchvision.models, 'vit_b_16'):
            model = torchvision.models.vit_b_16(pretrained=pretrained)
            feat_dim = None
            try:
                if hasattr(model, 'heads'):
                    if hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
                        feat_dim = model.heads.head.in_features
                        model.heads.head = nn.Linear(feat_dim, num_classes)
                    else:
                        model.heads = nn.Sequential(nn.LayerNorm(model.hidden_dim), nn.Linear(model.hidden_dim, num_classes))
                        feat_dim = model.hidden_dim
                else:
                    feat_dim = getattr(model, 'hidden_dim', None)
                    if feat_dim is None:
                        raise RuntimeError("Cannot find ViT feature dimension in torchvision model.")
                    model.heads = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, num_classes))
            except Exception as ex:
                raise RuntimeError(f"Failed adapting torchvision ViT: {ex}")
            return model, feat_dim
    except Exception:
        pass
    raise RuntimeError("No suitable ViT model builder found. Please install timm (recommended) or upgrade torchvision.")

# -------------------------
# Training & Evaluation utils
# -------------------------
def accuracy_from_outputs(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    return float((preds == targets_np).mean())

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy().tolist())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = float((np.array(all_preds) == np.array(all_targets)).mean()) if all_preds else 0.0
    return avg_loss, acc, all_preds, all_targets

# -------------------------
# Training loop
# -------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for imgs, targets in pbar:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == targets).sum().item()
        total_samples += batch_size
        pbar.set_postfix(loss=running_loss/total_samples, acc=running_corrects/total_samples)
    epoch_loss = running_loss / total_samples if total_samples else 0.0
    epoch_acc = running_corrects / total_samples if total_samples else 0.0
    return epoch_loss, epoch_acc

# -------------------------
# Helpers for optimizer freezing/unfreezing
# -------------------------
def freeze_backbone_except_head(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False
    if hasattr(model, 'head'):
        for name, param in model.head.named_parameters():
            param.requires_grad = True
    else:
        for key in ['heads', 'classifier', 'fc']:
            if hasattr(model, key):
                for name, param in getattr(model, key).named_parameters():
                    param.requires_grad = True
                break

def unfreeze_all(model: nn.Module):
    for _, param in model.named_parameters():
        param.requires_grad = True

# -------------------------
# Main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on processed gaze dataset (3-class).")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Processed dataset base directory.")
    parser.add_argument("--output_dir", type=str, default="outputs/vit", help="Directory to save outputs and models.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (already resized but transforms will ensure).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Number of epochs to train only head while backbone frozen.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for head training phase.")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Learning rate for fine-tuning full model.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thr_att", type=float, default=10.0, help="Threshold for Attentive (magnitude <= thr_att)")
    parser.add_argument("--thr_dist", type=float, default=25.0, help="Threshold for Distracted (thr_att < magnitude <= thr_dist)")
    parser.add_argument("--no_pretrained", action='store_true', help="Do not use imagenet pretrained weights.")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224", help="timm model name (or torchvision fallback).")
    parser.add_argument("--train_csv", type=str, default=None, help="Optional explicit path to train_labels.csv (defaults to processed_dir/train_labels.csv)")
    parser.add_argument("--val_csv", type=str, default=None, help="Optional explicit path to val_labels.csv")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional explicit path to test_labels.csv")
    parser.add_argument("--print_stats_only", action='store_true', help="Only print pitch/vert/horiz stats and exit (useful to tune thresholds).")
    parser.add_argument("--save_every_epoch", action='store_true', help="Also save checkpoint every epoch (in addition to best).")
    parser.add_argument("--patience_lr", type=int, default=3, help="Patience for ReduceLROnPlateau scheduler.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    processed_dir = os.path.abspath(args.processed_dir)
    ensure_dir(args.output_dir)
    train_csv = args.train_csv or os.path.join(processed_dir, "train_labels.csv")
    val_csv = args.val_csv or os.path.join(processed_dir, "val_labels.csv")
    test_csv = args.test_csv or os.path.join(processed_dir, "test_labels.csv")

    print("Arguments:", vars(args))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def print_angle_stats(csv_path, name):
        if not os.path.isfile(csv_path):
            print(f"[Stats] CSV not found: {csv_path}")
            return
        df = pd.read_csv(csv_path)
        for col in ['pitch', 'vert', 'horiz']:
            if col not in df.columns:
                continue
            arr = pd.to_numeric(df[col], errors='coerce').dropna().abs()
            percentiles = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100]) if len(arr) else [0]*9
            print(f"[{name}] {col} abs stats: count={len(arr)}, min={percentiles[0]:.2f}, p1={percentiles[1]:.2f}, p5={percentiles[2]:.2f}, p25={percentiles[3]:.2f}, median={percentiles[4]:.2f}, p75={percentiles[5]:.2f}, p95={percentiles[6]:.2f}, p99={percentiles[7]:.2f}, max={percentiles[8]:.2f}")

    print("\n=== Angle distribution stats (train/val/test) ===")
    print_angle_stats(train_csv, "train")
    print_angle_stats(val_csv, "val")
    print_angle_stats(test_csv, "test")
    print("=== End stats ===\n")

    if args.print_stats_only:
        print("Printed stats only (--print_stats_only). Exiting.")
        return

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    val_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_ds = GazeCSVImageDataset(train_csv, processed_dir, transform=train_transform,
                                  thr_att=args.thr_att, thr_dist=args.thr_dist, verbose=True)
    val_ds = GazeCSVImageDataset(val_csv, processed_dir, transform=val_transform,
                                thr_att=args.thr_att, thr_dist=args.thr_dist, verbose=True)
    test_ds = GazeCSVImageDataset(test_csv, processed_dir, transform=val_transform,
                                 thr_att=args.thr_att, thr_dist=args.thr_dist, verbose=True)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("No training samples found after filtering missing images. Check CSV and processed_dir paths.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    y_train = [rec['label'] for rec in train_ds.records]
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=np.array(y_train))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Computed class weights: {class_weights} (to be used in CrossEntropyLoss)")

    model, feat_dim = build_vit_model(args.model_name, num_classes=3, pretrained=(not args.no_pretrained), image_size=args.image_size)
    model.to(device)
    print(f"Model built: {args.model_name}. Feature dim: {feat_dim}")

    freeze_backbone_except_head(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience_lr, factor=0.5)

    history = []
    best_val_acc = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "best_vit.pth")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        phase = "head-only" if epoch <= args.freeze_epochs else "fine-tune"
        print(f"\nEpoch {epoch}/{args.epochs} — Phase: {phase}")

        if epoch == args.freeze_epochs + 1:
            print("Unfreezing backbone for fine-tuning...")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.fine_tune_lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience_lr, factor=0.5, verbose=True)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} results: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        history.append({
            'epoch': epoch,
            'phase': phase,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
                'class_weights': class_weights.tolist(),
            }
            torch.save(checkpoint, best_ckpt_path)
            print(f"Saved new best model (val_acc={val_acc:.4f}) to: {best_ckpt_path}")

        if args.save_every_epoch:
            ep_ckpt = os.path.join(args.output_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, ep_ckpt)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60.0:.2f} minutes. Best val_acc={best_val_acc:.4f}")

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(args.output_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"Saved training history to: {history_csv}")

    cmd_example = f"python {os.path.basename(__file__)} " + " ".join([f"--{k} {v}" for k, v in vars(args).items() if not isinstance(v, bool)]) \
                  + (" --no_pretrained" if args.no_pretrained else "")
    readme_text = (
        f"Run command used (approx):\n\n{cmd_example}\n\n"
        "Files saved here:\n"
        "- best_vit.pth : best model checkpoint (by val acc)\n"
        "- training_history.csv : per-epoch metrics\n"
        "- test_report.json : final test metrics (accuracy, precision, recall, f1, confusion matrix)\n"
        "- classification_report.txt : sklearn classification_report text\n\n"
        "Notes:\n- Use --print_stats_only to print distribution stats for pitch/vert/horiz to help choose thresholds.\n"
        "- No horizontal flip used because gaze/horiz would invert.\n"
    )
    with open(os.path.join(args.output_dir, "README_run.txt"), "w") as f:
        f.write(readme_text)
    print(f"Saved README to: {os.path.join(args.output_dir, 'README_run.txt')}")

    # -------------------------
    # Final evaluation on test set using best checkpoint
    # -------------------------
    if not os.path.isfile(best_ckpt_path):
        raise RuntimeError("Best checkpoint not found; training may have failed to save any checkpoint.")
    print(f"Loading best checkpoint from {best_ckpt_path} for final test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except Exception as e:
        print(f"Warning: model.load_state_dict failed with error: {e}. Attempting strict=False load.")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)

    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, device)

    # --- FIX: compute both weighted (scalar) metrics and per-class (arrays) metrics ---
    # Weighted (single numbers)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(test_targets, test_preds, average='weighted', zero_division=0)
    # Per-class arrays
    per_prec, per_rec, per_f1, support = precision_recall_fscore_support(test_targets, test_preds, average=None, zero_division=0)

    # classification_report (string) and confusion matrix
    per_class_report = classification_report(test_targets, test_preds, target_names=["Attentive", "Distracted", "Disengaged"], zero_division=0)
    conf_mat = confusion_matrix(test_targets, test_preds).tolist()

    # Build JSON-friendly report (convert numpy arrays to native lists)
    def to_pylist(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        try:
            return list(x)
        except Exception:
            return [x]

    test_accuracy = accuracy_score(test_targets, test_preds)
    test_report = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'precision_weighted': float(prec_w),
        'recall_weighted': float(rec_w),
        'f1_weighted': float(f1_w),
        'per_class_precision': [float(x) for x in to_pylist(per_prec)],
        'per_class_recall': [float(x) for x in to_pylist(per_rec)],
        'per_class_f1': [float(x) for x in to_pylist(per_f1)],
        'support': [int(x) for x in to_pylist(support)],
        'confusion_matrix': conf_mat,
        'num_test_samples': int(len(test_targets)),
        'best_val_acc': float(ckpt.get('val_acc', best_val_acc)),
        'args': vars(args),
    }

    with open(os.path.join(args.output_dir, "test_report.json"), "w") as jf:
        json.dump(test_report, jf, indent=2)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as rf:
        rf.write(per_class_report)

    print(f"Test evaluation complete. Results saved to {args.output_dir}:")
    print(json.dumps(test_report, indent=2))
    print("\nClassification report (per-class):\n")
    print(per_class_report)

if __name__ == "__main__":
    main()
