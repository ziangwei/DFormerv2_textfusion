#!/usr/bin/env python3
"""
NaN Loss Diagnostic Script

This script helps diagnose NaN loss issues by checking:
1. Data labels distribution
2. Text features for NaN/Inf values
3. Model outputs for numerical issues

Usage:
    python diagnose_nan.py --config local_configs.SUNRGBD.DFormerv2_B
"""

import argparse
import torch
import numpy as np
from importlib import import_module
from utils.dataloader.dataloader import get_train_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="train config file path")
parser.add_argument("--num-batches", type=int, default=10, help="number of batches to check")
parser.add_argument("--gpus", default=1, type=int, help="used gpu number")

args = parser.parse_args()
config = getattr(import_module(args.config), "C")

print("=" * 80)
print("NaN Loss Diagnostic Tool")
print("=" * 80)

# Mock engine for data loading
class MockEngine:
    def __init__(self):
        self.distributed = False
        self.local_rank = 0
        self.world_size = 1

engine = MockEngine()

# Load data
print("\n[1/4] Loading dataset...")
train_loader, _ = get_train_loader(engine, RGBXDataset, config)
print(f"Dataset loaded. Total batches: {len(train_loader)}")

# Check data labels
print(f"\n[2/4] Checking labels in first {args.num_batches} batches...")
all_background_count = 0
label_stats = {"min": [], "max": [], "unique": [], "background_ratio": []}

for i, minibatch in enumerate(train_loader):
    if i >= args.num_batches:
        break

    labels = minibatch["label"]
    text_feats = minibatch.get("text_features")

    # Label statistics
    bg_mask = (labels == config.background)
    bg_ratio = bg_mask.float().mean().item()
    label_stats["background_ratio"].append(bg_ratio)
    label_stats["min"].append(labels.min().item())
    label_stats["max"].append(labels.max().item())
    label_stats["unique"].append(len(torch.unique(labels)))

    if bg_ratio == 1.0:
        all_background_count += 1
        print(f"  WARNING: Batch {i} has 100% background pixels!")

    # Check text features
    if text_feats is not None:
        has_nan = torch.isnan(text_feats).any().item()
        has_inf = torch.isinf(text_feats).any().item()
        if has_nan:
            print(f"  WARNING: Batch {i} has NaN in text_features!")
        if has_inf:
            print(f"  WARNING: Batch {i} has Inf in text_features!")

print(f"\nLabel Statistics (over {args.num_batches} batches):")
print(f"  Background ratio: mean={np.mean(label_stats['background_ratio']):.2%}, "
      f"max={np.max(label_stats['background_ratio']):.2%}")
print(f"  Label range: [{int(np.min(label_stats['min']))}, {int(np.max(label_stats['max']))}]")
print(f"  Unique labels per batch: mean={np.mean(label_stats['unique']):.1f}")
print(f"  Batches with 100% background: {all_background_count}/{args.num_batches}")

if all_background_count > 0:
    print(f"\n  ⚠️  ISSUE FOUND: {all_background_count} batches have all background pixels!")
    print("     This will cause NaN loss without proper handling.")

# Test model forward pass
print(f"\n[3/4] Testing model forward pass...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)

model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
model = model.to(device)
model.eval()

print(f"Model created and moved to {device}")

print(f"\n[4/4] Running inference on sample batch...")
with torch.no_grad():
    minibatch = next(iter(train_loader))
    imgs = minibatch["data"].to(device)
    gts = minibatch["label"].to(device)
    modal_xs = minibatch["modal_x"].to(device)
    text_feats = minibatch.get("text_features")
    if text_feats is not None:
        text_feats = text_feats.to(device)

    try:
        loss = model(imgs, modal_xs, label=gts, text_features=text_feats)
        print(f"✓ Forward pass successful")
        print(f"  Loss value: {loss.item():.6f}")

        if torch.isnan(loss):
            print(f"  ⚠️  ISSUE FOUND: Loss is NaN!")
        elif torch.isinf(loss):
            print(f"  ⚠️  ISSUE FOUND: Loss is Inf!")
        else:
            print(f"  ✓ Loss is valid (not NaN/Inf)")

    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")

print("\n" + "=" * 80)
print("Diagnostic complete!")
print("=" * 80)

# Summary and recommendations
print("\n📋 Summary and Recommendations:\n")
recommendations = []

if all_background_count > 0:
    recommendations.append(
        "1. Background pixels issue detected. The fix in models/builder.py now handles this by\n"
        "   returning zero loss for all-background batches instead of NaN."
    )

if any(r > 0.95 for r in label_stats["background_ratio"]):
    recommendations.append(
        "2. High background ratio in some batches. Consider:\n"
        "   - Checking your data augmentation pipeline\n"
        "   - Verifying label files are correct\n"
        "   - Adjusting cropping strategy to include more valid pixels"
    )

if not recommendations:
    recommendations.append("✓ No major issues detected in the diagnostic checks!")

for rec in recommendations:
    print(rec)

print("\n💡 The following fixes have been applied:")
print("   ✓ Empty tensor protection in loss calculation (models/builder.py)")
print("   ✓ Text features NaN/Inf checking and fixing (models/builder.py)")
print("   ✓ Gradient clipping added (utils/train.py)")
print("   ✓ Text encoding NaN/Inf protection (utils/prompt_utils.py)")
