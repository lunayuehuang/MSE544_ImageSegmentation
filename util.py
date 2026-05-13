"""Utility helpers for the MSE544 U-Net image-segmentation tutorial.

Keeps the notebook focused on the learning-oriented cells (dataset class,
losses, training loop) by lifting the mechanical I/O, model definition, and
visualisation glue out of the README. Import what you need, e.g.:

    from util import (
        UNet, decode_predictions, mask_to_rgb,
        pad_to_multiple, predict_image,
        load_multiclass_mask, augment_array, AUG_TAGS, collect_pairs,
    )
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw


# ============================================================
# Label map (kept in sync with Part 1 Step A)
# ============================================================
LABEL_MAP = {
    'defect': 1,
}


# ============================================================
# Dataset-prep helpers (Part 1)
# ============================================================
def load_multiclass_mask(json_path, image_size, label_map=LABEL_MAP):
    """Rasterise LabelMe polygons into an (H, W) uint8 mask (0=bg, 1=defect)."""
    with open(json_path) as f:
        data = json.load(f)

    w, h = image_size
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for s in data['shapes']:
        label = s['label']
        if label not in label_map or s['shape_type'] != 'polygon':
            continue
        pts = [tuple(p) for p in s['points']]
        draw.polygon(pts, fill=label_map[label])

    return np.array(mask, dtype=np.uint8)


# Full augmentation: 6 transforms (original + horizontal/vertical flip + 3 rotations)
AUG_TAGS = ['orig', 'fliplr', 'flipud', 'rot90', 'rot180', 'rot270']


def augment_array(arr, tag):
    """Apply the named flip/rotation to a 2-D image or mask array."""
    if tag == 'fliplr':  return np.fliplr(arr)
    #if tag == 'flipud':  return np.flipud(arr)
    #if tag == 'rot90':   return np.rot90(arr, 1)
    #if tag == 'rot180':  return np.rot90(arr, 2)
    #if tag == 'rot270':  return np.rot90(arr, 3)
    return arr  # 'orig'


def collect_pairs(directory):
    """List image stems in `directory` that have a matching .json label file."""
    pairs = []
    for f in os.listdir(directory):
        if f.endswith('.json'):
            stem = f.replace('.json', '')
            img_path = os.path.join(directory, stem + '.png')
            if os.path.exists(img_path):
                pairs.append(stem)
    pairs.sort(key=lambda x: int(x) if x.isdigit() else x)
    return pairs


# ============================================================
# U-Net model (Part 2 Step D / Part 3 Step B)
# ============================================================
def make_group_norm(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            make_group_norm(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            make_group_norm(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.d1 = DoubleConv(1, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)
        self.d4 = DoubleConv(128, 256, dropout=0.10)
        self.bottleneck = DoubleConv(256, 512, dropout=0.25)

        self.u4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.uconv4 = DoubleConv(512, 256, dropout=0.10)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.uconv3 = DoubleConv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.uconv2 = DoubleConv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.uconv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        bn = self.bottleneck(self.pool(c4))

        x = self.uconv4(torch.cat([self.u4(bn), c4], dim=1))
        x = self.uconv3(torch.cat([self.u3(x), c3], dim=1))
        x = self.uconv2(torch.cat([self.u2(x), c2], dim=1))
        x = self.uconv1(torch.cat([self.u1(x), c1], dim=1))

        return self.out(x)   # (B, num_classes, H, W) logits


# ============================================================
# Prediction / visualisation helpers (Part 2 Steps G/I, Part 3 Step D)
# ============================================================
# 2-class palette: background, defect
PALETTE = np.array([
    [0,   0,   0],     # 0 background - black
    [255, 255, 0],     # 1 defect     - yellow
], dtype=np.uint8)


def decode_predictions(logits, class_thresholds=None):
    """Argmax with per-class confidence thresholds (low-confidence → background)."""
    if class_thresholds is None:
        class_thresholds = {1: 0.50}
    probs = F.softmax(logits, dim=1)
    confidence, preds = probs.max(dim=1)
    preds = preds.clone()
    for class_idx, threshold in class_thresholds.items():
        preds[(preds == class_idx) & (confidence < threshold)] = 0
    return preds


def mask_to_rgb(mask, palette=PALETTE):
    """Colourise an integer mask using the class palette."""
    mask = np.clip(mask, 0, len(palette) - 1)
    return palette[mask]


def pad_to_multiple(x, multiple=16):
    """Reflect-pad a (B,C,H,W) tensor so H and W are multiples of `multiple`."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect'), (h, w)


def predict_image(image_path, model, device, multiple=16):
    """Run full-image inference; returns (display_rgb, pred_mask, defect_prob)."""
    img_tf = T.ToTensor()
    display_image = Image.open(image_path).convert('RGB')
    model_image = display_image.convert('L')
    x = img_tf(model_image).unsqueeze(0).to(device)
    x, (orig_h, orig_w) = pad_to_multiple(x, multiple=multiple)

    with torch.no_grad():
        logits = model(x)
        pred = decode_predictions(logits)[0, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)
        defect_prob = F.softmax(logits, dim=1)[0, 1, :orig_h, :orig_w].cpu().numpy()

    return np.array(display_image), pred, defect_prob


def find_image_path(image_id, source_dir, extensions=('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
    """Locate a test image by stem across the supported extensions."""
    from pathlib import Path
    source_dir = Path(source_dir)
    for ext in extensions:
        candidate = source_dir / f'{image_id}{ext}'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not find image {image_id} in {source_dir}')
