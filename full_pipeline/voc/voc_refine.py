#!/usr/bin/env python3
"""
voc_refine.py
==============
Refinement network code shared by the train_* and val_* steps: the two architectures
(UNet+ASPP, UNet+ASPP+DINOv2), DINOv2 features, the training Dataset, the loss, the training loop
and the hybrid evaluation.

This is a port of the audited PASCAL-Context-59 train/val scripts (the VOC and Cityscapes
notebooks used the same logic). The math, the order of RNG-consuming calls and the
module/attribute names (so old checkpoints still load) are unchanged. The only differences are:
paths come from Paths(arm), dataset specifics come from section A of the *_common module, and the
hybrid val reads the SAM3 prediction + coarse masks from val_cache instead of recomputing them.
Identical across the three dataset folders except for the *_common import.
"""

import argparse
import json
import math
import os
import random
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights
from tqdm import tqdm

from voc_common import (
    ARCH_TITLES, CLASS_TO_EVAL_INDEX, DATASET, DATASET_TITLE, EVAL_NUM_CLASSES, ROOT, SEED,
    TRAIN_PARAMS, TRAIN_RAW_VALUE, Paths, PipelineError, add_common_args, banner, cache_key,
    collect_sample, compute_metrics, fast_confusion_matrix, fingerprints, get_ids, gt_path,
    image_path, is_complete, load_gt_eval, load_gt_train_raw, load_prompt_config,
    plot_bar_chart, plot_class_visualizations, prepare_dir, read_split, resolve_out_root,
    set_seed, setup_torch, split_hash, train_valid_mask, write_manifest,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2_MODEL_NAME = "dinov2_vits14"
DINOV2_EMBED_DIM = 384
DINOV2_PATCH_SIZE = 14
DINOV2_INPUT_SIZE = 518
DINOV2_GRID_SIZE = DINOV2_INPUT_SIZE // DINOV2_PATCH_SIZE  # 37
DINOV2_COMPRESS = 32
DINO_SHAPE = (DINOV2_EMBED_DIM, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE)

NUM_WORKERS = 0 if os.name == "nt" else 4


# ============================================================================
# Models -- attribute names and construction order identical to the original scripts
# ============================================================================
class ASPP(nn.Module):
    def __init__(self, in_ch=512, out_ch=256):
        super().__init__()

        def _branch(dilation):
            if dilation == 1:
                return nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
                                 nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.b1 = _branch(1)
        self.b6 = _branch(6)
        self.b12 = _branch(12)
        self.b18 = _branch(18)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                 nn.GroupNorm(32, out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        gap = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([self.b1(x), self.b6(x), self.b12(x), self.b18(x), gap], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


def _build_backbone(total_in, imagenet_init):
    """ResNet-34 with a widened first conv: ch 0-2 = ImageNet RGB weights, every extra channel 0."""
    backbone = tvm.resnet34(weights=ResNet34_Weights.DEFAULT if imagenet_init else None)
    orig_w = backbone.conv1.weight.data.clone()
    new_conv = nn.Conv2d(total_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if imagenet_init:
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_w
            new_conv.weight[:, 3:] = 0.0
    backbone.conv1 = new_conv
    return backbone


class _UNetASPPBody(nn.Module):
    def _init_body(self, backbone, out_channels):
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4
        self.aspp = ASPP(in_ch=512, out_ch=256)
        self.dec4 = DecoderBlock(256, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def _body(self, x, H, W):
        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottle = self.aspp(e4)
        d = self.dec4(bottle, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.final_up(d)
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(d)


class UNetASPP(_UNetASPPBody):
    """ResNet-34 + ASPP + UNet decoder. Input: RGB + N coarse channels."""

    def __init__(self, in_channels_base, out_channels, imagenet_init=True):
        super().__init__()
        self.in_channels_base = in_channels_base
        self._init_body(_build_backbone(in_channels_base, imagenet_init), out_channels)

    def forward(self, input_base):
        H, W = input_base.shape[-2:]
        return self._body(input_base, H, W)


class UNetASPPDINOv2(_UNetASPPBody):
    """UNet+ASPP + frozen DINOv2 ViT-S/14 features: 1x1 conv 384->32, bilinear upsample, concat."""

    def __init__(self, in_channels_base, out_channels, dinov2_dim=DINOV2_EMBED_DIM,
                 dinov2_compress=DINOV2_COMPRESS, imagenet_init=True):
        super().__init__()
        self.in_channels_base = in_channels_base
        self.dinov2_dim = dinov2_dim
        self.dinov2_compress = dinov2_compress
        self.total_in = in_channels_base + dinov2_compress
        self.dino_compress = nn.Sequential(
            nn.Conv2d(dinov2_dim, dinov2_compress, 1, bias=False),
            nn.BatchNorm2d(dinov2_compress),
            nn.ReLU(inplace=True),
        )
        self._init_body(_build_backbone(self.total_in, imagenet_init), out_channels)

    def forward(self, input_base, dino_feat):
        H, W = input_base.shape[-2:]
        dino_u = F.interpolate(self.dino_compress(dino_feat), size=(H, W), mode="bilinear", align_corners=False)
        return self._body(torch.cat([input_base, dino_u], dim=1), H, W)


def uses_dino(arch):
    return arch == "unetasppdinov2"


# ============================================================================
# DINOv2
# ============================================================================
def load_dinov2(device):
    print(f"Loading DINOv2 ({DINOV2_MODEL_NAME})...")
    try:
        model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL_NAME, trust_repo=True)
    except Exception as e:
        print(f"torch.hub.load failed: {e}\nFallback: local repo/checkpoint...")
        import sys
        repos = [ROOT / "weights" / "dinov2-repo", Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"]
        repo = next((p for p in repos if p.exists()), None)
        if repo is None:
            raise PipelineError(f"DINOv2 unavailable: no internet and no local repo in {repos}")
        sys.path.insert(0, str(repo))
        from dinov2.hub.backbones import dinov2_vits14 as _build
        model = _build(pretrained=False)
        weights = [ROOT / "weights" / "dinov2_vits14_pretrain.pth", repo / "dinov2_vits14_pretrain.pth"]
        w = next((p for p in weights if p.exists()), None)
        if w is None:
            raise PipelineError(f"DINOv2 weights not found in {weights}")
        model.load_state_dict(torch.load(w, map_location="cpu"))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"DINOv2 loaded ({sum(p.numel() for p in model.parameters()):,} params, frozen)")
    return model


@torch.no_grad()
def dinov2_features(model, image_pil, device):
    """PIL image -> [1, 384, 37, 37] patch-token feature map on `device`."""
    img_r = image_pil.convert("RGB").resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.BILINEAR)
    x = TF.normalize(TF.to_tensor(img_r), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    tokens = model.forward_features(x)["x_norm_patchtokens"]
    B, N, D = tokens.shape
    assert N == DINOV2_GRID_SIZE * DINOV2_GRID_SIZE and D == DINOV2_EMBED_DIM
    return tokens.transpose(1, 2).reshape(B, D, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE)


def _dino_ok(path):
    try:
        with np.load(path) as d:
            return d["features"].shape == DINO_SHAPE
    except Exception:
        return False


def build_dinov2_cache(train_ids, val_ids, cache_dir, device):
    """Shared by both arms (features depend on the image only). Stored fp16 [384, 37, 37].
    The model is always loaded, as in the original, so the RNG state downstream is the same."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = train_ids + val_ids
    model = load_dinov2(device)
    n_cached = n_new = 0
    errors = []
    for img_id in tqdm(targets, desc="Build DINOv2 cache"):
        out = cache_dir / f"{cache_key(img_id)}.npz"
        if out.exists() and _dino_ok(out):
            n_cached += 1
            continue
        try:
            feat = dinov2_features(model, Image.open(image_path(img_id, "train")), device)
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, features=feat.squeeze(0).to(torch.float16).cpu().numpy())
            tmp.replace(out)
            n_new += 1
        except Exception as e:
            errors.append((img_id, str(e)))
    print(f"DINOv2 cache: {n_cached} already existed, {n_new} new, {len(errors)} errors -> {cache_dir}")
    for e in errors[:5]:
        print(f"  {e}")
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ok = lambda i: _dino_ok(cache_dir / f"{cache_key(i)}.npz")
    train_ids = [i for i in train_ids if ok(i)]
    val_ids = [i for i in val_ids if ok(i)]
    print(f"Final internal-train: {len(train_ids)}   internal-val: {len(val_ids)}")
    return train_ids, val_ids


# ============================================================================
# Training data
# ============================================================================
class TargetSpec:
    """Everything derived from the target class list (picklable, for DataLoader workers)."""

    def __init__(self, target_classes):
        self.classes = list(target_classes)
        self.n = len(self.classes)
        self.raw_value = {c: TRAIN_RAW_VALUE[c] for c in self.classes}
        self.local = {c: i + 1 for i, c in enumerate(self.classes)}
        self.local_to_class = {v: k for k, v in self.local.items()}
        self.num_out = self.n + 1  # ch0 = other/bg


def build_augment(image_h, image_w):
    """RandomScale -> PadIfNeeded(0) -> RandomCrop -> ColorJitter; hflip done manually (DINOv2 sync).
    Pad value 0 for image and masks: that is what albumentations>=2.0 actually applied in the
    original runs (their `value=`/`mask_value=` arguments are ignored by 2.x)."""
    import albumentations as A
    major = int(A.__version__.split(".")[0])
    if major >= 2:
        pad = A.PadIfNeeded(min_height=image_h, min_width=image_w, border_mode=cv2.BORDER_CONSTANT,
                            fill=0, fill_mask=0, p=1.0)
    else:
        pad = A.PadIfNeeded(min_height=image_h, min_width=image_w, border_mode=cv2.BORDER_CONSTANT,
                            value=0, mask_value=0, p=1.0)
    return A.Compose([
        A.RandomScale(scale_limit=(-0.25, 0.25), p=0.7),
        pad,
        A.RandomCrop(height=image_h, width=image_w, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.6),
    ])


def compute_boundary_weight(target_np, alpha=1.5):
    gt = target_np.astype(bool)
    if not gt.any() or gt.all():
        return np.ones(target_np.shape, dtype=np.float32)
    dist = (distance_transform_edt(gt) + distance_transform_edt(~gt)).astype(np.float32)
    w = 1.0 / (dist + 1.0) ** alpha
    w_max = w.max()
    return (w / w_max).astype(np.float32) if w_max > 0 else w


class RefineDataset(Dataset):
    """input_base [3+N,H,W], dino_feat [384,37,37] (DINOv2 arch only), target_label [H,W] (0 = other),
    valid [1,H,W], coarses [N,H,W], bnd_w [N,H,W]."""

    def __init__(self, image_ids, spec, coarse_dir, image_h, image_w, dino_dir=None, aug=None):
        self.image_ids = image_ids
        self.spec = spec
        self.coarse_dir = Path(coarse_dir)
        self.dino_dir = Path(dino_dir) if dino_dir else None
        self.image_h, self.image_w = image_h, image_w
        self.aug = aug

    def __len__(self):
        return len(self.image_ids)

    def _build_label_map(self, raw):
        label = np.zeros(raw.shape, dtype=np.uint8)
        for c in self.spec.classes:
            label[raw == self.spec.raw_value[c]] = self.spec.local[c]
        return label

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        H, W, N = self.image_h, self.image_w, self.spec.n
        image_np = np.array(Image.open(image_path(img_id, "train")).convert("RGB"))
        raw = load_gt_train_raw(img_id, "train")
        with np.load(self.coarse_dir / f"{cache_key(img_id)}.npz") as d:
            coarses_np = np.stack([d[c].astype(np.uint8) for c in self.spec.classes], axis=0)
        dino_feat = None
        if self.dino_dir is not None:
            with np.load(self.dino_dir / f"{cache_key(img_id)}.npz") as d:
                dino_feat = d["features"].astype(np.float32)

        image_np = cv2.resize(image_np, (W, H), interpolation=cv2.INTER_LINEAR)
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
        coarses_np = np.stack([cv2.resize(coarses_np[c], (W, H), interpolation=cv2.INTER_NEAREST)
                               for c in range(N)], axis=0)
        valid_np = train_valid_mask(raw)
        label_np = self._build_label_map(raw)

        masks = [label_np, valid_np] + [coarses_np[c] for c in range(N)]
        if self.aug is not None:
            if random.random() < 0.5:
                image_np = image_np[:, ::-1].copy()
                masks = [m[:, ::-1].copy() for m in masks]
                if dino_feat is not None:
                    dino_feat = dino_feat[:, :, ::-1].copy()
            out = self.aug(image=image_np, masks=masks)
            image_np = out["image"]
            label_np, valid_np = out["masks"][0], out["masks"][1]
            coarses_np = np.stack(out["masks"][2:2 + N], axis=0)

        label_np = np.asarray(label_np, dtype=np.uint8)
        valid_np = (np.asarray(valid_np) > 0).astype(np.float32)
        coarses_np = (np.asarray(coarses_np) > 0).astype(np.float32)
        bnd_w = np.zeros((N, H, W), dtype=np.float32)
        for c in range(N):
            bnd_w[c] = compute_boundary_weight((label_np == c + 1).astype(np.uint8))

        rgb = TF.normalize(TF.to_tensor(image_np), IMAGENET_MEAN, IMAGENET_STD)
        coarses_t = torch.from_numpy(coarses_np)
        item = {
            "input_base": torch.cat([rgb, coarses_t], dim=0),
            "target_label": torch.from_numpy(label_np.astype(np.int64)),
            "valid": torch.from_numpy(valid_np).unsqueeze(0),
            "coarses": coarses_t,
            "bnd_w": torch.from_numpy(bnd_w),
            "image_id": img_id,
        }
        if dino_feat is not None:
            item["dino_feat"] = torch.from_numpy(dino_feat.astype(np.float32))
        return item


# ============================================================================
# Loss + metrics
# ============================================================================
class RefineLoss:
    """0.5 * focal CE + 0.3 * per-class Tversky + 0.2 * per-class boundary BCE."""

    def __init__(self, n_targets, ce_weights, tversky, boundary_lambda, focal_gamma):
        self.n = n_targets
        self.ce_weights = ce_weights
        self.tversky = tversky
        self.boundary_lambda = boundary_lambda
        self.focal_gamma = focal_gamma

    def masked_ce(self, logits, target_label, valid):
        ce = F.cross_entropy(logits, target_label, weight=self.ce_weights, reduction="none")
        if self.focal_gamma > 0:
            log_pt = F.log_softmax(logits, dim=1).gather(1, target_label.unsqueeze(1)).squeeze(1)
            ce = ce * (1.0 - log_pt.exp().clamp(0.0, 1.0)).pow(self.focal_gamma)
        v = valid.squeeze(1)
        return (ce * v).sum() / v.sum().clamp_min(1.0)

    def per_class(self, probs, target_label, valid, bnd_w):
        v = valid.squeeze(1).float()
        eps = 1.0
        dice_sum = torch.tensor(0.0, device=probs.device)
        bnd_sum = torch.tensor(0.0, device=probs.device)
        for c_idx in range(self.n):
            local_id = c_idx + 1
            p = probs[:, local_id] * v
            t = (target_label == local_id).float() * v
            alpha, beta = self.tversky[c_idx, 0], self.tversky[c_idx, 1]
            tp = (p * t).sum(dim=(1, 2))
            fp = (p * (1 - t)).sum(dim=(1, 2))
            fn = ((1 - p) * t).sum(dim=(1, 2))
            dice_sum = dice_sum + (1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()
            p_c = torch.clamp(p, 1e-7, 1.0 - 1e-7)
            bce = -(t * torch.log(p_c) + (1.0 - t) * torch.log(1.0 - p_c))
            wmap = bnd_w[:, c_idx] * v
            bnd_sum = bnd_sum + self.boundary_lambda[c_idx] * (bce * wmap).sum() / wmap.sum().clamp_min(1.0)
        return dice_sum / self.n, bnd_sum / self.n

    def __call__(self, logits, target_label, valid, bnd_w):
        ce = self.masked_ce(logits, target_label, valid)
        dice, bnd = self.per_class(F.softmax(logits, dim=1), target_label, valid, bnd_w)
        return 0.5 * ce + 0.3 * dice + 0.2 * bnd, ce.detach(), dice.detach(), bnd.detach()


def _iou(tp, fp, fn):
    d = tp + fp + fn
    return float(tp / d) if d > 0 else float("nan")


def _update_metrics(mdict, classes, pred_label, target_label, valid_bool, coarses_bool):
    for c_idx, cls in enumerate(classes):
        local_id = c_idx + 1
        p = (pred_label == local_id) & valid_bool
        t = (target_label == local_id) & valid_bool
        c = coarses_bool[:, c_idx] & valid_bool
        m = mdict[cls]
        m["tp"] += int((p & t).sum().item())
        m["fp"] += int((p & ~t).sum().item())
        m["fn"] += int((~p & t).sum().item())
        m["c_tp"] += int((c & t).sum().item())
        m["c_fp"] += int((c & ~t).sum().item())
        m["c_fn"] += int((~c & t).sum().item())


def _summarize(mdict, classes):
    per_iou = {c: _iou(mdict[c]["tp"], mdict[c]["fp"], mdict[c]["fn"]) for c in classes}
    per_coarse = {c: _iou(mdict[c]["c_tp"], mdict[c]["c_fp"], mdict[c]["c_fn"]) for c in classes}
    miou = float(np.nanmean(list(per_iou.values())))
    coarse_miou = float(np.nanmean(list(per_coarse.values())))
    return {"per_iou": per_iou, "per_coarse": per_coarse, "miou": miou,
            "coarse_miou": coarse_miou, "delta": miou - coarse_miou}


# ============================================================================
# Training step
# ============================================================================
def _loss_params(spec, device):
    tv = TRAIN_PARAMS["tversky"]
    bl = TRAIN_PARAMS["boundary_lambda"]
    boost = TRAIN_PARAMS["ce_boost"]
    for name, d in (("tversky", tv), ("boundary_lambda", bl), ("ce_boost", boost)):
        extra = sorted(set(d) - set(spec.classes))
        if extra:
            print(f"[warn] TRAIN_PARAMS['{name}'] has classes that are not targets (ignored): {extra}")
    tversky = {c: tuple(tv.get(c, TRAIN_PARAMS["default_tversky"])) for c in spec.classes}
    boundary = {c: float(bl.get(c, TRAIN_PARAMS["default_boundary_lambda"])) for c in spec.classes}
    ce_boost = {c: float(b) for c, b in boost.items() if c in spec.classes}
    return tversky, boundary, ce_boost


def _filter_ready(ids, spec, coarse_dir, tag):
    ok, skipped = [], []
    for img_id in tqdm(ids, desc=f"Validate {tag}"):
        if not image_path(img_id, "train").exists():
            skipped.append((img_id, "missing_image"))
            continue
        try:
            with np.load(coarse_dir / f"{cache_key(img_id)}.npz") as d:
                good = set(spec.classes) <= set(d.files)
        except Exception:
            good = False
        if not good:
            skipped.append((img_id, "missing_or_bad_coarse_npz"))
            continue
        if gt_path(img_id, "train") is None:
            skipped.append((img_id, "missing_gt"))
            continue
        ok.append(img_id)
    print(f"  {tag}: usable {len(ok)}/{len(ids)}")
    for s in skipped[:5]:
        print(f"    skipped: {s}")
    return ok


def _compute_ce_weights(train_ids, spec, ce_boost, device):
    cls_pixels = {c: 0 for c in range(spec.num_out)}
    presence = {c: 0 for c in spec.classes}
    for img_id in tqdm(train_ids, desc="Scan GT for CE weights"):
        raw = load_gt_train_raw(img_id, "train")
        valid = train_valid_mask(raw).astype(bool)
        v_total = int(valid.sum())
        target_px = 0
        for c in spec.classes:
            px = int(((raw == spec.raw_value[c]) & valid).sum())
            cls_pixels[spec.local[c]] += px
            target_px += px
            if px > 0:
                presence[c] += 1
        cls_pixels[0] += v_total - target_px

    total = sum(cls_pixels.values())
    print("Pixel counts (internal-train):")
    for c in range(spec.num_out):
        name = "other/bg" if c == 0 else spec.local_to_class[c]
        img_info = "" if c == 0 else f"  in {presence[name]} images"
        print(f"  ch {c:2d} ({name:14s}): {cls_pixels[c]:>13,d} px ({100.0 * cls_pixels[c] / max(total, 1):5.2f}%){img_info}")

    freq = np.array([cls_pixels[c] for c in range(spec.num_out)], dtype=np.float64)
    inv = 1.0 / np.clip(freq, 1, None)
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.2, 5.0)
    inv = inv / inv.mean()
    if ce_boost:
        for c, b in ce_boost.items():
            inv[spec.local[c]] *= b
        inv = inv / inv.mean()
    print(f"CE class weights (clipped [0.2, 5.0], boosts {ce_boost or 'none'}, mean=1):")
    for c in range(spec.num_out):
        name = "other/bg" if c == 0 else spec.local_to_class[c]
        print(f"  ch {c:2d} ({name:14s}): {inv[c]:.3f}")
    return torch.tensor(inv, dtype=torch.float32, device=device)


def train_main(arch):
    p = add_common_args(argparse.ArgumentParser(description=f"Train {ARCH_TITLES[arch]} ({DATASET})"))
    p.add_argument("--epochs", type=int, default=None, help="override the epoch count (smoke tests)")
    args = p.parse_args()

    out_root = resolve_out_root(args)
    paths = Paths(args.arm, out_root)
    pc = load_prompt_config(args.arm)
    fps = fingerprints(args.arm, args.limit)
    fp = fps[f"train:{arch}"]
    ckpt_dir = paths.ckpt_dir(arch)
    title = ARCH_TITLES[arch]

    banner(f"TRAIN {title} -- {DATASET_TITLE}, arm={args.arm}")
    if is_complete(ckpt_dir, fp) and (ckpt_dir / "best.pth").exists() and not args.force:
        print(f"[skip] already complete: {ckpt_dir}")
        return
    if not is_complete(paths.coarse_cache, fps["coarse"]):
        raise PipelineError(f"coarse cache for arm '{args.arm}' is missing or outdated: {paths.coarse_cache}\n"
                            f"  run get_coarse_{DATASET}.py --arm {args.arm} first")
    if not paths.split_json.exists():
        raise PipelineError(f"{paths.split_json} not found -- run get_coarse_{DATASET}.py first")
    prepare_dir(ckpt_dir, "train", fp, force=args.force)

    P = TRAIN_PARAMS
    H, W = P["image_h"], P["image_w"]
    epochs = args.epochs or P["epochs"]
    warmup, patience = P["warmup_epochs"], P["patience"]
    spec = TargetSpec(pc.target_classes)
    tversky, boundary, ce_boost = _loss_params(spec, None)

    device = setup_torch(global_bf16_autocast=False)
    set_seed(SEED)
    print(f"Targets (local 1..{spec.n}): {spec.classes}")
    print(f"Image {H}x{W}  batch {P['batch_size']}  epochs {epochs} (warmup {warmup} + cosine)  patience {patience}")
    print(f"Tversky: {tversky}\nBoundary lambda: {boundary}\nFocal gamma: {P['focal_gamma']}")

    split_tr, split_va = read_split(paths.split_json)
    all_train = set(get_ids("train", args.limit))
    tr_ids = [i for i in split_tr if i in all_train]
    va_ids = [i for i in split_va if i in all_train]
    print(f"Internal train/val from split.json: {len(tr_ids)} / {len(va_ids)}")
    tr_ids = _filter_ready(tr_ids, spec, paths.coarse_cache, "internal-train")
    va_ids = _filter_ready(va_ids, spec, paths.coarse_cache, "internal-val")
    if not tr_ids:
        raise PipelineError("no usable internal-train images")

    dino = uses_dino(arch)
    if dino:
        tr_ids, va_ids = build_dinov2_cache(tr_ids, va_ids, paths.dinov2_cache, device)

    ce_weights = _compute_ce_weights(tr_ids, spec, ce_boost, device)
    loss_fn = RefineLoss(
        spec.n, ce_weights,
        torch.tensor([tversky[c] for c in spec.classes], dtype=torch.float32, device=device),
        torch.tensor([boundary[c] for c in spec.classes], dtype=torch.float32, device=device),
        P["focal_gamma"])

    dino_dir = paths.dinov2_cache if dino else None
    train_ds = RefineDataset(tr_ids, spec, paths.coarse_cache, H, W, dino_dir, aug=build_augment(H, W))
    val_ds = RefineDataset(va_ids, spec, paths.coarse_cache, H, W, dino_dir, aug=None)
    b = train_ds[0]
    assert b["input_base"].shape == (3 + spec.n, H, W)
    print(f"Sample[0] {b['image_id']}: input_base {tuple(b['input_base'].shape)}, "
          f"labels {sorted(b['target_label'].unique().tolist())}, valid fg {float(b['valid'].mean()):.4f}")

    if dino:
        model = UNetASPPDINOv2(3 + spec.n, spec.num_out).to(device)
    else:
        model = UNetASPP(3 + spec.n, spec.num_out).to(device)
    print(f"{type(model).__name__}: in {3 + spec.n} (+{DINOV2_COMPRESS} DINOv2)" if dino else
          f"{type(model).__name__}: in {3 + spec.n}", f"| out {spec.num_out} | "
          f"trainable params {sum(q.numel() for q in model.parameters() if q.requires_grad):,}")
    with torch.no_grad():
        _x = torch.zeros(1, 3 + spec.n, H, W, device=device)
        _y = model(_x, torch.zeros(1, *DINO_SHAPE, device=device)) if dino else model(_x)
        assert _y.shape == (1, spec.num_out, H, W)
    del _x, _y

    pin = device.type == "cuda"
    kw = dict(batch_size=P["batch_size"], num_workers=NUM_WORKERS, pin_memory=pin, drop_last=False)
    if NUM_WORKERS > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    _b = next(iter(train_loader))
    assert _b["input_base"].shape[1] == 3 + spec.n
    del _b
    print(f"Train batches: {len(train_loader)} ({len(train_ds)} imgs, aug) | Val batches: {len(val_loader)} ({len(val_ds)} imgs)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.float16)) if use_amp else nullcontext

    def forward(batch):
        x = batch["input_base"].to(device, non_blocking=True)
        if dino:
            return model(x, batch["dino_feat"].to(device, non_blocking=True))
        return model(x)

    ckpt_meta = dict(
        dataset=DATASET, arm=args.arm, arch=arch, fingerprint=fp, prompts=pc.as_dict(),
        target_classes=spec.classes, target_to_raw_value=spec.raw_value, target_to_local=spec.local,
        boundary_lambda=boundary, tversky_params=tversky, ce_boost=ce_boost,
        focal_gamma=P["focal_gamma"], in_channels_base=3 + spec.n, num_out_classes=spec.num_out,
        image_h=H, image_w=W, ce_weights=ce_weights.detach().cpu().numpy().tolist(),
        arch_version=f"multi_class_softmax_{DATASET}_{arch}_{args.arm}_v1",
    )
    if dino:
        ckpt_meta.update(in_channels_total=3 + spec.n + DINOV2_COMPRESS, dinov2_model_name=DINOV2_MODEL_NAME,
                         dinov2_embed_dim=DINOV2_EMBED_DIM, dinov2_patch_size=DINOV2_PATCH_SIZE,
                         dinov2_input_size=DINOV2_INPUT_SIZE, dinov2_grid_size=DINOV2_GRID_SIZE,
                         dinov2_compress=DINOV2_COMPRESS)

    history, best_miou, best_epoch, since_best = [], -1.0, -1, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tr = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "bnd": 0.0}
        tr_n = 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs} [train]", leave=False)
        for batch in bar:
            tgt = batch["target_label"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            bnd_w = batch["bnd_w"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = forward(batch)
                loss, ce_d, dice_d, bnd_d = loss_fn(logits, tgt, valid, bnd_w)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            bs = tgt.size(0)
            for k, v in (("loss", loss), ("ce", ce_d), ("dice", dice_d), ("bnd", bnd_d)):
                tr[k] += v.item() * bs
            tr_n += bs
            bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()

        model.eval()
        mdict = {c: {"tp": 0, "fp": 0, "fn": 0, "c_tp": 0, "c_fp": 0, "c_fn": 0} for c in spec.classes}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{epochs} [val  ]", leave=False):
                tgt = batch["target_label"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                coarses = batch["coarses"].to(device, non_blocking=True)
                with autocast_ctx():
                    logits = forward(batch)
                _update_metrics(mdict, spec.classes, logits.argmax(dim=1), tgt,
                                valid.squeeze(1) >= 0.5, coarses >= 0.5)

        n = max(tr_n, 1)
        s = _summarize(mdict, spec.classes)
        history.append(dict(epoch=epoch, train_loss=tr["loss"] / n, ce=tr["ce"] / n, dice=tr["dice"] / n,
                            bnd=tr["bnd"] / n, miou=s["miou"], coarse_miou=s["coarse_miou"], delta=s["delta"],
                            lr=float(optimizer.param_groups[0]["lr"]),
                            per_iou={k: float(v) for k, v in s["per_iou"].items()},
                            per_coarse={k: float(v) for k, v in s["per_coarse"].items()}))
        print(f"\nEpoch {epoch:2d}/{epochs}  loss {tr['loss'] / n:.4f} (ce {tr['ce'] / n:.4f} dice {tr['dice'] / n:.4f} "
              f"bnd {tr['bnd'] / n:.4f})\n  [internal-val] coarse mIoU {s['coarse_miou']:.4f} -> refined "
              f"{s['miou']:.4f} ({s['delta']:+.4f})")
        for c in spec.classes:
            ci, ri = s["per_coarse"][c], s["per_iou"][c]
            print(f"    {c:14s}: {ci:.4f} -> {ri:.4f} ({ri - ci:+.4f})")

        ckpt = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(), epoch=epoch, miou=s["miou"],
                    coarse_miou=s["coarse_miou"], **ckpt_meta)
        torch.save(ckpt, ckpt_dir / "last.pth")
        if s["miou"] > best_miou:
            best_miou, best_epoch, since_best = s["miou"], epoch, 0
            torch.save(ckpt, ckpt_dir / "best.pth")
            print(f"  BEST checkpoint (epoch {epoch}, mIoU {best_miou:.4f})")
        else:
            since_best += 1
            print(f"  no improvement ({since_best}/{patience}, best epoch {best_epoch}, mIoU {best_miou:.4f})")
        with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if since_best >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nTraining done | best epoch {best_epoch} | internal-val mIoU {best_miou:.4f} | "
          f"{(time.time() - t0) / 60:.1f} min")
    _plot_curves(history, best_epoch, best_miou, spec.classes, title, ckpt_dir / "training_curves.png")
    write_manifest(ckpt_dir, "train", fp, "complete", arch=arch, best_epoch=best_epoch,
                   best_internal_val_miou=best_miou, epochs_run=len(history), n_train=len(tr_ids),
                   n_internal_val=len(va_ids), split_sha=split_hash(paths.split_json))


def _plot_curves(history, best_epoch, best_miou, classes, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not history:
        return
    ep = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for key, style, label in (("train_loss", "o-", "Total"), ("ce", "s--", "Focal CE"),
                              ("dice", "^--", "Tversky"), ("bnd", "d--", "Boundary")):
        axes[0].plot(ep, [r[key] for r in history], style, label=label)
    axes[0].set_title("Training losses")
    axes[1].plot(ep, [r["coarse_miou"] for r in history], "o-", label="Coarse (SAM3)", color="#d9534f")
    axes[1].plot(ep, [r["miou"] for r in history], "s-", label=f"Refined ({title})", color="#2ca02c")
    axes[1].set_title(f"Internal-val mIoU ({len(classes)} targets)")
    for c in classes:
        axes[2].plot(ep, [r["per_iou"][c] for r in history], "-", label=c, alpha=0.8)
    axes[2].set_title("Per-class IoU (refined)")
    axes[2].legend(fontsize=6, ncol=2)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    plt.suptitle(f"{title} {DATASET_TITLE} -- best epoch {best_epoch} | internal-val mIoU {best_miou:.4f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================================
# Hybrid val step
# ============================================================================
def val_main(arch):
    p = add_common_args(argparse.ArgumentParser(description=f"Hybrid val SAM3 + {ARCH_TITLES[arch]} ({DATASET})"))
    p.add_argument("--ckpt", default=None,
                   help="evaluate this checkpoint instead of outputs/<arm>/<arch>/checkpoint/best.pth "
                        "(e.g. an old one, to compare); results go to <arch>/val_external/")
    args = p.parse_args()

    out_root = resolve_out_root(args)
    paths = Paths(args.arm, out_root)
    fps = fingerprints(args.arm, args.limit)
    title = ARCH_TITLES[arch]
    hybrid_label = f"SAM3 + {title} (hybrid)"

    if args.ckpt:
        ckpt_path = Path(args.ckpt).resolve()
        out_dir = paths.arm_dir / arch / "val_external"
        fp = f"external:{ckpt_path}:{ckpt_path.stat().st_mtime if ckpt_path.exists() else 0}"
    else:
        ckpt_path = paths.best_ckpt(arch)
        out_dir = paths.val_dir(arch)
        fp = fps[f"val:{arch}"]
        if not is_complete(paths.ckpt_dir(arch), fps[f"train:{arch}"]):
            raise PipelineError(f"no finished training for {arch}/{args.arm}: {paths.ckpt_dir(arch)}")

    banner(f"HYBRID VAL SAM3 + {title} -- {DATASET_TITLE}, arm={args.arm}")
    if is_complete(out_dir, fp) and not args.force:
        print(f"[skip] already complete: {out_dir}")
        return
    if not is_complete(paths.val_cache, fps["baseline"]):
        raise PipelineError(f"val cache for arm '{args.arm}' is missing or outdated: {paths.val_cache}\n"
                            f"  run sam3_baseline_{DATASET}.py --arm {args.arm} first")
    if not ckpt_path.exists():
        raise PipelineError(f"checkpoint not found: {ckpt_path}")
    prepare_dir(out_dir, "val", fp, force=args.force)

    device = setup_torch(global_bf16_autocast=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    targets = list(ckpt["target_classes"])
    local = ckpt["target_to_local"]
    n_out, in_base = ckpt["num_out_classes"], ckpt["in_channels_base"]
    H, W = ckpt["image_h"], ckpt["image_w"]
    dino = "dinov2_embed_dim" in ckpt
    if dino != uses_dino(arch):
        raise PipelineError(f"{ckpt_path} is {'a DINOv2' if dino else 'a non-DINOv2'} checkpoint, but --arch {arch}")
    if dino:
        model = UNetASPPDINOv2(in_base, n_out, ckpt["dinov2_embed_dim"], ckpt["dinov2_compress"], imagenet_init=False)
    else:
        model = UNetASPP(in_base, n_out, imagenet_init=False)
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for q in model.parameters():
        q.requires_grad = False
    local_to_global = {lid: CLASS_TO_EVAL_INDEX[name] for name, lid in local.items()}
    print(f"Checkpoint: {ckpt_path}\n  arch_version {ckpt.get('arch_version')} | epoch {ckpt.get('epoch')} | "
          f"internal-val mIoU {ckpt.get('miou', float('nan')):.4f}\n  targets ({len(targets)}): {targets}\n  image {H}x{W}")
    if "prompts" in ckpt:
        print(f"  trained with prompts from: {ckpt['prompts'].get('source')}")
    dino_model = load_dinov2(device) if dino else None

    val_ids = get_ids("val", args.limit)
    conf_mat = np.zeros((EVAL_NUM_CLASSES, EVAL_NUM_CLASSES), dtype=np.int64)
    class_samples = {}
    skipped, no_cache, errors = [], [], []
    t_start = time.time()
    with torch.no_grad():
        for img_id in tqdm(val_ids, desc=f"Hybrid val {arch} ({args.arm})", dynamic_ncols=True):
            ip = image_path(img_id, "val")
            if not ip.exists() or gt_path(img_id, "val") is None:
                skipped.append(img_id)
                continue
            cpath = paths.val_cache / f"{cache_key(img_id)}.npz"
            try:
                with np.load(cpath) as d:
                    sam3_pred = d["sam3_pred"]
                    coarse = {c: d[f"coarse__{c}"] for c in targets}
            except Exception:
                no_cache.append(img_id)
                continue
            try:
                gt = load_gt_eval(img_id, "val")
                image = Image.open(ip).convert("RGB")
                width, height = image.size
                image_resized = cv2.resize(np.array(image), (W, H), interpolation=cv2.INTER_LINEAR)
                coarse_stack = np.stack([cv2.resize(coarse[c], (W, H), interpolation=cv2.INTER_NEAREST)
                                         for c in targets], axis=0).astype(np.float32)
                rgb_t = TF.normalize(TF.to_tensor(image_resized), IMAGENET_MEAN, IMAGENET_STD)
                x = torch.cat([rgb_t, torch.from_numpy(coarse_stack)], dim=0).unsqueeze(0).to(device)
                logits = model(x, dinov2_features(dino_model, image, device)) if dino else model(x)
                local_small = logits.argmax(dim=1).squeeze(0).byte().cpu().numpy()
                local_pred = cv2.resize(local_small, (width, height), interpolation=cv2.INTER_NEAREST)
                pred = sam3_pred.copy()
                for lid, gidx in local_to_global.items():
                    pred[local_pred == lid] = gidx
                conf_mat += fast_confusion_matrix(gt, pred)
                collect_sample(class_samples, img_id, ip, gt, pred)
            except Exception:
                errors.append(img_id)
                tqdm.write(f"ERROR {img_id}:")
                traceback.print_exc()

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min | skipped {len(skipped)} (image/GT missing) | "
          f"no val-cache entry {len(no_cache)} | errors {len(errors)}")
    if no_cache:
        print(f"[warn] images without a val_cache entry are NOT evaluated (they failed in the SAM3 pass too): {no_cache[:10]}")

    run_title = f"{DATASET_TITLE} val -- SAM3 + {title} hybrid ({len(targets)} targets), {args.arm}"
    iou, class_df, miou = compute_metrics(conf_mat, out_dir, run_title, target_classes=targets, hybrid_label=hybrid_label)
    plot_bar_chart(class_df, miou, out_dir / "per_class_iou_bar_chart.png", run_title, hybrid_label=hybrid_label)
    plot_class_visualizations(class_samples, iou, out_dir / "class_visualizations", f"Prediction ({hybrid_label})",
                              target_classes=targets, hybrid_tag=hybrid_label)
    if errors:
        raise RuntimeError(f"{len(errors)} images failed during hybrid val -- metrics are incomplete; re-run.")
    write_manifest(out_dir, "val", fp, "complete", arch=arch, checkpoint=str(ckpt_path),
                   checkpoint_epoch=ckpt.get("epoch"), n_evaluated=len(val_ids) - len(skipped) - len(no_cache),
                   skipped=skipped, no_cache=no_cache, miou=miou)
