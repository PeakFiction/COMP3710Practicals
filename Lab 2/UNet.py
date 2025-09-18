# train_unet.py
# COMP3710 Lab 2 — Part 4.3.2 (UNet)
# Zero-dep beyond: torch, numpy, PIL
# Trains on OASIS PNGs under /home/groups/comp3710/OASIS:
#   keras_png_slices_train/              (images)
#   keras_png_slices_seg_train/          (masks)
#   keras_png_slices_validate/           (images)
#   keras_png_slices_seg_validate/       (masks)
#   keras_png_slices_test/               (images)        [unused for training]
#   keras_png_slices_seg_test/           (masks)         [unused for training]
#
# If *validate* split is missing, falls back to random split of the train set.

import os
import re
import glob
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# CLI
# ---------------------------

def get_args():
    p = argparse.ArgumentParser(description="UNet on OASIS PNG slices")
    p.add_argument("--data-dir", type=str, default="/home/groups/comp3710/OASIS",
                   help="OASIS root or split dir. Default: /home/groups/comp3710/OASIS")
    p.add_argument("--outdir", type=str, default="./unet_outputs")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.2, help="Used only if validate split is missing")
    p.add_argument("--limit", type=int, default=0, help="Debug limit on total training pairs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    return p.parse_args()

# ---------------------------
# FS utils
# ---------------------------

EXTS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _list_images(root_dir: str, recursive: bool = True) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    out = []
    if recursive:
        for rp, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(EXTS):
                    out.append(os.path.join(rp, f))
    else:
        for ext in EXTS:
            out += glob.glob(os.path.join(root_dir, f"*{ext}"))
    out.sort()
    return out

def _strip_mask_suffix(stem: str) -> str:
    # Remove ONE trailing block like _seg, -seg, _mask2, _label_3
    return re.sub(r'(?i)(?:[_-])(seg|mask|label)(?:[_-]?\d+)?$', '', stem)

def _normalize_key(path: str) -> str:
    s = Path(path).stem           # e.g., "case_001_slice_0.nii"
    s = s.replace(".nii", "")     # -> "case_001_slice_0"
    # drop leading "case_" or "seg_"
    s = re.sub(r'(?i)^(case|seg)[_-]', '', s)
    # also strip any trailing mask-like suffixes if present
    s = re.sub(r'(?i)(?:[_-])(seg|mask|label)(?:[_-]?\d+)?$', '', s)
    return s.lower()


def _split_dirs_from_root(root: str) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Return mapping for splits -> (img_dir, mask_dir).
    Accepts both *_seg_* and *_mask variants. Does not assume flat structure; only checks top-level names.
    """
    def cand(*names):
        for n in names:
            p = os.path.join(root, n)
            if os.path.isdir(p):
                return p
        return None

    train_img = cand("keras_png_slices_train")
    val_img   = cand("keras_png_slices_validate", "keras_png_slices_val", "keras_png_slices_valid", "keras_png_slices_validation")
    test_img  = cand("keras_png_slices_test")

    train_msk = cand("keras_png_slices_seg_train", "keras_png_slices_train_mask")
    val_msk   = cand("keras_png_slices_seg_validate", "keras_png_slices_validate_mask", "keras_png_slices_seg_val")
    test_msk  = cand("keras_png_slices_seg_test", "keras_png_slices_test_mask")

    return {
        "train": (train_img, train_msk),
        "validate": (val_img, val_msk),
        "test": (test_img, test_msk),
    }

def _pair_split(img_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    imgs  = _list_images(img_dir, recursive=True)
    masks = _list_images(mask_dir, recursive=True)
    if not imgs or not masks:
        return []
    mindex = {_normalize_key(m): m for m in masks}
    pairs = []
    misses = 0
    for ip in imgs:
        key = _normalize_key(ip)
        mp = mindex.get(key)
        if mp:
            pairs.append((ip, mp))
        else:
            # allow exact stem fallback
            mp2 = mindex.get(Path(ip).stem.lower())
            if mp2:
                pairs.append((ip, mp2))
            else:
                misses += 1
    # Diagnostic if bad pairing
    if len(pairs) == 0:
        print(f"[PAIR-DEBUG] ZERO pairs in split with img_dir={img_dir} (#imgs={len(imgs)}) mask_dir={mask_dir} (#masks={len(masks)})")
        print("[PAIR-DEBUG] Example image stems:", [Path(x).stem for x in imgs[:5]])
        print("[PAIR-DEBUG] Example mask stems :", [Path(x).stem for x in masks[:5]])
    else:
        print(f"[PAIR] {len(pairs)} pairs from img={img_dir} mask={mask_dir} (misses={misses})")
    return pairs

def find_pairs_by_convention(data_dir: str) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]]]:
    """
    Preferred: use provided OASIS train/validate directories under data_dir.
    Fallbacks:
      - If only 'train' exists, return val by random split later.
      - If data_dir itself *is* a split dir, try to infer its pair dir by name substitution.
      - Final fallback: global scan inside data_dir for any *_mask/seg siblings next to images (slow).
    """
    # Case A: data_dir is the OASIS root
    split = _split_dirs_from_root(data_dir)
    train_img, train_msk = split["train"]
    val_img,   val_msk   = split["validate"]

    train_pairs, val_pairs = [], []

    if train_img and train_msk:
        train_pairs = _pair_split(train_img, train_msk)

    if val_img and val_msk:
        val_pairs = _pair_split(val_img, val_msk)

    if train_pairs:
        return train_pairs, val_pairs  # val_pairs may be empty

    # Case B: data_dir is one split dir (e.g., .../keras_png_slices_train or .../keras_png_slices_seg_train)
    name = os.path.basename(os.path.abspath(data_dir)).lower()
    parent = os.path.abspath(os.path.join(data_dir, ".."))
    if "seg" in name or "mask" in name:
        # we are in mask dir; try to find sibling image dir
        buddy = name.replace("seg_", "").replace("_mask", "")
        img_dir = os.path.join(parent, buddy)
        if os.path.isdir(img_dir):
            train_pairs = _pair_split(img_dir, data_dir)
            if train_pairs:
                return train_pairs, []
    else:
        # we are in image dir; try to find sibling mask dir
        m1 = name.replace("slices_", "slices_seg_")
        m2 = name + "_mask"
        for mdir in (os.path.join(parent, m1), os.path.join(parent, m2)):
            if os.path.isdir(mdir):
                train_pairs = _pair_split(data_dir, mdir)
                if train_pairs:
                    return train_pairs, []

    # Case C: fully generic slow fallback
    print("[WARN] Using slow fallback scan. Consider pointing --data-dir at the OASIS root.")
    all_files = []
    for rp, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(EXTS):
                all_files.append(os.path.join(rp, f))
    all_files.sort()
    mask_like = re.compile(r'(?i).*(?:[_-])(seg|mask|label)(?:[_-]?\d+)?\.[^.]+$')
    images = [f for f in all_files if not mask_like.match(f)]
    masks  = [f for f in all_files if     mask_like.match(f)]
    mindex = {_normalize_key(m): m for m in masks}
    train_pairs = []
    for ip in images:
        mp = mindex.get(_normalize_key(ip))
        if mp:
            train_pairs.append((ip, mp))
    if train_pairs:
        print(f"[PAIR-FALLBACK] {len(train_pairs)} generic pairs from {data_dir}")
        return train_pairs, []
    return [], []

# ---------------------------
# I/O helpers
# ---------------------------

def pil_load_gray(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("L").resize((size, size), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0

def pil_load_mask(path: str, size: int) -> np.ndarray:
    m = Image.open(path).convert("L").resize((size, size), resample=Image.NEAREST)
    return np.asarray(m, dtype=np.int32)

# ---------------------------
# Dataset
# ---------------------------

class SegDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str,str]], img_size: int,
                 label_map: Optional[Dict[int,int]] = None, augment: bool = True, seed: int = 42):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self._rng = np.random.RandomState(seed)
        self.label_map = label_map or self._derive_label_map()

    def _derive_label_map(self) -> Dict[int, int]:
        # Scan a subset of masks for unique raw label values
        vals = set()
        step = max(1, len(self.pairs)//200) if self.pairs else 1
        for _, mp in self.pairs[::step]:
            arr = pil_load_mask(mp, self.img_size)
            vals.update(np.unique(arr).tolist())

        if not vals:
            vals = {0, 1}

        # Normalize common binary convention: 255 -> 1
        if 255 in vals:
            vals.discard(255)
            vals.add(1)

        # Ensure background exists
        vals.add(0)

        # Deterministic ordering: background first, then ascending
        ordered = [0] + sorted(v for v in vals if v != 0)

        # Map to contiguous class ids with background=0
        remap = {0: 0}
        for i, v in enumerate(ordered[1:], start=1):
            remap[v] = i

        return remap


    @property
    def n_classes(self) -> int:
        return len(set(self.label_map.values()))

    def __len__(self) -> int:
        return len(self.pairs)

    def _maybe_aug(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return x, y
        if self._rng.rand() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        if self._rng.rand() < 0.5:
            x = np.flip(x, axis=0).copy()
            y = np.flip(y, axis=0).copy()
        return x, y

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img = pil_load_gray(ip, self.img_size)                 # (H,W) float32 [0,1]
        msk_raw = pil_load_mask(mp, self.img_size)             # (H,W) int
        msk = np.zeros_like(msk_raw, dtype=np.int64)
        for raw_val, cid in self.label_map.items():
            msk[msk_raw == raw_val] = cid
        img, msk = self._maybe_aug(img, msk)
        img_t = torch.from_numpy(img).unsqueeze(0)             # (1,H,W)
        msk_t = torch.from_numpy(msk)                          # (H,W)
        return img_t, msk_t

# ---------------------------
# Model
# ---------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=True, base=32):
        super().__init__()
        self.inc = DoubleConv(n_channels, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base*8, base*16 // factor)
        self.up1 = Up(base*16, base*8 // factor, bilinear)
        self.up2 = Up(base*8, base*4 // factor, bilinear)
        self.up3 = Up(base*4, base*2 // factor, bilinear)
        self.up4 = Up(base*2, base, bilinear)
        self.outc = OutConv(base, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ---------------------------
# Loss / Metrics / Viz
# ---------------------------

def one_hot(y: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=n).permute(0,3,1,2).float()

def dice_per_class(logits: torch.Tensor, y: torch.Tensor, eps=1e-6) -> torch.Tensor:
    ncls = logits.shape[1]
    p = F.softmax(logits, dim=1)
    y1 = one_hot(y, ncls).to(p.device)
    dims = (0,2,3)
    inter = (p*y1).sum(dim=dims)
    denom = p.sum(dim=dims) + y1.sum(dim=dims)
    return (2*inter + eps) / (denom + eps)   # (C,)

class DiceCELoss(nn.Module):
    def __init__(self, weight_ce=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight_ce)
    def forward(self, logits, y):
        ce = self.ce(logits, y)
        d  = dice_per_class(logits, y).mean()
        return (1.0 - d) + ce, d.detach()

def make_palette(nc:int) -> np.ndarray:
    base = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],
                     [0,255,255],[128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128]], dtype=np.uint8)
    if nc <= len(base): return base[:nc]
    rng = np.random.RandomState(123); extra = rng.randint(0,256,size=(nc-len(base),3),dtype=np.uint8)
    return np.vstack([base, extra])

def overlay(gray: np.ndarray, pred: np.ndarray, pal: np.ndarray, alpha=0.5) -> np.ndarray:
    gray_rgb = (np.stack([gray,gray,gray], -1)*255).astype(np.uint8)
    color = pal[pred]
    return (alpha*color + (1-alpha)*gray_rgb).astype(np.uint8)

def save_pred_grid(x: torch.Tensor, logits: torch.Tensor, out_png: str, pal: np.ndarray, max_k=4):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with torch.no_grad():
        k = min(max_k, x.shape[0])
        for i in range(k):
            g = x[i,0].cpu().numpy()
            pred = torch.argmax(logits[i:i+1], dim=1)[0].cpu().numpy().astype(np.int32)
            Image.fromarray(overlay(g, pred, pal)).save(out_png.replace(".png", f"_k{i}.png"))

# ---------------------------
# Data loader builder
# ---------------------------

def build_loaders(data_dir: str, img_size: int, batch: int, workers: int,
                  val_split: float, limit: int, seed: int, augment: bool):
    train_pairs, val_pairs = find_pairs_by_convention(data_dir)

    if not train_pairs:
        raise RuntimeError(f"No train image/mask pairs found under {data_dir}")

    # Limit if requested
    if limit > 0 and limit < len(train_pairs):
        train_pairs = train_pairs[:limit]

    # If validate split missing, use random split from train
    if not val_pairs:
        rng = np.random.RandomState(seed)
        idx = np.arange(len(train_pairs))
        rng.shuffle(idx)
        n_val = max(1, int(len(idx)*val_split))
        val_pairs = [train_pairs[i] for i in idx[:n_val]]
        train_pairs = [train_pairs[i] for i in idx[n_val:]]
        print(f"[SPLIT] validate split missing -> random split: train={len(train_pairs)} val={len(val_pairs)}")
    else:
        print(f"[SPLIT] using OASIS validate split: train={len(train_pairs)} val={len(val_pairs)}")

    # Label mapping derived from TRAIN to keep consistency
    tmp_ds = SegDataset(train_pairs, img_size, label_map=None, augment=False, seed=seed)
    label_map = tmp_ds.label_map
    n_classes = tmp_ds.n_classes
    print(f"[LABELS] raw->cls map {label_map} | n_classes={n_classes}")

    train_ds = SegDataset(train_pairs, img_size, label_map=label_map, augment=augment, seed=seed)
    val_ds   = SegDataset(val_pairs,   img_size, label_map=label_map, augment=False,   seed=seed)

    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers,
                          pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers,
                          pin_memory=True)
    return train_ld, val_ld, n_classes, label_map

# ---------------------------
# Train / Eval
# ---------------------------

def compute_ce_weights(loader, n_classes: int) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float64)
    for i, (_, y) in enumerate(loader):
        y_np = y.numpy()
        for c in range(n_classes):
            counts[c] += (y_np == c).sum()
        if i >= 9:  # enough
            break
    freq = counts / max(counts.sum(), 1.0)
    inv = 1.0 / np.clip(freq, 1e-6, None)
    inv = inv / inv.sum() * n_classes
    return torch.tensor(inv, dtype=torch.float32)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    dices = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            per_c = dice_per_class(logits, y)  # (C,)
            dices.append(per_c.unsqueeze(0))
    if not dices:
        return 0.0, [0.0]
    d = torch.cat(dices, dim=0).mean(dim=0)      # (C,)
    return float(d.mean().cpu().item()), [float(v) for v in d.cpu().numpy().tolist()]

def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    ensure_dir(args.outdir)

    print(f"[INFO] Data root: {args.data_dir}")
    print(f"[INFO] Outdir:    {args.outdir}")
    print(f"[INFO] Device:    {device}")

    train_ld, val_ld, n_classes, label_map = build_loaders(
        args.data_dir, args.img_size, args.batch_size, args.workers,
        args.val_split, args.limit, args.seed, augment=(not args.no_aug)
    )
    with open(os.path.join(args.outdir, "label_map.json"), "w") as f:
        json.dump({str(k): int(v) for k, v in label_map.items()}, f, indent=2)

    model = UNet(n_channels=1, n_classes=n_classes, bilinear=True, base=32).to(device)
    ce_w = compute_ce_weights(train_ld, n_classes).to(device)
    crit = DiceCELoss(weight_ce=ce_w)
    opt  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_ep = 0
    best_mdice = -1.0
    if args.ckpt and os.path.isfile(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if args.amp and "scaler" in ck and ck["scaler"] is not None:
            try: scaler.load_state_dict(ck["scaler"])
            except: pass
        start_ep  = ck.get("epoch", 0) + 1
        best_mdice = ck.get("best_mdice", -1.0)
        print(f"[RESUME] {args.ckpt} @ epoch {start_ep} best mDice={best_mdice:.4f}")

    if args.eval_only:
        md, per = evaluate(model, val_ld, device)
        print(f"[EVAL] mDice={md:.4f} " + " ".join([f"C{c}:{per[c]:.3f}" for c in range(len(per))]))
        return

    log_csv = os.path.join(args.outdir, "train_log.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w") as f:
            f.write("epoch,loss,mdice," + ",".join([f"dice_c{c}" for c in range(n_classes)]) + "\n")

    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        model.train()
        run_loss = 0.0; nb = 0

        for x, y in train_ld:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss, _ = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.item()); nb += 1

        md, per = evaluate(model, val_ld, device)
        avg_loss = run_loss / max(1, nb)
        msg = f"[EPOCH {ep+1}/{args.epochs}] loss={avg_loss:.4f} mDice={md:.4f} | " + " ".join([f"C{c}:{per[c]:.3f}" for c in range(len(per))]) + f" | time={time.time()-t0:.1f}s"
        print(msg)

        with open(log_csv, "a") as f:
            f.write(f"{ep+1},{avg_loss:.6f},{md:.6f}," + ",".join([f"{v:.6f}" for v in per]) + "\n")

        # Save prediction overlays on the first validation batch
        pal = make_palette(n_classes)
        with torch.no_grad():
            for i, (vx, vy) in enumerate(val_ld):
                vx = vx.to(device, non_blocking=True)
                vout = model(vx)
                save_pred_grid(vx, vout, os.path.join(args.outdir, "pred_vis", f"e{ep+1:03d}.png"), pal, max_k=4)
                break

        # Save best + periodic
        state = {
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if args.amp else None,
            "best_mdice": best_mdice,
            "n_classes": n_classes,
            "label_map": label_map,
            "img_size": args.img_size,
        }
        if md > best_mdice:
            best_mdice = md
            state["best_mdice"] = best_mdice
            torch.save(state, os.path.join(args.outdir, "best.pt"))
        if (ep+1) % max(1, args.save_every) == 0:
            torch.save(state, os.path.join(args.outdir, f"epoch_{ep+1:03d}.pt"))

    print(f"[DONE] Best mDice={best_mdice:.4f}")

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ensure_dir(args.outdir)
    train_loop(args)
