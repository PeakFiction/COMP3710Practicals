# DAWNBench-style CIFAR-10 trainer (from-scratch ResNet-18)
# Fast + >93% test acc on A100 with AMP. No torchvision.models used.

import math, time, random, argparse, os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

# ---------------------------
# Reproducibility & utils
# ---------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class AvgMeter:
    def __init__(self): self.n=0; self.s=0.0
    def update(self, v, k=1): self.s += float(v)*k; self.n += k
    @property
    def avg(self): return self.s / max(1, self.n)

# ---------------------------
# CIFAR-10 ResNet-18 (from scratch)
# ---------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.short = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.short = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.short(x)
        return F.relu(out, inplace=True)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # CIFAR: 32x32, no 7x7
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ---------------------------
# MixUp / CutMix (optional)
# ---------------------------
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1); cy = random.randint(0, H-1)
    x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W); y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def apply_mix(batch, targets, alpha=0.2, cutmix_prob=0.5):
    if alpha <= 0: return batch, targets, None, None, 1.0
    lam = np_lam = torch.distributions.Beta(alpha, alpha).sample().item()
    if random.random() < cutmix_prob:
        # CutMix
        bs, _, H, W = batch.size()
        index = torch.randperm(bs, device=batch.device)
        x1,y1,x2,y2 = rand_bbox(W,H,lam)
        batch[:, :, y1:y2, x1:x2] = batch[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2-x1)*(y2-y1) / (W*H))
        return batch, targets, targets[index], lam, 2
    else:
        # MixUp
        bs = batch.size(0)
        index = torch.randperm(bs, device=batch.device)
        batch = batch * lam + batch[index] * (1 - lam)
        return batch, targets, targets[index], lam, 1

# ---------------------------
# Training
# ---------------------------
def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--mix", type=float, default=0.2)     # alpha; 0 to disable
    p.add_argument("--cutmix_p", type=float, default=0.5)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data: AutoAugment + standard CIFAR crop/flip + per-channel normalize
    normalize = transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    train_tfms = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_tfms = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_tfms)
    test_set  = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_tfms)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=1024,     shuffle=False, num_workers=args.workers, pin_memory=True)

    model = ResNet18(num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    # Cosine schedule with linear warmup (5 epochs)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)
    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_schedule)

    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_m, acc_m = AvgMeter(), AvgMeter()
        t0 = time.time()
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if args.mix > 0:
                xb, y1, y2, lam, mode = apply_mix(xb, yb, alpha=args.mix, cutmix_prob=args.cutmix_p)
            else:
                y1=y2=None; lam=1.0; mode=0

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(xb)
                if mode==0:
                    loss = crit(logits, yb)
                else:
                    loss = lam*crit(logits, y1) + (1-lam)*crit(logits, y2)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step(); step += 1

            loss_m.update(loss.item(), xb.size(0))
            acc_m.update(accuracy(logits, yb), xb.size(0))

            if (i+1) % args.log_every == 0:
                print(f"epoch {epoch:03d} step {i+1:04d}/{len(train_loader)} "
                      f"lr {sched.get_last_lr()[0]:.4f} loss {loss_m.avg:.4f} acc {acc_m.avg:.3f}")

        # eval
        model.eval()
        te_loss, te_acc = AvgMeter(), AvgMeter()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                te_loss.update(loss.item(), xb.size(0))
                te_acc.update(accuracy(logits, yb), xb.size(0))

        print(f"[epoch {epoch:03d}] train loss {loss_m.avg:.4f} acc {acc_m.avg:.3f} | "
              f"test loss {te_loss.avg:.4f} acc {te_acc.avg:.4f} | time {time.time()-t0:.1f}s")

    # Save for demo inference
    os.makedirs("artifacts", exist_ok=True)
    torch.save({"model": model.state_dict()}, "artifacts/cifar10_resnet18.pt")
    print("Saved -> artifacts/cifar10_resnet18.pt")

if __name__ == "__main__":
    set_seed(42); main()
