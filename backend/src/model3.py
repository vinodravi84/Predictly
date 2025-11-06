#!/usr/bin/env python3
"""
simple_trainer.py (updated)

- Robust caching from legacy or new image folders
- Optional --preload into RAM
- Minimal, stable training loop with checkpointing and early stopping
- Supports EfficientNet-B0, ResNet34, ResNet50
"""
import os, sys, time, random, math
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---------------- Config ----------------
DATASET_DIR = Path("./dataset")
TRAIN_CSV = DATASET_DIR / "train.csv"
TEST_CSV = DATASET_DIR / "test.csv"
TRAIN_IMG_DIR = DATASET_DIR / "images/train"
TEST_IMG_DIR = DATASET_DIR / "images/test"
LEGACY_TRAIN_NPY = DATASET_DIR / "traine"
LEGACY_TEST_NPY  = DATASET_DIR / "teste"

CACHE_DIR = Path("./cache_model_simple")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = CACHE_DIR / "model_best.pth"
OOF_NPZ = CACHE_DIR / "oof.npz"
TEST_PREDS_NPY = CACHE_DIR / "test_preds.npy"

IMG_SIZE = 224
DEFAULT_BATCH = 16
DEFAULT_EPOCHS = 15
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 1e-5
CLIP_NORM = 3.0
PATIENCE = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE.startswith("cuda") else False
NUM_WORKERS = 2

# ---------------- Utils ----------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def smape_np(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom==0, 1e-8, denom)
    return np.mean(np.abs(y_true - y_pred)/denom) * 100.0

# ---------------- Caching ----------------
def cache_images(df, img_folder: Path, npy_folder: Path, img_col="sample_id", label_col=None,
                 img_ext=".jpg", img_size=IMG_SIZE, force=False, show_progress=True):
    npy_folder.mkdir(parents=True, exist_ok=True)
    npy_paths, targets, ids = [], [], []
    iterator = tqdm(df.iterrows(), desc=f"Caching -> {npy_folder.name}") if show_progress else df.iterrows()
    for _, row in iterator:
        sid = str(row[img_col])
        img_path = img_folder / f"{sid}{img_ext}"
        npy_path = npy_folder / f"{sid}.npy"
        if not npy_path.exists() or force:
            try:
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB").resize((img_size,img_size))
                else:
                    img = Image.new("RGB", (img_size,img_size), (127,127,127))
                np.save(npy_path, np.asarray(img, dtype=np.uint8))
            except Exception:
                np.save(npy_path, np.zeros((img_size,img_size,3),dtype=np.uint8))
        npy_paths.append(str(npy_path))
        ids.append(sid)
        if label_col is not None:
            targets.append(float(row[label_col]))
    targets_arr = np.array(targets, dtype=np.float32) if targets else None
    return npy_paths, targets_arr, ids

def gather_npy_from_existing(npy_folder: Path, df, img_col="sample_id", npy_ext=".npy", show_progress=True):
    npy_paths, ids, missing = [], [], 0
    iterator = tqdm(df[img_col].tolist(), desc=f"Using existing npy -> {npy_folder.name}") if show_progress else df[img_col].tolist()
    for sid in iterator:
        p = npy_folder / f"{sid}{npy_ext}"
        if p.exists():
            npy_paths.append(str(p))
        else:
            npy_paths.append(None)
            missing += 1
        ids.append(str(sid))
    return npy_paths, ids, missing

# ---------------- Dataset ----------------
class SimpleImageDataset(Dataset):
    def __init__(self, npy_paths, targets=None, transform=None, preload_cache=None):
        self.npy_paths = npy_paths
        self.targets = None if targets is None else np.array(targets,dtype=np.float32)
        self.transform = transform
        self.train = targets is not None
        self.preload = preload_cache is not None
        self.preload_cache = preload_cache or {}

    def __len__(self): return len(self.npy_paths)
    def __getitem__(self, idx):
        arr = self.preload_cache[idx] if self.preload and idx in self.preload_cache else \
              np.load(self.npy_paths[idx]) if self.npy_paths[idx] else np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)
        if arr.ndim==2: arr = np.stack([arr]*3,axis=-1)
        img = Image.fromarray(arr)
        if self.transform: img = self.transform(img)
        return (img, torch.tensor(self.targets[idx],dtype=torch.float32)) if self.train else img

# ---------------- Transforms ----------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.05)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- Model ----------------
class SimpleRegressor(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, head_dim=512):
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name=="resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone_name=="resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
        else:
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(in_features, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(head_dim, 1)
        )
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x).squeeze(1)
        return x

# ---------------- Checkpoint loader ----------------
def load_checkpoint_with_bar(ckpt_path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt = torch.load(ckpt_path,map_location='cpu')
    model_state = ckpt.get("model", ckpt.get("state_dict",None))
    if model_state is None: raise RuntimeError("Checkpoint missing model weights")
    target_state = model.state_dict()
    for k in tqdm(model_state.keys(), desc="Loading weights"):
        if k in target_state and model_state[k].shape==target_state[k].shape:
            try: target_state[k].copy_(model_state[k])
            except: target_state[k].copy_(model_state[k].to(target_state[k].dtype))
    model.load_state_dict(target_state)
    for obj, name in zip([optimizer,scheduler,scaler],["optimizer","scheduler","scaler"]):
        state = ckpt.get(name,None)
        if obj and state:
            try: obj.load_state_dict(state)
            except: print(f"⚠️ Couldn't fully restore {name}")
    return {"epoch": ckpt.get("epoch",0), "best_smape": ckpt.get("best_smape",None)}

# ---------------- Train/Val ----------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train(); running_loss=0.0; n=0
    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast(): loss = criterion(model(imgs), targets)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer); scaler.update()
        else:
            loss = criterion(model(imgs), targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM); optimizer.step()
        bs = imgs.size(0); running_loss += float(loss.item())*bs; n+=bs
    return running_loss/max(1,n)

def validate(model, loader, device):
    model.eval(); preds,targs=[],[]
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            out = model(imgs).detach().cpu().numpy()
            preds.append(out); targs.append(targets.numpy())
    if not preds: return None,None
    return np.concatenate(preds), np.concatenate(targs)

def predict_test(model, loader, device):
    model.eval(); preds=[]
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Test"): imgs=imgs.to(device); preds.append(model(imgs).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])

# ---------------- Main ----------------
def run_train(args):
    set_seed(args.seed)
    train_df = pd.read_csv(TRAIN_CSV).dropna(subset=["sample_id","price"])
    test_df = pd.read_csv(TEST_CSV).dropna(subset=["sample_id"]) if TEST_CSV.exists() else pd.DataFrame(columns=["sample_id"])

    use_legacy_train = LEGACY_TRAIN_NPY.exists() and any(LEGACY_TRAIN_NPY.glob("*.npy"))
    use_legacy_test = LEGACY_TEST_NPY.exists() and any(LEGACY_TEST_NPY.glob("*.npy"))

    if use_legacy_train and not args.force_cache:
        train_npy_paths_raw, train_ids_ordered, missing_train = gather_npy_from_existing(LEGACY_TRAIN_NPY, train_df)
        if missing_train>0:
            train_npy_paths, train_targets, train_ids = cache_images(train_df, TRAIN_IMG_DIR, LEGACY_TRAIN_NPY, label_col="price", force=args.force_cache)
        else:
            train_npy_paths = train_npy_paths_raw
            train_targets = np.array(train_df['price'].tolist(),dtype=np.float32)
            train_ids = train_ids_ordered
    else:
        train_npy_paths, train_targets, train_ids = cache_images(train_df, TRAIN_IMG_DIR, CACHE_DIR/"train_npy", label_col="price", force=args.force_cache)

    if use_legacy_test and not args.force_cache:
        test_npy_paths_raw, test_ids_ordered, missing_test = gather_npy_from_existing(LEGACY_TEST_NPY, test_df)
        if missing_test>0:
            test_npy_paths, _, test_ids = cache_images(test_df, TEST_IMG_DIR, LEGACY_TEST_NPY, force=args.force_cache)
        else:
            test_npy_paths = test_npy_paths_raw
            test_ids = test_ids_ordered
    else:
        test_npy_paths, _, test_ids = cache_images(test_df, TEST_IMG_DIR, CACHE_DIR/"test_npy", force=args.force_cache)

    if len([p for p in train_npy_paths if p])==0: print("No train data cached. abort."); return

    preload_cache = None
    if args.preload:
        preload_cache={}
        paths_to_preload=[p for p in train_npy_paths if p]+[p for p in test_npy_paths if p]
        preloaded_list=[]
        for p in tqdm(paths_to_preload, desc="Preloading .npy"): 
            try: preloaded_list.append(np.load(p))
            except: preloaded_list.append(np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8))
        # map train indices
        preload_cache = {i: preloaded_list[i] for i in range(len(train_npy_paths))}

    # split
    n=len(train_npy_paths); val_size=max(1,int(0.1*n)); train_size=n-val_size
    idxs=list(range(n)); random.shuffle(idxs)
    train_idxs,val_idxs=idxs[:train_size],idxs[train_size:]
    train_ds=SimpleImageDataset([train_npy_paths[i] for i in train_idxs],
                                train_targets[train_idxs],
                                transform=train_tfms,
                                preload_cache={k:preload_cache[k] for k in train_idxs} if preload_cache else None)
    val_ds=SimpleImageDataset([train_npy_paths[i] for i in val_idxs],
                              train_targets[val_idxs],
                              transform=val_tfms,
                              preload_cache={k:preload_cache[k] for k in val_idxs} if preload_cache else None)
    test_ds=SimpleImageDataset(test_npy_paths, transform=val_tfms) if len(test_npy_paths)>0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) if test_ds else None

    model = SimpleRegressor(backbone_name=args.backbone, pretrained=True).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,args.epochs))
    scaler = torch.cuda.amp.GradScaler() if args.amp and DEVICE.startswith("cuda") else None

    start_epoch = 0; best_smape=float("inf"); epochs_no_improve=0
    if args.resume and os.path.exists(args.resume):
        print("Loading checkpoint...")
        meta = load_checkpoint_with_bar(args.resume, model, optimizer, scheduler, scaler, DEVICE)
        start_epoch=int(meta.get("epoch",0))
        best_smape=float(meta.get("best_smape",best_smape))
        print(f"Resumed from epoch {start_epoch}, best_smape={best_smape}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        t0=time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_preds, val_targs = validate(model, val_loader, DEVICE)
        val_smape = smape_np(val_targs,val_preds) if val_preds is not None else float("inf")
        print(f"TrainLoss={train_loss:.4f} | Val SMAPE={val_smape:.4f}% | time={(time.time()-t0):.1f}s")

        scheduler.step()
        if val_smape < best_smape - 1e-6:
            best_smape = val_smape
            epochs_no_improve = 0
            torch.save({"model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "scheduler":scheduler.state_dict(),
                        "scaler":scaler.state_dict() if scaler else None,
                        "epoch":epoch+1,
                        "best_smape":best_smape}, str(BEST_CKPT))
            print(f"Saved BEST checkpoint -> {BEST_CKPT}")
        else:
            epochs_no_improve +=1
            print(f"No improvement {epochs_no_improve}/{PATIENCE}")
        if epochs_no_improve>=PATIENCE:
            print("Early stopping triggered."); break

    # Final evaluation
    if BEST_CKPT.exists():
        ckpt=torch.load(BEST_CKPT,map_location='cpu')
        try: model.load_state_dict(ckpt["model"])
        except: pass
    val_preds, val_targs = validate(model, val_loader, DEVICE)
    if val_preds is not None: np.savez_compressed(OOF_NPZ, sample_id=np.array([train_ids[i] for i in val_idxs]), val_pred=val_preds, val_true=val_targs)
    if test_loader:
        test_preds = predict_test(model, test_loader, DEVICE)
        np.save(TEST_PREDS_NPY, test_preds)
    print("Training complete. Best_val_smape=", best_smape)

# ---------------- CLI ----------------
def get_args():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--preload", action="store_true")
    p.add_argument("--backbone", type=str, default="efficientnet_b0", help="efficientnet_b0 | resnet34 | resnet50")
    return p.parse_args()

if __name__=="__main__":
    args=get_args(); args.resume=args.resume if args.resume else None
    print(f"Device: {DEVICE} | num_workers: {NUM_WORKERS} | pin_memory: {PIN_MEMORY}")
    run_train(args)
