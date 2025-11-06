#!/usr/bin/env python3
"""
Smart Product Pricing Challenge - Image Model
----------------------------------------------
✅ Uses cached .npy features if available, else creates them
✅ Handles dataset/train and dataset/test folders with sample_id.jpg naming
✅ Configurable paths & params at top
✅ SMAPE metric for evaluation
✅ Saves test_preds.npy (on test set)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# =============================================================
# CONFIG
# =============================================================
class CFG:
    train_csv = "train.csv"
    test_csv = "test.csv"
    train_img_dir = "dataset/train"
    test_img_dir = "dataset/test"
    train_cache_dir = "dataset/traine"
    test_cache_dir = "dataset/teste"
    model_save_path = "best_model.pt"
    test_pred_path = "test_preds.npy"

    img_size = 224
    batch_size = 32
    epochs = 10
    lr = 1e-4
    seed = 42
    val_split = 0.1
    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================
# UTILS
# =============================================================

def seed_everything(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100 * np.mean(numerator / np.clip(denominator, 1e-8, None))


# =============================================================
# DATASET + CACHE
# =============================================================

def build_transform(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((CFG.img_size, CFG.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CFG.img_size, CFG.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


def load_or_create_cache(df, img_dir, cache_dir, transform):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "images.npy")

    if os.path.exists(cache_path):
        print(f"🔹 Loading cached data from {cache_path}")
        images = np.load(cache_path)
    else:
        print(f"⚙️ Creating cache at {cache_path} ...")
        images = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(img_dir, f"{row['sample_id']}.jpg")
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            images.append(image.numpy())
        images = np.stack(images)
        np.save(cache_path, images)
        print(f"✅ Cache saved to {cache_path}")
    return images


# =============================================================
# MODEL
# =============================================================

class PriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


# =============================================================
# TRAIN LOOP
# =============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, prices in loader:
        imgs, prices = imgs.to(device), prices.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, prices)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    preds_list, true_list = [], []
    with torch.no_grad():
        for imgs, prices in loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            preds_list.append(preds)
            true_list.append(prices.numpy())
    preds_all = np.concatenate(preds_list)
    true_all = np.concatenate(true_list)
    return smape(true_all, preds_all)


def predict(model, loader, device):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for (imgs,) in loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            preds_list.append(preds)
    return np.concatenate(preds_list)


# =============================================================
# MAIN
# =============================================================

def main():
    seed_everything(CFG.seed)

    print("🚀 Loading data...")
    train_df = pd.read_csv(CFG.train_csv)
    test_df = pd.read_csv(CFG.test_csv)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Build transforms
    train_tf = build_transform(is_train=True)
    test_tf = build_transform(is_train=False)

    # Load or create cache
    train_imgs = load_or_create_cache(train_df, CFG.train_img_dir, CFG.train_cache_dir, train_tf)
    test_imgs = load_or_create_cache(test_df, CFG.test_img_dir, CFG.test_cache_dir, test_tf)

    y = train_df["price"].values
    X_train, X_val, y_train, y_val = train_test_split(
        train_imgs, y, test_size=CFG.val_split, random_state=CFG.seed
    )

    # Convert to torch tensors
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_ds = torch.utils.data.TensorDataset(torch.tensor(test_imgs, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, num_workers=CFG.num_workers)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, num_workers=CFG.num_workers)

    # Model setup
    device = CFG.device
    model = PriceModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    criterion = nn.L1Loss()

    best_smape = 1e9

    print("🧠 Starting training...")
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_smape = validate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{CFG.epochs} | Loss: {train_loss:.4f} | SMAPE: {val_smape:.2f}")

        if val_smape < best_smape:
            best_smape = val_smape
            torch.save(model.state_dict(), CFG.model_save_path)
            print(f"✅ New best model saved with SMAPE {val_smape:.2f}")

    # Test prediction
    print("🧩 Generating test predictions...")
    model.load_state_dict(torch.load(CFG.model_save_path))
    test_preds = predict(model, test_loader, device)
    np.save(CFG.test_pred_path, test_preds)
    print(f"✅ Saved test predictions to {CFG.test_pred_path}")


if __name__ == "__main__":
    main()
