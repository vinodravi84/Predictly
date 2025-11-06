#!/usr/bin/env python3
"""
app.py - FastAPI server to serve:
 - Model3 image regressor (cache_model_simple/model_best.pth)
 - Text/meta predictor (cache_merge artifacts)

Endpoints:
 - GET  /health
 - POST /predict       (multipart/form-data or application/json)
 - POST /predict/batch (application/json, list of inputs)
 - OpenAPI docs at /docs
"""
from typing import Optional, List, Any, Dict
from pathlib import Path
import io
import time
import unicodedata
import glob
import traceback
import gc

import numpy as np
import pandas as pd
from PIL import Image

import joblib
import lightgbm as lgb

import torch
import torch.nn as nn
from torchvision import transforms, models

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os


# ---------------- Paths / config ----------------
ROOT = Path(".")
CACHE_IMG = ROOT / "cache_model_simple"
MERGE_CACHE = ROOT / "cache_merge"
DATASET_DIR = ROOT / "dataset"

IMG_CHECKPOINT = CACHE_IMG / "model_best.pth"
TFIDF_JOB = MERGE_CACHE / "meta_tfidf.joblib"
SVD_JOB = MERGE_CACHE / "meta_tfidf_svd.joblib"
SCALER_JOB = MERGE_CACHE / "meta_scaler.joblib"
RIDGE_FOLDS_GLOB = MERGE_CACHE / "meta_ridge_fold*.joblib"
LGB_FOLDS_GLOB = MERGE_CACHE / "meta_lgb_fold*.txt"
RESIDUAL_LGB_JOB = MERGE_CACHE / "residual_lgb.joblib"
BLEND_INFO = MERGE_CACHE / "meta_blend_info.joblib"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

app = FastAPI(title="TripNBook Model API", version="1.0")

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],              # GET, POST, PUT, DELETE...
    allow_headers=["*"],              # allow Authorization, Content-Type, etc.
)

def log(*args, **kwargs):
    print("[app]", *args, **kwargs)

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = "" if pd.isna(s) else str(s)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = unicodedata.normalize("NFKC", s)
    return s

# ---------------- Image model ----------------
class SimpleRegressor(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=False, head_dim=512):
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone_name == "resnet50":
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
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x).squeeze(1)
        return x

val_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.05)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def try_load_torch_weights(checkpoint_path: Path):
    """
    Try to load with weights_only=True (safer), fallback to normal torch.load if not supported.
    """
    try:
        # newer PyTorch supports weights_only argument
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        # older PyTorch doesn't support weights_only -> normal load
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        # final fallback
        return torch.load(checkpoint_path, map_location="cpu")

def load_image_model(checkpoint_path: Path = IMG_CHECKPOINT, backbone="efficientnet_b0"):
    if not checkpoint_path.exists():
        log("Image checkpoint missing:", checkpoint_path)
        return None
    model = SimpleRegressor(backbone_name=backbone, pretrained=False)
    try:
        ckpt = try_load_torch_weights(checkpoint_path)
        model_state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(model_state)
    except Exception as e:
        log("Full load failed; trying partial load fallback:", e)
        try:
            ckpt2 = torch.load(checkpoint_path, map_location="cpu")
            sd = ckpt2.get("model", ckpt2.get("state_dict", ckpt2))
            target = model.state_dict()
            for k in sd:
                if k in target and sd[k].shape == target[k].shape:
                    target[k].copy_(sd[k])
            model.load_state_dict(target)
        except Exception as e2:
            log("Partial load failed:", e2)
            return None
    model.to(DEVICE)
    model.eval()
    log("Loaded image model from", checkpoint_path)
    return model

def predict_image(model, pil_img: Image.Image) -> float:
    img_t = val_tfms(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_t).detach().cpu().numpy().reshape(-1)[0]
    return float(out)

# ---------------- Meta/Text artifacts ----------------
def load_meta_artifacts():
    artifacts = {}
    # TF-IDF + SVD
    if TFIDF_JOB.exists() and SVD_JOB.exists():
        try:
            artifacts["tfidf"] = joblib.load(TFIDF_JOB)
            artifacts["svd"] = joblib.load(SVD_JOB)
            log("Loaded TF-IDF and SVD.")
        except Exception as e:
            log("Failed TFIDF/SVD load:", e)
            artifacts["tfidf"] = None; artifacts["svd"] = None
    else:
        log("TFIDF/SVD missing in cache.")
        artifacts["tfidf"] = None; artifacts["svd"] = None

    # scaler
    if SCALER_JOB.exists():
        try:
            artifacts["scaler"] = joblib.load(SCALER_JOB)
        except Exception as e:
            log("Failed loading scaler:", e)
            artifacts["scaler"] = None
    else:
        artifacts["scaler"] = None

    # ridge folds (use str() for glob)
    artifacts["ridge_folds"] = []
    for p in sorted(glob.glob(str(RIDGE_FOLDS_GLOB))):
        try:
            artifacts["ridge_folds"].append(joblib.load(p))
        except Exception as e:
            log("Failed loading ridge fold:", p, e)

    # lgb folds
    artifacts["lgb_folds"] = []
    for p in sorted(glob.glob(str(LGB_FOLDS_GLOB))):
        try:
            artifacts["lgb_folds"].append(lgb.Booster(model_file=p))
        except Exception as e:
            log("Failed loading lgb fold:", p, e)

    # residual lgb
    if RESIDUAL_LGB_JOB.exists():
        try:
            artifacts["residual"] = joblib.load(RESIDUAL_LGB_JOB)
        except Exception as e:
            log("Failed loading residual LGB:", e)
            artifacts["residual"] = None
    else:
        artifacts["residual"] = None

    # blend info
    if BLEND_INFO.exists():
        try:
            artifacts["blend_info"] = joblib.load(BLEND_INFO)
        except Exception as e:
            log("Failed loading blend info:", e)
            artifacts["blend_info"] = None
    else:
        artifacts["blend_info"] = None

    # train median fallback
    if (DATASET_DIR / "train.csv").exists():
        try:
            df = pd.read_csv(DATASET_DIR / "train.csv", usecols=["price"])
            artifacts["train_median_price"] = float(df["price"].median())
        except Exception:
            artifacts["train_median_price"] = 0.0
    else:
        artifacts["train_median_price"] = 0.0

    log("Meta artifacts loaded.")
    return artifacts

def build_meta_feature_for_single(text: str, artifacts: dict):
    median = artifacts.get("train_median_price", 0.0) or 0.0
    base = np.array([median, median, median], dtype=float).reshape(1,3)
    meta_mean = np.nanmean(base, axis=1).reshape(-1,1)
    meta_std  = np.nanstd(base, axis=1).reshape(-1,1)
    meta_min  = np.nanmin(base, axis=1).reshape(-1,1)
    meta_max  = np.nanmax(base, axis=1).reshape(-1,1)
    meta_nnan = np.isnan(base).sum(axis=1).reshape(-1,1)

    tf = artifacts.get("tfidf")
    svd = artifacts.get("svd")
    if tf is None or svd is None:
        tf_svd_vec = np.zeros((1, getattr(svd, "n_components", 128) if svd is not None else 128), dtype=float)
    else:
        txt = normalize_text(text)
        try:
            tfv = tf.transform([txt])
            tf_svd_vec = svd.transform(tfv)
        except Exception as e:
            log("TFIDF transform error:", e)
            tf_svd_vec = np.zeros((1, getattr(svd, "n_components", 128)), dtype=float)

    meta_x = np.hstack([base, meta_mean, meta_std, meta_min, meta_max, meta_nnan, tf_svd_vec])
    return meta_x

def predict_text_price(catalog_content: str, artifacts: dict):
    meta_x = build_meta_feature_for_single(catalog_content, artifacts)
    scaler = artifacts.get("scaler")
    if scaler is not None:
        try:
            meta_x_s = scaler.transform(meta_x)
        except Exception:
            meta_x_s = meta_x
    else:
        meta_x_s = meta_x

    ridge_preds = []
    for r in artifacts.get("ridge_folds", []):
        try:
            rv_log = r.predict(meta_x_s)
            ridge_preds.append(np.expm1(rv_log)[0])
        except Exception as e:
            log("ridge predict error:", e)
    ridge_pred = float(np.mean(ridge_preds)) if ridge_preds else float(artifacts.get("train_median_price", 0.0))

    lgb_preds = []
    for b in artifacts.get("lgb_folds", []):
        try:
            lp_log = b.predict(meta_x)[0]
            lgb_preds.append(np.expm1(lp_log))
        except Exception as e:
            log("lgb predict error:", e)
    lgb_pred = float(np.mean(lgb_preds)) if lgb_preds else float(artifacts.get("train_median_price", 0.0))

    # residual correction
    residual_model = artifacts.get("residual")
    final_price = None
    if residual_model is not None:
        try:
            meta_for_res = np.vstack([np.array([lgb_pred, ridge_pred])]).astype(float)
            res_log = residual_model.predict(meta_for_res)[0]
            combo = 0.5*lgb_pred + 0.5*ridge_pred
            final_price = float(np.expm1(np.log1p(combo) + res_log))
        except Exception as e:
            log("residual predict error:", e)
    if final_price is None:
        blend_info = artifacts.get("blend_info")
        if blend_info and isinstance(blend_info, tuple):
            try:
                best_w = blend_info[0]
                final_price = float(best_w[0]*lgb_pred + best_w[1]*ridge_pred)
            except Exception:
                final_price = float(0.5*lgb_pred + 0.5*ridge_pred)
        else:
            final_price = float(0.5*lgb_pred + 0.5*ridge_pred)

    return final_price, {"lgb_pred": lgb_pred, "ridge_pred": ridge_pred}

# ---------------- Startup: load artifacts ----------------
IMAGE_MODEL = None
META_ARTIFACTS = None

@app.on_event("startup")
def startup_event():
    global IMAGE_MODEL, META_ARTIFACTS
    log("Startup: loading artifacts. This can take a few seconds...")
    IMAGE_MODEL = load_image_model()  # may be None
    META_ARTIFACTS = load_meta_artifacts()
    log("Startup complete. Device:", DEVICE)

# ---------------- Request/Response models ----------------
class PredictInput(BaseModel):
    catalog_content: Optional[str] = None
    image_weight: Optional[float] = 0.2

class PredictOutput(BaseModel):
    image_pred: Optional[float]
    text_pred: Optional[float]
    combined_pred: float
    image_weight: float
    time_s: float
    details: Dict[str, Any] = {}

# ---------------- Endpoints ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict", response_model=PredictOutput)
async def predict(
    request: Request,
    image: Optional[UploadFile] = File(None),
    catalog_content: Optional[str] = Form(None),
    image_weight: Optional[float] = Form(0.2)
):
    start = time.time()
    # allow JSON body with {"catalog_content": "..."} in case user POSTs application/json
    if catalog_content is None:
        try:
            body = await request.json()
            if isinstance(body, dict):
                catalog_content = body.get("catalog_content", None)
        except Exception:
            pass

    catalog_content = normalize_text(catalog_content)

    image_pred = None
    text_pred = None
    details = {"image_model_loaded": IMAGE_MODEL is not None, "meta_loaded": META_ARTIFACTS is not None}

    if image is not None:
        try:
            content = await image.read()
            pil = pil_from_bytes(content)
            if IMAGE_MODEL is not None:
                image_pred = predict_image(IMAGE_MODEL, pil)
            else:
                details["image_error"] = "image model not loaded"
        except Exception as e:
            details["image_error"] = str(e)

    if catalog_content and catalog_content.strip():
        try:
            text_pred, text_details = predict_text_price(catalog_content, META_ARTIFACTS)
            details["text_details"] = text_details
        except Exception as e:
            details["text_error"] = str(e)

    if image_pred is None and text_pred is None:
        raise HTTPException(status_code=400, detail="No valid input provided. Upload image or provide catalog_content.")

    image_weight = float(image_weight or 0.2)
    image_weight = max(0.0, min(1.0, image_weight))

    if image_pred is not None and text_pred is not None:
        combined = image_weight * image_pred + (1.0 - image_weight) * text_pred
    elif image_pred is not None:
        combined = float(image_pred)
    else:
        combined = float(text_pred)

    elapsed = time.time() - start
    return PredictOutput(
        image_pred=float(image_pred) if image_pred is not None else None,
        text_pred=float(text_pred) if text_pred is not None else None,
        combined_pred=float(combined),
        image_weight=image_weight,
        time_s=elapsed,
        details=details
    )

# Batch predictions: accept list of {"catalog_content": "..."} (no images)
class BatchItem(BaseModel):
    catalog_content: Optional[str] = None
    image_weight: Optional[float] = 0.5

class BatchOutput(BaseModel):
    results: List[PredictOutput]

@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(items: List[BatchItem]):
    results = []
    for it in items:
        cat = normalize_text(it.catalog_content)
        try:
            text_pred, text_details = predict_text_price(cat, META_ARTIFACTS)
            image_pred = None
            image_weight = float(it.image_weight or 0.5)
            combined = text_pred
            details = {"image_model_loaded": IMAGE_MODEL is not None, "meta_loaded": META_ARTIFACTS is not None, "text_details": text_details}
            results.append(PredictOutput(
                image_pred=None,
                text_pred=float(text_pred),
                combined_pred=float(combined),
                image_weight=image_weight,
                time_s=0.0,
                details=details
            ))
        except Exception as e:
            results.append(PredictOutput(
                image_pred=None,
                text_pred=None,
                combined_pred=0.0,
                image_weight=float(it.image_weight or 0.5),
                time_s=0.0,
                details={"error": str(e)}
            ))
    return BatchOutput(results=results)

# small index
@app.get("/", include_in_schema=False)
def index():
    return {
        "msg": "TripNBook Model API - use /docs for Swagger UI. Endpoints: /health, /predict, /predict/batch"
    }

# ---------------- Run note ----------------
# Run with: python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
