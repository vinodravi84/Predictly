#!/usr/bin/env python3
"""
Model 1 — upgraded: uses local SBERT if present at LOCAL_SBERT_PATH
or downloads SentenceTransformer 'all-mpnet-base-v2' automatically when run in VSCode.
Auto-installs sentence-transformers if missing (best-effort).
"""
import os
import gc
import re
import random
import math
import subprocess
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import torch
import warnings
warnings.filterwarnings("ignore")

# ---- sklearn / lightgbm / model helpers (FIXED imports) ----
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
import lightgbm as lgb


# ---------------- CONFIG ----------------
DATA_DIR = "./dataset"                    # path containing train.csv & test.csv (set to your dataset)
CACHE_DIR = "cache_model1"
LOCAL_SBERT_PATH = Path("/kaggle/input/sentencetransformers/other/default/1/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
HF_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Hugging Face model to download if local not found
TFIDF = True
N_TFIDF = 40000
N_SVD = 256
SEED = 42
N_FOLDS = 5
BATCH_EMB = 128
# Determine device based on torch availability and CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_BOOST_ROUND = 15000
EARLY_STOPPING_ROUNDS = 250
VERBOSE_EVAL = 250

LGB_CONFIGS = [
    {
        "device":"gpu",
        "objective": "regression",
        "learning_rate": 0.01,
        "num_leaves": 128,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
        "min_child_samples": 20,
        "reg_alpha": 0.3,
        "reg_lambda": 0.5,
        "random_state": SEED,
        "n_jobs": -1,
    },
    {
        "objective": "regression",
        "learning_rate": 0.01,
        "num_leaves": 64,
        "colsample_bytree": 0.8,
        "subsample": 0.9,
        "subsample_freq": 1,
        "min_child_samples": 30,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": SEED + 7,
        "n_jobs": -1,
    }
]

REMOVE_OUTLIERS = True
OUTLIER_QUANTILE = 0.995
USE_RIDGE_BLEND = True
BLEND_GRID = np.linspace(0.0, 1.0, 11)
TARGET_ENC_MIN_COUNT = 20

os.makedirs(CACHE_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ---------------- HELPERS ----------------
def log(msg):
    print(f"[INFO] {msg}", flush=True)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / np.clip(denom, 1e-8, None)) * 100.0

def clean_text(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"http\S+", " ", x)
    x = re.sub(r"[^a-z0-9\s\.\-/×××××]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def ensure_package(pkg_name, import_name=None):
    """
    Try to import import_name (or pkg_name). If it fails, pip install pkg_name and import again.
    Returns the imported module or raises ImportError.
    """
    import_name = import_name or pkg_name
    try:
        return importlib.import_module(import_name)
    except Exception:
        log(f"Package '{pkg_name}' not found — attempting to pip install it (best-effort).")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            return importlib.import_module(import_name)
        except Exception as e:
            raise ImportError(f"Failed to install/import {pkg_name}: {e}")

# ---------------- SBERT/EMBEDDING UTIL ----------------
def load_or_download_sbert(local_path: Path, hf_model_name: str, device: str):
    """
    Attempts:
      1) Load local SentenceTransformer if local_path exists and sentence-transformers is installed.
      2) Else, ensures sentence-transformers package is installed and loads hf_model_name (downloads if needed).
    Returns: (sbert_model_or_None, embed_method_str)
    """
    SentenceTransformer = None
    # try import without auto-install first to preserve environment stability
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
    except Exception:
        # attempt auto-install
        try:
            st = ensure_package("sentence-transformers")
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            log(f"Could not import or install sentence-transformers: {e}")
            return None, "no_sentence_transformers"

    # Prefer local path if available
    if local_path.exists() and local_path.is_dir():
        try:
            log(f"Loading local SBERT from: {local_path}")
            sbert = SentenceTransformer(str(local_path), device=device)
            log("Local SBERT loaded successfully.")
            return sbert, "sbert_local"
        except Exception as e:
            log(f"Failed to load local SBERT: {e} — will try HF download fallback.")

    # Try to download/load from HF
    try:
        log(f"Loading SBERT from Hugging Face: '{hf_model_name}' (this will download model files if needed)...")
        sbert = SentenceTransformer(hf_model_name, device=device)
        log(f"Downloaded/loaded SBERT model '{hf_model_name}' successfully.")
        return sbert, "sbert_hf"
    except Exception as e:
        log(f"Failed to download/load SBERT '{hf_model_name}': {e}")
        return None, "sbert_failed"

def encode_in_batches(model, texts, batch_size=128, convert_to_numpy=True, show_progress=True):
    """
    Robust batching for SentenceTransformer.encode with fallback to manual loops.
    """
    try:
        return model.encode(texts, batch_size=batch_size, convert_to_numpy=convert_to_numpy, show_progress_bar=show_progress)
    except Exception:
        # manual batching
        out = []
        it = tqdm(range(0, len(texts), batch_size)) if show_progress else range(0, len(texts), batch_size)
        for i in it:
            batch = texts[i:i+batch_size]
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            out.append(emb)
        return np.vstack(out)

# ---------------- numeric & token features ----------------
def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df["catalog_content"].fillna("").astype(str)
    pq = text.str.extract(r'(?:pack|pk|pkt|packs?|packets?)\s*(?:of\s*)?(\d{1,4})')[0]
    px = text.str.extract(r'(\d{1,4})\s*[x×]\s*\d*')[0]
    df["pack_qty"] = pq.fillna(px)
    df["pack_qty"] = pd.to_numeric(df["pack_qty"], errors="coerce").fillna(1.0)
    all_nums = text.str.findall(r'\d+')
    df["numeric_sum"] = all_nums.apply(lambda lst: sum(int(x) for x in lst) if lst else 0)
    df["first_num"] = all_nums.apply(lambda lst: int(lst[0]) if lst else 0)
    df["last_num"] = all_nums.apply(lambda lst: int(lst[-1]) if lst else 0)
    df["digits_count"] = all_nums.apply(lambda lst: sum(len(x) for x in lst) if lst else 0)
    df["token_len"] = text.str.split().apply(len).fillna(0).astype(int)
    df["first_token"] = text.str.split().str[0].fillna("unknown")
    df["last_token"] = text.str.split().str[-1].fillna("unknown")
    df["has_unit_mg"] = text.str.contains(r'\bmg\b').astype(int)
    df["has_unit_g"]  = text.str.contains(r'\bg\b').astype(int)
    df["has_unit_kg"] = text.str.contains(r'\bkg\b').astype(int)
    df["has_unit_ml"] = text.str.contains(r'\bml\b').astype(int)
    df["has_unit_l"]  = text.str.contains(r'\bl\b|\blitre\b|\bliter\b').astype(int)
    df["has_unit_tablet"] = text.str.contains(r'\btablet\b|\btab\b').astype(int)
    df["has_unit_strip"]  = text.str.contains(r'\bstrip\b').astype(int)
    df["has_unit_pcs"]    = text.str.contains(r'\bpcs\b|\bpc\b|\bpieces\b').astype(int)
    df["has_digit"] = text.str.contains(r'\d').astype(int)
    num_cols = ["pack_qty","numeric_sum","first_num","last_num","digits_count","token_len"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def target_encode_kfold(train_df, test_df, column, target, n_splits=5, seed=SEED, min_count=TARGET_ENC_MIN_COUNT):
    rng = np.random.RandomState(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_df), dtype=float)
    global_median = float(np.median(target))
    grp_counts = train_df.groupby(column).size().to_dict()
    qcut = np.clip(pd.qcut(target, q=10, labels=False, duplicates="drop"), 0, 9)
    for tr_idx, val_idx in skf.split(train_df, qcut):
        grp = train_df.iloc[tr_idx].groupby(column)[target.name].median().to_dict()
        vals = train_df.iloc[val_idx][column].map(grp).fillna(global_median).values
        counts = train_df.iloc[val_idx][column].map(lambda x: grp_counts.get(x,0)).values
        vals = np.where(counts < min_count, global_median, vals)
        oof[val_idx] = vals
    full_map = train_df.groupby(column)[target.name].median().to_dict()
    test_enc = test_df[column].map(full_map).fillna(global_median).values
    test_counts = test_df[column].map(lambda x: train_df.groupby(column).size().to_dict().get(x,0)).values
    test_enc = np.where(test_counts < min_count, global_median, test_enc)
    return oof, test_enc

# ---------------- LOAD DATA ----------------
log("Loading CSVs...")
train = pd.read_csv(Path(DATA_DIR)/"train.csv", usecols=["sample_id","catalog_content","price"])
test  = pd.read_csv(Path(DATA_DIR)/"test.csv",  usecols=["sample_id","catalog_content"])

log("Cleaning text...")
for df, name in [(train, "train"), (test, "test")]:
    df["catalog_content"] = df["catalog_content"].fillna("").astype(str)
    df["catalog_content"] = [clean_text(x) for x in tqdm(df["catalog_content"], desc=f"{name} cleaning")]

log("Extracting numeric & token features...")
train = extract_numeric_features(train)
test  = extract_numeric_features(test)

# numeric matrix
num_features = [
    "pack_qty","numeric_sum","first_num","last_num",
    "digits_count","token_len","has_unit_mg","has_unit_g",
    "has_unit_kg","has_unit_ml","has_unit_l","has_unit_tablet",
    "has_unit_strip","has_unit_pcs","has_digit"
]
scaler = StandardScaler()
train_num = scaler.fit_transform(train[num_features])
test_num  = scaler.transform(test[num_features])

# target clipping and log
price = train["price"].values.copy()
if REMOVE_OUTLIERS:
    cap = np.quantile(price, OUTLIER_QUANTILE)
    log(f"Clipping target prices above {OUTLIER_QUANTILE} quantile -> {cap:.2f}")
    price = np.minimum(price, cap)
y_log = np.log1p(price)
y_series = pd.Series(price, name="price")

# ---------------- target encoding for 'first_token' ----------------
log("Creating KFold target encoding for first_token (OOF-safe)...")
first_te_oof, first_te_test = target_encode_kfold(train, test, "first_token", y_series, n_splits=N_FOLDS, seed=SEED, min_count=TARGET_ENC_MIN_COUNT)
train["first_token_te"] = first_te_oof
test["first_token_te"] = first_te_test
train_te_log = np.log1p(train["first_token_te"].values)
test_te_log = np.log1p(test["first_token_te"].values)
train_num = np.hstack([train_num, train_te_log.reshape(-1,1)])
test_num  = np.hstack([test_num, test_te_log.reshape(-1,1)])

train_inter = np.vstack([
    train["pack_qty"].values * train["token_len"].values,
    train["numeric_sum"].values / (train["first_num"].replace(0,1).values + 1),
]).T
test_inter = np.vstack([
    test["pack_qty"].values * test["token_len"].values,
    test["numeric_sum"].values / (test["first_num"].replace(0,1).values + 1),
]).T

train_num = np.hstack([train_num, train_inter])
test_num  = np.hstack([test_num, test_inter])

log(f"Numeric feature matrix shapes: train {train_num.shape} test {test_num.shape}")

# ---------------- EMBEDDINGS (prefer local SBERT, else download HF) ----------------
emb_cache_train = Path(CACHE_DIR)/"train_emb.npy"
emb_cache_test  = Path(CACHE_DIR)/"test_emb.npy"

# load or download sbert
sbert, embed_method = load_or_download_sbert(LOCAL_SBERT_PATH, HF_MODEL_NAME, DEVICE)

if sbert is not None:
    try:
        if emb_cache_train.exists() and emb_cache_test.exists():
            train_emb = np.load(emb_cache_train); test_emb = np.load(emb_cache_test); log("Loaded cached SBERT embeddings.")
        else:
            log("Computing SBERT train embeddings (this will use the model loaded above)...")
            train_emb = encode_in_batches(sbert, train["catalog_content"].tolist(), batch_size=BATCH_EMB, convert_to_numpy=True, show_progress=True)
            np.save(emb_cache_train, train_emb)
            log("Computing SBERT test embeddings...")
            test_emb = encode_in_batches(sbert, test["catalog_content"].tolist(), batch_size=BATCH_EMB, convert_to_numpy=True, show_progress=True)
            np.save(emb_cache_test, test_emb)
        embed_method = "sbert_used"
    except Exception as e:
        log(f"SBERT embedding failure: {e}. Falling back to TF-IDF/SVD embeddings.")
        sbert = None
        embed_method = "sbert_failed_fallback"

if sbert is None:
    # fallback: TF-IDF -> SVD embeddings (no HF network calls)
    emb_tfidf_path = Path(CACHE_DIR)/f"emb_tfidf_{N_TFIDF}.joblib"
    emb_svd_path = Path(CACHE_DIR)/f"emb_svd_{N_SVD}.joblib"
    if emb_cache_train.exists() and emb_cache_test.exists():
        train_emb = np.load(emb_cache_train); test_emb = np.load(emb_cache_test); embed_method = "tfidf_svd_cache"
        log("Loaded cached TFIDF-SVD embeddings.")
    else:
        log("Computing TF-IDF -> SVD embeddings (fallback)...")
        emb_tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=N_TFIDF, min_df=3, max_df=0.95, strip_accents="unicode", sublinear_tf=True)
        X_all = pd.concat([train["catalog_content"], test["catalog_content"]]).astype(str)
        emb_tfidf.fit(X_all)
        joblib.dump(emb_tfidf, emb_tfidf_path)
        X_train_tf = emb_tfidf.transform(train["catalog_content"])
        X_test_tf  = emb_tfidf.transform(test["catalog_content"])
        svd_emb = TruncatedSVD(n_components=N_SVD, random_state=SEED)
        svd_emb.fit(X_train_tf)
        joblib.dump(svd_emb, emb_svd_path)
        train_emb = svd_emb.transform(X_train_tf)
        test_emb = svd_emb.transform(X_test_tf)
        np.save(emb_cache_train, train_emb); np.save(emb_cache_test, test_emb)
        embed_method = "tfidf_svd"

log(f"Embedding method: {embed_method} | shapes: train {train_emb.shape}, test {test_emb.shape}")

# ---------------- TF-IDF features (separate) ----------------
train_tfidf = test_tfidf = None
if TFIDF:
    tfidf_path = Path(CACHE_DIR)/"tfidf_vectorizer.joblib"
    svd_path   = Path(CACHE_DIR)/"svd_model.joblib"
    if tfidf_path.exists() and svd_path.exists():
        tfidf = joblib.load(tfidf_path); svd = joblib.load(svd_path)
        log("Loaded cached TF-IDF features.")
    else:
        log("Fitting TF-IDF (features) + SVD...")
        tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=N_TFIDF, min_df=3, max_df=0.9, strip_accents="unicode", sublinear_tf=True)
        tfidf.fit(train["catalog_content"])
        joblib.dump(tfidf, tfidf_path)
        svd = TruncatedSVD(n_components=N_SVD, random_state=SEED)
        svd.fit(tfidf.transform(train["catalog_content"]))
        joblib.dump(svd, svd_path)
    train_tfidf = svd.transform(tfidf.transform(train["catalog_content"]))
    test_tfidf  = svd.transform(tfidf.transform(test["catalog_content"]))
    log(f"TF-IDF features shapes: train {train_tfidf.shape} test {test_tfidf.shape}")

# ---------------- stack final X ----------------
if TFIDF and (train_tfidf is not None):
    X_train = np.hstack([train_emb, train_tfidf, train_num])
    X_test  = np.hstack([test_emb, test_tfidf, test_num])
else:
    X_train = np.hstack([train_emb, train_num])
    X_test  = np.hstack([test_emb, test_num])
del train_emb, test_emb
gc.collect()
log(f"Final feature shapes: X_train={X_train.shape} X_test={X_test.shape}")

# ---------------- rest of original pipeline (unchanged) ----------------
# ... (the remainder of your training loop, skf, lgb training, blending, calibration, saving artifacts)
# For brevity I will include the rest exactly as in your original script,
# starting with price binning, the lgb feval, resume handling, training loop, and final save.

# ---------------- Stratified KFold bins ----------------
price_raw = train["price"].values
n_bins = 20
try:
    price_bins = pd.qcut(price_raw, q=n_bins, labels=False, duplicates="drop")
except Exception:
    price_bins = pd.cut(price_raw, bins=n_bins, labels=False)
price_bins = price_bins.astype(int)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def smape_lgb(y_pred_log, dataset):
    y_true_log = dataset.get_label()
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    val = smape(y_true, y_pred)
    return "SMAPE", float(val), False

# resume / partials
oof = np.zeros(len(X_train))
test_preds = np.zeros(X_test.shape[0])
oof_partial = Path(CACHE_DIR)/"oof_partial.npy"
test_partial = Path(CACHE_DIR)/"test_partial.npy"
resume_marker = Path(CACHE_DIR)/"last_completed_fold.txt"
if oof_partial.exists(): oof = np.load(oof_partial); log("Loaded partial OOF")
if test_partial.exists(): test_preds = np.load(test_partial); log("Loaded partial TEST preds")
last_completed = int(resume_marker.read_text().strip()) if resume_marker.exists() else 0
if last_completed>0: log(f"Resuming from fold {last_completed+1}")

log("Starting training with fold ensembles and calibration...")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, price_bins), 1):
    if fold <= last_completed:
        log(f"Skipping fold {fold} (already done)"); continue
    log(f"--- Fold {fold}/{N_FOLDS} ---")
    X_tr, X_val = X_train[trn_idx], X_train[val_idx]
    y_tr_log, y_val_log = y_log[trn_idx], y_log[val_idx]

    val_preds_log_list = []
    test_preds_orig_list = []

    for cfg_idx, base_cfg in enumerate(LGB_CONFIGS, 1):
        cfg = dict(base_cfg)
        cfg.setdefault("metric", "rmse")
        dtrain = lgb.Dataset(X_tr, label=y_tr_log)
        dval   = lgb.Dataset(X_val, label=y_val_log)
        try:
            model = lgb.train(cfg,
                              dtrain,
                              num_boost_round=NUM_BOOST_ROUND,
                              valid_sets=[dtrain, dval],
                              valid_names=["train","valid"],
                              feval=smape_lgb,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                              verbose_eval=VERBOSE_EVAL)
        except TypeError:
            model = lgb.train(cfg,
                              dtrain,
                              num_boost_round=NUM_BOOST_ROUND,
                              valid_sets=[dtrain, dval],
                              valid_names=["train","valid"],
                              feval=smape_lgb,
                              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(VERBOSE_EVAL)])

        best_it = getattr(model, "best_iteration", None) or getattr(model, "best_iter", None) or NUM_BOOST_ROUND
        log(f"cfg#{cfg_idx} best_iter={best_it}")

        val_pred_log = model.predict(X_val, num_iteration=best_it)
        test_pred_log = model.predict(X_test, num_iteration=best_it)

        val_preds_log_list.append(val_pred_log)
        test_preds_orig_list.append(np.expm1(test_pred_log))

        try:
            model.save_model(str(Path(CACHE_DIR)/f"fold{fold}_cfg{cfg_idx}.txt"))
        except Exception:
            pass

        del model
        gc.collect()

    val_preds_orig_ensemble = np.mean([np.expm1(v) for v in val_preds_log_list], axis=0)
    test_preds_orig_ensemble = np.mean(test_preds_orig_list, axis=0)

    if USE_RIDGE_BLEND and TFIDF and (train_tfidf is not None):
        ridge_feats = np.hstack([train_tfidf, train_num])
        X_ridge_tr = ridge_feats[trn_idx]; X_ridge_val = ridge_feats[val_idx]; X_ridge_test = np.hstack([test_tfidf, test_num])
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(X_ridge_tr, y_tr_log)
        ridge_val_log = ridge.predict(X_ridge_val)
        ridge_test_log = ridge.predict(X_ridge_test)
        best_w, best_sm = 1.0, 1e9
        for w in BLEND_GRID:
            combo_val = w * val_preds_orig_ensemble + (1-w) * np.expm1(ridge_val_log)
            sm = smape(np.expm1(y_val_log), combo_val)
            if sm < best_sm:
                best_sm = sm; best_w = w
        log(f"Ridge blend weight (LGB contribution): {best_w:.2f}, val SMAPE: {best_sm:.4f}%")
        combo_val_orig = best_w * val_preds_orig_ensemble + (1-best_w) * np.expm1(ridge_val_log)
        combo_test_orig = best_w * test_preds_orig_ensemble + (1-best_w) * np.expm1(ridge_test_log)
    else:
        combo_val_orig = val_preds_orig_ensemble
        combo_test_orig = test_preds_orig_ensemble

    # multiplicative calibration
    alphas = np.linspace(0.7, 1.3, 61)
    best_alpha, best_alpha_sm = 1.0, 1e9
    y_val_true = np.expm1(y_val_log)
    for a in alphas:
        sm = smape(y_val_true, combo_val_orig * a)
        if sm < best_alpha_sm:
            best_alpha_sm = sm; best_alpha = a
    log(f"Calibration alpha chosen: {best_alpha:.4f}, val SMAPE after alpha: {best_alpha_sm:.4f}%")
    combo_val_orig *= best_alpha
    combo_test_orig *= best_alpha

    oof[val_idx] = np.log1p(combo_val_orig)
    test_preds += combo_test_orig / N_FOLDS

    np.save(Path(CACHE_DIR)/"oof_partial.npy", oof)
    np.save(Path(CACHE_DIR)/"test_partial.npy", test_preds)
    resume_marker.write_text(str(fold))

    fold_sm = smape(y_val_true, combo_val_orig)
    log(f"Fold {fold} SMAPE: {fold_sm:.4f}%")
    gc.collect()

log("All folds complete ✅")

# finalize
oof_is_log = np.nanmedian(oof) > 1.0
oof_orig = np.expm1(oof) if oof_is_log else oof.copy()
test_preds_orig = test_preds.copy()
final_smape = smape(train["price"].values, oof_orig)
log(f"\nFinal OOF SMAPE: {final_smape:.4f}%")

# save artifacts & submission
np.save(Path(CACHE_DIR)/"train_features.npy", X_train)
np.save(Path(CACHE_DIR)/"test_features.npy", X_test)
np.save(Path(CACHE_DIR)/"oof_orig_preds.npy", oof_orig)
np.save(Path(CACHE_DIR)/"test_orig_preds.npy", test_preds_orig)
np.savez(Path(CACHE_DIR)/"oof_test_preds.npz", oof=oof_orig, test=test_preds_orig)

sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_preds_orig})
sub.to_csv("submission_model1_public.csv", index=False)
log(f"Saved submission_model1_public.csv and caches in {CACHE_DIR}")
log("Done ✅")
