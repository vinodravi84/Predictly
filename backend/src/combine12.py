#!/usr/bin/env python3
"""
combine12.py - robust merging / stacking for Model1 + Model2 caches

Features:
- Fully robust loader for .npy and .npz archives (extracts sensible arrays)
- Detects embeddings/features vs 1D preds (2D features won't be mistaken as preds)
- Assembles OOF from per-fold val preds when present
- Fallback quick proxy training on features if OOF missing
- Uses LightGBM with callback-style early stopping to avoid API mismatch
- Prints Model1 & Model2 SMAPE right after loading
- Re-uses TF-IDF+SVD artifacts from Model1 if present to build meta text features
- Saves meta artifacts and submission into cache_merge/
"""
import os, sys, math, time, gc, random
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import lightgbm as lgb

# ---------------- CONFIG ----------------
DATA_DIR = Path("dataset")
CACHE1 = Path("cache_model1")
CACHE2 = Path("cache_model2")
MERGE_CACHE = Path("cache_merge")
OUT_SUB = Path("submission_ensemble.csv")

N_FOLDS = 5
SEED = 42
VERBOSE = True

# LightGBM meta config (use callbacks style)
LGB_META = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 128,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": SEED,
    "n_jobs": max(1, (os.cpu_count() or 1) - 1)
}
NUM_BOOST_ROUND = 10000
EARLY_STOPPING = 300
VERBOSE_EVAL = 100

TFIDF_MAX = 30000
TFIDF_SVD = 128

PROXY_FOLDS = 3  # small CV for proxy fallback

random.seed(SEED)
np.random.seed(SEED)

# ---------------- HELPERS ----------------
def log(msg: str):
    if VERBOSE:
        print(f"[merge] {msg}", flush=True)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.clip(denom, 1e-8, None)) * 100.0)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# robust loader: .npy or .npz -> plain ndarray (1D or 2D)
def robust_load_array(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    try:
        if path.suffix.lower() == ".npz":
            arc = np.load(path, allow_pickle=False)
            prefer = ("oof","oof_orig","oof_preds","test","test_preds","preds","arr_0")
            for k in prefer:
                if k in arc:
                    arr = np.array(arc[k])
                    return arr
            files = getattr(arc, "files", [])
            if files:
                return np.array(arc[files[0]])
            return None
        else:
            arr = np.load(path, allow_pickle=False)
            return np.array(arr)
    except Exception as e:
        log(f"robust_load_array failed for {path}: {e}")
        return None

def is_embeddings_like(arr: np.ndarray) -> bool:
    if arr is None:
        return False
    if arr.ndim == 2 and arr.shape[1] > 1:
        return True
    return False

def ensure_1d_preds(arr: Optional[np.ndarray], expected_len: int, name: str) -> Optional[np.ndarray]:
    """Return 1D arr of expected_len or None if arr looks like features (2D) or invalid."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] > 1:
        log(f"{name} is 2D (shape={a.shape}) - treating as features, not preds.")
        return None
    a = a.reshape(-1)
    if a.shape[0] == expected_len:
        return a
    if a.shape[0] > expected_len:
        log(f"{name} longer ({a.shape[0]}) than expected ({expected_len}) -> truncating.")
        return a[:expected_len]
    med = float(np.nanmedian(a)) if a.size>0 else 0.0
    log(f"{name} shorter ({a.shape[0]}) than expected ({expected_len}) -> padding with median {med:.4f}.")
    pad = np.full(expected_len - a.shape[0], med, dtype=a.dtype)
    return np.concatenate([a, pad])

def list_cache_files(cache_dir: Path) -> List[Path]:
    if not cache_dir.exists(): return []
    files = [p for p in cache_dir.rglob("*") if p.is_file()]
    files_sorted = sorted(files, key=lambda p: p.as_posix())
    return files_sorted

def find_artifacts(cache_dir: Path):
    """
    Search known candidate filenames and return dict:
      { 'oof': Path|None, 'test': Path|None, 'fold_val': [Paths], 'fold_test': [Paths], 'train_feat': Path|None, 'test_feat': Path|None }
    """
    out = {"oof": None, "test": None, "fold_val": [], "fold_test": [], "train_feat": None, "test_feat": None, "other": []}
    if not cache_dir.exists(): return out
    for p in list_cache_files(cache_dir):
        n = p.name.lower()
        out["other"].append(p)
        if any(x in n for x in ("oof_orig_preds.npy","oof_orig.npy","oof.npy","model2_oof_preds.npy","model_oof_preds.npy")):
            out["oof"] = p
        if n.endswith(".npz") and ("oof" in n or "oof_test" in n):
            out["oof"] = p
        if any(x in n for x in ("test_orig_preds.npy","test_orig.npy","test.npy","model2_test_preds.npy","model_test_preds.npy")):
            out["test"] = p
        if "fold" in n and "val" in n and p.suffix==".npy":
            out["fold_val"].append(p)
        if "fold" in n and "test" in n and p.suffix==".npy":
            out["fold_test"].append(p)
        if p.name in ("train_features.npy","train_features.npz","train_emb.npy","train_emb.npz"):
            out["train_feat"] = p
        if p.name in ("test_features.npy","test_emb.npy","test_features.npz","test_emb.npz"):
            out["test_feat"] = p
    out["fold_val"] = sorted(out["fold_val"])
    out["fold_test"] = sorted(out["fold_test"])
    return out

# Assemble OOF from fold val preds if possible
def assemble_oof_from_folds(val_paths: List[Path], test_paths: List[Path], n_train: int, n_test: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not val_paths:
        return None, None
    arrays = []
    for p in val_paths:
        a = robust_load_array(p)
        if a is None:
            continue
        arrays.append(a)
    fulls = [a for a in arrays if a is not None and a.shape[0]==n_train]
    if fulls:
        stacked = np.vstack([f.reshape(-1) for f in fulls])
        oof = np.nanmedian(stacked, axis=0)
    else:
        lens = [a.shape[0] for a in arrays if a is not None]
        if sum(lens) == n_train:
            oof = np.full(n_train, np.nan)
            pos = 0
            for a in arrays:
                l = a.shape[0]
                oof[pos:pos+l] = a
                pos += l
        else:
            log("Cannot assemble OOF from fold files (size mismatch).")
            oof = None
    test_preds = None
    if test_paths:
        t_acc = None; cnt = 0
        for p in test_paths:
            a = robust_load_array(p)
            if a is None: continue
            if t_acc is None: t_acc = np.zeros_like(a, dtype=float)
            t_acc += a; cnt += 1
        if t_acc is not None and cnt>0:
            test_preds = t_acc / cnt
    return oof, test_preds

# Quick proxy training to produce OOF/test if features exist but OOF missing
def quick_proxy_train(train_feats: np.ndarray, test_feats: np.ndarray, y_log: np.ndarray, folds=PROXY_FOLDS):
    from sklearn.model_selection import KFold
    cfg = dict(LGB_META)
    kf = KFold(n_splits=min(folds,3), shuffle=True, random_state=SEED)
    oof = np.zeros(train_feats.shape[0])
    tpred = np.zeros(test_feats.shape[0])
    for tr, va in kf.split(train_feats):
        dtr = lgb.Dataset(train_feats[tr], label=y_log[tr])
        dva = lgb.Dataset(train_feats[va], label=y_log[va])
        callbacks = [lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)]
        bst = lgb.train(cfg, dtr, num_boost_round=2000, valid_sets=[dva], valid_names=["val"], callbacks=callbacks)
        it = getattr(bst, "best_iteration", None) or 2000
        oof[va] = np.expm1(bst.predict(train_feats[va], num_iteration=it))
        tpred += np.expm1(bst.predict(test_feats, num_iteration=it)) / kf.get_n_splits()
        del bst; gc.collect()
    return oof, tpred

# detect GPU presence (torch preferred)
def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# ---------------- MAIN FLOW ----------------
ensure_dir(MERGE_CACHE)
log(f"merge cache -> {MERGE_CACHE.resolve()}")

# read CSVs
train_csv = DATA_DIR / "train.csv"
test_csv  = DATA_DIR / "test.csv"
if not train_csv.exists() or not test_csv.exists():
    raise FileNotFoundError("Please place train.csv and test.csv inside dataset/")

train_df = pd.read_csv(train_csv, usecols=["sample_id","catalog_content","price"])
test_df  = pd.read_csv(test_csv, usecols=["sample_id","catalog_content"])
n_train = len(train_df); n_test = len(test_df)
y_price = train_df["price"].values.copy()
y_log = np.log1p(y_price)

# discover artifacts
log("Inspecting model caches...")
m1 = find_artifacts(CACHE1)
m2 = find_artifacts(CACHE2)
log(f"Model1: oof={m1['oof']}, test={m1['test']}, train_feat={m1['train_feat']}, val-folds={len(m1['fold_val'])}")
log(f"Model2: oof={m2['oof']}, test={m2['test']}, train_feat={m2['train_feat']}, val-folds={len(m2['fold_val'])}")

# load or assemble oofs & tests
m1_oof = robust_load_array(m1["oof"])
m1_test = robust_load_array(m1["test"])
if m1_oof is None:
    mo, mt = assemble_oof_from_folds(m1["fold_val"], m1["fold_test"], n_train, n_test)
    if mo is not None:
        m1_oof = mo
    if m1_test is None and mt is not None:
        m1_test = mt
if m1_test is not None and is_embeddings_like(m1_test):
    log("Model1 test artifact looks like embeddings/features -> ignoring as test preds.")
    m1_test = None

m2_oof = robust_load_array(m2["oof"])
m2_test = robust_load_array(m2["test"])
if m2_oof is None:
    mo, mt = assemble_oof_from_folds(m2["fold_val"], m2["fold_test"], n_train, n_test)
    if mo is not None:
        m2_oof = mo
    if m2_test is None and mt is not None:
        m2_test = mt
if m2_test is not None and is_embeddings_like(m2_test):
    log("Model2 test artifact looks like embeddings/features -> ignoring as test preds.")
    m2_test = None

# If OOF missing but train features exist, run quick proxy
if m1_oof is None and m1["train_feat"] is not None and m1["test_feat"] is not None:
    log("Model1: OOF missing BUT train_features/test_features present -> training quick proxy LGB")
    trf = robust_load_array(m1["train_feat"]); tef = robust_load_array(m1["test_feat"])
    if trf is not None and tef is not None and trf.ndim==2:
        m1_oof, m1_test = quick_proxy_train(trf, tef, np.log1p(y_price), folds=PROXY_FOLDS)

if m2_oof is None and m2["train_feat"] is not None and m2["test_feat"] is not None:
    log("Model2: OOF missing BUT train_features/test_features present -> training quick proxy LGB")
    trf = robust_load_array(m2["train_feat"]); tef = robust_load_array(m2["test_feat"])
    if trf is not None and tef is not None and trf.ndim==2:
        m2_oof, m2_test = quick_proxy_train(trf, tef, np.log1p(y_price), folds=PROXY_FOLDS)

# finalize shapes & convert to 1D preds
m1_oof = ensure_1d_preds(m1_oof, n_train, "m1_oof")
m2_oof = ensure_1d_preds(m2_oof, n_train, "m2_oof")
m1_test = ensure_1d_preds(m1_test, n_test, "m1_test")
m2_test = ensure_1d_preds(m2_test, n_test, "m2_test")

# print base smapes
if m1_oof is not None:
    try:
        mask = ~np.isnan(m1_oof)
        log(f"Model1 OOF SMAPE: {smape(y_price[mask], m1_oof[mask]):.4f}%")
    except Exception as e:
        log(f"Model1 SMAPE compute failed: {e}")
else:
    log("Model1 OOF not available.")

if m2_oof is not None:
    try:
        mask = ~np.isnan(m2_oof)
        log(f"Model2 OOF SMAPE: {smape(y_price[mask], m2_oof[mask]):.4f}%")
    except Exception as e:
        log(f"Model2 SMAPE compute failed: {e}")
else:
    log("Model2 OOF not available.")

# Build base matrix (if only one model exists, duplicate it to avoid degenerate stacking)
base_train_cols = []
base_test_cols = []
if m1_oof is not None:
    base_train_cols.append(m1_oof)
    base_test_cols.append(m1_test if m1_test is not None else np.full(n_test, np.nan))
else:
    base_train_cols.append(np.full(n_train, np.nan))
    base_test_cols.append(np.full(n_test, np.nan))

if m2_oof is not None:
    base_train_cols.append(m2_oof)
    base_test_cols.append(m2_test if m2_test is not None else np.full(n_test, np.nan))
else:
    base_train_cols.append(np.full(n_train, np.nan))
    base_test_cols.append(np.full(n_test, np.nan))

BASE_TRAIN = np.vstack(base_train_cols).T
BASE_TEST  = np.vstack(base_test_cols).T

# Add simple aggregations as meta features
meta_mean = np.nanmean(BASE_TRAIN, axis=1).reshape(-1,1)
meta_std = np.nanstd(BASE_TRAIN, axis=1).reshape(-1,1)
meta_min = np.nanmin(BASE_TRAIN, axis=1).reshape(-1,1)
meta_max = np.nanmax(BASE_TRAIN, axis=1).reshape(-1,1)
meta_nnan = np.isnan(BASE_TRAIN).sum(axis=1).reshape(-1,1)

# ---------------- TF-IDF -> SVD meta features (reuse Model1 artifacts first) ----------------
log("Preparing TF-IDF -> SVD meta features (will try to reuse Model1 artifacts first).")

# Candidate Model1 artifact names (from Model1 script)
m1_tfidf_path = CACHE1 / "tfidf_vectorizer.joblib"
m1_svd_path   = CACHE1 / "svd_model.joblib"
# optional cached reduced arrays Model1 might have saved
m1_train_tfidf_svd = CACHE1 / "train_tfidf_svd.npy"
m1_test_tfidf_svd  = CACHE1 / "test_tfidf_svd.npy"

# Local merge-cache locations (where we'll store a copy if we compute/fetch)
meta_tfidf_job = MERGE_CACHE / "meta_tfidf.joblib"
meta_svd_job   = MERGE_CACHE / "meta_tfidf_svd.joblib"
meta_train_svd = MERGE_CACHE / "meta_train_tfidf_svd.npy"
meta_test_svd  = MERGE_CACHE / "meta_test_tfidf_svd.npy"

train_tfidf_svd = None
test_tfidf_svd  = None

# 1) Prefer to load the exact TF-IDF + SVD from Model1 cache (keeps features identical)
if m1_tfidf_path.exists() and m1_svd_path.exists():
    try:
        log(f"Loading TF-IDF and SVD from Model1 cache: {m1_tfidf_path.name}, {m1_svd_path.name}")
        tf = joblib.load(m1_tfidf_path)
        svd = joblib.load(m1_svd_path)
        train_tfidf_svd = svd.transform(tf.transform(train_df["catalog_content"].astype(str)))
        test_tfidf_svd  = svd.transform(tf.transform(test_df["catalog_content"].astype(str)))
        # save a copy into MERGE_CACHE for reproducibility
        joblib.dump(tf, meta_tfidf_job)
        joblib.dump(svd, meta_svd_job)
        np.save(meta_train_svd, train_tfidf_svd)
        np.save(meta_test_svd, test_tfidf_svd)
        log("Loaded and cached Model1 TF-IDF+SVD successfully.")
    except Exception as e:
        log(f"Failed loading Model1 TF-IDF+SVD: {e} -- will try other fallbacks.")

# 2) If Model1 saved reduced arrays, try to reuse them
if train_tfidf_svd is None and m1_train_tfidf_svd.exists() and m1_test_tfidf_svd.exists():
    try:
        train_tfidf_svd = np.load(m1_train_tfidf_svd)
        test_tfidf_svd  = np.load(m1_test_tfidf_svd)
        joblib.dump(None, meta_tfidf_job)  # mark that we used cached arrays (optional)
        np.save(meta_train_svd, train_tfidf_svd)
        np.save(meta_test_svd, test_tfidf_svd)
        log("Loaded precomputed TF-IDF-SVD reduced arrays from Model1 cache.")
    except Exception as e:
        log(f"Failed loading Model1 reduced arrays: {e}")

# 3) If still missing, but Model1 has tfidf+svd under different names, try common alternatives
if train_tfidf_svd is None:
    alt_tfidf = CACHE1 / "tfidf_vectorizer.pkl"
    alt_svd   = CACHE1 / "svd_model.pkl"
    if alt_tfidf.exists() and alt_svd.exists():
        try:
            tf = joblib.load(alt_tfidf)
            svd = joblib.load(alt_svd)
            train_tfidf_svd = svd.transform(tf.transform(train_df["catalog_content"].astype(str)))
            test_tfidf_svd  = svd.transform(tf.transform(test_df["catalog_content"].astype(str)))
            joblib.dump(tf, meta_tfidf_job); joblib.dump(svd, meta_svd_job)
            np.save(meta_train_svd, train_tfidf_svd); np.save(meta_test_svd, test_tfidf_svd)
            log("Loaded alternative TF-IDF+SVD names from Model1 cache.")
        except Exception as e:
            log(f"Alt TFIDF/SVD load failed: {e}")

# 4) Last fallback: fit a fresh TF-IDF + SVD and cache into MERGE_CACHE
if train_tfidf_svd is None:
    log("No TF-IDF/SVD found in Model1 cache -> fitting fresh TF-IDF + SVD for meta (this may take time).")
    tf = TfidfVectorizer(ngram_range=(1,2), max_features=TFIDF_MAX, min_df=3, max_df=0.95, sublinear_tf=True, strip_accents="unicode")
    Xall = pd.concat([train_df["catalog_content"], test_df["catalog_content"]]).astype(str)
    tf.fit(Xall)
    joblib.dump(tf, meta_tfidf_job)
    X_train_tf = tf.transform(train_df["catalog_content"].astype(str))
    svd = TruncatedSVD(n_components=TFIDF_SVD, random_state=SEED)
    svd.fit(X_train_tf)
    joblib.dump(svd, meta_svd_job)
    train_tfidf_svd = svd.transform(X_train_tf)
    test_tfidf_svd  = svd.transform(tf.transform(test_df["catalog_content"].astype(str)))
    np.save(meta_train_svd, train_tfidf_svd)
    np.save(meta_test_svd, test_tfidf_svd)
    log("Fitted and cached TF-IDF+SVD into merge cache.")

log(f"TF-IDF-SVD shapes: {train_tfidf_svd.shape} / {test_tfidf_svd.shape}")

# assemble final meta matrices
META_X_train = np.hstack([BASE_TRAIN, meta_mean, meta_std, meta_min, meta_max, meta_nnan, train_tfidf_svd])
META_X_test  = np.hstack([BASE_TEST,  np.nanmean(BASE_TEST, axis=1).reshape(-1,1), np.nanstd(BASE_TEST, axis=1).reshape(-1,1),
                          np.nanmin(BASE_TEST, axis=1).reshape(-1,1), np.nanmax(BASE_TEST, axis=1).reshape(-1,1),
                          np.isnan(BASE_TEST).sum(axis=1).reshape(-1,1), test_tfidf_svd])
log(f"META_X_train shape: {META_X_train.shape} ; META_X_test shape: {META_X_test.shape}")

np.save(MERGE_CACHE / "meta_X_train.npy", META_X_train)
np.save(MERGE_CACHE / "meta_X_test.npy", META_X_test)

# scale for Ridge/MLP if needed
scaler = StandardScaler()
META_X_train_s = scaler.fit_transform(META_X_train)
META_X_test_s  = scaler.transform(META_X_test)
joblib.dump(scaler, MERGE_CACHE / "meta_scaler.joblib")

# Stratified KFold on price quantiles for meta training
n_bins = 20
try:
    bins = pd.qcut(y_price, q=n_bins, labels=False, duplicates="drop")
except Exception:
    bins = pd.cut(y_price, bins=n_bins, labels=False)
bins = np.asarray(bins, dtype=int)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# prepare accumulators
meta_oof_dict = {"lgb": np.full(n_train, np.nan), "ridge": np.full(n_train, np.nan)}
meta_test_acc = {"lgb": np.zeros(n_test), "ridge": np.zeros(n_test)}

log(f"Starting meta training (folds={N_FOLDS}) - GPU available: {has_gpu()}")

fold_i = 0
for tr_idx, val_idx in skf.split(META_X_train, bins):
    fold_i += 1
    log(f"--- Meta fold {fold_i}/{N_FOLDS} ---")
    Xtr = META_X_train[tr_idx]; Xval = META_X_train[val_idx]
    Xtr_s = META_X_train_s[tr_idx]; Xval_s = META_X_train_s[val_idx]
    ytr_log = np.log1p(y_price[tr_idx]); yval_log = np.log1p(y_price[val_idx])

    # LightGBM using callback-style early stopping
    dtr = lgb.Dataset(Xtr, label=ytr_log)
    dval = lgb.Dataset(Xval, label=yval_log)
    cfg = dict(LGB_META)
    if has_gpu():
        cfg["device"] = "gpu"; cfg["device_type"] = "gpu"; cfg["gpu_platform_id"]=0; cfg["gpu_device_id"]=0
    callbacks = [lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(VERBOSE_EVAL)]
    try:
        bst = lgb.train(cfg, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dtr,dval], valid_names=["train","valid"], callbacks=callbacks)
    except Exception as e:
        log(f"LGB train fallback: {e} -> retrying CPU-only config")
        cfg2 = dict(LGB_META)
        callbacks = [lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(VERBOSE_EVAL)]
        bst = lgb.train(cfg2, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dtr,dval], valid_names=["train","valid"], callbacks=callbacks)
    it = getattr(bst, "best_iteration", None) or NUM_BOOST_ROUND
    val_pred_log = bst.predict(Xval, num_iteration=it)
    test_pred_log = bst.predict(META_X_test, num_iteration=it)
    val_pred = np.expm1(val_pred_log); test_pred = np.expm1(test_pred_log)

    # multiplicative calibration small grid
    alphas = np.linspace(0.9, 1.1, 41)
    best_a, best_sm = 1.0, 1e9
    y_val_true = np.expm1(yval_log)
    for a in alphas:
        s = smape(y_val_true, val_pred * a)
        if s < best_sm:
            best_sm = s; best_a = a
    log(f"LGB fold {fold_i} alpha={best_a:.4f} SMAPE={best_sm:.4f}%")
    val_pred *= best_a; test_pred *= best_a

    meta_oof_dict["lgb"][val_idx] = val_pred
    meta_test_acc["lgb"] += test_pred / N_FOLDS
    try:
        bst.save_model(str(MERGE_CACHE / f"meta_lgb_fold{fold_i}.txt"))
    except Exception:
        pass
    del bst, dtr, dval; gc.collect()

    # Ridge on scaled matrix
    r = Ridge(alpha=1.0, random_state=SEED)
    r.fit(Xtr_s, ytr_log)
    rval_log = r.predict(Xval_s); rtest_log = r.predict(META_X_test_s)
    rval = np.expm1(rval_log); rtest = np.expm1(rtest_log)
    best_a, best_sm = 1.0, 1e9
    for a in alphas:
        s = smape(y_val_true, rval * a)
        if s < best_sm:
            best_sm = s; best_a = a
    log(f"Ridge fold {fold_i} alpha={best_a:.4f} SMAPE={best_sm:.4f}%")
    rval *= best_a; rtest *= best_a
    meta_oof_dict["ridge"][val_idx] = rval
    meta_test_acc["ridge"] += rtest / N_FOLDS
    joblib.dump(r, MERGE_CACHE / f"meta_ridge_fold{fold_i}.joblib")
    del r; gc.collect()

# ensure no NaNs remain
for k,v in meta_oof_dict.items():
    if np.isnan(v).any():
        med = np.nanmedian(v[~np.isnan(v)])
        v[np.isnan(v)] = med
        log(f"Filled NaNs in meta_oof {k} with median {med:.4f}")

# assemble meta OOF matrix (columns: lgb, ridge)
meta_oof_mat = np.vstack([meta_oof_dict["lgb"], meta_oof_dict["ridge"]]).T
meta_test_mat = np.vstack([meta_test_acc["lgb"], meta_test_acc["ridge"]]).T
joblib.dump(meta_oof_mat, MERGE_CACHE / "meta_oof_matrix.npy")
joblib.dump(meta_test_mat, MERGE_CACHE / "meta_test_matrix.npy")
log(f"meta_oof_mat shape {meta_oof_mat.shape}")

# simple convex blend search via grid on 2 dims (coarse)
log("Searching for best convex blend weights for meta learners (lgb, ridge).")
best_sm, best_w = 1e9, None
for w in np.linspace(0.0, 1.0, 101):
    combo = w * meta_oof_mat[:,0] + (1.0 - w) * meta_oof_mat[:,1]
    s = smape(y_price, combo)
    if s < best_sm:
        best_sm = s; best_w = (w, 1.0-w)
log(f"Best convex blend OOF SMAPE {best_sm:.4f} with weights {best_w}")

# apply blend to test
final_test_combo = best_w[0] * meta_test_mat[:,0] + best_w[1] * meta_test_mat[:,1]
final_oof_combo = best_w[0] * meta_oof_mat[:,0] + best_w[1] * meta_oof_mat[:,1]

# final residual LightGBM on log-space residuals
log("Training residual LGB on log-residuals to correct bias.")
res_target = np.log1p(y_price) - np.log1p(final_oof_combo + 1e-12)
dres = lgb.Dataset(meta_oof_mat, label=res_target)
cfg = dict(LGB_META)
if has_gpu():
    cfg["device"]="gpu"; cfg["device_type"]="gpu"; cfg["gpu_platform_id"]=0; cfg["gpu_device_id"]=0
callbacks = [lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(VERBOSE_EVAL)]
bst_res = lgb.train(cfg, dres, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dres], valid_names=["train"], callbacks=callbacks)
it = getattr(bst_res, "best_iteration", None) or NUM_BOOST_ROUND
res_oof_log = bst_res.predict(meta_oof_mat, num_iteration=it)
res_test_log = bst_res.predict(meta_test_mat, num_iteration=it)

final_oof = np.expm1(np.log1p(final_oof_combo) + res_oof_log)
final_test = np.expm1(np.log1p(final_test_combo) + res_test_log)

final_sm = smape(y_price, final_oof)
log(f"Final ensemble OOF SMAPE after residual correction: {final_sm:.4f}%")

# save artifacts and submission
ensure_dir(MERGE_CACHE)
np.save(MERGE_CACHE / "final_oof.npy", final_oof)
np.save(MERGE_CACHE / "final_test.npy", final_test)
joblib.dump(best_w, MERGE_CACHE / "meta_blend_weights.joblib")
try:
    bst_res.save_model(str(MERGE_CACHE / "residual_lgb.txt"))
except Exception:
    pass
joblib.dump(bst_res, MERGE_CACHE / "residual_lgb.joblib")
pd.DataFrame({"sample_id": test_df["sample_id"], "price": final_test}).to_csv(OUT_SUB, index=False)
log(f"Saved submission -> {OUT_SUB.resolve()}")
log("Done ✅")
