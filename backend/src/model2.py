#!/usr/bin/env python3
"""
Memory-sane Model2 MPNet — updated for small GPUs (4GB)
- Preserves LoRA and frozen modes, per-fold resume, cache validation
- Adds gradient_checkpointing, OOM backoff (batch/seqlen), aggressive cleanup
- Tries retries when CUDA OOM occurs (auto-adjusts batch/max_length)
"""
import os, gc, json, math, random, joblib, re, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATA_DIR = "dataset"
CACHE_DIR = "cache_model2"
os.makedirs(CACHE_DIR, exist_ok=True)

MODE = "lora"           # "lora" or "frozen"
N_FOLDS = 5
SEED = 42

# MEMORY-FRIENDLY DEFAULTS (tweak these if you have more GPU memory)
BASE_BATCH_SIZE = 8         # starting batch (will auto-backoff if OOM)
BASE_MAX_LENGTH = 96        # starting max tokens (was 128)
GRADIENT_ACCUMULATION_STEPS = 1   # simulate larger batch if needed
EPOCHS = 3
LR = 2e-4

# LoRA / model settings
MPNET_NAME = "sentence-transformers/all-mpnet-base-v2"
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# frozen-mode SVD / TFIDF dims
EMB_SVD_DIM = 192
CHAR_TFIDF_MAX_FEATURES = 20000
CHAR_SVD_DIM = 64

# GBDT params for frozen mode
USE_CATBOOST = True
CAT_PARAMS = {"iterations":5000, "learning_rate":0.03, "depth":8, "loss_function":"RMSE", "verbose":200}
LGB_PARAMS = {"objective":"regression", "metric":"rmse", "learning_rate":0.02, "num_leaves":128, "n_jobs":-1}

random.seed(SEED); np.random.seed(SEED)

# set PyTorch allocator environment tweak to reduce fragmentation (recommended by error)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------- HELPERS ----------------
def clean_text(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"http\S+", " ", x)
    x = re.sub(r"[^a-z0-9\s\.\-×/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def smape(y_true, y_pred):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(num / np.clip(den, 1e-8, None)) * 100.0

def extract_basic_numeric(df):
    s = df["catalog_content"].fillna("").astype(str)
    out = pd.DataFrame(index=df.index)
    pq = s.str.extract(r'(?:pack|pk|pkt|packs?|packets?)\s*(?:of\s*)?(\d{1,4})')[0]
    px = s.str.extract(r'(\d{1,4})\s*[x×]\s*\d*')[0]
    out["pack_qty"] = pd.to_numeric(pq.fillna(px).fillna(1), errors="coerce").fillna(1)
    out["token_len"] = s.str.split().apply(len).fillna(0).astype(int)
    out["char_len"] = s.str.len().fillna(0).astype(int)
    return out

def save_resume_state(completed_folds, oof_arr, test_arr):
    state = {"completed_folds": sorted(list(completed_folds))}
    with open(Path(CACHE_DIR)/"resume.json", "w") as f:
        json.dump(state, f)
    np.save(Path(CACHE_DIR)/"oof_partial.npy", oof_arr)
    np.save(Path(CACHE_DIR)/"test_partial.npy", test_arr)

def load_resume_state():
    resume_f = Path(CACHE_DIR)/"resume.json"
    if resume_f.exists():
        try:
            state = json.load(open(resume_f))
            completed = set(state.get("completed_folds", []))
        except Exception:
            completed = set()
    else:
        completed = set()
    oof_p = Path(CACHE_DIR)/"oof_partial.npy"
    test_p = Path(CACHE_DIR)/"test_partial.npy"
    oof_arr = np.load(oof_p) if oof_p.exists() else None
    test_arr = np.load(test_p) if test_p.exists() else None
    return completed, oof_arr, test_arr

def find_latest_checkpoint_dir(base_out_dir: Path):
    if not base_out_dir.exists(): return None
    ckpts = [d for d in base_out_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
    if not ckpts:
        return None
    def keyfn(p):
        parts = p.name.split("-")
        try:
            return int(parts[-1])
        except:
            return 0
    ckpts.sort(key=keyfn)
    return ckpts[-1]

def validate_npy(path: Path, expected_len: int):
    if not path.exists():
        return False
    try:
        arr = np.load(path)
        if getattr(arr, "shape", None) and arr.shape[0] == expected_len:
            return True
        else:
            print(f"[WARN] Cache {path} exists but shape {getattr(arr,'shape',None)} != expected {expected_len}. Deleting corrupt cache.")
            try:
                path.unlink()
            except Exception:
                pass
            return False
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}. Deleting and recomputing.")
        try:
            path.unlink()
        except Exception:
            pass
        return False

def cleanup_torch():
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        gc.collect()

# ---------------- DEP CHECKS ----------------
HAS_TRANSFORMERS = False
HAS_PEFT = False
HAS_SENTENCE_TRANSFORMERS = False
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, default_data_collator
    HAS_TRANSFORMERS = True
    import packaging.version as _pv
    TRANSFORMERS_VERSION = getattr(transformers, "__version__", "0.0.0")
except Exception:
    HAS_TRANSFORMERS = False
    TRANSFORMERS_VERSION = "0.0.0"

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False

# If LoRA requested but dependencies missing -> fallback
if MODE == "lora" and not (HAS_TRANSFORMERS and HAS_PEFT):
    print("[WARN] Transformers/PEFT not available. Falling back to MODE='frozen'.")
    MODE = "frozen"

# ---------------- LOAD DATA ----------------
print("[INFO] Loading CSVs...")
train = pd.read_csv(Path(DATA_DIR)/"train.csv", usecols=["sample_id","catalog_content","price"])
test  = pd.read_csv(Path(DATA_DIR)/"test.csv",  usecols=["sample_id","catalog_content"])

print("[INFO] Cleaning text...")
for df in (train, test):
    df["catalog_content"] = df["catalog_content"].fillna("").astype(str).map(clean_text)

train_num = extract_basic_numeric(train)
test_num = extract_basic_numeric(test)

price_raw = train["price"].values.copy()
y_log = np.log1p(price_raw)

# KFold splits (deterministic)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(kf.split(train))

# Resume state load
completed_folds, oof_partial, test_partial = load_resume_state()
if oof_partial is None:
    oof = np.zeros(len(train))
else:
    oof = oof_partial.copy()
if test_partial is None:
    test_preds = np.zeros(len(test))
else:
    test_preds = test_partial.copy()

print(f"[INFO] Completed folds on resume: {sorted(list(completed_folds))}")

# ---------------- MODE: LORA ----------------
if MODE == "lora":
    print("[MODE] LoRA (single-GPU, resumable, memory-optimized)")

    import torch
    # dataset wrapper
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, enc, labels=None):
            self.enc = enc
            self.labels = labels
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            if self.labels is not None:
                item["labels"] = self.labels[idx]
            return item

    tokenizer = AutoTokenizer.from_pretrained(MPNET_NAME, use_fast=True)

    # helper to attempt training with backoff
    def attempt_fold_train(tr_idx, val_idx, fold_idx):
        # define per-fold dynamic settings
        batch_candidates = [BASE_BATCH_SIZE, max(1, BASE_BATCH_SIZE//2), 1]
        maxlen_candidates = [BASE_MAX_LENGTH, max(32, BASE_MAX_LENGTH//2)]
        last_exception = None

        for batch_size in batch_candidates:
            for max_len in maxlen_candidates:
                print(f"[Attempt] Fold {fold_idx} trying batch={batch_size}, max_len={max_len}")
                cleanup_torch()
                try:
                    # prepare tokenization with current max_len
                    X_tr_texts = train.loc[tr_idx, "catalog_content"].tolist()
                    X_val_texts = train.loc[val_idx, "catalog_content"].tolist()
                    y_tr = np.log1p(train.loc[tr_idx, "price"].values).astype(np.float32)
                    y_val = np.log1p(train.loc[val_idx, "price"].values).astype(np.float32)

                    enc_tr = tokenizer(X_tr_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                    enc_val = tokenizer(X_val_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                    enc_test = tokenizer(test["catalog_content"].tolist(), padding=True, truncation=True, max_length=max_len, return_tensors="pt")

                    ds_tr = TextDataset(enc_tr, labels=torch.tensor(y_tr))
                    ds_val = TextDataset(enc_val, labels=torch.tensor(y_val))
                    ds_test = TextDataset(enc_test, labels=None)

                    # build model and apply LoRA adapters
                    model = AutoModelForSequenceClassification.from_pretrained(MPNET_NAME, num_labels=1)
                    # enable gradient checkpointing if available
                    if hasattr(model, "gradient_checkpointing_enable"):
                        try:
                            model.gradient_checkpointing_enable()
                        except Exception:
                            pass

                    model.to("cuda" if torch.cuda.is_available() else "cpu")

                    peft_cfg = LoraConfig(
                        r=LORA_R, lora_alpha=LORA_ALPHA,
                        target_modules=["query", "key", "value", "q", "k", "v", "dense"],
                        lora_dropout=LORA_DROPOUT, bias="none", task_type="SEQ_CLS"
                    )
                    model = get_peft_model(model, peft_cfg)
                    model.print_trainable_parameters()

                    # Build TrainingArguments robustly (try modern API, fallback to minimal)
                    try:
                        training_args = TrainingArguments(
                            output_dir=str(Path(CACHE_DIR)/f"lora_fold{fold_idx}"),
                            evaluation_strategy="epoch",
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                            num_train_epochs=EPOCHS,
                            learning_rate=LR,
                            weight_decay=0.01,
                            fp16=True if torch.cuda.is_available() else False,
                            save_strategy="epoch",
                            logging_strategy="steps",
                            logging_steps=200,
                            remove_unused_columns=False,
                            save_total_limit=3,
                            load_best_model_at_end=False,
                            report_to="none",
                            seed=SEED + fold_idx
                        )
                    except TypeError as e:
                        training_args = TrainingArguments(
                            output_dir=str(Path(CACHE_DIR)/f"lora_fold{fold_idx}"),
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            num_train_epochs=EPOCHS,
                            learning_rate=LR,
                            fp16=True if torch.cuda.is_available() else False,
                            save_total_limit=3,
                            seed=SEED + fold_idx,
                            logging_steps=200,
                        )

                    def compute_metrics(eval_pred):
                        preds = eval_pred.predictions.squeeze()
                        labels = eval_pred.label_ids
                        preds_orig = np.expm1(preds)
                        labels_orig = np.expm1(labels)
                        return {"smape": smape(labels_orig, preds_orig)}

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=ds_tr,
                        eval_dataset=ds_val,
                        tokenizer=tokenizer,
                        data_collator=default_data_collator,
                        compute_metrics=compute_metrics
                    )

                    # resume checkpoint if exists
                    resume_ckpt = find_latest_checkpoint_dir(Path(CACHE_DIR)/f"lora_fold{fold_idx}")
                    if resume_ckpt:
                        print(f"[INFO] Found checkpoint for fold {fold_idx}: {resume_ckpt.name} — resuming from it.")
                        try:
                            trainer.train(resume_from_checkpoint=str(resume_ckpt))
                        except TypeError:
                            trainer.train(str(resume_ckpt))
                    else:
                        trainer.train()

                    # predictions
                    val_out = trainer.predict(ds_val)
                    test_out = trainer.predict(ds_test)
                    val_preds_log = val_out.predictions.squeeze()
                    test_preds_log = test_out.predictions.squeeze()

                    val_preds_orig = np.expm1(val_preds_log)
                    test_preds_orig = np.expm1(test_preds_log)

                    # calibration
                    y_val_true = np.expm1(y_val)
                    alphas = np.linspace(0.8, 1.2, 81)
                    best_a, best_sm = 1.0, 1e9
                    for a in alphas:
                        s = smape(y_val_true, val_preds_orig * a)
                        if s < best_sm:
                            best_sm = s; best_a = a
                    print(f"[Fold {fold_idx}] calibration alpha={best_a:.4f} -> smape {best_sm:.4f}%")
                    val_preds_orig *= best_a
                    test_preds_orig *= best_a

                    # cleanup heavy objects before returning results
                    try:
                        trainer.save_model(str(Path(CACHE_DIR)/f"lora_fold{fold_idx}/final_model"))
                    except Exception:
                        pass

                    # return arrays scaled as original (not log)
                    return val_preds_orig, test_preds_orig

                except RuntimeError as e:
                    last_exception = e
                    tb = traceback.format_exc()
                    print("[OOM/RuntimeError] Fold attempt failed:", str(e))
                    print(tb.splitlines()[-1])
                    # aggressively cleanup then continue to next candidate
                    try:
                        del model, trainer, ds_tr, ds_val, ds_test, enc_tr, enc_val, enc_test
                    except Exception:
                        pass
                    cleanup_torch()
                    # fallback to smaller batch / smaller max_len (next loop iteration)
                    time.sleep(2)
                    continue
                except Exception as e:
                    last_exception = e
                    print("[ERROR] Unexpected failure during training attempt:", e)
                    tb = traceback.format_exc()
                    print(tb)
                    try:
                        del model, trainer, ds_tr, ds_val, ds_test
                    except Exception:
                        pass
                    cleanup_torch()
                    break  # non-OOM unexpected error: break out
            # end for max_len
        # end for batch_size
        # if we reach here we failed all attempts
        raise last_exception if last_exception is not None else RuntimeError("Unknown failure in attempt_fold_train()")

    # iterate folds in fixed reproducible order
    for fold_idx, (tr_idx, val_idx) in enumerate(splits, 1):
        out_dir = Path(CACHE_DIR)/f"lora_fold{fold_idx}"
        val_file = Path(CACHE_DIR)/f"fold{fold_idx}_val_preds.npy"
        test_file = Path(CACHE_DIR)/f"fold{fold_idx}_test_preds.npy"

        if fold_idx in completed_folds and validate_npy(val_file, len(val_idx)) and validate_npy(test_file, len(test)):
            print(f"[LOOP] Fold {fold_idx} already completed — loading cached preds.")
            fold_val = np.load(val_file)
            fold_test = np.load(test_file)
            oof[val_idx] = np.log1p(fold_val)
            test_preds += fold_test / N_FOLDS
            continue

        print(f"\n[LOOP] Fold {fold_idx}/{N_FOLDS} — starting/resuming")

        try:
            val_preds_orig, test_preds_orig = attempt_fold_train(tr_idx, val_idx, fold_idx)
        except Exception as e:
            print(f"[FATAL] Fold {fold_idx} failed after retries: {e}")
            # Save state and re-raise to stop — you can edit to skip fold instead
            save_resume_state(completed_folds, oof, test_preds)
            raise

        # persist per-fold preds
        np.save(val_file, val_preds_orig)
        np.save(test_file, test_preds_orig)

        # Update aggregated arrays and resume state
        oof[val_idx] = np.log1p(val_preds_orig)
        test_preds += test_preds_orig / N_FOLDS
        completed_folds.add(fold_idx)
        save_resume_state(completed_folds, oof, test_preds)

        # final cleanup
        cleanup_torch()
        time.sleep(1)

    # finalize
    oof_orig = np.expm1(oof)
    final_smape = smape(train["price"].values, oof_orig)
    print(f"\n[LoRA] Final OOF SMAPE: {final_smape:.4f}%")

    np.save(Path(CACHE_DIR)/"model2_oof_preds.npy", oof_orig)
    np.save(Path(CACHE_DIR)/"model2_test_preds.npy", test_preds)
    pd.DataFrame({"sample_id": test["sample_id"], "price": test_preds}).to_csv(Path("submission_model2.csv"), index=False)
    print(f"[LoRA] Saved artifacts to {CACHE_DIR}")

# ---------------- MODE: FROZEN ----------------
else:
    print("[MODE] Frozen embeddings (single-GPU/CPU friendly)")

    # compute / load embeddings
    has_sbert = HAS_SENTENCE_TRANSFORMERS

    emb_train_path = Path(CACHE_DIR)/"train_emb.npy"
    emb_test_path  = Path(CACHE_DIR)/"test_emb.npy"
    svd_emb_path   = Path(CACHE_DIR)/"model2_emb_svd.joblib"

    # Validate existing embeddings (shape check)
    if validate_npy(emb_train_path, len(train)) and validate_npy(emb_test_path, len(test)) and svd_emb_path.exists():
        try:
            emb_train = np.load(emb_train_path)
            emb_test  = np.load(emb_test_path)
            svd_emb   = joblib.load(svd_emb_path)
            if getattr(emb_train, "shape", None) and emb_train.shape[0] == len(train):
                print("[Frozen] Loaded cached embeddings & SVD.")
            else:
                raise Exception("Train embedding shape mismatch after load.")
            emb_train = svd_emb.transform(emb_train)
            emb_test  = svd_emb.transform(emb_test)
        except Exception as e:
            print(f"[WARN] Cache load problem: {e}. Recomputing embeddings.")
            try:
                emb_train_path.unlink()
            except:
                pass
            try:
                emb_test_path.unlink()
            except:
                pass
            if svd_emb_path.exists():
                try: svd_emb_path.unlink()
                except: pass
            emb_train = emb_test = None
    else:
        emb_train = emb_test = None

    if emb_train is None or emb_test is None:
        print("[Frozen] Computing MPNet embeddings (CPU/GPU — might take time).")
        if has_sbert:
            from sentence_transformers import SentenceTransformer
            model_s = SentenceTransformer(MPNET_NAME, device="cuda" if (torch and torch.cuda.is_available()) else "cpu")
            emb_all_train = model_s.encode(train["catalog_content"].tolist(), batch_size=max(1, BASE_BATCH_SIZE//2), show_progress_bar=True, convert_to_numpy=True)
            emb_all_test  = model_s.encode(test["catalog_content"].tolist(), batch_size=max(1, BASE_BATCH_SIZE//2), show_progress_bar=True, convert_to_numpy=True)
        else:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(MPNET_NAME, use_fast=True)
            model_hf = AutoModel.from_pretrained(MPNET_NAME).to("cuda" if (torch and torch.cuda.is_available()) else "cpu")
            model_hf.eval()
            def mean_pool_texts(texts):
                import torch as _torch
                all_emb = []
                for i in tqdm(range(0, len(texts), max(1, BASE_BATCH_SIZE//2))):
                    batch = texts[i:i+max(1, BASE_BATCH_SIZE//2)]
                    enc = tokenizer(batch, padding=True, truncation=True, max_length=BASE_MAX_LENGTH, return_tensors="pt")
                    input_ids = enc["input_ids"].to("cuda" if (torch and torch.cuda.is_available()) else "cpu")
                    mask = enc["attention_mask"].to("cuda" if (torch and torch.cuda.is_available()) else "cpu")
                    with _torch.no_grad():
                        out = model_hf(input_ids=input_ids, attention_mask=mask, return_dict=True)
                        last = out.last_hidden_state
                        mask_f = mask.unsqueeze(-1).expand(last.size()).float()
                        summed = _torch.sum(last * mask_f, 1)
                        counts = _torch.clamp(mask_f.sum(1), min=1e-9)
                        emb = (summed / counts).cpu().numpy()
                        all_emb.append(emb)
                    # free GPU memory for each chunk
                    del enc, input_ids, mask, out, last, mask_f, summed, counts, emb
                    gc.collect()
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return np.vstack(all_emb)
            emb_all_train = mean_pool_texts(train["catalog_content"].tolist())
            emb_all_test  = mean_pool_texts(test["catalog_content"].tolist())
        print("[Frozen] Reducing embeddings with SVD...")
        svd_emb = TruncatedSVD(n_components=EMB_SVD_DIM, random_state=SEED)
        svd_emb.fit(np.vstack([emb_all_train, emb_all_test]))
        emb_train = svd_emb.transform(emb_all_train)
        emb_test  = svd_emb.transform(emb_all_test)
        np.save(emb_train_path, emb_train)
        np.save(emb_test_path, emb_test)
        joblib.dump(svd_emb, svd_emb_path)
        try:
            del model_s, model_hf
        except:
            pass
        gc.collect()

    # char TF-IDF + SVD
    char_train_path = Path(CACHE_DIR)/"model2_char_train.npy"
    char_test_path  = Path(CACHE_DIR)/"model2_char_test.npy"
    char_svd_path   = Path(CACHE_DIR)/"model2_char_svd.joblib"
    char_vect_path  = Path(CACHE_DIR)/"model2_char_vect.joblib"

    if validate_npy(char_train_path, len(train)) and validate_npy(char_test_path, len(test)) and char_svd_path.exists():
        char_train = np.load(char_train_path)
        char_test  = np.load(char_test_path)
        char_svd = joblib.load(char_svd_path)
        print("[Frozen] Loaded cached char TF-IDF & SVD.")
    else:
        X_all = pd.concat([train["catalog_content"], test["catalog_content"]]).astype(str)
        from sklearn.feature_extraction.text import TfidfVectorizer
        char_vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), max_features=CHAR_TFIDF_MAX_FEATURES, sublinear_tf=True)
        char_vect.fit(X_all)
        X_train_char = char_vect.transform(train["catalog_content"])
        X_test_char  = char_vect.transform(test["catalog_content"])
        char_svd = TruncatedSVD(n_components=CHAR_SVD_DIM, random_state=SEED)
        char_train = char_svd.fit_transform(X_train_char)
        char_test  = char_svd.transform(X_test_char)
        np.save(char_train_path, char_train); np.save(char_test_path, char_test)
        joblib.dump(char_svd, char_svd_path); joblib.dump(char_vect, char_vect_path)

    # assemble features
    X_train = np.hstack([emb_train, char_train, train_num.values])
    X_test  = np.hstack([emb_test,  char_test,  test_num.values])

    # standardize numeric tail columns
    scaler_path = Path(CACHE_DIR)/"model2_scaler.joblib"
    num_tail = train_num.shape[1]
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_train[:, -num_tail:] = scaler.transform(X_train[:, -num_tail:])
        X_test[:, -num_tail:]  = scaler.transform(X_test[:, -num_tail:])
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.vstack([X_train[:, -num_tail:], X_test[:, -num_tail:]]))
        X_train[:, -num_tail:] = scaler.transform(X_train[:, -num_tail:])
        X_test[:, -num_tail:]  = scaler.transform(X_test[:, -num_tail:])
        joblib.dump(scaler, scaler_path)

    # KFold train with per-fold resume similar to LoRA approach
    try:
        from catboost import CatBoostRegressor, Pool
        HAVE_CATBOOST = True
    except Exception:
        HAVE_CATBOOST = False
        import lightgbm as lgb

    for fold_idx, (tr_idx, val_idx) in enumerate(splits, 1):
        val_file = Path(CACHE_DIR)/f"fold{fold_idx}_val_preds.npy"
        test_file = Path(CACHE_DIR)/f"fold{fold_idx}_test_preds.npy"

        if fold_idx in completed_folds and validate_npy(val_file, len(val_idx)) and validate_npy(test_file, len(test)):
            print(f"[FROZEN] Fold {fold_idx} already completed — loading cached preds.")
            fold_val = np.load(val_file)
            fold_test = np.load(test_file)
            oof[val_idx] = np.log1p(fold_val)
            test_preds += fold_test / N_FOLDS
            continue

        print(f"\n[FROZEN] Fold {fold_idx}/{N_FOLDS} — training")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]

        fold_model_path = Path(CACHE_DIR)/f"model2_frozen_fold{fold_idx}.pkl"

        if HAVE_CATBOOST and USE_CATBOOST:
            model = CatBoostRegressor(**CAT_PARAMS)
            pool_tr = Pool(X_tr, y_tr)
            pool_val = Pool(X_val, y_val)
            model.fit(pool_tr, eval_set=pool_val, use_best_model=True, verbose=200)
            val_pred_log = model.predict(X_val)
            test_pred_log = model.predict(X_test)
            try:
                model.save_model(str(Path(CACHE_DIR)/f"model2_cat_fold{fold_idx}.cbm"))
            except Exception:
                pass
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=20000, valid_sets=[dval], early_stopping_rounds=300, verbose_eval=200)
            best_it = getattr(model, "best_iteration", None) or 20000
            val_pred_log = model.predict(X_val, num_iteration=best_it)
            test_pred_log = model.predict(X_test, num_iteration=best_it)
            try:
                model.save_model(str(Path(CACHE_DIR)/f"model2_lgb_fold{fold_idx}.txt"))
            except Exception:
                pass

        val_pred_orig = np.expm1(val_pred_log)
        test_pred_orig = np.expm1(test_pred_log)

        # calibration
        y_val_true = np.expm1(y_val)
        best_a, best_sm = 1.0, 1e9
        for a in np.linspace(0.8, 1.2, 81):
            s = smape(y_val_true, val_pred_orig * a)
            if s < best_sm:
                best_sm = s; best_a = a
        print(f"[Fold {fold_idx}] calibration alpha={best_a:.4f} -> smape {best_sm:.4f}%")
        val_pred_orig *= best_a
        test_pred_orig *= best_a

        # save per-fold preds and update resume state
        np.save(val_file, val_pred_orig)
        np.save(test_file, test_pred_orig)
        oof[val_idx] = np.log1p(val_pred_orig)
        test_preds += test_pred_orig / N_FOLDS
        completed_folds.add(fold_idx)
        save_resume_state(completed_folds, oof, test_preds)

        del model
        gc.collect()

    oof_orig = np.expm1(oof)
    final_smape = smape(train["price"].values, oof_orig)
    print(f"\n[FROZEN] Final OOF SMAPE: {final_smape:.4f}%")

    np.save(Path(CACHE_DIR)/"model2_oof_preds.npy", oof_orig)
    np.save(Path(CACHE_DIR)/"model2_test_preds.npy", test_preds)
    pd.DataFrame({"sample_id": test["sample_id"], "price": test_preds}).to_csv(Path("submission_model2.csv"), index=False)
    print(f"[FROZEN] Saved artifacts to {CACHE_DIR}")

print("\nAll done. You can re-run this script to resume if interrupted.")
