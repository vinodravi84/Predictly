# 🧠 Smart Price Prediction System


**🌐 Live Demo:** [https://predictly-ten.vercel.app/](https://predictly-ten.vercel.app/)

Built with:

A full-stack AI-powered web application that predicts the **price of a product** based on its **image** and **text description**.

Built with:
- **FastAPI** (Python backend with ML models)
- **React.js** (interactive frontend)
- **PyTorch, LightGBM, SBERT, and LoRA fine-tuned MPNET** (under the hood)

---

## 🚀 Overview

This system combines multiple machine learning models (Model1, Model2, Model3) and an ensemble layer (`combine12.py`) to estimate prices accurately from product data.

The backend serves both:
- **Text-based price estimation** using TF-IDF, SVD, Ridge, and LightGBM.
- **Image-based price estimation** using an EfficientNet-B0 regression model.
- A **FastAPI REST API** that integrates both and delivers real-time predictions to the React frontend.

The frontend allows users to upload an image or type/paste product details to get an instant predicted price.

---

## 🧩 Project Structure
```
root/
│
├── app.py                    # FastAPI backend server
├── model1.py                 # Text regression model (SBERT + LightGBM)
├── model2.py                 # LoRA fine-tuned MPNET model
├── model3.py                 # Image regression (EfficientNet/ResNet)
├── combine12.py              # Ensemble meta-model combining Model1 & Model2
│
├── cache_model1/             # Cached embeddings/features for Model1
├── cache_model2/             # Cached artifacts for Model2
├── cache_model_simple/       # Image model checkpoints
├── cache_merge/              # Ensemble artifacts (TF-IDF, SVD, Ridge, LGBM)
│
├── dataset/                  # train.csv, test.csv, and images/
│   ├── train.csv
│   ├── test.csv
│   └── images/
│       ├── train/
│       └── test/
│
├── frontend/                 # React app
│   ├── src/
│   │   ├── App.js
│   │   └── components/
│   │       └── PredictForm.js
│   └── public/
│
└── README.md
```

---

## 🧠 Model Summary

| Model | Purpose | Core Tech | Output |
|--------|----------|------------|---------|
| **Model1** | Text-only regression | SBERT embeddings + LightGBM | Price |
| **Model2** | Transformer fine-tuned regression | LoRA over MPNet | Price |
| **Model3** | Image regression | EfficientNet / ResNet | Price |
| **combine12.py** | Ensemble layer | Ridge + LightGBM meta learner | Final prediction |
| **app.py** | API server | FastAPI + Torch + LGBM | REST endpoints |

---

## ⚙️ Backend Setup (FastAPI)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

Typical dependencies include:
```
fastapi
uvicorn
torch
torchvision
sentence-transformers
lightgbm
joblib
pandas
numpy
scikit-learn
Pillow
tqdm
```

### 2️⃣ Run the Server
```bash
uvicorn app:app --reload
```

Server runs by default on:

🔗 http://127.0.0.1:8000

You can test endpoints via:

📘 http://127.0.0.1:8000/docs

---

## 💻 Frontend Setup (React.js)

### 1️⃣ Navigate to frontend
```bash
cd frontend
```

### 2️⃣ Install dependencies
```bash
npm install
```

### 3️⃣ Start the frontend
```bash
npm start
```

Frontend runs by default on:

🌐 http://localhost:3000

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predicts price from single text or image |
| `/predict/batch` | POST | Predicts multiple entries from JSON array |
| `/docs` | GET | Swagger UI for API testing |

**Example cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "catalog_content=Vitamin C tablets 500mg pack of 10" \
  -F "image=@./sample.jpg"
```

---

## 🧪 Ensemble Workflow

1. **Model1** extracts SBERT embeddings and numeric features → LightGBM regression.

2. **Model2** fine-tunes MPNET using LoRA → text regression.

3. **Model3** trains an image regressor (EfficientNet-B0).

4. **combine12.py** merges all outputs → builds meta TF-IDF + Ridge/LGBM ensemble.

5. **app.py** serves predictions from cached meta artifacts.

---

## 📦 Artifacts & Outputs

| Folder | Description |
|--------|-------------|
| `cache_model1/` | TF-IDF, SVD, and embeddings |
| `cache_model2/` | Transformer checkpoints |
| `cache_model_simple/` | EfficientNet weights |
| `cache_merge/` | Meta ensemble artifacts |
| `dataset/` | Training/test data and product images |

---

## 🧰 Environment Variables

Create a `.env` file (optional):
```bash
ALLOWED_ORIGINS=http://localhost:3000
REACT_APP_API_BASE=http://127.0.0.1:8000
```

---

## 📊 Performance Metrics

- **Evaluation metric:** SMAPE (Symmetric Mean Absolute Percentage Error)
- **LightGBM:** GPU optimized with early stopping
- **Transformer:** LoRA fine-tuning for low-memory GPUs
- **Image model:** EfficientNet-B0 (pretrained on ImageNet)

---

## 🧱 Future Enhancements

- ✅ Add multilingual product text handling
- ✅ Optimize TF-IDF caching for production
- ✅ Dockerize backend + frontend
- ✅ Add Redis queue for async predictions
- ✅ Deploy on AWS/GCP

---

## 📄 License

This project is licensed under the MIT License.



