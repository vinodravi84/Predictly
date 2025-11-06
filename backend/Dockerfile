# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Working Directory ----------
WORKDIR /app

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy Files ----------
COPY . /app

# ---------- Install Python Dependencies ----------
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Environment Variables ----------
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# ---------- Expose Port ----------
EXPOSE 7860

# ---------- Run Server ----------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
