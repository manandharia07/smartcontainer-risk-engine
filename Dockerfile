# ──────────────────────────────────────────────────────────────
# SmartContainer Risk Engine — Dockerfile
# ──────────────────────────────────────────────────────────────
# Base image: slim Python 3.11 for a smaller image size
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# ── Install system dependencies needed by LightGBM / SHAP ─────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies first (layer caching) ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source code ──────────────────────────────
COPY dashboard.py .
COPY train_model.py .
COPY predict.py .

# ── Copy pre-trained model artifacts ──────────────────────────
# These are baked into the image so the dashboard starts immediately
# without needing to train first.
COPY risk_model.pkl .
COPY isolation_forest.pkl .
COPY label_encoders.pkl .
COPY risk_rate_tables.pkl .


# Streamlit configuration: disable CORS & set server options
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# ── Default command: launch the dashboard ─────────────────────
CMD ["streamlit", "run", "dashboard.py"]
