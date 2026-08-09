```dockerfile
# ==========================================================
# Base Image
# ==========================================================

FROM python:3.11-slim

# ==========================================================
# Environment Variables
# ==========================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ==========================================================
# Create Non-Root User
# ==========================================================

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser

# ==========================================================
# Working Directory
# ==========================================================

WORKDIR /app

# ==========================================================
# Install System Dependencies
# ==========================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# Copy Requirements
# ==========================================================

COPY requirements.txt /app/requirements.txt

# ==========================================================
# Install Python Packages
# ==========================================================

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# ==========================================================
# Copy Application Files
# ==========================================================

COPY --chown=appuser:appgroup app/ /app/app/
COPY --chown=appuser:appgroup model/ /app/model/
COPY --chown=appuser:appgroup data/ /app/data/

# ==========================================================
# Secure Application Files
# ==========================================================

RUN find /app -type d -exec chmod 755 {} \; \
    && find /app -type f -exec chmod 644 {} \;

# ==========================================================
# Run Container as Non-Root User
# ==========================================================

USER appuser

# ==========================================================
# Expose FastAPI Port
# ==========================================================

EXPOSE 8000

# ==========================================================
# Health Check
# ==========================================================

HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=10s \
    --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# ==========================================================
# Run FastAPI
# ==========================================================

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
