FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AI_PROVIDER=disabled \
    PORT=7860 \
    HF_HOME=/app/model-cache \
    TRANSFORMERS_CACHE=/app/model-cache

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 7860
CMD ["sh", "-c", "python init_db.py && uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
