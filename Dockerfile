FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scanner/ ./scanner/
COPY frontend/ ./frontend/
COPY backend.py .
COPY config.yaml .

RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
