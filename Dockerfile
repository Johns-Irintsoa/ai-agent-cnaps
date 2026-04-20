FROM python:3.11-slim

WORKDIR /app

RUN echo "deb http://deb.debian.org/debian bookworm contrib non-free" \
        >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y \
        p7zip-full \
        unrar \
        tesseract-ocr \
        tesseract-ocr-fra \
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharger le modèle d'embedding au build (évite le téléchargement au runtime)
ARG EMBEDDINGS_MODEL=BAAI/bge-m3
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDINGS_MODEL}', cache_folder='/app/models')"

COPY src/ ./src/
COPY cnaps_urls.json .

CMD ["python", "src/main.py"]
