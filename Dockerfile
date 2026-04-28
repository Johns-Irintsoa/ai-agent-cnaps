# syntax=docker/dockerfile:1

# ============================================
# STAGE 1 : Builder
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Augmenter le timeout pour les apt-get
RUN echo 'Acquire::http::Timeout "10";' > /etc/apt/apt.conf.d/99timeout \
    && echo 'Acquire::ftp::Timeout "10";' >> /etc/apt/apt.conf.d/99timeout

RUN apt-get update && apt-get install -y \
        build-essential \
        libmagic-dev \
        libtesseract-dev \
        libpoppler-cpp-dev \
        gcc \
        g++ \
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Configuration PIP pour plus de robustesse
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=15 \
    PIP_NO_CACHE_DIR=0

RUN pip install --upgrade pip --quiet

# ── Groupe 1 : Outils stables ──────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=600 --retries=15 \
        python-dotenv \
        "httpx==0.27.2" \
        rarfile

# ── Groupe 2 : Framework web ───────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=600 --retries=15 \
        fastapi \
        "uvicorn[standard]" \
        pydantic-settings

# ── Groupe 3 : Parsing de documents & Docling ──
# Installer pylatexenc en premier (dépendance critique de docling)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=900 --retries=15 \
        pylatexenc

# Installer les dépendances lourdes
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=900 --retries=15 \
        torch --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=900 --retries=15 \
        torchvision --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=900 --retries=15 \
        transformers \
        scipy \
        numpy

# Installer docling et SES DÉPENDANCES (sans --no-deps)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=900 --retries=15 \
        docling \
        docling-core \
        docling-parse \
        docling-ibm-models \
        rapidocr-onnxruntime \
        onnxruntime \
        "beautifulsoup4==4.12.3" \
        "pypdf==4.3.1" \
        "python-docx==1.1.2"

# Installer explicitement les dépendances manquantes de docling
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=900 --retries=15 \
        marko>=2.1.2 \
        openpyxl>=3.1.5 \
        pluggy>=1.0.0 \
        polyfactory>=2.22.2 \
        python-pptx>=1.0.2 \
        rapidocr>=3.8 \
        'typer<0.22.0,>=0.12.5'

# ── Groupe 4a : Unstructured (lourd ~800MB) ────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=900 --retries=15 \
        "unstructured[image]"

# ── Groupe 4b : ChromaDB (lourd ~200MB) ────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=900 --retries=15 \
        chromadb

# ── Groupe 5 : LangChain (change souvent) ──────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=600 --retries=15 \
        langchain \
        langchain-core \
        langchain-text-splitters \
        langchain-openai \
        langchain-community \
        langchain-chroma \
        langchain-unstructured

# ── Groupe 6 : Modèle spaCy ──
RUN python -m spacy download en_core_web_sm

# Vérification finale que docling peut être importé
RUN python -c "from docling.document_converter import DocumentConverter; print('Docling importé avec succès')"

USER root
RUN chown -R appuser:appuser /opt/venv
USER appuser

# ============================================
# STAGE 2 : Runtime
# ============================================
FROM python:3.11-slim

RUN echo "deb http://deb.debian.org/debian bookworm non-free" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y \
        p7zip-full \
        unrar \
        tesseract-ocr \
        tesseract-ocr-fra \
        poppler-utils \
        libmagic1 \
        libtesseract5 \
        libpoppler-cpp2 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        wget \
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY cnaps_urls.json .

# Création du répertoire pour les modèles Docling
RUN mkdir -p /home/appuser/.cache/docling && chown -R 1000:1000 /home/appuser/.cache/docling

RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "src/main.py"]