# syntax=docker/dockerfile:1

# ============================================
# STAGE 1 : Builder
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Augmenter le timeout pour apt-get
RUN echo 'Acquire::http::Timeout "30";' > /etc/apt/apt.conf.d/99timeout \
    && apt-get update && apt-get install -y \
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

# Configuration PIP pour éviter les déconnexions (IncompleteRead)
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=20 \
    PIP_NO_CACHE_DIR=0

RUN pip install --upgrade pip --quiet

# ── Groupe 1 : Outils stables & Web ──────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --progress-bar off \
        python-dotenv "httpx==0.27.2" rarfile \
        fastapi "uvicorn[standard]" pydantic-settings

# ── Groupe 2 : IA & Parsing (Le coeur du problème) ──
# Installation de Torch CPU en premier pour alléger la suite
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --progress-bar off \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

# INSTALLATION GROUPÉE : On laisse pip résoudre le conflit Typer entre Docling et Transformers
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --progress-bar off \
        docling \
        transformers \
        scipy \
        numpy \
        pylatexenc \
        "beautifulsoup4==4.12.3" \
        "pypdf==4.3.1" \
        "python-docx==1.1.2" \
        "unstructured[image]"

# ── Groupe 3 : Autres outils lourds & LangChain ──
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --progress-bar off \
        chromadb \
        langchain \
        langchain-core \
        langchain-openai \
        langchain-community \
        langchain-chroma \
        langchain-unstructured

# ── Groupe 4 : Modèle spaCy ──
RUN pip install spacy && python -m spacy download en_core_web_sm

# Vérification immédiate de l'import Docling
RUN python -c "from docling.document_converter import DocumentConverter; print('Docling importé avec succès !')"

# ============================================
# STAGE 2 : Runtime
# ============================================
FROM python:3.11-slim

# Créer l'utilisateur avant tout pour les permissions
RUN useradd -m -u 1000 appuser

RUN echo "deb http://deb.debian.org/debian bookworm non-free" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y \
        p7zip-full unrar tesseract-ocr tesseract-ocr-fra \
        poppler-utils libmagic1 libtesseract5 libpoppler-cpp2 \
        libgl1 libglib2.0-0 libgomp1 wget \
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Récupérer l'environnement virtuel du builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY cnaps_urls.json .

# Préparer les répertoires de cache pour les modèles IA
RUN mkdir -p /home/appuser/.cache/docling /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser/.cache /app

USER appuser

# Le script sera exécuté avec le venv actif par défaut via le PATH
CMD ["python", "src/main.py"]