# syntax=docker/dockerfile:1

# ============================================
# STAGE 1 : Builder
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

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

RUN pip install --upgrade pip --quiet

# ── Groupe 1 : Outils stables ──────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=300 --retries=5 \
        python-dotenv \
        "httpx==0.27.2" \
        rarfile

# ── Groupe 2 : Framework web ───────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=300 --retries=5 \
        fastapi \
        "uvicorn[standard]"

# ── Groupe 3 : Parsing de documents ───────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=300 --retries=5 \
        "beautifulsoup4==4.12.3" \
        "pypdf==4.3.1" \
        "python-docx==1.1.2"

# ── Groupe 4a : Unstructured (lourd ~800MB) ────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=600 --retries=5 \
        "unstructured[image]"

# ── Groupe 4b : ChromaDB (lourd ~200MB) ────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=600 --retries=5 \
        chromadb

# ── Groupe 5 : LangChain (change souvent) ──────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=300 --retries=5 \
        langchain \
        langchain-core \
        langchain-text-splitters \
        langchain-openai \
        langchain-community \
        langchain-chroma

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
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY cnaps_urls.json .

RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "src/main.py"]