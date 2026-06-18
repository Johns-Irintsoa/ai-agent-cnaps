# syntax=docker/dockerfile:1
# Dockerfile optimisé pour le développement local avec découpage des dépendances
# Les requirements sont séparés en 3 fichiers pour maximiser le cache Docker

# -----------------------------------------------------------------------------
# Stage de construction (builder)
# -----------------------------------------------------------------------------
FROM img-torch-base AS builder

# Éviter les interactions tzdata et autres prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build

# Dépendances système nécessaires à la compilation des paquets Python
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libmagic-dev \
        libtesseract-dev \
        libpoppler-cpp-dev \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Créer l’environnement virtuel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Mettre à jour pip (optionnel mais recommandé)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip

# Copier et installer les dépendances par groupe (du plus stable au plus volatile)

# 1. Socle de base (rarement changé)
COPY requirements/base.txt /tmp/base.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/base.txt

# 2. ML / LLM / Vector DB (assez stable)
COPY requirements/ml.txt /tmp/ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/ml.txt

# 3. Traitement de documents (le plus souvent modifié)
COPY requirements/processing.txt /tmp/processing.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/processing.txt

# 4. Mises à jour ponctuelles (Oracle, etc.)
COPY requirements/update1.txt /tmp/update1.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/update1.txt

# -----------------------------------------------------------------------------
# Stage d’exécution (runtime)
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Créer un utilisateur non‑root
RUN useradd -m -u 1000 appuser

# Installer les dépendances système nécessaires au runtime (sans les compilateurs)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    echo "deb http://deb.debian.org/debian bookworm non-free" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        p7zip-full unrar tesseract-ocr tesseract-ocr-fra \
        poppler-utils libmagic1 libtesseract5 libpoppler-cpp2 \
        libgl1 libglib2.0-0 libgomp1 wget libaio1t64 unzip \
    && rm -rf /var/lib/apt/lists/*

# Oracle Instant Client — zip téléchargé manuellement dans instantclient/ (non commité)
COPY instantclient/instantclient-basic-linux.x64-23.26.2.0.0.zip /tmp/ic.zip
RUN unzip -q /tmp/ic.zip -d /opt/oracle \
    && rm /tmp/ic.zip \
    && echo /opt/oracle/instantclient_23_26 > /etc/ld.so.conf.d/oracle-instantclient.conf \
    && ln -sf /usr/lib/x86_64-linux-gnu/libaio.so.1t64 /usr/lib/x86_64-linux-gnu/libaio.so.1 \
    && ldconfig

# Copier l’environnement virtuel depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Créer les dossiers de cache utilisateur (pas de chown -R sur /opt/venv)
RUN mkdir -p /home/appuser/.cache/docling /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser \
    && chown appuser:appuser /app

# Copier le code source et les fichiers de configuration
COPY src/ /app/src/
COPY cnaps_urls.json /app/
# Utilisateur non‑root
USER appuser

# Exposer le port de l’application (informative)
EXPOSE 8000

# Commande par défaut (sans --reload, le compose l’ajoutera pour le dev)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]