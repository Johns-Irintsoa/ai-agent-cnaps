FROM python:3.11-slim

WORKDIR /app

RUN echo "deb http://deb.debian.org/debian bookworm contrib non-free" \
        >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y \
        p7zip-full \
        unrar \
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/main.py"]
