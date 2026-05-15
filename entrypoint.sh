#!/bin/bash

echo "Waiting for Redis..."
until python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('redis-cache',6379)); s.close()" 2>/dev/null; do
    sleep 1
done
echo "Redis is ready."

# Donner la permissions a l'user appuser sur le dossier vector_cnaps_db
if [ -d "/app/vector_cnaps_db" ]; then
    chown -R appuser:appuser /app/vector_cnaps_db
fi
# Lancer la commande principale (uvicorn)
exec "$@"