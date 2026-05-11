#!/bin/bash
# Donner la permissions a l'user appuser sur le dossier vector_cnaps_db
if [ -d "/app/vector_cnaps_db" ]; then
    chown -R appuser:appuser /app/vector_cnaps_db
fi
# Lancer la commande principale (uvicorn)
exec "$@"