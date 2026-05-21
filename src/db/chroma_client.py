"""
Client ChromaDB : HttpClient (Docker) ou PersistentClient (dev local).
Singleton avec retry + backoff exponentiel, calqué sur redis_client.py.

Les env vars sont lues dans _connect() (pas au niveau module) pour éviter
les problèmes d'ordre d'import avec load_dotenv().
"""
import logging
import os
import time
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)


class ChromaClient:
    """Singleton pour la connexion ChromaDB (HTTP ou locale)."""

    _instance: Optional[chromadb.ClientAPI] = None

    @classmethod
    def get_client(cls) -> chromadb.ClientAPI:
        """Retourne le client ChromaDB unique (création si nécessaire)."""
        if cls._instance is None:
            cls._instance = cls._connect()
        return cls._instance

    @classmethod
    def _connect(cls) -> chromadb.ClientAPI:
        """Établit la connexion avec retry et backoff exponentiel.
        Les env vars sont lues ici pour être sûr que load_dotenv() est passé.
        """
        host    = os.getenv("CHROMA_HOST", "")
        port    = int(os.getenv("CHROMA_PORT", "8000"))
        retries = int(os.getenv("CHROMA_RETRIES", "3"))
        backoff = float(os.getenv("CHROMA_BACKOFF", "0.5"))
        persist = os.getenv("VECTOR_DB_DIR", "./vector_cnaps_db")

        last_exception = None

        for attempt in range(retries):
            try:
                if host:
                    client = chromadb.HttpClient(host=host, port=port)
                    client.heartbeat()  # vérifie que le serveur répond
                    logger.info("ChromaClient connecté à http://%s:%d", host, port)
                else:
                    client = chromadb.PersistentClient(path=persist)
                    logger.info("ChromaClient local : %s", persist)
                return client
            except Exception as e:
                last_exception = e
                logger.warning(
                    "Tentative %d/%d échouée pour ChromaDB : %s. Nouvel essai dans %.1fs",
                    attempt + 1, retries, e, backoff,
                )
                time.sleep(backoff)
                backoff *= 2  # backoff exponentiel

        raise last_exception

    @classmethod
    def close(cls) -> None:
        """Réinitialise le singleton (HttpClient n'a pas de méthode close explicite)."""
        cls._instance = None
        logger.info("ChromaClient réinitialisé")

    @classmethod
    def healthcheck(cls) -> bool:
        """Vérifie que ChromaDB est accessible et répond."""
        try:
            client = cls.get_client()
            host = os.getenv("CHROMA_HOST", "")
            if host:
                client.heartbeat()
            return True
        except Exception:
            return False
