"""
Client Redis asynchrone avec pool de connexions, configuration centralisée et healthcheck.
"""
import asyncio
import logging
import os
from typing import Optional

from redis import asyncio as aioredis

logger = logging.getLogger(__name__)

# Configuration issue des variables d'environnement
REDIS_URL = os.getenv("REDIS_URL", "redis://redis-cache:6379/0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "2.0"))


class RedisClient:
    """Singleton pour la connexion Redis asynchrone."""

    _instance: Optional[aioredis.Redis] = None

    @classmethod
    async def get_client(cls) -> aioredis.Redis:
        """Retourne le client Redis unique (création si nécessaire)."""
        if cls._instance is None:
            cls._instance = await cls._connect()
        return cls._instance

    @classmethod
    async def _connect(cls) -> aioredis.Redis:
        """Établit la connexion avec retry et backoff."""
        retries = 5
        backoff = 1.0
        last_exception = None
        for attempt in range(retries):
            try:
                client = await aioredis.from_url(
                    REDIS_URL,
                    password=REDIS_PASSWORD or None,
                    max_connections=REDIS_MAX_CONNECTIONS,
                    socket_timeout=REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
                    socket_keepalive=True,
                    health_check_interval=30,
                    decode_responses=True,
                    retry_on_timeout=True,
                )
                await client.ping()
                logger.info("RedisClient connecté à %s", REDIS_URL)
                return client
            except Exception as e:
                last_exception = e
                logger.warning("Tentative %d/%d échouée pour Redis : %s. Nouvel essai dans %.1fs",
                                attempt+1, retries, e, backoff)
                await asyncio.sleep(backoff)
                backoff *= 2  # double à chaque essai
        raise last_exception
    @classmethod
    async def close(cls):
        """Ferme la connexion Redis proprement."""
        if cls._instance:
            await cls._instance.close()
            await cls._instance.connection_pool.disconnect()
            cls._instance = None
            logger.info("RedisClient fermé")

    @classmethod
    async def healthcheck(cls) -> bool:
        """Vérifie que Redis est accessible et répond au PING."""
        try:
            client = await cls.get_client()
            await client.ping()
            return True
        except Exception:
            return False