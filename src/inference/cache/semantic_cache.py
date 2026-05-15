"""
Cache sémantique utilisant Redis + RediSearch pour la recherche vectorielle.
Dépend de RedisClient et LRUEmbeddingCache.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple




import numpy as np
from redis import asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from redis.commands.search.field import TextField, VectorField

from redis.commands.search.index_definition import IndexDefinition
from redis.commands.search.query import Query

from ...db.redis_client import RedisClient
from .embedding_cache import LRUEmbeddingCache
from ...models.embedding import embedding_manager

# Configuration
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
CACHE_MAX_ANSWER_TOKENS = int(os.getenv("CACHE_MAX_ANSWER_TOKENS", "4000"))
REDIS_INDEX_NAME = os.getenv("REDIS_INDEX_NAME", "idx:semantic_cache")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "semcache:")
RETRY_ATTEMPTS = int(os.getenv("CACHE_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.getenv("CACHE_RETRY_BACKOFF", "0.5"))

logger = logging.getLogger(__name__)


class SemanticCache:
    """Cache sémantique asynchrone."""

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._embedding_cache = LRUEmbeddingCache(maxsize=100)
        self._vector_dim: Optional[int] = None
        self._initialized = False

    async def _ensure_redis(self):
        """Assure que le client Redis est disponible."""
        if self._redis is None:
            self._redis = await RedisClient.get_client()
        return self._redis

    async def _init_index(self):
        redis = await self._ensure_redis()
        try:
            await redis.ft(REDIS_INDEX_NAME).info()
            logger.debug("Index Redis existe déjà : %s", REDIS_INDEX_NAME)
            return
        except Exception:
            logger.info("Création de l'index Redis %s", REDIS_INDEX_NAME)

        # Déterminer la dimension de l'embedding
        loop = asyncio.get_running_loop()
        sample = await loop.run_in_executor(
            None, embedding_manager.model.embed_query, "test"
        )
        self._vector_dim = len(sample)
        logger.info("Dimension des embeddings détectée : %d", self._vector_dim)

        # Définir les champs
        schema = (
            TextField("question"),
            TextField("answer"),
            TextField("metadata"),
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": self._vector_dim,
                "DISTANCE_METRIC": "COSINE",
                "M": 16,
                "EF_CONSTRUCTION": 200
            })
        )
        # Définir l'index (sans IndexType, le type HASH est implicite car aucun paramètre ne le définit)
        definition = IndexDefinition(prefix=[REDIS_KEY_PREFIX])  # index_type n'est pas requis
        try:
            await redis.ft(REDIS_INDEX_NAME).create_index(schema, definition=definition)
            logger.info("Index Redis créé avec succès")
        except Exception as e:
            logger.error("Échec création index Redis : %s", e)
            raise

    async def connect(self):
        """Initialise la connexion Redis et l'index."""
        if not self._initialized:
            await self._init_index()
            self._initialized = True
            logger.info("Cache sémantique prêt")

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Retourne l'embedding en utilisant le cache local LRU."""
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        emb = await loop.run_in_executor(
            None, embedding_manager.model.embed_query, text
        )
        emb_array = np.array(emb, dtype=np.float32)
        self._embedding_cache.set(text, emb_array)
        return emb_array

    async def _execute_with_retry(self, func, *args, **kwargs):
        """Exécute une commande Redis avec retry et backoff."""
        last_exception = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return await func(*args, **kwargs)
            except (ConnectionError, TimeoutError, RedisError) as e:
                last_exception = e
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Tentative %d/%d pour Redis échouée : %s. Nouvel essai dans %.2fs",
                    attempt + 1, RETRY_ATTEMPTS, e, wait
                )
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error("Erreur critique Redis : %s", e, exc_info=True)
                raise
        raise last_exception

    async def get(self, query: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Récupère depuis le cache si une question similaire existe.
        Retourne (answer, metadata) ou None.
        """
        if not self._initialized:
            logger.warning("Cache non initialisé, on ignore la lecture")
            return None

        try:
            emb = await self._get_embedding(query)
            emb_bytes = emb.tobytes()
            redis = await self._ensure_redis()

            search_query = (
                Query(f"* => [KNN 5 @embedding $vec AS score]")
                .return_fields("question", "answer", "metadata", "score")
                .dialect(2)
            )

            res = await self._execute_with_retry(
                redis.ft(REDIS_INDEX_NAME).search,
                search_query,
                query_params={"vec": emb_bytes}
            )


            best = None
            best_similarity = -1.0
            for doc in res.docs:
                similarity = 1.0 - float(doc.score)
                if similarity >= CACHE_SIMILARITY_THRESHOLD and similarity > best_similarity:
                    best = doc
                    best_similarity = similarity

            if best:
                metadata = json.loads(best.metadata) if best.metadata else {}
                logger.info("Cache HIT (sim=%.4f) : %s", best_similarity, best.question[:80])
                return (best.answer, metadata)
            logger.debug("Cache MISS pour : %s", query[:80])
            return None

        except Exception as e:
            logger.error("Erreur lecture cache : %s", e, exc_info=True)
            return None

    async def set(self, query: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Stocke une réponse dans le cache."""
        if not self._initialized:
            logger.warning("Cache non initialisé, écriture ignorée")
            return False

        if len(answer.split()) > CACHE_MAX_ANSWER_TOKENS:
            logger.info("Réponse trop longue (%d tokens) → non mise en cache", len(answer.split()))
            return False

        try:
            emb = await self._get_embedding(query)
            emb_bytes = emb.tobytes()
            doc_id = f"{REDIS_KEY_PREFIX}{abs(hash(query) % 10**8)}"
            data = {
                "question": query,
                "answer": answer,
                "metadata": json.dumps(metadata or {}),
                "embedding": emb_bytes,
            }
            redis = await self._ensure_redis()
            await self._execute_with_retry(redis.hset, doc_id, mapping=data)
            await self._execute_with_retry(redis.expire, doc_id, CACHE_TTL_SECONDS)
            logger.info("Cache SET pour : %s", query[:80])
            return True
        except Exception as e:
            logger.error("Erreur écriture cache : %s", e, exc_info=True)
            return False

    async def healthcheck(self) -> bool:
        """Vérifie la santé du cache (Redis + index)."""
        if not self._initialized:
            return False
        try:
            redis = await self._ensure_redis()
            await redis.ping()
            await redis.ft(REDIS_INDEX_NAME).info()
            return True
        except Exception:
            return False

    async def close(self):
        """Ferme le client Redis (via RedisClient)."""
        await RedisClient.close()
        self._redis = None
        self._initialized = False
        logger.info("Cache sémantique fermé")


# Singleton
semantic_cache = SemanticCache()