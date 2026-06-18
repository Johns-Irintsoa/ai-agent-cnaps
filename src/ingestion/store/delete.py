import logging
from typing import List

from ...db.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


def _get_collection(collection_name: str):
    client = ChromaClient.get_client()
    return client.get_or_create_collection(name=collection_name)


def delete_by_source(source: str, collection_name: str) -> int:
    """Supprime tous les chunks correspondant à la source (filename PDF ou URL).
    Cherche dans les deux champs metadata : 'source' (PDF) et 'source_url' (HTML)."""
    try:
        col = _get_collection(collection_name)
        where = {"$or": [{"source": source}, {"source_url": source}]}
        result = col.get(where=where, include=[])
        ids = result.get("ids", [])
        if not ids:
            logger.warning("delete_by_source: aucun chunk trouve pour %s", source)
            return 0
        col.delete(where=where)
        logger.info("delete_by_source: %d chunks supprimes pour %s", len(ids), source)
        return len(ids)
    except Exception as e:
        logger.error("delete_by_source failed: %s", e)
        return 0


def delete_by_ids(ids: List[str], collection_name: str) -> int:
    """Supprime les chunks par leurs IDs. Retourne le nombre d'IDs soumis."""
    try:
        if not ids:
            return 0
        col = _get_collection(collection_name)
        col.delete(ids=ids)
        logger.info("delete_by_ids: %d chunks supprimés", len(ids))
        return len(ids)
    except Exception as e:
        logger.error("delete_by_ids failed: %s", e)
        return 0


def delete_all(collection_name: str) -> int:
    """Supprime tous les documents de la collection. Retourne le nombre supprimé."""
    try:
        col = _get_collection(collection_name)
        result = col.get(include=[])
        ids = result.get("ids", [])
        if not ids:
            logger.info("delete_all: collection déjà vide")
            return 0
        col.delete(ids=ids)
        logger.info("delete_all: %d documents supprimés de %s", len(ids), collection_name)
        return len(ids)
    except Exception as e:
        logger.error("delete_all failed: %s", e)
        return 0
