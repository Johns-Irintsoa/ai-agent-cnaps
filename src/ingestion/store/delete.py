import logging
from typing import List

from ...db.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


def _get_collection(collection_name: str):
    client = ChromaClient.get_client()
    return client.get_or_create_collection(name=collection_name)


def delete_by_source_url(source_url: str, collection_name: str) -> int:
    """Supprime tous les chunks dont la metadata source_url correspond. Retourne le nombre supprimé."""
    try:
        col = _get_collection(collection_name)
        result = col.get(where={"source_url": source_url}, include=[])
        ids = result.get("ids", [])
        if not ids:
            logger.warning("delete_by_source_url: aucun document trouvé pour %s", source_url)
            return 0
        col.delete(where={"source_url": source_url})
        logger.info("delete_by_source_url: %d chunks supprimés pour %s", len(ids), source_url)
        return len(ids)
    except Exception as e:
        logger.error("delete_by_source_url failed: %s", e)
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
