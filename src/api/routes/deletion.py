import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException

from ..schemas import DeleteBySourceRequest, DeleteByIdsRequest, DeleteResponse
from ...ingestion.store.delete import delete_by_source, delete_by_ids, delete_all

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Suppression"])


@router.delete("/ingestion/documents/by-source", response_model=DeleteResponse)
async def delete_documents_by_source(request: DeleteBySourceRequest):
    """Supprime tous les chunks d'un fichier PDF par son nom (metadata source)."""
    try:
        collection_name = os.getenv("COLLECTION_NAME", "rag_cnaps")
        count = await asyncio.to_thread(delete_by_source, request.source, collection_name)
        if count > 0:
            from ...inference.bm25_retriever import bm25_index
            await asyncio.to_thread(bm25_index.refresh)
        return DeleteResponse(
            status="success",
            deleted_count=count,
            message=f"{count} chunk(s) supprime(s) pour {request.source}",
        )
    except Exception as e:
        logger.error("delete_documents_by_source failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



@router.delete("/ingestion/documents/by-ids", response_model=DeleteResponse)
async def delete_documents_by_ids(request: DeleteByIdsRequest):
    """Supprime des chunks par leurs IDs ChromaDB."""
    try:
        collection_name = os.getenv("COLLECTION_NAME", "rag_cnaps")
        count = await asyncio.to_thread(delete_by_ids, request.ids, collection_name)
        if count > 0:
            from ...inference.bm25_retriever import bm25_index
            await asyncio.to_thread(bm25_index.refresh)
        return DeleteResponse(
            status="success",
            deleted_count=count,
            message=f"{count} chunk(s) supprimé(s)",
        )
    except Exception as e:
        logger.error("delete_documents_by_ids failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/ingestion/collection", response_model=DeleteResponse)
async def delete_entire_collection():
    """Supprime TOUS les documents de la collection ChromaDB."""
    try:
        collection_name = os.getenv("COLLECTION_NAME", "rag_cnaps")
        count = await asyncio.to_thread(delete_all, collection_name)
        if count > 0:
            from ...inference.bm25_retriever import bm25_index
            await asyncio.to_thread(bm25_index.refresh)
        return DeleteResponse(
            status="success",
            deleted_count=count,
            message=f"Collection '{collection_name}' vidée — {count} document(s) supprimé(s)",
        )
    except Exception as e:
        logger.error("delete_entire_collection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
