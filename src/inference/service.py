import asyncio
import logging

from .query_retriever import get_query_vector, search_by_vector
from .bm25_retriever import bm25_index
from .multi_query_retriever import reciprocal_rank_fusion
from .reranking import get_reranked_documents
from .prompting import generate_answer
from .timer import RAGTimer
from .models import QueryMetaData, SourceItem
from typing import Dict, Any
from .cache.semantic_cache import semantic_cache

logger = logging.getLogger(__name__)


async def ask_question(user_query: str) -> Dict[str, Any]:
    """
    Pipeline RAG :
    0. Cache sémantique
    1. Embedding de la query
    2. Retrieval ChromaDB (vecteur pré-calculé)
    3. Retrieval BM25
    4. RRF Fusion
    5. Reranking cosinus
    6. Génération LLM
    """
    timer = RAGTimer()

    # 0. Cache sémantique
    async with timer.ameasure("Cache lookup"):
        cached = await semantic_cache.get(user_query)
    if cached:
        answer, meta_dict = cached
        cached_metadata = QueryMetaData(**meta_dict) if meta_dict else None
        return {
            "answer": answer,
            "metadata": cached_metadata.model_dump() if cached_metadata else None,
            "evaluation": {},
            "from_cache": True,
        }

    # 1. Embedding — vecteur réutilisé pour ChromaDB (pas de double encoding)
    async with timer.ameasure("Embedding"):
        query_vector = await asyncio.to_thread(get_query_vector, user_query)

    # 2. ChromaDB retrieval — k=8 pour laisser le reranker choisir parmi plus de candidats
    async with timer.ameasure("ChromaDB retrieval"):
        vector_docs = await asyncio.to_thread(search_by_vector, query_vector, 8)

    # 3. BM25 retrieval
    async with timer.ameasure("BM25"):
        bm25_docs = await asyncio.to_thread(bm25_index.search, user_query, 8)

    # 4. RRF Fusion (CPU pur — exécution directe, pas de thread)
    with timer.measure("RRF Fusion"):
        fused_docs = reciprocal_rank_fusion([vector_docs, bm25_docs])[:8]

    # 5. Reranking → 3 meilleurs documents
    async with timer.ameasure("Reranking"):
        reranked_docs = await asyncio.to_thread(
            get_reranked_documents, user_query, fused_docs, 3
        )

    # 6. Génération LLM
    async with timer.ameasure("Generation LLM"):
        answer, tokens = await asyncio.to_thread(generate_answer, user_query, reranked_docs)

    # Métadonnées : source principale + toutes les sources uniques rerankées
    metadata = None
    if reranked_docs:
        seen_urls: set = set()
        sources = []
        for doc in reranked_docs:
            m = doc.metadata
            url = m.get("source_url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(SourceItem(
                    source_url=url,
                    title=m.get("title", "Titre inconnu"),
                    date_posted=m.get("date_posted", "Date inconnue"),
                ))
        top = reranked_docs[0].metadata
        metadata = QueryMetaData(
            title=top.get("title", "Titre inconnu"),
            date_posted=top.get("date_posted", "Date inconnue"),
            sources=sources,
        )

    # Rapport de timing dans les logs + retour dans evaluation
    timing = timer.report()

    # Rapport des tokens dans les logs
    estimated_label = " (estimé)" if tokens.get("estimated") else ""
    logger.info(
        "TOKEN USAGE%s — prompt: %d | completion: %d | total: %d",
        estimated_label,
        tokens.get("prompt_tokens", 0),
        tokens.get("completion_tokens", 0),
        tokens.get("total_tokens", 0),
    )

    # 7. Mise en cache (fire-and-forget)
    asyncio.create_task(
        semantic_cache.set(user_query, answer, metadata.model_dump() if metadata else None)
    )

    return {
        "answer": answer,
        "metadata": metadata.model_dump() if metadata else None,
        "evaluation": {
            "timing": timing,
            "tokens": tokens,
        },
        "from_cache": False,
    }
