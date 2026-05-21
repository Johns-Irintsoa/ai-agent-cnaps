import asyncio
import logging

from src.inference.query_retriever import get_query_vector, search_by_vector
from src.inference.bm25_retriever import bm25_index
from src.inference.multi_query_retriever import reciprocal_rank_fusion
from src.inference.reranking import get_reranked_documents
from src.inference.prompting import generate_answer
from src.inference.timer import RAGTimer
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
    cached = await semantic_cache.get(user_query)
    if cached:
        answer, _ = cached
        return {"answer": answer, "evaluation": {}, "from_cache": True}

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
    asyncio.create_task(semantic_cache.set(user_query, answer))

    return {
        "answer": answer,
        "evaluation": {
            "timing": timing,
            "tokens": tokens,
        },
        "from_cache": False,
    }
