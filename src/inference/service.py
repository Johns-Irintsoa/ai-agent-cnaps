import asyncio
import logging
import re
from typing import Dict, Any, Optional

from .query_retriever import get_query_vector, search_by_vector
from .bm25_retriever import bm25_index
from .multi_query_retriever import reciprocal_rank_fusion
from .reranking import get_reranked_documents
from .prompting import generate_answer
from .timer import RAGTimer
from .models import QueryMetaData, SourceItem
from .cache.semantic_cache import semantic_cache
from .SQLAgent.SQL_agent import run_sql_agent

logger = logging.getLogger(__name__)

_COTISATION_KEYWORDS = {"période", "cotisation", "cotisé", "cotisations", "dernière", "dernier"}


def _is_cotisation_intent(message: str) -> bool:
    words = set(message.lower().split())
    return bool(words & _COTISATION_KEYWORDS)


def _extract_matricule(message: str) -> Optional[str]:
    match = re.search(r'\b(\d{5,7})\b', message)
    return match.group(1) if match else None


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
    # Routing SQL Agent (stateless)
    matricule = _extract_matricule(user_query)
    if matricule:
        return await run_sql_agent(user_query)

    if _is_cotisation_intent(user_query):
        return {
            "answer": "Pour consulter votre dernière période de cotisation, veuillez me fournir votre numéro de matricule.",
            "needs_matricule": True,
            "metadata": None,
            "evaluation": {},
            "from_cache": False,
        }

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
