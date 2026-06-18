import asyncio
import logging
import re
from enum import Enum
from typing import Dict, Any, Optional

import unicodedata

from .query_retriever import get_query_vector, search_by_vector
from .bm25_retriever import bm25_index
from .multi_query_retriever import reciprocal_rank_fusion
from .reranking import get_reranked_documents
from .prompting import generate_answer
from .timer import RAGTimer
from .models import QueryMetaData, SourceItem
from .cache.semantic_cache import semantic_cache
from .SQLAgent.SQL_agent import run_sql_agent
from src.db.oracle_client import OracleClient

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    SQL_PERSONAL = "sql_personal"
    RAG = "rag"
    OUT_OF_SCOPE = "out_of_scope"


_COTISATION_KEYWORDS = {
    "cotisation", "cotisations", "cotise", "cotises",
    "periode", "periodes", "derniere", "dernier",
    "versement", "versements", "salaire", "salaires",
    "affilie", "affiliation", "solde", "prestation", "prestations",
}

_OUT_OF_SCOPE_KEYWORDS = {
    "meteo", "sport", "cuisine", "recette", "film", "musique",
    "politique", "religion", "crypto", "bitcoin", "football",
    "cinema", "serie", "jeu", "jeux", "voyage",
}

_MSG_OUT_OF_SCOPE = "Cette question est en dehors de mon domaine. Je suis Lucy, l'assistante CNaPS — je réponds uniquement aux questions liées à la retraite, aux cotisations et aux prestations CNaPS."

_BASE_RESULT: Dict[str, Any] = {
    "needs_auth": False,
    "needs_matricule": False,
    "metadata": None,
    "evaluation": {},
    "from_cache": False,
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


def _is_cotisation_intent(message: str) -> bool:
    words = set(_normalize(message).split())
    return bool(words & _COTISATION_KEYWORDS)


def init_triage(message: str) -> Intent:
    words = set(_normalize(message).split())

    if words & _OUT_OF_SCOPE_KEYWORDS:
        return Intent.OUT_OF_SCOPE

    if _is_cotisation_intent(message):
        return Intent.SQL_PERSONAL

    return Intent.RAG


async def ask_question(
    user_query: str,
    matricule: Optional[str] = None,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pipeline principal — routing via init_triage() :
      OUT_OF_SCOPE  → réponse hors périmètre
      SQL_PERSONAL  → authentification Oracle puis SQL Agent
      RAG           → pipeline RAG (embedding → retrieval → reranking → LLM)
    """
    intent = init_triage(user_query)

    # --- OUT_OF_SCOPE ---
    if intent == Intent.OUT_OF_SCOPE:
        return {**_BASE_RESULT, "answer": _MSG_OUT_OF_SCOPE}

    # --- SQL_PERSONAL ---
    if intent == Intent.SQL_PERSONAL:
        if not matricule or not password:
            return {
                **_BASE_RESULT,
                "needs_auth": True,
                "answer": "Pour accéder à vos données personnelles, veuillez vous authentifier avec votre matricule et mot de passe.",
            }
        if not await asyncio.to_thread(OracleClient.verify_auth, matricule, password):
            logger.warning("Authentification Oracle échouée pour matricule=%s", matricule)
            return {**_BASE_RESULT, "answer": _MSG_OUT_OF_SCOPE}
        return await run_sql_agent(user_query)

    # --- RAG ---
    timer = RAGTimer()

    # 0. Cache sémantique
    async with timer.ameasure("Cache lookup"):
        cached = await semantic_cache.get(user_query)
    if cached:
        answer, meta_dict = cached
        cached_metadata = QueryMetaData(**meta_dict) if meta_dict else None
        return {
            **_BASE_RESULT,
            "answer": answer,
            "metadata": cached_metadata.model_dump() if cached_metadata else None,
            "from_cache": True,
        }

    # 1. Embedding
    async with timer.ameasure("Embedding"):
        query_vector = await asyncio.to_thread(get_query_vector, user_query)

    # 2. ChromaDB retrieval
    async with timer.ameasure("ChromaDB retrieval"):
        vector_docs = await asyncio.to_thread(search_by_vector, query_vector, 8)

    # 3. BM25 retrieval
    async with timer.ameasure("BM25"):
        bm25_docs = await asyncio.to_thread(bm25_index.search, user_query, 8)

    # 4. RRF Fusion
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

    # Métadonnées
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

    timing = timer.report()

    estimated_label = " (estimé)" if tokens.get("estimated") else ""
    logger.info(
        "TOKEN USAGE%s — prompt: %d | completion: %d | total: %d",
        estimated_label,
        tokens.get("prompt_tokens", 0),
        tokens.get("completion_tokens", 0),
        tokens.get("total_tokens", 0),
    )

    asyncio.create_task(
        semantic_cache.set(user_query, answer, metadata.model_dump() if metadata else None)
    )

    return {
        **_BASE_RESULT,
        "answer": answer,
        "metadata": metadata.model_dump() if metadata else None,
        "evaluation": {
            "timing": timing,
            "tokens": tokens,
        },
    }
