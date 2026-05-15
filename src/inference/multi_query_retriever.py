"""
Module de retrieval multi-query avec parallélisation des requêtes.
Adapté pour FastAPI asynchrone.
"""

import asyncio
import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

from ..models.llm import LLMClient
from ..models.embedding import embedding_manager

# ---------------------------------------------------------------------------
# LOGGING & CONSTANTES
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_client = LLMClient()
embedding_model = embedding_manager.model

# Prompt pour générer 3 reformulations
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Tu es un assistant spécialisé dans les documents administratifs et légaux malgaches.
INSTRUCTION IMPORTANTE : Tu dois OBLIGATOIREMENT répondre en FRANÇAIS uniquement.

Génère exactement 3 reformulations en FRANÇAIS de la question suivante pour améliorer la recherche documentaire.
Chaque reformulation doit chercher la même information avec des mots différents.
Retourne UNIQUEMENT les 3 questions en FRANÇAIS, une par ligne, sans numérotation, sans puce, sans explication.

Question : {question}

Reformulations en français :"""
)

# ---------------------------------------------------------------------------
# 1. GÉNÉRATION DES VARIANTES (synchrone, mais exécutée dans un thread)
# ---------------------------------------------------------------------------
def generate_query_variations(question: str) -> List[str]:
    """Appelle le LLM pour générer 3 reformulations + inclut la question originale."""
    prompt = MULTI_QUERY_PROMPT.format(question=question)
    response = llm_client.model.invoke(prompt)
    lines = response.content.strip().split("\n")
    variations = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
    # On garde au maximum 3 variations
    variations = variations[:3]
    # On retourne la liste complète : [question_originale, var1, var2, var3]
    return [question] + variations


# ---------------------------------------------------------------------------
# 2. RECHERCHE INDIVIDUELLE (synchrone, à paralléliser)
# ---------------------------------------------------------------------------
def _search_single_query(query: str, collection_name: str, top_k: int) -> List[Document]:
    """Exécute une recherche vectorielle synchrone pour une requête donnée."""
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=os.getenv("VECTOR_DB_DIR"),
    )
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever.invoke(query)


# ---------------------------------------------------------------------------
# 3. RECIPROCAL RANK FUSION (RRF) - inchangée
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(results_list: List[List[Document]], k: int = 60) -> List[Document]:
    scores: dict = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_key = doc.metadata.get("chunk_id", doc.page_content[:80])
            if doc_key not in scores:
                scores[doc_key] = {"score": 0.0, "doc": doc}
            scores[doc_key]["score"] += 1.0 / (k + rank + 1)

    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    logger.info(f"🔀 RRF : {sum(len(r) for r in results_list)} chunks → {len(sorted_results)} uniques")
    return [item["doc"] for item in sorted_results]


# ---------------------------------------------------------------------------
# 4. FONCTION PRINCIPALE ASYNCHRONE (à utiliser dans FastAPI)
# ---------------------------------------------------------------------------
async def multi_query_retriever_async(
    query: str,
    top_k_per_query: int = 4,
    top_k_final: int = 4,
) -> List[Document]:
    """
    Version asynchrone parallélisant les 4 requêtes (originale + 3 variantes).
    """
    logger.info(f"🔍 Multi-query (async) pour : '{query}'")
    collection_name = os.getenv("COLLECTION_NAME", "rag_cnaps")

    # Étape 1 : générer les variantes (bloquant, à déléguer à un thread)
    loop = asyncio.get_running_loop()
    queries = await loop.run_in_executor(None, generate_query_variations, query)
    logger.info(f"📝 Requêtes à paralléliser : {queries}")

    # Étape 2 : paralléliser les 4 recherches vectorielles
    tasks = [
        loop.run_in_executor(None, _search_single_query, q, collection_name, top_k_per_query)
        for q in queries
    ]
    results_per_query = await asyncio.gather(*tasks)

    # Étape 3 : fusion RRF
    fused_docs = reciprocal_rank_fusion(results_per_query)

    # Étape 4 : top K final
    final_docs = fused_docs[:top_k_final]
    logger.info(f"✅ {len(final_docs)} chunks finaux retournés")
    return final_docs


# ---------------------------------------------------------------------------
# 5. VERSION SYNCHRONE DE COMPATIBILITÉ (si nécessaire)
# ---------------------------------------------------------------------------
def multi_query_retriever(
    query: str,
    top_k_per_query: int = 4,
    top_k_final: int = 4,
) -> List[Document]:
    """
    Version synchrone (exécution séquentielle) pour compatibilité.
    Si vous utilisez FastAPI, préférez la version asynchrone.
    """
    logger.info(f"🔍 Multi-query (sync) pour : '{query}'")
    collection_name = os.getenv("COLLECTION_NAME", "rag_cnaps")
    queries = generate_query_variations(query)
    logger.info(f"📝 {len(queries)} requêtes : {queries}")

    all_results = []
    for q in queries:
        docs = _search_single_query(q, collection_name, top_k_per_query)
        all_results.append(docs)

    fused_docs = reciprocal_rank_fusion(all_results)
    final_docs = fused_docs[:top_k_final]
    logger.info(f"✅ {len(final_docs)} chunks finaux")
    return final_docs