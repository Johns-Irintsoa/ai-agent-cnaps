"""
Module de retrieval multi-query adapté pour :
- LLM     : LLMClient existant (Mistral-7B via llama.cpp)
- VectorDB: ChromaDB
- Stratégie : MultiQueryRetriever LangChain + RRF séquentiel
- Contrainte : CPU only
"""

import logging
import os
from typing import List
 
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_chroma import Chroma

# Initialize Models 
from ..models.llm import LLMClient
from ..models.embedding import embedding_manager

llm_client = LLMClient() 
embedding_model = embedding_manager.model  

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. PARSER PERSONNALISÉ — extrait les variantes de la réponse du LLM
# ---------------------------------------------------------------------------

class QueriesParser(BaseOutputParser[List[str]]):
    """
    Parse la réponse du LLM en liste de questions reformulées.
    Remplace le parser_key déprécié de MultiQueryRetriever.
    """
 
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return [
            line.strip()
            for line in lines
            if line.strip() and len(line.strip()) > 10
        ]
 
    @property
    def _type(self) -> str:
        return "queries_parser"

# ---------------------------------------------------------------------------
# 2. PROMPT FRANÇAIS — adapté au corpus CNaPS (décrets, tableaux, salaires)
# ---------------------------------------------------------------------------

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
# 3. RETRIEVER DE BASE — ChromaDB + EmbeddingManager
# ---------------------------------------------------------------------------

def build_base_retriever(
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_cnaps"),
    top_k: int = 4,
):
    """
    Initialise le retriever ChromaDB en réutilisant l'EmbeddingManager singleton.

    On passe embedding_manager.model directement à Chroma pour éviter
    de réinstancier le modèle BGE-M3 (déjà chargé en RAM).

    Args:
        collection_name : Nom de la collection ChromaDB.
        top_k           : Nombre de chunks à récupérer par variante.

    Returns:
        Retriever LangChain connecté à ChromaDB.
    """
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_manager.model, 
        persist_directory=os.getenv("VECTOR_DB_DIR"),
    )

    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


# ---------------------------------------------------------------------------
# 4. RECIPROCAL RANK FUSION (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Fusionne plusieurs listes de chunks via Reciprocal Rank Fusion.

    Formule : score(chunk) = Σ 1 / (k + rank)
    Un chunk qui apparaît haut dans plusieurs listes obtient un score élevé.

    Args:
        results_list : Liste de listes de Documents récupérés.
        k            : Constante de lissage RRF (défaut : 60).

    Returns:
        Liste de Documents triés par score RRF décroissant, dédupliqués.
    """
    scores: dict = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_key = doc.metadata.get("chunk_id", doc.page_content[:80])
            if doc_key not in scores:
                scores[doc_key] = {"score": 0.0, "doc": doc}
            scores[doc_key]["score"] += 1.0 / (k + rank + 1)

    sorted_results = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    logger.info(
        f"🔀 RRF : {sum(len(r) for r in results_list)} chunks "
        f"→ {len(sorted_results)} chunks uniques rerankés"
    )

    return [item["doc"] for item in sorted_results]

# ---------------------------------------------------------------------------
# 5. BUILD MULTI QUERY RETRIEVER
# ---------------------------------------------------------------------------

def build_multi_query_retriever(
    base_retriever,
    llm_client
) -> MultiQueryRetriever:
    """
    Construit le MultiQueryRetriever en réutilisant ton LLMClient existant.

    On passe llm_client.model (la property qui expose le ChatOpenAI)
    car MultiQueryRetriever attend un Runnable LangChain.

    Args:
        base_retriever : Retriever ChromaDB de base.
        llm_client     : Ton LLMClient existant.

    Returns:
        MultiQueryRetriever configuré avec prompt français.
    """
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_client.model,                        # ← .model expose le ChatOpenAI
        prompt=MULTI_QUERY_PROMPT,
        # parser_output_fn=QueriesParser().parse,
        include_original=True,                       # Inclut la requête originale
    )

# ---------------------------------------------------------------------------
# 6. FONCTION PRINCIPALE
# ---------------------------------------------------------------------------

def multi_query_retriever(
    query: str,
    top_k_per_query: int = 4,
    top_k_final: int = 4,
) -> List[Document]:
    """
    Pipeline complet de retrieval multi-query avec RAG Fusion (RRF).

    Étapes :
        1. Génère 3 reformulations via LLMClient.model (Mistral-7B)
        2. Récupère top_k_per_query chunks par variante via ChromaDB + BGE-M3
        3. Fusionne tous les résultats via RRF
        4. Retourne les top_k_final chunks les plus pertinents

    Adapté CPU 16 Go RAM :
        - Séquentiel (pas de parallélisme)
        - Réutilise embedding_manager.model déjà chargé en RAM
        - Réutilise llm_client.model déjà chargé en RAM
        - include_original=True évite un appel LLM supplémentaire

    Args:
        query           : Requête utilisateur.
        llm_client      : Instance LLMClient (créée si None).
        collection_name : Collection ChromaDB cible.
        top_k_per_query : Chunks récupérés par variante.
        top_k_final     : Chunks finaux retournés après RRF.

    Returns:
        Liste des top_k_final Documents les plus pertinents.

    Example:
        >>> client = LLMClient()
        >>> docs = multi_query_retriever(
        ...     query="Quel est le salaire minimum d'un ouvrier spécialisé ?",
        ...     llm_client=client,
        ... )
    """

    global llm_client
    if llm_client is None:
        llm_client = LLMClient()

    logger.info(f"🔍 Multi-query pour : '{query}'")

    # ── Étape 1 : Composants ────────────────────────────────────────────────
    collection_name = os.getenv("COLLECTION_NAME", "rag_cnaps")
    base_retriever = build_base_retriever(collection_name, top_k_per_query)
    mq_retriever   = build_multi_query_retriever(base_retriever, llm_client)

    # ── Étape 2 : Génération des variantes ──────────────────────────────────
    # generated_queries = mq_retriever.generate_queries(
    #     question=query,
    #     run_manager=None,
    # )
    # logger.info(f"📝 {len(generated_queries)} variantes : {generated_queries}")

    unique_docs = mq_retriever.invoke(query)
    logger.info(f"📦 {len(unique_docs)} chunks après MultiQueryRetriever")

    # ── Étape 3 : Retrieval séquentiel par variante ─────────────────────────
    results_by_source: dict = {}
    for doc in unique_docs:
        source = doc.metadata.get("source", "unknown")
        if source not in results_by_source:
            results_by_source[source] = []
        results_by_source[source].append(doc)

    all_results = list(results_by_source.values())

    # Fallback si tous les docs ont la même source
    if len(all_results) <= 1:
        all_results = [unique_docs]

    # ── Étape 4 : Fusion RRF ────────────────────────────────────────────────
    fused_docs = reciprocal_rank_fusion(all_results)

    # ── Étape 5 : Top K final ───────────────────────────────────────────────
    final_docs = fused_docs[:top_k_final]
    logger.info(f"✅ {len(final_docs)} chunks finaux retournés")

    return final_docs