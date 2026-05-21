from ..models.embedding import embedding_manager
from langchain_chroma import Chroma
from ..VectorDB.initialize import get_vector_db
from typing import List
from langchain_core.documents import Document
import os

#1 load models
embeddingModel = embedding_manager.model

# Cette fonction est utilisée pour transformer une question en vecteur numérique
def get_query_vector(query: str):
    """
    Transforme une question textuelle en un vecteur numérique (liste de floats).
    """
    vector = embeddingModel.embed_query(query)
    return vector

# Cette fonction effectue une recherche de similarité dans la base de données vectorielle et retourne les documents les plus proches
def search_similar_documents(query: str, k: int = 5) -> List[Document]:
    """
    Effectue une recherche de similarité et retourne les k documents les plus proches.
    """
    db = get_vector_db(embeddingModel)
    results = db.similarity_search(query, k=k)
    return results


def search_by_vector(vector: list, k: int = 5) -> List[Document]:
    """
    Recherche ChromaDB à partir d'un vecteur pré-calculé.
    À utiliser après get_query_vector() pour séparer le timing embedding / retrieval.
    """
    db = get_vector_db(embeddingModel)
    return db.similarity_search_by_vector(vector, k=k)


def search_hybrid(query: str, k: int = 5) -> List[Document]:
    """
    Recherche hybride : combine la recherche vectorielle (Chroma) et BM25
    via Reciprocal Rank Fusion (RRF). Retourne les k meilleurs documents.
    """
    from .bm25_retriever import bm25_index
    from .multi_query_retriever import reciprocal_rank_fusion

    vector_results = search_similar_documents(query, k=k)
    bm25_results = bm25_index.search(query, top_k=k)

    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    return fused[:k]


