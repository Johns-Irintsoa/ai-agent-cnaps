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
    # La méthode embed_query est optimisée pour transformer une seule chaîne
    vector = embeddingModel.embed_query(query)
    return vector

# Cette fonction effectue une recherche de similarité dans la base de données vectorielle et retourne les documents les plus proches
def search_similar_documents(query: str, k: int = 5) -> List[Document]:
    """
    Effectue une recherche de similarité et retourne les k documents les plus proches.
    """
    db = get_vector_db(embeddingModel)
    
    # LangChain gère l'embedding de la query automatiquement ici
    # car embedding_function a été passée lors de l'initialisation de la DB
    results = db.similarity_search(query, k=k)
    
    return results


