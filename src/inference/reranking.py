from ..models.embedding import embedding_manager
from typing import List
from langchain_core.documents import Document
import numpy as np

def get_reranked_documents(query: str, initial_docs: List[Document], k_final: int = 3) -> List[Document]:
    """
    Reranking via similarité cosinus sur les embeddings.
    """
    if not initial_docs:
        return []

    try:
        # 1. Embedding de la query et des documents via le modèle d'embedding existant
        all_texts = [query] + [doc.page_content for doc in initial_docs]
        embeddings = embedding_manager.model.embed_documents(all_texts)

        # 2. Séparation query / documents
        query_vec = np.array(embeddings[0])
        doc_vecs = np.array(embeddings[1:])

        # 3. Similarité cosinus
        scores = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-10
        )

        # 4. Tri et sélection des k_final meilleurs
        ranked_indices = np.argsort(scores)[::-1][:k_final]

        final_docs = []
        for idx in ranked_indices:
            doc = initial_docs[idx]
            doc.metadata["rerank_score"] = float(scores[idx])
            final_docs.append(doc)

        return final_docs

    except Exception as e:
        print(f"Erreur lors du reranking : {e}")
        return initial_docs[:k_final]