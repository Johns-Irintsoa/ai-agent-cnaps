"""
BM25 retriever — construit un index BM25 sur tous les chunks Chroma.
Conçu pour s'intégrer dans search_hybrid() via reciprocal_rank_fusion().
"""

import logging
import os
import threading
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ..models.embedding import embedding_manager
from ..db.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Index:
    """
    Index BM25 construit lazily depuis ChromaDB.
    Thread-safe : un seul rebuild à la fois via _lock.
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._docs: List[Document] = []
        self._lock = threading.Lock()
        self._built = False

    def build(self) -> None:
        """Charge tous les chunks depuis Chroma et construit l'index BM25."""
        with self._lock:
            store = Chroma(
                collection_name=os.getenv("COLLECTION_NAME", "rag_cnaps"),
                embedding_function=embedding_manager.model,
                client=ChromaClient.get_client(),
            )
            data = store._collection.get(include=["documents", "metadatas"])

            if not data["documents"]:
                logger.warning("BM25 : aucun document trouvé dans ChromaDB — index vide")
                self._bm25 = None
                self._docs = []
                self._built = True
                return

            docs = []
            for content, meta, doc_id in zip(
                data["documents"], data["metadatas"], data["ids"]
            ):
                meta = dict(meta) if meta else {}
                if "chunk_id" not in meta:
                    meta["chunk_id"] = doc_id
                docs.append(Document(page_content=content, metadata=meta))

            corpus = [_tokenize(doc.page_content) for doc in docs]
            self._bm25 = BM25Okapi(corpus)
            self._docs = docs
            self._built = True
            logger.info("BM25 : index construit (%d documents)", len(docs))

    def refresh(self) -> None:
        """Reconstruit l'index après ingestion de nouveaux documents."""
        logger.info("BM25 : refresh demandé")
        self._built = False
        self.build()

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retourne les top_k documents triés par score BM25 décroissant.
        Retourne List[Document] compatible avec reciprocal_rank_fusion().
        """
        self._ensure_built()

        if self._bm25 is None or not self._docs:
            logger.warning("BM25 : index vide, recherche ignorée")
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                break
            doc = self._docs[idx]
            doc.metadata["bm25_score"] = float(scores[idx])
            results.append(doc)

        logger.info("BM25 : %d résultats pour '%s'", len(results), query[:60])
        return results


bm25_index = BM25Index()
