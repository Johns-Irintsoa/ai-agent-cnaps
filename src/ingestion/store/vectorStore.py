"""
VectorStore — Stockage et recherche de documents dans ChromaDB pour le pipeline CNaPS.

Utilise LangChain + ChromaDB avec HuggingFaceEmbeddings comme fonction d'embedding.
Les vecteurs sont persistés dans data/chromadb/v1/{collection_name}/.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemins par défaut
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # ai-agent-cnaps/
_DEFAULT_PERSIST_DIR = _PROJECT_ROOT / "data" / "chromadb" / "v1"
_DEFAULT_COLLECTION  = "cnaps_web_data"

# ---------------------------------------------------------------------------
# Configuration embedding (partagée avec embedding.py)
# ---------------------------------------------------------------------------

_EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "huggingface.co/gpustack/bge-m3-gguf")


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__( self, collection_name: str = _DEFAULT_COLLECTION, persist_directory: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            collection_name:   Nom de la collection ChromaDB.
                               Défaut : "cnaps_web_data".
            persist_directory: Répertoire de persistance des vecteurs.
                               Défaut : data/chromadb/v1/ (relatif à la racine du projet).
        """
        self.collection_name   = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory else _DEFAULT_PERSIST_DIR
        self.store: Chroma     = None
        self._embedding_fn: OpenAIEmbeddings = None
        self._initialize_store()

    # ---------------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------------

    def _initialize_store(self) -> None:
        """
        Crée ou charge la collection ChromaDB existante.

        - Si la collection existe déjà dans persist_directory, elle est chargée
          telle quelle (idempotent).
        - Sinon, une nouvelle collection vide est créée.
        - Le répertoire de persistance est créé automatiquement si absent.
        """
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        log.info(f"Chargement du modèle d'embedding : {_EMBEDDINGS_MODEL}")
        self._embedding_fn = OpenAIEmbeddings(
            base_url=os.environ["LLM_BASE_URL"],
            model=_EMBEDDINGS_MODEL,
            api_key=os.environ.get("LLM_API_KEY", "no-key"),
            check_embedding_ctx_length=False,
        )

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embedding_fn,
            persist_directory=str(self.persist_directory),
        )

        count = self.store._collection.count()
        log.info(
            f"VectorStore initialisé — collection='{self.collection_name}' "
            f"| répertoire='{self.persist_directory}' "
            f"| {count} document(s) existant(s)"
        )

    # ---------------------------------------------------------------------------
    # Ajout de documents
    # ---------------------------------------------------------------------------

    def add_documents(self, documents: list[Any], embeddings: np.ndarray) -> int:
        """
        Ajoute des documents et leurs embeddings pré-calculés dans la collection ChromaDB.

        Les embeddings sont fournis explicitement (issus de EmbeddingManager.generate_embeddings)
        plutôt que recalculés en interne, ce qui permet de réutiliser des vecteurs déjà produits.

        Args:
            documents:  Liste de Document LangChain (issus de chuncking.py).
            embeddings: Tableau numpy des embeddings correspondants,
                        shape (len(documents), embedding_dim).

        Returns:
            Nombre de documents ajoutés.

        Raises:
            ValueError: Si le nombre de documents ne correspond pas au nombre d'embeddings.
        """
        if not documents:
            log.warning("add_documents : liste vide, rien à ajouter.")
            return 0

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Nombre de documents ({len(documents)}) différent du nombre "
                f"d'embeddings ({len(embeddings)})"
            )

        log.info(f"Ajout de {len(documents)} document(s) dans '{self.collection_name}'...")

        ids:             list[str]        = []
        metadatas:       list[dict]       = []
        documents_text:  list[str]        = []
        embeddings_list: list[list[float]]= []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"]      = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.store._collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            total = self.store._collection.count()
            log.info(f"Ajout terminé. Total dans la collection : {total}")
            return len(documents)

        except Exception as e:
            log.error(f"Erreur lors de l'ajout dans ChromaDB : {e}")
            raise

    # ---------------------------------------------------------------------------
    # Recherche
    # ---------------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Recherche les k documents les plus proches sémantiquement de la requête.

        Args:
            query:  Texte de la requête.
            k:      Nombre de résultats à retourner (défaut : 4).
            filter: Filtre optionnel sur les métadonnées ChromaDB.
                    Exemple : {"file_type": "pdf"} ou {"source": "rapport.pdf"}

        Returns:
            Liste de Document triés par similarité décroissante.
        """
        return self.store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """
        Recherche les k documents les plus proches avec leur score de similarité.

        Args:
            query:  Texte de la requête.
            k:      Nombre de résultats.
            filter: Filtre optionnel sur les métadonnées.

        Returns:
            Liste de tuples (Document, score) — score entre 0 et 1 (plus haut = plus proche).
        """
        return self.store.similarity_search_with_relevance_scores(query, k=k, filter=filter)

    # ---------------------------------------------------------------------------
    # Utilitaires
    # ---------------------------------------------------------------------------

    def count(self) -> int:
        """Retourne le nombre total de documents dans la collection."""
        return self.store._collection.count()

    def reset_collection(self) -> None:
        """
        Supprime et recrée la collection (efface tous les vecteurs existants).
        Utile pour réindexer depuis zéro.
        """
        log.warning(f"Réinitialisation de la collection '{self.collection_name}'...")
        self.store.delete_collection()
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embedding_fn,
            persist_directory=str(self.persist_directory),
        )
        log.info("Collection réinitialisée.")

    def as_retriever(self, k: int = 4, filter: Optional[dict] = None):
        """
        Retourne un retriever LangChain compatible avec les chaînes RAG.

        Args:
            k:      Nombre de documents à récupérer par requête.
            filter: Filtre optionnel sur les métadonnées.

        Returns:
            VectorStoreRetriever utilisable dans une LangChain chain.
        """
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return self.store.as_retriever(search_kwargs=search_kwargs)
