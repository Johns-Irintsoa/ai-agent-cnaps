import os
from typing import List

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Répertoire de cache local pour les modèles HuggingFace (défini dans le Dockerfile)
_CACHE_FOLDER = os.getenv("SENTENCE_TRANSFORMERS_HOME", None)

# Modèle par défaut depuis le .env (format HuggingFace Hub : "BAAI/bge-m3")
_DEFAULT_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-m3")


class EmbeddingManager:
    """Handles document embedding generation using HuggingFaceEmbeddings (LangChain)"""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        """
        Initialize the embedding manager.

        Args:
            model_name: HuggingFace Hub model ID.
                        Defaults to the EMBEDDINGS_MODEL env variable.
                        Exemple : "BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"
        """
        self.model_name = model_name
        self.model: HuggingFaceEmbeddings = None
        self._load_model()

    def _load_model(self):
        """Load the HuggingFaceEmbeddings model (SentenceTransformer backend)."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=_CACHE_FOLDER,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            dim = self.model.client.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {dim}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim).
        """
        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts...")
        vectors = self.model.embed_documents(texts)
        embeddings = np.array(vectors)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


## initialize the embedding manager

embedding_manager = EmbeddingManager()
embedding_manager
