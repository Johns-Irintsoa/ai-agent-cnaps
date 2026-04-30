import os
from typing import List

import numpy as np
from langchain_openai import OpenAIEmbeddings

_DEFAULT_MODEL = os.getenv("EMBEDDINGS_MODEL", "huggingface.co/gpustack/bge-m3-gguf")


class EmbeddingManager:
    """Handles document embedding generation via Docker Model Runner (OpenAI-compatible API)."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self.model: OpenAIEmbeddings = None
        self._load_model()

    def _load_model(self):
        print(f"Initializing embedding model: {self.model_name}")
        self.model = OpenAIEmbeddings(
            base_url=os.environ["LLM_BASE_URL"],
            model=self.model_name,
            api_key=os.environ.get("LLM_API_KEY", "no-key"),
            check_embedding_ctx_length=False,
        )
        print("Embedding model ready.")

embedding_manager = EmbeddingManager()
