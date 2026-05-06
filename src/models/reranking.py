import os

_DEFAULT_MODEL = os.getenv("LLM_RERANKING_MODEL", "huggingface.co/gpustack/bge-reranker-v2-m3-gguf:latest")

class RerankingManager:
    """Stocke la config du modèle de reranking — utilise l'endpoint embeddings de DMR."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self.base_url = os.environ["LLM_BASE_URL"]  # ex: http://model-runner.docker.internal/engines/v1
        self.api_key = os.environ.get("LLM_API_KEY", "no-key")
        print(f"Reranking model ready: {self.model_name}")

reranking_manager = RerankingManager()