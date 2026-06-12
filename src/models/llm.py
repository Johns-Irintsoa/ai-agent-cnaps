import logging
import os
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self._llm = ChatOpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ["LLM_MODEL"],
            api_key=os.environ.get("LLM_API_KEY", "no-key"),
            temperature=0.3,
        )

    @property
    def model(self):
        """Expose l'instance pour les fonctionnalités avancées comme with_structured_output."""
        return self._llm

    def invoke(self, message: str) -> str:
        response = self._llm.invoke(message)
        return response.content

    def invoke_with_usage(self, message: str) -> tuple[str, dict]:
        """Invoque le LLM et retourne (contenu, token_counts).

        token_counts = {
            "prompt_tokens":     int,
            "completion_tokens": int,
            "total_tokens":      int,
            "estimated":         bool,  # True si le modèle n'a pas retourné d'usage
        }
        """
        response = self._llm.invoke(message)
        content = response.content

        usage: dict = {}

        # Priorité 1 — usage_metadata LangChain >= 0.2 (input_tokens / output_tokens)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            m = response.usage_metadata
            usage = {
                "prompt_tokens":     m.get("input_tokens", 0),
                "completion_tokens": m.get("output_tokens", 0),
                "total_tokens":      m.get("total_tokens", 0),
                "estimated":         False,
            }

        # Priorité 2 — response_metadata au format OpenAI (prompt_tokens / completion_tokens)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            tu = response.response_metadata.get("token_usage", {})
            if tu:
                usage = {
                    "prompt_tokens":     tu.get("prompt_tokens", 0),
                    "completion_tokens": tu.get("completion_tokens", 0),
                    "total_tokens":      tu.get("total_tokens", 0),
                    "estimated":         False,
                }

        # Fallback — estimation par comptage de caractères (~4 chars/token)
        if not usage:
            prompt_est     = len(message) // 4
            completion_est = len(content) // 4
            usage = {
                "prompt_tokens":     prompt_est,
                "completion_tokens": completion_est,
                "total_tokens":      prompt_est + completion_est,
                "estimated":         True,
            }
            logger.warning(
                "Usage tokens non disponible depuis le modèle — estimation : "
                "prompt=%d  completion=%d  total=%d (≈4 chars/token)",
                prompt_est, completion_est, prompt_est + completion_est,
            )

        return content, usage
