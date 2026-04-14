import os
from langchain_ollama import ChatOllama


class OllamaLLM:
    def __init__(self):
        self._llm = ChatOllama(
            base_url=os.environ["OLLAMA_BASE_URL"],
            model=os.environ["OLLAMA_MODEL"],
        )

    def invoke(self, message: str) -> str:
        response = self._llm.invoke(message)
        return response.content
