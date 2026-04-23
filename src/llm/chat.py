import os
from langchain_openai import ChatOpenAI


class LLMClient:
    def __init__(self):
        self._llm = ChatOpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ["LLM_MODEL"],
            api_key=os.environ.get("LLM_API_KEY", "no-key"),
        )

    def invoke(self, message: str) -> str:
        response = self._llm.invoke(message)
        return response.content
