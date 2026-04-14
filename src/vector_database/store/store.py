import os
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document


class VectorDatabase:
    def __init__(self):
        self._embeddings = OllamaEmbeddings(
            base_url=os.environ["OLLAMA_BASE_URL"],
            model=os.environ["EMBEDDINGS_MODEL"],
        )
        self._store = InMemoryVectorStore(embedding=self._embeddings)

    def add_documents(self, texts: list[str]) -> None:
        docs = [Document(page_content=text) for text in texts]
        self._store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 4) -> list[str]:
        results = self._store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
