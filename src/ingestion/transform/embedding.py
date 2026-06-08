
from dotenv import load_dotenv
from langchain_chroma import Chroma

from ...db.chroma_client import ChromaClient
from ...models.embedding import embedding_manager

load_dotenv()


def embed_chunks(json_chunks, collection_name="rag_cnaps"):
    """
    Vectorise et stocke une liste de chunks dans ChromaDB.
    Accepte soit des dicts (pipeline PDF) soit des WebPageContentChunked (pipeline web).
    """
    embeddingModel = embedding_manager.model

    texts = []
    ids = []
    metadatas = []
    for chunk in json_chunks:
        if isinstance(chunk, dict):
            text = chunk["content"]
            chunk_id = chunk["chunk_id"]
            meta = chunk["metadata"].copy()
        else:
            text = chunk.document
            chunk_id = chunk.id
            meta = dict(chunk.metadata)

        for key, val in meta.items():
            if isinstance(val, list):
                meta[key] = str(val)

        texts.append(text)
        ids.append(chunk_id)
        metadatas.append(meta)

    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddingModel,
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        client=ChromaClient.get_client(),
    )

    print(f"✅ {len(json_chunks)} chunks vectorises et stockes dans '{collection_name}'.")
    return vector_db
