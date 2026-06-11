
import logging

from dotenv import load_dotenv
from langchain_chroma import Chroma

from ...db.chroma_client import ChromaClient
from ...models.embedding import embedding_manager

load_dotenv()

logger = logging.getLogger(__name__)


def embed_chunks(json_chunks, collection_name="rag_cnaps"):
    """
    Vectorise et stocke une liste de chunks dans ChromaDB.
    Accepte soit des dicts (pipeline PDF) soit des WebPageContentChunked (pipeline web).
    """
    embeddingModel = embedding_manager.model

    # Déduplication par ID (défense contre les doublons en amont)
    seen_ids: set = set()
    deduped = []
    for chunk in json_chunks:
        chunk_id = chunk["chunk_id"] if isinstance(chunk, dict) else chunk.id
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            deduped.append(chunk)
    if len(deduped) < len(json_chunks):
        logger.warning(
            "embed_chunks: %d chunk(s) dupliqué(s) supprimé(s)",
            len(json_chunks) - len(deduped),
        )
    json_chunks = deduped

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

    client = ChromaClient.get_client()
    collection = client.get_or_create_collection(collection_name)
    embeddings = embeddingModel.embed_documents(texts)
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"✅ {len(json_chunks)} chunks vectorises et stockes dans '{collection_name}'.")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddingModel,
        client=client,
    )
