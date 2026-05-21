import os
from langchain_chroma import Chroma
from ..db.chroma_client import ChromaClient

def get_vector_db(embedding_model):
    """
    Initialise et charge le Vector Store Chroma existant.
    Se connecte au serveur ChromaDB HTTP (Docker) ou en local selon CHROMA_HOST.
    """
    return Chroma(
        collection_name=os.getenv("COLLECTION_NAME", "rag_cnaps"),
        embedding_function=embedding_model,
        client=ChromaClient.get_client(),
    )
