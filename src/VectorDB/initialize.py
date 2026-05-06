import os
from langchain_chroma import Chroma

def get_vector_db(embedding_model):
    """
    Initialise et charge le Vector Store Chroma existant.
    """
    return Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        persist_directory=os.getenv("VECTOR_DB_DIR"),
        embedding_function=embedding_model
    )
