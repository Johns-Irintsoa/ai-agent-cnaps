
from dotenv import load_dotenv
from langchain_chroma import Chroma

from ...db.chroma_client import ChromaClient
from ...models.embedding import embedding_manager

load_dotenv()  # Charge les variables d'environnement depuis le fichier .env

def embed_chunks(json_chunks, collection_name="rag_cnaps"):
    """
    Prend la liste de JSON, vectorise le contenu et sauvegarde dans ChromaDB.
    """
    
    # 1. Configuration du modèle d'embedding
    embeddingModel = embedding_manager.model
    
    # 2. Préparation des données pour ChromaDB
    texts = [chunk["content"] for chunk in json_chunks]
    ids = [chunk["chunk_id"] for chunk in json_chunks]
    
    # ChromaDB n'accepte pas les dictionnaires imbriqués dans les métadonnées.
    # On doit "aplatir" votre dictionnaire metadata.
    metadatas = []
    for chunk in json_chunks:
        flat_meta = chunk["metadata"].copy()
        # On convertit les listes en strings car Chroma ne supporte que str, int, float ou bool
        if "page_numbers" in flat_meta:
            flat_meta["page_numbers"] = str(flat_meta["page_numbers"])
        metadatas.append(flat_meta)

    # 3. Création et stockage dans la base vectorielle
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddingModel,
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        client=ChromaClient.get_client(),
    )

    print(f"✅ {len(json_chunks)} chunks ont été vectorisés et stockés dans '{collection_name}'.")
    return vector_db