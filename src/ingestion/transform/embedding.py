import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings # Ou un modèle gratuit comme SentenceTransformers

load_dotenv()  # Charge les variables d'environnement depuis le fichier .env

def embed_chunks(json_chunks, collection_name="rag_cnaps"):
    """
    Prend la liste de JSON, vectorise le contenu et sauvegarde dans ChromaDB.
    """
    
    # 1. Configuration du modèle d'embedding
    # Note: Assurez-vous d'avoir votre clé API si vous utilisez OpenAI
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL"))
    
    # 2. Préparation des données pour ChromaDB
    # On sépare le texte, les métadonnées et les IDs
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
    # 'persist_directory' permet de sauvegarder la base sur le disque local
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        persist_directory=os.getenv("VECTOR_DB_DIR")
    )

    print(f"✅ {len(json_chunks)} chunks ont été vectorisés et stockés dans '{collection_name}'.")
    return vector_db

# --- Exemple d'intégration ---
# chunks_json = chuncking_md_data(md_output, "mon_document.pdf")
# db = embed_chunks(chunks_json)
