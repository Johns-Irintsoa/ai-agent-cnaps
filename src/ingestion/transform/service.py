import os
from parsing import _pdf_docling
from splitting import chuncking_md_data
from embedding import embed_chunks
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Charge les variables d'environnement depuis le fichier .env

def transform_pdf(file_path):

    """
    Pipeline complet : Extraction -> Chunking -> Vectorisation.
    
    Args:
        file_path (str): Chemin vers le fichier PDF à traiter.
    """
    filename = os.path.basename(file_path)
    
    # 1. Extraction (Parsing) via Docling
    print(f"--- Étape 1 : Parsing de {filename} ---")
    markdown_content = _pdf_docling(file_path) 
    
    if not markdown_content:
        print("Erreur : Aucun contenu extrait.")
        return None

    # 2. Découpage (Chunking)
    print(f"--- Étape 2 : Découpage en chunks ---")
    json_chunks = chuncking_md_data(
        md_text=markdown_content, 
        filename=filename,
        max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100))
    )

    # 3. Stockage Vectoriel (Embedding)
    print(f"--- Étape 3 : Vectorisation et stockage dans ChromaDB ---")
    vector_db = embed_chunks(
        json_chunks=json_chunks, 
        collection_name=os.getenv("COLLECTION_NAME", "rag_cnaps")
    )

    print(f"--- Terminé ! Document '{filename}' prêt pour le RAG ---")
    return vector_db

# Exemple d'usage :
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# print(_PROJECT_ROOT)
sample_pdf = _PROJECT_ROOT / "data/unstructured/pdf/265df2015-CNaPS_600fd1b4ca3383.50730780.pdf"
text_result = transform_pdf(sample_pdf)
# print(text_result)
