import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .parsing import _pdf_docling
from .splitting import chuncking_md_data
from .embedding import embed_chunks

load_dotenv()


# ---------------------------------------------------------------------------
# Pipeline PDF
# ---------------------------------------------------------------------------


def transform_pdf(file_path: str) -> Optional[object]:
    """
    Pipeline complet pour un fichier PDF : Extraction → Chunking → Vectorisation.

    Args:
        file_path: Chemin vers le fichier PDF à traiter.

    Returns:
        Instance ChromaDB VectorStore ou None si le traitement échoue.
    """
    filename = os.path.basename(file_path)

    print(f"--- Étape 1 : Parsing de {filename} ---")
    markdown_content = _pdf_docling(file_path)

    if not markdown_content:
        print("Erreur : Aucun contenu extrait.")
        return None

    print("--- Étape 2 : Découpage en chunks ---")
    json_chunks = chuncking_md_data(
        md_text=markdown_content,
        filename=filename,
        max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
    )

    print("--- Étape 3 : Vectorisation et stockage dans ChromaDB ---")
    vector_db = embed_chunks(
        json_chunks=json_chunks,
        collection_name=os.getenv("COLLECTION_NAME", "rag_cnaps"),
    )

    print(f"--- Terminé ! Document '{filename}' prêt pour le RAG ---")
    return vector_db

