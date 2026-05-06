import os
from typing import List, Dict, Any

from .utils import (
    extract_signature_blocks,
    split_tables_from_text,
    process_text_part,
    build_chunk_item,
    link_chunks,
)

from langchain_core.documents import Document

def chuncking_md_data(
    md_text: str,
    filename: str = "document_source",
    max_chunk_size: int = os.getenv("CHUNCKING_MAX_TOKENS"),
    chunk_overlap: int = os.getenv("CHUNCKING_OVERLAP_TOKENS"),
) -> List[Dict[str, Any]]:
    """
    Point d'entrée principal : découpe un document Markdown en chunks structurés.
 
    Pipeline complet :
      1. Extraction préalable des blocs de signature/tampon
      2. Séparation tableaux / texte
      3. Pour les tableaux  → chunk atomique (jamais découpé)
      4. Pour le texte      → headers puis taille avec séparateurs légaux
      5. Construction des items JSON avec toutes les métadonnées
      6. Liaison prev/next entre chunks
 
    Args:
        md_text:        Contenu Markdown issu de Docling.
        filename:       Nom du fichier source (pour les métadonnées).
        max_chunk_size: Taille maximum en caractères par chunk (défaut : 1000).
        chunk_overlap:  Chevauchement en caractères entre chunks (défaut : 100).
 
    Returns:
        Liste de dicts JSON représentant les chunks enrichis.
    """
 
    # ── Étape 1 : pré-extraction des signatures (fix tampon/date) ──────────
    signature_blocks = extract_signature_blocks(md_text)
 
    # ── Étape 2 : séparation tableaux / texte ──────────────────────────────
    parts = split_tables_from_text(md_text)
 
    final_json_list: List[Dict] = []
 
    for part in parts:
 
        if part["type"] == "table":
            # ── Texte → pipeline headers + taille ──────────────────────────
            docs_to_process = process_text_part(
                part["content"], max_chunk_size, chunk_overlap
            )
            for d in docs_to_process:
                d.metadata["is_table"] = True
        else:
            # Texte classique
            docs_to_process = process_text_part(
                part["content"], max_chunk_size, chunk_overlap
            )
 
        for doc in docs_to_process:
            current_index = len(final_json_list)
            item = build_chunk_item(
                content=doc.page_content,
                metadata_extra=doc.metadata,
                current_index=current_index,
                final_json_list=final_json_list,
                filename=filename,
                signature_blocks=signature_blocks,
            )
            # Lien "next" du chunk précédent
            if current_index > 0:
                final_json_list[current_index - 1]["metadata"]["next_chunk_id"] = (
                    item["chunk_id"]
                )
            final_json_list.append(item)
 
    # ── Étape finale : liens next complets + total_chunks ──────────────────
    return link_chunks(final_json_list)