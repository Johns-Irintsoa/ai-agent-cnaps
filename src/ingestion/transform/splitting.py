import os
from typing import Any, Dict, List

from .models import WebPageContentChunked
from .utils import (
    build_chunk_item,
    extract_signature_blocks,
    flatten_metadata,
    generate_chunk_id,
    link_chunks,
    normalize_md_text,
    prefix_chunk_with_context,
    process_text_part,
    split_by_headers,
    split_by_tokens,
    split_tables_from_text,
)


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


def chunking_md_html(page) -> List[WebPageContentChunked]:

    # Phase 1 — Normalize & flatten metadata
    clean_text = normalize_md_text(page.contenu_md)
    flat_meta = flatten_metadata(page)

    # Phase 2 — Semantic split: headers first, then token-aware size
    header_docs = split_by_headers(clean_text)
    token_docs = split_by_tokens(header_docs, max_tokens=450, overlap_tokens=50)

    # Phase 3 — Context enrichment: prefix each chunk with page title
    enriched_texts = [
        prefix_chunk_with_context(doc.page_content, flat_meta["title"])
        for doc in token_docs
    ]

    # Phase 4 — ID generation & assembly
    total = len(enriched_texts)
    return [
        WebPageContentChunked(
            id=generate_chunk_id(flat_meta["source_url"], i),
            document=text,
            metadata={**flat_meta, "chunk_index": i, "total_chunks": total},
        )
        for i, text in enumerate(enriched_texts)
    ]