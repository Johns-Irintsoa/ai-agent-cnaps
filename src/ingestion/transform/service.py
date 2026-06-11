import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .parsing import _pdf_docling
from .splitting import chuncking_md_data, chunking_md_html
from .embedding import embed_chunks
from ..transform.parsing import _parse_html
from ..scraping.WebPageScrapper import extract_urls
from ..scraping.models import WebPageContent

load_dotenv()

logger = logging.getLogger(__name__)

# Extract PDF
def transform_pdf(file_path: str) -> Optional[object]:
    """
    Pipeline complet pour un fichier PDF : Extraction → Chunking → Vectorisation.

    Args:
        file_path: Chemin vers le fichier PDF à traiter.

    Returns:
        Instance ChromaDB VectorStore ou None si le traitement échoue.
    """
    filename = os.path.basename(file_path)
    try:
        logger.info("Etape 1 : Parsing de %s", filename)
        markdown_content = _pdf_docling(file_path)
        if not markdown_content:
            logger.error("Etape 1 echouee : Aucun contenu extrait pour %s", filename)
            return None

        logger.info("Etape 2 : Decoupage en chunks")
        json_chunks = chuncking_md_data(
            md_text=markdown_content,
            filename=filename,
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
        )

        logger.info("Etape 3 : Vectorisation dans ChromaDB")
        vector_db = embed_chunks(
            json_chunks=json_chunks,
            collection_name=os.getenv("COLLECTION_NAME", "rag_cnaps"),
        )
        logger.info("Termine : '%s' pret pour le RAG", filename)
        return vector_db
    except Exception as e:
        logger.error("transform_pdf echoue pour %s : %s", filename, e, exc_info=True)
        return None

# Extract Web Pages from www.cnaps.mg
def transform_web_page(collection_name: str = None) -> Optional[object]:
    if collection_name is None:
        collection_name = os.getenv("COLLECTION_NAME", "rag_cnaps")

    logger.info("--- Etape 1 : Extraction des pages web CNaPS ---")
    extracted_pages = extract_urls()

    if not extracted_pages:
        logger.error("Etape 1 echouee : Aucune page extraite.")
        return None
    logger.info("Etape 1 OK : %d page(s) extraites.", len(extracted_pages))

    logger.info("--- Etape 2 : Decoupage de %d page(s) en chunks ---", len(extracted_pages))
    all_chunks = []
    for page in extracted_pages:
        try:
            chunks = chunking_md_html(page)
            all_chunks.extend(chunks)
            logger.info("  chunked %s → %d chunks", page.metadata.source_url, len(chunks))
        except Exception as e:
            logger.error("  erreur chunking pour %s : %s", page.metadata.source_url, e, exc_info=True)

    if not all_chunks:
        logger.error("Etape 2 echouee : Aucun chunk produit.")
        return None
    logger.info("Etape 2 OK : %d chunks au total.", len(all_chunks))

    logger.info("--- Etape 3 : Vectorisation de %d chunk(s) dans '%s' ---", len(all_chunks), collection_name)
    try:
        vector_db = embed_chunks(all_chunks, collection_name=collection_name)
    except Exception as e:
        logger.error("Etape 3 echouee : %s", e, exc_info=True)
        return None

    logger.info("--- Termine ! %d chunks web stockes dans '%s' ---", len(all_chunks), collection_name)
    return vector_db


def transform_url(url: str, classes: List[str], collection_name: Optional[str] = None) -> int:
    """Parse directement une URL avec les classes CSS fournies, chunk et stocke dans ChromaDB."""
    coll_name = collection_name or os.getenv("COLLECTION_NAME", "rag_cnaps")
    try:
        page = WebPageContent(url=url, classes=classes)
        extracted = _parse_html(page)
        if extracted is None:
            logger.warning("transform_url: aucun contenu extrait depuis %s", url)
            return 0

        chunks = chunking_md_html(extracted)
        if not chunks:
            logger.warning("transform_url: aucun chunk généré depuis %s", url)
            return 0

        embed_chunks(chunks, coll_name)
        logger.info("transform_url: %d chunks ingérés depuis %s", len(chunks), url)
        return len(chunks)
    except Exception as e:
        logger.error("transform_url failed for %s: %s", url, e)
        return 0

