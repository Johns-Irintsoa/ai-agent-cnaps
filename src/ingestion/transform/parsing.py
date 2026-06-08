import logging
import os
from typing import Optional

import trafilatura

from ..scraping.models import WebPageContentExtracted, WebPageMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF → Markdown
# ---------------------------------------------------------------------------

def _pdf_docling(file_path: str) -> Optional[str]:
    from docling.document_converter import DocumentConverter

    print(f"Extraction via Docling pour : {os.path.basename(file_path)}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


# ---------------------------------------------------------------------------
# HTML (URL) → WebPageContentExtracted via trafilatura
# ---------------------------------------------------------------------------

def _parse_html(url: str) -> Optional[WebPageContentExtracted]:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        logger.warning("_parse_html: aucun contenu telecharge depuis %s", url)
        return None

    texte_markdown = trafilatura.extract(downloaded, output_format="markdown", include_links=True)
    metas_brutes = trafilatura.extract_metadata(downloaded)

    if not texte_markdown:
        logger.warning("_parse_html: extraction vide depuis %s", url)
        return None

    logger.info("_parse_html: extraction reussie depuis %s", url)
    return WebPageContentExtracted(
        contenu_md=texte_markdown,
        metadata=WebPageMetadata(
            source_url=url,
            title=metas_brutes.title if metas_brutes else "Titre inconnu",
            date_posted=metas_brutes.date if metas_brutes else "Date inconnue",
        ),
    )


