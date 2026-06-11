import logging
import os
from typing import Optional

import trafilatura
from bs4 import BeautifulSoup

from ..scraping.models import WebPageContent, WebPageContentExtracted, WebPageMetadata

from .utils import get_class_contained, _fragments_to_markdown, convert_date_from_html

logger = logging.getLogger(__name__)

_FALLBACK_CONTENT = (
    "Voici la page concernant cette information que vous demandez : {title}. Publié le {date}."
)


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
# HTML (classes CSS ciblées) → WebPageContentExtracted via trafilatura
# ---------------------------------------------------------------------------


def _parse_html(page: WebPageContent) -> Optional[WebPageContentExtracted]:
    html = trafilatura.fetch_url(page.url)
    if not html:
        logger.warning("_parse_html: echec pour %s", page.url)
        return None

    soup = BeautifulSoup(html, "html.parser")
    title_tag = None
    date_tag = None
    contenu_md = None

    if page.classes:
        seen: set = set()
        fragments = []
        for cls in page.classes:
            if title_tag is None and get_class_contained("title", cls):
                title_tag = soup.find(class_=cls)
            if date_tag is None and get_class_contained("date", cls):
                date_tag = soup.find(class_=cls)
            for tag in soup.find_all(class_=cls):
                if id(tag) not in seen:
                    seen.add(id(tag))
                    fragments.append(str(tag))
        frag_soup = BeautifulSoup("".join(fragments), "html.parser")
        contenu_md = _fragments_to_markdown(frag_soup)
    else:
        contenu_md = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=True,
            include_formatting=True,
        )

    if title_tag is None:
        title_tag = soup.find("title")

    title_from_class = title_tag.get_text(strip=True) if title_tag else None
    date_from_class = convert_date_from_html(date_tag.get_text() if date_tag else None)

    metas = None
    if not (title_from_class or date_from_class):
        metas = trafilatura.extract_metadata(html)

    title = title_from_class or (metas.title if metas else "Titre inconnu")
    date = date_from_class or (metas.date if metas else "Date inconnue")

    if not contenu_md:
        logger.warning("_parse_html: extraction vide pour %s, document synthetique cree", page.url)
        contenu_md = _FALLBACK_CONTENT.format(title=title, date=date)

    logger.info("_parse_html: extraction reussie depuis %s", page.url)
    return WebPageContentExtracted(
        contenu_md=contenu_md,
        metadata=WebPageMetadata(
            source_url=page.url,
            title=title,
            date_posted=date,
        ),
    )
