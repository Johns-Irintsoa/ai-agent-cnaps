import json
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .models import WebPageContent, WebPageContentExtracted, WebPageContentMd

logger = logging.getLogger(__name__)

_CNAPS_URLS_PATH = Path(__file__).resolve().parents[3] / "cnaps_urls.json"


def get_urls_from_json(json_path: Path = _CNAPS_URLS_PATH) -> List[WebPageContent]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("cnaps_urls", [])
    logger.info(f"get_urls_from_json: {len(entries)} URL(s) trouvees dans {json_path.name}")
    return [WebPageContent.model_validate(entry) for entry in entries]


def _fetch_paginated_urls(page: WebPageContent) -> List[str]:
    article_urls: List[str] = []
    next_url: Optional[str] = page.url

    while next_url:
        try:
            response = requests.get(next_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"_fetch_paginated_urls: echec pour {next_url} — {e}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        for css_class in page.classes:
            for container in soup.find_all(class_=css_class):
                for a_tag in container.find_all("a", href=True):
                    absolute = urljoin(page.url, a_tag["href"])
                    if absolute not in article_urls:
                        article_urls.append(absolute)

        next_tag = soup.select_one(page.pagination_selector) if page.pagination_selector else None
        next_url = urljoin(next_url, next_tag["href"]) if next_tag else None

    logger.info(f"_fetch_paginated_urls: {len(article_urls)} URL(s) depuis {page.url}")
    return article_urls


def get_url_from_paginated_page() -> List[str]:
    pages = [p for p in get_urls_from_json() if p.is_paginated]
    return get_all_urls(pages)


def get_all_urls(pages: List[WebPageContent]) -> List[str]:
    urls: List[str] = []
    for page in pages:
        if page.is_paginated:
            urls.extend(_fetch_paginated_urls(page))
        else:
            urls.append(page.url)
    logger.info(f"get_all_urls: {len(urls)} URL(s) au total")
    return urls


def extract_urls() -> List[WebPageContentExtracted]:
    from ..transform.parsing import _parse_html

    pages = get_urls_from_json()
    urls = get_all_urls(pages)
    results: List[WebPageContentExtracted] = []
    for url in urls:
        extracted = _parse_html(url)
        if extracted:
            results.append(extracted)
    logger.info(f"extract_urls: {len(results)} page(s) extraites sur {len(urls)}")
    return results


def scrapper(page: WebPageContent) -> Optional[WebPageContentMd]:
    try:
        response = requests.get(page.url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"scrapper: echec pour {page.url} — {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    if page.classes:
        seen_ids: set = set()
        parts = []
        for css_class in page.classes:
            for tag in soup.find_all(class_=css_class):
                if id(tag) not in seen_ids:
                    seen_ids.add(id(tag))
                    parts.append(tag.get_text(separator="\n", strip=True))
        content = "\n\n".join(parts)
    else:
        body = soup.find("body")
        content = body.get_text(separator="\n", strip=True) if body else ""

    return WebPageContentMd(url=page.url, content=content)
