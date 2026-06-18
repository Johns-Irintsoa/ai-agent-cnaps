import json
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup

# cnaps.mg uses a certificate not present in minimal Docker CA bundles
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from .models import WebPageContent, WebPageContentExtracted, WebPageFromJSON

logger = logging.getLogger(__name__)

_CNAPS_URLS_PATH = Path(__file__).resolve().parents[3] / "cnaps_urls.json"


def get_urls_from_json(json_path: Path = _CNAPS_URLS_PATH) -> List[WebPageFromJSON]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("cnaps_urls", [])
    logger.info(
        f"get_urls_from_json: {len(entries)} URL(s) trouvees dans {json_path.name}"
    )
    return [WebPageFromJSON.model_validate(entry) for entry in entries]


def _extract_article_links(
    soup: BeautifulSoup, base_url: str, css_classes: List[str]
) -> List[str]:
    return [
        urljoin(base_url, a["href"])
        for cls in css_classes
        for container in soup.find_all(class_=cls)
        for a in container.find_all("a", href=True)
    ]


def _get_total_pages(soup: BeautifulSoup, selector: Optional[str]) -> int:
    if not selector:
        return 1
    page_nums = [
        int(a.get_text(strip=True))
        for a in soup.select(f"{selector} a")
        if a.get_text(strip=True).isdigit()
    ]
    return max(page_nums, default=1)


def _fetch_list_urls(page: WebPageFromJSON) -> List[str]:
    logger.info(f"_fetch_list_urls: scraping {page.url}")
    try:
        response = requests.get(page.url, timeout=10, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"_fetch_list_urls: echec pour {page.url} — {e}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    seen: set = set()
    urls = []
    for url in _extract_article_links(soup, page.url, page.classes):
        if url not in seen:
            seen.add(url)
            urls.append(url)
    logger.info(f"_fetch_list_urls: {len(urls)} URL(s) depuis {page.url}")
    return urls


def _fetch_paginated_urls(page: WebPageFromJSON) -> List[str]:
    seen: set = set()
    article_urls: List[str] = []

    logger.info(f"_fetch_paginated_urls: scraping {page.url}")
    try:
        response = requests.get(page.url, timeout=10, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"_fetch_paginated_urls: echec pour {page.url} — {e}")
        return article_urls

    soup = BeautifulSoup(response.text, "html.parser")
    for url in _extract_article_links(soup, page.url, page.classes):
        if url not in seen:
            seen.add(url)
            article_urls.append(url)

    total_pages = _get_total_pages(soup, page.pagination_selector)
    logger.info(f"_fetch_paginated_urls: {total_pages} page(s) detectee(s)")

    for page_num in range(2, total_pages + 1):
        page_url = f"{page.url}?page={page_num}"
        logger.info(f"_fetch_paginated_urls: scraping {page_url}")
        try:
            response = requests.get(page_url, timeout=10, verify=False)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"_fetch_paginated_urls: echec pour {page_url} — {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for url in _extract_article_links(soup, page.url, page.classes):
            if url not in seen:
                seen.add(url)
                article_urls.append(url)

    logger.info(f"_fetch_paginated_urls: {len(article_urls)} URL(s) depuis {page.url}")
    return article_urls


def get_all_urls(pages: List[WebPageFromJSON]) -> List[WebPageContent]:
    result: List[WebPageContent] = []
    seen: set = set()
    for page in pages:
        if page.is_contained_list:
            article_classes = page.item_classes if page.item_classes is not None else page.classes
            urls = _fetch_paginated_urls(page) if page.pagination_selector else _fetch_list_urls(page)
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    result.append(WebPageContent(url=url, classes=article_classes))
        else:
            if page.url not in seen:
                seen.add(page.url)
                result.append(WebPageContent(url=page.url, classes=page.classes))
    logger.info(f"get_all_urls: {len(result)} URL(s) au total (après déduplication)")
    return result


def extract_urls() -> List[WebPageContentExtracted]:
    from ..transform.parsing import _parse_html

    pages = get_urls_from_json()
    all_pages = get_all_urls(pages)
    results: List[WebPageContentExtracted] = []
    for page in all_pages:
        extracted = _parse_html(page)
        if extracted:
            results.append(extracted)
    logger.info(f"extract_urls: {len(results)} page(s) extraites sur {len(all_pages)}")
    return results
