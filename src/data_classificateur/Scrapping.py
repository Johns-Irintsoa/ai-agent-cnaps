import httpx
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from config import CONFIG
from data_classificateur.DataModel import FileEntry
log = logging.getLogger(__name__)
BASE_URL = "https://www.cnaps.mg"
HEADERS  = {"User-Agent": "Mozilla/5.0 (compatible; CNaPS-Bot/1.0)"}


def ext_from_url(url: str) -> str:
    path = url.lower().split("?")[0]
    for ext in (".pdf", ".docx", ".doc", ".xls", ".xlsx", ".zip", ".rar", ".txt"):
        if path.endswith(ext):
            return ext
    return ""

def scrape_page(page_url: str) -> list[FileEntry]:
    """
    Scrape une page CNaPS.

    Logique de parsing :
    1. Parcourt séquentiellement tous les éléments du DOM
    2. Chaque <h2 class="list-option___title text-left"> → nouveau groupe
    3. Chaque <a href="...fichier.ext"> → nouveau FileEntry dans le groupe courant
    4. Le libellé est pris dans le texte du nœud précédant le <a>
    """
    log.info(f"Scraping {page_url} ...")
    resp = httpx.get(page_url, headers=HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    entries: list[FileEntry] = []
    current_group = "Sans titre"
    seen_urls: set[str] = set()

    container = soup.find("main") or soup.find("div", id="content") or soup.body
    if not container:
        log.warning(f"  Contenu principal introuvable sur {page_url}")
        return entries

    for el in container.find_all(True):
        # Détection du titre de groupe
        if el.name == "h2":
            classes = el.get("class", [])
            # Classe exacte CNaPS
            is_group_title = (
                "list-option___title" in classes
                or "text-left" in classes
                # fallback : h2 hors nav/header/footer
                or not el.find_parent(["nav", "header", "footer"])
            )
            if is_group_title:
                txt = el.get_text(strip=True)
                if txt and len(txt) < 150:
                    current_group = txt
                    log.debug(f"  Groupe : {current_group}")

        # Détection d'un lien de fichier
        if el.name == "a":
            href = el.get("href", "")
            if not href or href.startswith(("#", "javascript", "mailto")):
                continue

            abs_url = urljoin(BASE_URL, href)
            ext = ext_from_url(abs_url)
            if not ext:
                continue
            if abs_url in seen_urls:
                continue
            seen_urls.add(abs_url)

            # Libellé : cherche le texte dans les éléments précédents
            label = ""
            # Cherche dans le parent direct ou le sibling précédent
            for candidate in [
                el.find_previous_sibling(),
                el.parent.find_previous_sibling() if el.parent else None,
            ]:
                if candidate:
                    txt = candidate.get_text(strip=True)
                    if txt and len(txt) > 3:
                        label = txt[:200]
                        break
            if not label:
                label = el.get_text(strip=True) or abs_url.split("/")[-1]

            entries.append(FileEntry(
                page_url=page_url,
                group_title=current_group,
                file_label=label,
                file_url=abs_url,
                file_type=ext.lstrip("."),
            ))
            log.debug(f"  + [{current_group}] {label[:50]} ({ext})")

    log.info(f"  → {len(entries)} fichier(s) trouvé(s)")
    return entries

def scrape_all_pages() -> list[FileEntry]:
    all_entries: list[FileEntry] = []
    for page_url in CONFIG.pages:
        try:
            all_entries.extend(scrape_page(page_url))
        except Exception as e:
            log.error(f"Erreur scraping {page_url} : {e}")
    log.info(f"\nTotal : {len(all_entries)} fichier(s) à classifier")
    return all_entries

def detect_type(url: str, ct: str) -> str:
    u, c = url.lower().split("?")[0], ct.lower()
    if u.endswith(".pdf") or "application/pdf" in c:   return "pdf"
    if u.endswith(".docx") or "wordprocessingml" in c: return "docx"
    if u.endswith(".doc") or "msword" in c:            return "doc"
    if u.endswith(".xls") or "ms-excel" in c:          return "xls"
    if u.endswith(".xlsx") or "spreadsheetml" in c:    return "xlsx"
    if u.endswith(".zip") or "zip" in c:               return "zip"
    if u.endswith(".rar") or "rar" in c:               return "rar"
    if u.endswith(".txt") or "text/plain" in c:        return "txt"
    return "inconnu"