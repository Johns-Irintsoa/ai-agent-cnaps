"""
UnstructuredLoader — Chargement de documents pour la vector database CNaPS.

Deux fonctions principales :
  - load_documents_from_dirs(directories) : charge PDF, DOCX, images, CSV, Excel, TXT
    depuis une liste de chemins de dossiers.
  - load_html_from_urls(urls) : recupere et parse le contenu textuel de pages HTML
    depuis une liste d'URLs.

Les deux fonctions retournent des list[Document] (LangChain) prets pour le chunking
et l'indexation dans la vector database.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader,
    WebBaseLoader,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping extension → loader LangChain
# ---------------------------------------------------------------------------

_LOADER_MAP: dict[str, type] = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc":  Docx2txtLoader,
    ".csv":  CSVLoader,
    ".txt":  TextLoader,
    ".xls":  UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".png":  UnstructuredImageLoader,
    ".jpg":  UnstructuredImageLoader,
    ".jpeg": UnstructuredImageLoader,
    ".gif":  UnstructuredImageLoader,
    ".webp": UnstructuredImageLoader,
}

# Extensions supportees (pour le filtrage rapide)
SUPPORTED_EXTENSIONS = set(_LOADER_MAP.keys())


# ---------------------------------------------------------------------------
# Utilitaire interne
# ---------------------------------------------------------------------------

def _load_file(file_path: str) -> list[Document]:
    """
    Charge un fichier unique en utilisant le loader LangChain adapte a son extension.
    Utilise UnstructuredFileLoader comme fallback pour les types non reconnus.

    Args:
        file_path: Chemin absolu ou relatif vers le fichier.

    Returns:
        Liste de Document. Retourne [] en cas d'erreur.
    """
    ext = Path(file_path).suffix.lower()
    loader_cls = _LOADER_MAP.get(ext, UnstructuredFileLoader)

    try:
        loader = loader_cls(file_path)
        docs = loader.load()
        # Ajouter le chemin source dans les metadonnees si absent
        for doc in docs:
            doc.metadata.setdefault("source", file_path)
        log.debug(f"  Charge : {Path(file_path).name} ({len(docs)} chunk(s)) [{loader_cls.__name__}]")
        return docs
    except Exception as e:
        log.warning(f"  Erreur chargement {file_path} : {e}")
        return []


# ---------------------------------------------------------------------------
# Chargement depuis des dossiers
# ---------------------------------------------------------------------------

def load_documents_from_dirs(
    directories: list[str],
    extensions: Optional[list[str]] = None,
    recursive: bool = True,
) -> list[Document]:
    """
    Charge tous les documents depuis une liste de dossiers.

    Formats supportes nativement :
      PDF, DOCX, DOC, CSV, TXT, XLS, XLSX, PNG, JPG, JPEG, GIF, WEBP.
    Tout autre type de fichier est tente via UnstructuredFileLoader.

    Args:
        directories: Liste de chemins de dossiers a parcourir.
                     Exemple : ["data/unstructured/formulaire/pdf",
                                "data/unstructured/rapport/pdf"]
        extensions:  Liste optionnelle d'extensions a inclure (avec le point).
                     Exemple : [".pdf", ".docx"]
                     Si None, tous les fichiers supportes sont charges.
        recursive:   Si True (defaut), parcourt les sous-dossiers recursivement.

    Returns:
        Liste de Document LangChain avec metadonnees (source, file_type, directory).
        Retourne [] si aucun fichier trouve ou si toutes les erreurs sont ignorees.

    Exemple :
        docs = load_documents_from_dirs(
            directories=["data/unstructured/formulaire"],
            extensions=[".pdf", ".docx"],
        )
    """
    allowed_exts = {e.lower() for e in extensions} if extensions else None
    all_docs: list[Document] = []
    total_files = 0
    errors = 0

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            log.warning(f"Dossier introuvable, ignore : {directory}")
            continue
        if not dir_path.is_dir():
            log.warning(f"Chemin n'est pas un dossier, ignore : {directory}")
            continue

        log.info(f"Parcours du dossier : {directory}")
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(dir_path.glob(pattern)):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if allowed_exts and ext not in allowed_exts:
                continue
            # Ignorer les fichiers caches et temporaires
            if file_path.name.startswith((".", "~", "__")):
                continue

            total_files += 1
            docs = _load_file(str(file_path))

            if docs:
                # Enrichir les metadonnees avec le type et le dossier source
                for doc in docs:
                    doc.metadata["file_type"] = ext.lstrip(".")
                    doc.metadata["directory"] = str(dir_path)
                all_docs.extend(docs)
            else:
                errors += 1

    log.info(
        f"Chargement termine : {len(all_docs)} document(s) depuis {total_files} fichier(s) "
        f"({errors} erreur(s)) dans {len(directories)} dossier(s)"
    )
    return all_docs


# ---------------------------------------------------------------------------
# Chargement depuis des URLs HTML
# ---------------------------------------------------------------------------

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CNaPS-Bot/1.0)"}
_MIN_TEXT_CHARS = 30   # nb min de caracteres pour qu'une classe soit consideree utile


def _has_useful_text(element) -> bool:
    """
    Verifie si un element BeautifulSoup contient du texte brut utile.

    Un element est considere inutile si :
      - son texte (apres stripping) est trop court (< _MIN_TEXT_CHARS)
      - il ne contient que des balises <a>, <img>, <button> sans texte associe
    """
    text = element.get_text(separator=" ", strip=True)
    if len(text) < _MIN_TEXT_CHARS:
        return False

    # Compter les caracteres de texte pur hors balises enfants non-textuels
    non_text_tags = {"img", "video", "audio", "iframe", "script", "style", "svg"}
    for tag in element.find_all(non_text_tags):
        tag.decompose()

    clean_text = element.get_text(separator=" ", strip=True)
    return len(clean_text) >= _MIN_TEXT_CHARS


def _extract_text_from_classes(
    html: str,
    classes: list[str],
    url: str,
) -> str:
    """
    Extrait le texte brut des elements correspondant aux classes CSS indiquees,
    en ignorant automatiquement celles qui ne contiennent pas de texte utile.

    Gere les classes multi-tokens ("foo bar" = element qui possede les deux classes).

    Args:
        html:    Contenu HTML brut de la page.
        classes: Liste de classes CSS a cibler (ex: ["content-article__texte", "faq-content__items"]).
        url:     URL source (pour les logs).

    Returns:
        Texte brut concatene depuis toutes les classes utiles, separe par double saut de ligne.
    """
    soup = BeautifulSoup(html, "html.parser")
    collected: list[str] = []
    seen_ids: set[int] = set()

    for class_name in classes:
        parts = class_name.split()

        if len(parts) > 1:
            # Multi-classe : l'element doit posseder TOUTES les classes listees
            matches = soup.find_all(
                lambda tag, p=parts: all(c in tag.get("class", []) for c in p)
            )
        else:
            matches = soup.find_all(class_=class_name)

        useful_count = 0
        for el in matches:
            if id(el) in seen_ids:
                continue
            seen_ids.add(id(el))

            if not _has_useful_text(el):
                log.debug(f"  Classe '{class_name}' ignoree (pas de texte utile) sur {url}")
                continue

            text = el.get_text(separator="\n", strip=True)
            if text:
                collected.append(text)
                useful_count += 1

        if useful_count > 0:
            log.debug(f"  Classe '{class_name}' : {useful_count} element(s) utile(s) extrait(s)")

    return "\n\n".join(collected)


def load_html_from_urls(
    urls: list[str],
    classes: Optional[list[str]] = None,
    continue_on_error: bool = True,
) -> list[Document]:
    """
    Recupere et parse le contenu textuel de pages HTML depuis une liste d'URLs.

    Si 'classes' est fourni, seul le texte contenu dans les elements correspondant
    a ces classes CSS est extrait. Les classes qui ne contiennent pas de texte brut
    utile (images, liens seuls, contenu vide) sont automatiquement ignorees.

    Si 'classes' est None, WebBaseLoader charge la page complete (comportement original).

    Args:
        urls:              Liste d'URLs a charger.
                           Exemple : ["https://www.cnaps.mg/fr/faq",
                                      "https://www.cnaps.mg/fr/contacter/"]
        classes:           Liste optionnelle de classes CSS a cibler dans le HTML.
                           Supporte les classes multi-tokens :
                             "liste-result__item grid-result__item"
                             → elements qui possedent LES DEUX classes.
                           Les classes sans texte brut sont automatiquement ignorees.
                           Exemple : ["content-article__texte", "faq-content__items"]
        continue_on_error: Si True (defaut), ignore les URLs qui echouent.

    Returns:
        Liste de Document LangChain avec metadonnees (source, url, classes_used).
        Retourne [] si toutes les URLs echouent ou ne contiennent pas de texte utile.

    Exemples :
        # Sans filtrage de classes (charge toute la page)
        docs = load_html_from_urls([
            "https://www.cnaps.mg/fr/faq",
        ])

        # Avec filtrage par classes CSS
        docs = load_html_from_urls(
            urls=["https://www.cnaps.mg/fr/faq"],
            classes=["faq-content__items", "content-article__texte"],
        )
    """
    if not urls:
        log.warning("load_html_from_urls : liste d'URLs vide.")
        return []

    all_docs: list[Document] = []
    errors = 0

    for url in urls:
        try:
            log.info(f"Chargement HTML : {url}")

            if classes:
                # ── Mode filtre : extraction manuelle par classes CSS ──────────
                resp = httpx.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
                resp.raise_for_status()

                text = _extract_text_from_classes(resp.text, classes, url)

                if not text.strip():
                    log.warning(f"  Aucun texte utile extrait depuis {url} avec les classes {classes}")
                    continue

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "url": url,
                        "classes_used": classes,
                    },
                )
                all_docs.append(doc)
                log.info(f"  {url} → 1 document ({len(text)} caracteres)")

            else:
                # ── Mode complet : WebBaseLoader charge toute la page ─────────
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    doc.metadata.setdefault("source", url)
                    doc.metadata["url"] = url
                all_docs.extend(docs)
                log.info(f"  {url} → {len(docs)} document(s)")

        except Exception as e:
            errors += 1
            log.warning(f"  Erreur chargement URL {url} : {e}")
            if not continue_on_error:
                raise

    log.info(
        f"Chargement HTML termine : {len(all_docs)} document(s) depuis "
        f"{len(urls) - errors}/{len(urls)} URL(s) ({errors} erreur(s))"
    )
    return all_docs
