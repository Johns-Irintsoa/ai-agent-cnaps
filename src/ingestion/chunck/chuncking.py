"""
Chuncking — Découpage de documents par type pour la vector database CNaPS.

Fonctions principales :
  - split_html(documents)      : découpe HTML par en-têtes H1/H2/H3
  - split_markdown(documents)  : découpe Markdown par titres #/##/###
  - split_tabular(documents)   : découpe CSV/Excel ligne par ligne
  - split_text(documents)      : découpe générique (PDF, DOCX, TXT, OCR)
  - split_documents(documents) : dispatcher automatique selon le type de fichier

Toutes les fonctions acceptent une list[Document] LangChain et retournent
une list[Document] prête pour l'indexation dans la vector database.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes par défaut
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200

_HTML_HEADERS = [("h1", "H1"), ("h2", "H2"), ("h3", "H3")]
_MD_HEADERS   = [("#", "H1"), ("##", "H2"), ("###", "H3")]

# Types de fichiers tabulaires (pas de chevauchement utile)
_TABULAR_TYPES = {"csv", "xls", "xlsx"}
# Types HTML
_HTML_TYPES    = {"html", "htm"}
# Types Markdown
_MD_TYPES      = {"md", "markdown"}


# ---------------------------------------------------------------------------
# Détection du type depuis les métadonnées
# ---------------------------------------------------------------------------

def _infer_type(doc: Document) -> str:
    """
    Retourne le type de fichier (sans point) depuis les métadonnées du document.

    Priorité :
      1. doc.metadata["file_type"]
      2. Extension de doc.metadata["source"]
      3. "txt" par défaut
    """
    ft = doc.metadata.get("file_type", "").lower().strip()
    if ft:
        return ft
    source = doc.metadata.get("source", "").lower()
    suffix = Path(source).suffix.lstrip(".")
    return suffix if suffix else "txt"


# ---------------------------------------------------------------------------
# Splitters spécialisés
# ---------------------------------------------------------------------------

def split_html(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Découpe des documents HTML en chunks structurés par en-têtes H1/H2/H3.

    Deux passes :
      1. HTMLHeaderTextSplitter : segmente selon les balises de titre,
         enrichit les métadonnées avec le contexte hiérarchique (H1, H2, H3).
      2. RecursiveCharacterTextSplitter : second passage pour limiter
         la taille des segments issus de la première passe.

    Args:
        documents:     Liste de Document dont le contenu est du HTML.
        chunk_size:    Taille maximale d'un chunk (en caractères).
        chunk_overlap: Chevauchement entre chunks consécutifs.

    Returns:
        Liste de Document découpés, avec métadonnées enrichies.
    """
    if not documents:
        return []

    header_splitter = HTMLHeaderTextSplitter(headers_to_split_on=_HTML_HEADERS)
    char_splitter   = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        try:
            # Première passe : découpe par en-têtes
            header_chunks = header_splitter.split_text(doc.page_content)

            # Propager les métadonnées du document source
            for chunk in header_chunks:
                chunk.metadata = {**doc.metadata, **chunk.metadata}

            # Deuxième passe : limiter la taille
            final_chunks = char_splitter.split_documents(header_chunks)
            all_chunks.extend(final_chunks)

        except Exception as e:
            log.warning(f"split_html : erreur sur {doc.metadata.get('source', '?')} : {e}")
            # Fallback : découpe générique
            fallback = char_splitter.split_documents([doc])
            all_chunks.extend(fallback)

    log.info(f"split_html : {len(documents)} doc(s) → {len(all_chunks)} chunk(s)")
    return all_chunks


def split_markdown(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Découpe des documents Markdown en chunks structurés par titres #/##/###.

    Deux passes :
      1. MarkdownHeaderTextSplitter : segmente selon les titres Markdown,
         enrichit les métadonnées avec le contexte hiérarchique.
      2. RecursiveCharacterTextSplitter : second passage pour limiter la taille.

    Args:
        documents:     Liste de Document dont le contenu est du Markdown.
        chunk_size:    Taille maximale d'un chunk (en caractères).
        chunk_overlap: Chevauchement entre chunks consécutifs.

    Returns:
        Liste de Document découpés, avec métadonnées enrichies.
    """
    if not documents:
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        try:
            header_chunks = header_splitter.split_text(doc.page_content)

            # Propager les métadonnées du document source
            for chunk in header_chunks:
                chunk.metadata = {**doc.metadata, **chunk.metadata}

            final_chunks = char_splitter.split_documents(header_chunks)
            all_chunks.extend(final_chunks)

        except Exception as e:
            log.warning(f"split_markdown : erreur sur {doc.metadata.get('source', '?')} : {e}")
            fallback = char_splitter.split_documents([doc])
            all_chunks.extend(fallback)

    log.info(f"split_markdown : {len(documents)} doc(s) → {len(all_chunks)} chunk(s)")
    return all_chunks


def split_tabular(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[Document]:
    """
    Découpe des documents tabulaires (CSV, Excel) ligne par ligne.

    Utilise CharacterTextSplitter avec '\\n' comme séparateur et un
    chevauchement nul (les lignes CSV/Excel sont indépendantes).

    Args:
        documents:  Liste de Document dont le contenu est tabulaire.
        chunk_size: Nombre maximum de caractères par chunk.

    Returns:
        Liste de Document découpés.
    """
    if not documents:
        return []

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=0,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)
    log.info(f"split_tabular : {len(documents)} doc(s) → {len(chunks)} chunk(s)")
    return chunks


def split_text(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Découpe générique pour PDF, DOCX, TXT et texte OCR (images).

    Utilise RecursiveCharacterTextSplitter qui essaie de préserver les
    paragraphes, phrases et mots avant de couper arbitrairement.

    Args:
        documents:     Liste de Document.
        chunk_size:    Taille maximale d'un chunk (en caractères).
        chunk_overlap: Chevauchement entre chunks consécutifs.

    Returns:
        Liste de Document découpés.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    log.info(f"split_text : {len(documents)} doc(s) → {len(chunks)} chunk(s)")
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher principal
# ---------------------------------------------------------------------------

def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Dispatcher automatique : détecte le type de chaque document et applique
    le splitter adapté.

    Règles de routage (basées sur doc.metadata["file_type"] ou l'extension
    de doc.metadata["source"]) :
      - html, htm              → split_html
      - md, markdown           → split_markdown
      - csv, xls, xlsx         → split_tabular
      - tout autre type        → split_text  (pdf, docx, txt, png, jpg, ...)

    Les documents sont regroupés par type avant d'être envoyés au splitter
    correspondant, puis les résultats sont concaténés dans l'ordre original.

    Args:
        documents:     Liste de Document LangChain (types potentiellement mixtes).
        chunk_size:    Taille maximale d'un chunk (en caractères).
        chunk_overlap: Chevauchement entre chunks consécutifs (ignoré pour tabulaires).

    Returns:
        Liste de Document découpés, prêts pour l'indexation.

    Exemple :
        docs = load_documents_from_dirs(["data/unstructured/formulaire"])
        chunks = split_documents(docs, chunk_size=800, chunk_overlap=100)
    """
    if not documents:
        log.warning("split_documents : liste de documents vide.")
        return []

    # Regrouper les documents par type
    groups: dict[str, list[Document]] = {
        "html":    [],
        "md":      [],
        "tabular": [],
        "text":    [],
    }

    for doc in documents:
        ft = _infer_type(doc)
        if ft in _HTML_TYPES:
            groups["html"].append(doc)
        elif ft in _MD_TYPES:
            groups["md"].append(doc)
        elif ft in _TABULAR_TYPES:
            groups["tabular"].append(doc)
        else:
            groups["text"].append(doc)

    log.info(
        f"split_documents : {len(documents)} doc(s) répartis — "
        f"html={len(groups['html'])}, md={len(groups['md'])}, "
        f"tabular={len(groups['tabular'])}, text={len(groups['text'])}"
    )

    all_chunks: list[Document] = []

    if groups["html"]:
        all_chunks.extend(split_html(groups["html"], chunk_size, chunk_overlap))
    if groups["md"]:
        all_chunks.extend(split_markdown(groups["md"], chunk_size, chunk_overlap))
    if groups["tabular"]:
        all_chunks.extend(split_tabular(groups["tabular"], chunk_size))
    if groups["text"]:
        all_chunks.extend(split_text(groups["text"], chunk_size, chunk_overlap))

    log.info(f"split_documents : total {len(all_chunks)} chunk(s) produit(s)")
    return all_chunks
