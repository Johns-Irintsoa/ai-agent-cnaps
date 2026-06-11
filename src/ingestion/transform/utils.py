import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# 1. CONSTANTES & CONFIGURATION POUR LE CHUNKING
# ---------------------------------------------------------------------------

# Séparateurs utilisés pour les documents légaux malgaches / francophones
LEGAL_SEPARATORS = [
    "\nArticle ",  # Chaque article du décret
    "\nArt\. ",  # Variante abrégée
    "\n- Vu ",  # Préambule "Vu la loi…"
    "\nDECRETE",  # Bloc DECRETE
    "\nANNEXE",  # Bloc Annexe
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    " ",
    "",
]

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Patterns regex pour détecter les blocs de signature et de tampon
SIGNATURE_PATTERNS = [
    r"Fait à .+?le \d{1,2}\s+\w+\s+\d{4}",  # "Fait à Antananarivo, le 05 mai 2015"
    r"Antananarivo,?\s+le\s+\d{1,2}\s+\w+\s+\d{4}",  # "Antananarivo, le 22 MAY 2015"
    r"PAR LE PREMIER MINISTRE[\s\S]{0,300}?(?=\n\n|\Z)",  # Bloc Premier Ministre
    r"Pour ampliation conforme[\s\S]{0,400}?(?=\n\n|\Z)",  # Bloc ampliation + tampon
    r"LE SECRETAIRE GENERAL[\s\S]{0,200}?(?=\n\n|\Z)",  # Secrétaire Général
]

# ---------------------------------------------------------------------------
# 2. DÉTECTION DES BLOCS DE SIGNATURE / TAMPON
# ---------------------------------------------------------------------------


def extract_signature_blocks(md_text: str) -> List[str]:
    """
    Extrait les blocs de signature et de tampon du texte Markdown.

    Ces blocs contiennent des informations critiques (dates de signature,
    noms des signataires, dates de tampon) qui doivent être préservées
    comme chunks atomiques pour éviter qu'elles soient noyées ou coupées.

    Args:
        md_text: Contenu Markdown complet du document.

    Returns:
        Liste des blocs de signature trouvés (texte brut).
    """
    extracted = []
    for pattern in SIGNATURE_PATTERNS:
        matches = re.findall(pattern, md_text, re.DOTALL | re.IGNORECASE)
        extracted.extend([m.strip() for m in matches if m.strip()])
    return extracted


def is_signature_chunk(content: str, signature_blocks: List[str]) -> bool:
    """
    Vérifie si un chunk contient un bloc de signature/tampon extrait.

    Args:
        content:          Contenu du chunk à tester.
        signature_blocks: Liste des blocs de signature extraits.

    Returns:
        True si le chunk contient au moins un bloc de signature.
    """
    return any(sig[:50] in content for sig in signature_blocks if len(sig) >= 10)


# ---------------------------------------------------------------------------
# 3. SÉPARATION TABLEAUX / TEXTE
# ---------------------------------------------------------------------------


def split_tables_from_text(md_text: str) -> List[Dict[str, str]]:
    """
    Sépare les tableaux Markdown du texte courant.

    Les tableaux (ex: grilles de salaires) sont atomiques : ils ne doivent
    jamais être coupés entre deux chunks pour conserver leur cohérence.

    Args:
        md_text: Contenu Markdown complet.

    Returns:
        Liste ordonnée de dicts {"type": "table"|"text", "content": str}.
    """
    # Un bloc tableau = lignes qui commencent et finissent par |
    table_pattern = re.compile(r"((?:\|.+\|\n?)+)", re.MULTILINE)

    parts = []
    last_end = 0

    for match in table_pattern.finditer(md_text):
        # Texte avant le tableau
        before = md_text[last_end : match.start()].strip()
        if before:
            parts.append({"type": "text", "content": before})

        # Tableau complet = 1 chunk atomique
        parts.append({"type": "table", "content": match.group(0).strip()})
        last_end = match.end()

    # Texte restant après le dernier tableau
    remaining = md_text[last_end:].strip()
    if remaining:
        parts.append({"type": "text", "content": remaining})

    return parts


# ---------------------------------------------------------------------------
# 4. DÉCOUPAGE PAR HEADERS MARKDOWN
# ---------------------------------------------------------------------------


def split_by_headers(text: str) -> List[Document]:
    """
    Découpe le texte selon la hiérarchie des titres Markdown (#, ##, ###).

    Permet de garder la cohérence sémantique en regroupant d'abord
    le contenu par section avant de le redécouper par taille.

    Args:
        text: Texte Markdown à découper.

    Returns:
        Liste de Documents LangChain avec métadonnées de headers.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# 5. DÉCOUPAGE PAR TAILLE (SÉCURITÉ)
# ---------------------------------------------------------------------------


def split_by_size(
    documents: List[Document],
    max_chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Redécoupe les Documents trop longs en respectant les séparateurs légaux.

    Utilise LEGAL_SEPARATORS pour ne pas couper brutalement au milieu
    d'un article ou d'une phrase juridique.

    Args:
        documents:      Liste de Documents issus du découpage par headers.
        max_chunk_size: Taille max en caractères par chunk.
        chunk_overlap:  Chevauchement en caractères entre chunks consécutifs.

    Returns:
        Liste de Documents redécoupés si nécessaire.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=LEGAL_SEPARATORS,
        length_function=len,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 6. DÉCOUPAGE SPÉCIAL : BLOCS DE SIGNATURE
# ---------------------------------------------------------------------------


def split_signature_sections(md_text: str) -> List[Document]:
    """
    Isole les sections de signature en chunks dédiés.

    Chaque bloc de signature (date de signature, signataires, tampon)
    est transformé en un Document atomique avec métadonnée is_signature=True.

    Args:
        md_text: Texte Markdown complet.

    Returns:
        Liste de Documents correspondant aux blocs de signature.
    """
    docs = []
    for pattern in SIGNATURE_PATTERNS:
        for match in re.finditer(pattern, md_text, re.DOTALL | re.IGNORECASE):
            content = match.group(0).strip()
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"is_signature": True},
                    )
                )
    return docs


# ---------------------------------------------------------------------------
# 7. CONSTRUCTION DES CHUNKS FINAUX
# ---------------------------------------------------------------------------


def build_chunk_item(
    content: str,
    metadata_extra: Dict[str, Any],
    current_index: int,
    final_json_list: List[Dict],
    filename: str,
    signature_blocks: List[str],
) -> Dict[str, Any]:
    """
    Construit un item chunk structuré avec toutes ses métadonnées.

    Args:
        content:          Contenu textuel du chunk.
        metadata_extra:   Métadonnées supplémentaires (headers, is_table…).
        current_index:    Index courant dans la liste finale.
        final_json_list:  Liste des chunks déjà construits (pour liens prev/next).
        filename:         Nom du fichier source.
        signature_blocks: Liste des blocs de signature pour détection.

    Returns:
        Dict représentant le chunk avec toutes ses métadonnées.
    """
    chunk_id = str(uuid.uuid4())

    # Reconstruction de la hiérarchie depuis les métadonnées de headers
    hierarchy = " > ".join(
        [val for key, val in metadata_extra.items() if "Header" in key]
    )

    prev_id: Optional[str] = (
        final_json_list[current_index - 1]["chunk_id"] if current_index > 0 else None
    )

    return {
        "chunk_id": chunk_id,
        "content": content,
        "metadata": {
            "source": filename,
            "page_numbers": [],
            "creation_date": datetime.now().isoformat(),
            "hierarchical_context": hierarchy or "Root",
            "chunk_index": current_index + 1,
            "is_table": metadata_extra.get("is_table", False),
            "is_signature_block": (
                metadata_extra.get("is_signature", False)
                or is_signature_chunk(content, signature_blocks)
            ),
            "contains_table": "|" in content,
            "previous_chunk_id": prev_id,
            "next_chunk_id": None,  # mis à jour au prochain tour
        },
    }


def link_chunks(final_json_list: List[Dict]) -> List[Dict]:
    """
    Met à jour les liens next_chunk_id entre chunks consécutifs
    et ajoute total_chunks à tous les items.

    Args:
        final_json_list: Liste des chunks sans liens next.

    Returns:
        Liste des chunks avec liens prev/next complets et total_chunks.
    """
    total = len(final_json_list)
    for i, item in enumerate(final_json_list):
        item["metadata"]["total_chunks"] = total
        if i < total - 1:
            item["metadata"]["next_chunk_id"] = final_json_list[i + 1]["chunk_id"]
    return final_json_list


# ---------------------------------------------------------------------------
# 8. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------


def process_text_part(
    text: str,
    max_chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Pipeline de découpage pour une partie textuelle (non-tableau).

    Étapes :
      1. Découpage par headers Markdown
      2. Redécoupage par taille avec séparateurs légaux

    Args:
        text:           Texte Markdown à traiter.
        max_chunk_size: Taille max par chunk.
        chunk_overlap:  Chevauchement entre chunks.

    Returns:
        Liste de Documents prêts à être transformés en items JSON.
    """
    header_splits = split_by_headers(text)
    return split_by_size(header_splits, max_chunk_size, chunk_overlap)


def clean_ocr_text(text: str) -> str:
    """
    Nettoie les artefacts OCR courants dans les PDFs scannés.
    Ajoute les espaces manquants autour des dates et mots collés.
    """
    text = re.sub(r"(\d{1,2})([A-Z]{3})(\d{4})", r"\1 \2 \3", text)

    text = re.sub(r"(\w),le(\d)", r"\1, le \2", text)

    text = re.sub(r",le\s+", ", le ", text)

    return text


# ---------------------------------------------------------------------------
# 9. HELPERS POUR LE PIPELINE HTML (chunking_md_html)
# ---------------------------------------------------------------------------


def normalize_md_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def flatten_metadata(page) -> Dict[str, Any]:
    return {
        "source_url": page.metadata.source_url,
        "title": page.metadata.title,
        "date_posted": page.metadata.date_posted,
    }


def split_by_tokens(
    documents: List[Document],
    max_tokens: int = 512,
    overlap_tokens: int = 77,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def prefix_chunk_with_context(chunk_text: str, title: str) -> str:
    return f"Document : {title}\nContenu : {chunk_text}"


def generate_chunk_id(source_url: str, index: int) -> str:
    return f"{source_url}#chunk{index}"

def get_class_contained(str1: str, str2: str) -> bool:
    return str1.lower() in str2.lower()


# ---------------------------------------------------------------------------
# 10. CONVERSION DE DATE HTML → ISO
# ---------------------------------------------------------------------------

_FRENCH_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2,
    "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12,
}


def convert_date_from_html(raw: Optional[str]) -> Optional[str]:
    """
    Convertit une date HTML brute française en ISO YYYY-MM-DD.
    "Publié le vendredi 31 mars 2023" → "2023-03-31"

    Le regex cible directement le triplet jour/mois/année — les préfixes
    ("Publié le") et noms de jours sont ignorés naturellement par le moteur.
    Retourne None si la conversion échoue.
    """
    if not raw:
        return None
    text = re.sub(r"\s+", " ", raw).strip()
    match = re.search(r"(\d{1,2})\s+([a-zéûôîèùàâêïü]+)\s+(\d{4})", text, re.IGNORECASE)
    if not match:
        return None
    day, month_fr, year = match.groups()
    month_num = _FRENCH_MONTHS.get(month_fr.lower())
    if not month_num:
        return None
    try:
        from datetime import date as _date
        return _date(int(year), month_num, int(day)).isoformat()
    except ValueError:
        return None


def _fragments_to_markdown(frag_soup: BeautifulSoup) -> str:
    lines = []
    for el in frag_soup.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol"]):
        if el.find_parent(["ul", "ol"]):
            continue
        text = el.get_text(strip=True)
        if not text:
            continue
        tag = el.name
        if tag == "h1":
            lines.append(f"# {text}")
        elif tag == "h2":
            lines.append(f"## {text}")
        elif tag == "h3":
            lines.append(f"### {text}")
        elif tag == "h4":
            lines.append(f"#### {text}")
        elif tag == "p":
            lines.append(text)
        elif tag in ("ul", "ol"):
            for li in el.find_all("li"):
                lines.append(f"- {li.get_text(strip=True)}")
    return "\n\n".join(lines)

