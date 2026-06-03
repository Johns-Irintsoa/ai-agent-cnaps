import os
from typing import Optional

from docling.document_converter import DocumentConverter


# ---------------------------------------------------------------------------
# PDF → Markdown
# ---------------------------------------------------------------------------

def _pdf_docling(file_path: str) -> Optional[str]:
    """
    Extrait le contenu d'un PDF (texte + tableaux) en Markdown via Docling.

    Docling détecte les zones de tableaux et fait l'OCR à l'intérieur.

    Args:
        file_path: Chemin absolu vers le fichier PDF.

    Returns:
        Texte Markdown complet ou None si l'extraction échoue.
    """
    print(f"Extraction via Docling pour : {os.path.basename(file_path)}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()
