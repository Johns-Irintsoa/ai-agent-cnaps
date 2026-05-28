import os
from pathlib import Path
from docling.document_converter import DocumentConverter

def _pdf_docling(file_path):
    print(f"Extraction via docling pour : {os.path.basename(file_path)}")
    converter = DocumentConverter()
    # Docling détecte les zones de tableaux et fait l'OCR à l'intérieur
    result = converter.convert(file_path)
    # Retourne le document entier, tableaux inclus, en Markdown propre
    return result.document.export_to_markdown()

