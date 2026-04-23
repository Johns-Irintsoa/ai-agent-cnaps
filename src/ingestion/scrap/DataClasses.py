from dataclasses import dataclass
from typing import Optional

@dataclass
class FileEntry:
    page_url: str        # URL de la page source
    group_title: str     # Titre du groupe (ex: "Cotisations")
    file_label: str      # Libellé du fichier
    file_url: str        # URL de téléchargement absolue
    file_type: str = ""  # pdf / docx / rar / zip / xls / ...

@dataclass
class ClassificationResult:
    page_url: str
    group_title: str
    file_label: str
    file_url: str
    file_type: str
    source_name: str     # nom fichier ou membre archive
    categorie: str
    confiance: float
    raison: str
    extrait: str         # 200 premiers chars du texte
    erreur: Optional[str] = None