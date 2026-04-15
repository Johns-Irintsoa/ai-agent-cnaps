from dataclasses import dataclass, field
import os

@dataclass
class ConfigClassificationWebData:
    ollama_base_url: str     = os.getenv("OLLAMA_BASE_URL", "")
    ollama_model: str        = os.getenv("OLLAMA_MODEL", "")
    max_text_chars: int      = 2000       # chars envoyés au LLM
    request_delay: float     = 0.5        # throttle entre appels LLM (s)
    download_timeout: int    = 60         # timeout HTTP (s)
    stream_max_bytes: int    = 512*1024   # 512 KB max pour fichiers simples
    archive_member_max: int  = 5*1024*1024  # 5 MB max par membre archive
    archive_max_members: int = 3          # max PDF lus par archive
    pages: list = field(default_factory=lambda: [
        "https://www.cnaps.mg/fr/telechargement",
        "https://www.cnaps.mg/fr/document",
        "https://www.cnaps.mg/fr/magazine-cnaps",
    ])
    categories: list = field(default_factory=lambda: [
        "formulaire",   # document avec champs à remplir
        "rapport",      # analyse, bilan, statistiques
        "loi_decret",   # texte réglementaire, code, convention
        "magazine",     # revue, publication périodique CNaPS
        "procedure",    # guide étape par étape
        "notice",       # mode d'emploi, fiche technique
        "autre",
    ])


CONFIG_CLASSIFICATION = ConfigClassificationWebData()
CONFIG = CONFIG_CLASSIFICATION  # alias utilisé dans data_classificateur/
