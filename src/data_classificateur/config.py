from dataclasses import dataclass, field
import os


@dataclass
class ConfigClassificationWebData:
    llm_base_url: str        = os.getenv("LLM_BASE_URL", "")
    llm_model: str           = os.getenv("LLM_MODEL", "")
    llm_api_key: str         = os.getenv("LLM_API_KEY", "no-key")
    max_text_chars: int      = 2000        # chars envoyés au LLM
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
        "formulaire",
        "tableau",
        "texte",
        "autre",
    ])


CONFIG_CLASSIFICATION = ConfigClassificationWebData()
CONFIG = CONFIG_CLASSIFICATION  # alias utilisé dans data_classificateur/
