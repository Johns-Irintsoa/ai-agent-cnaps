import logging
from pathlib import Path

from langchain_core.documents import Document

from ingestion.Utils import convert_json_to_list
from ingestion.load.UnstructuredLoader import load_html_from_url

log = logging.getLogger(__name__)

# cnaps_urls.json se trouve a la racine du projet (2 niveaux au-dessus de src/ingestion/)
_CNAPS_URLS_PATH = Path(__file__).resolve().parents[2] / "cnaps_urls.json"


def load_web_data() -> list[Document]:
    """
    Charge toutes les pages web CNaPS definies dans cnaps_urls.json.

    Lit le fichier cnaps_urls.json via convert_json_to_list, puis appelle
    load_html_from_url pour chaque entree UrlCnapsWeb.

    Returns:
        Liste de Document LangChain prets pour le chunking et l'indexation.
    """
    url_objects = convert_json_to_list(str(_CNAPS_URLS_PATH))
    log.info(f"load_web_data : {len(url_objects)} URL(s) a charger")

    all_docs: list[Document] = []
    for url_obj in url_objects:
        docs = load_html_from_url(url_obj)
        all_docs.extend(docs)

    log.info(f"load_web_data : {len(all_docs)} document(s) total")
    return all_docs

