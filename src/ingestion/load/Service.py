import logging
from pathlib import Path

from langchain_core.documents import Document

from ingestion.Utils import convert_json_to_list
from ingestion.load.UnstructuredLoader import load_html_from_url, _load_file, pdf_image_text, _load_pdf_from_dirs

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


_TEST_PDF_PATH = Path(__file__).resolve().parents[3] / "data" / "unstructured" / "pdf" / "coordonneesdesrepresentationscnaps-CNaPS_6932c6799c88a7.04384774.pdf"


def load_file_test() -> list[Document]:
    log.info(f"load_file_test : chargement de {_TEST_PDF_PATH}")
    docs = _load_file(str(_TEST_PDF_PATH))
    log.info(f"load_file_test : {len(docs)} document(s) charges")
    return docs


def load_pdf_with_image() -> list[Document]:
    log.info(f"load_pdf_with_image : OCR sur {_TEST_PDF_PATH}")
    docs = pdf_image_text(str(_TEST_PDF_PATH))
    log.info(f"load_pdf_with_image : {len(docs)} page(s) extraites via OCR")
    return docs

def load_pdf_data(pdf_path: str) -> list[Document]:
    log.info(f"load_pdf_data : chargement de {pdf_path}")
    docs = _load_pdf_from_dirs(pdf_path)
    log.info(f"load_pdf_data : {len(docs)} document(s) charges")
    return docs
