"""
Process — Fonctions d'ingestion de donnees CNaPS.
"""

import logging

log = logging.getLogger(__name__)


def from_web() -> dict:
    """
    Lance le pipeline complet de recuperation des donnees web CNaPS.

    Flux :
        cnaps_urls.json (18 URLs)
          -> scrape_all_pages_from_json()
          -> pour chaque document : download + classify + save
          -> data/unstructured/{categorie}/{ftype}/

    Returns:
        dict avec "status" et "message".
    """
    from ingestion.scrap.Scrapper import run_scrapper

    try:
        log.info("Demarrage ingestion web CNaPS...")
        stats = run_scrapper()
        log.info("Ingestion web terminee.")
        msg = (
            f"Ingestion terminee : {stats['docs_saved']} docs + "
            f"{stats['archive_saved']} membres d'archives sauvegardes "
            f"({stats['errors']} erreurs sur {stats['entries']} entrees)."
        )
        return {"status": "success", "message": msg}
    except Exception as e:
        log.error(f"Erreur ingestion web : {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def scrap_from_web() -> dict:
    """
    Telecharge les documents scrapes et les sauvegarde par type de fichier
    sans classification LLM (data/unstructured/{ftype}/{filename}).

    Returns:
        dict avec "status" et "message".
    """
    from ingestion.scrap.Scrapper import download_data_from_web

    try:
        log.info("Demarrage telechargement web CNaPS (sans LLM)...")
        stats = download_data_from_web()
        log.info("Telechargement web termine.")
        msg = (
            f"Telechargement termine : {stats['saved']} fichier(s) sauvegardes "
            f"({stats['errors']} erreurs sur {stats['entries']} entrees)."
        )
        return {"status": "success", "message": msg}
    except Exception as e:
        log.error(f"Erreur telechargement web : {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
