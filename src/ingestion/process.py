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
    from ingestion.scrapper import run_scrapper

    try:
        log.info("Demarrage ingestion web CNaPS...")
        run_scrapper()
        log.info("Ingestion web terminee.")
        return {
            "status": "success",
            "message": "Ingestion web terminee. Voir les logs pour le detail.",
        }
    except Exception as e:
        log.error(f"Erreur ingestion web : {e}")
        return {
            "status": "error",
            "message": str(e),
        }
